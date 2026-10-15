#!/usr/bin/env python3
"""
07_qc_plddt_pocket.py
=====================

Control de calidad de los modelos de AlphaFold de los 33 homólogos de
T. cruzi del pipeline PocketVec, **residuo por residuo y restringido
a los residuos del bolsillo**, no sólo el pLDDT global del modelo.

Motivación
----------
El pLDDT promedio de toda la cadena esconde el caso que importa: un modelo
con buen promedio global pero con loops flexibles y poco confiables justo
alrededor del sitio activo. Si el bolsillo está mal modelado, la geometría
que después alimenta a rbcavity/rDock (y por lo tanto el fingerprint
PocketVec) no vale, por más que el promedio global se vea bien.

De dónde sale cada cosa (todo ya definido en el pipeline, no se inventa nada)
-----------------------------------------------------------------------------
  - Lista de proteínas : paths.qc_targets_csv (config.yaml), columnas
                         cruzi_af_code / cruzi_uniprot / pair_id.
  - Modelo AF          : <paths.preparation_dir>/<code>/raw/<code>.pdb,
                         el mismo archivo que baja
                         pipeline_steps.fetch_alphafold_structure().
  - pLDDT por residuo  : JSON de confianza de AlphaFold DB
                         (AF-<uniprot>-F1-confidence_v<N>.json), resuelto por
                         la API de AFDB igual que _query_afdb_api(). Si no se
                         puede bajar, se cae a la columna B-factor del PDB de
                         AF, que es donde AlphaFold escribe el mismo pLDDT.
                         La columna plddt_fuente dice cuál se usó en cada fila.
  - PAE                : JSON de PAE de AlphaFold DB
                         (AF-<uniprot>-F1-predicted_aligned_error_v<N>.json).
                         No está en el PDB, así que si no se puede bajar la
                         fila queda con PAE vacío y se dice explícitamente.
  - Residuos del bolsillo : el bolsillo es el MISMO que ya elige el pipeline
                         para los AF de cruzi -- pipeline_steps.
                         select_best_pocket_centroid(criterion="druggability"),
                         o sea el pocket de fpocket con mayor Druggability
                         Score. Sus residuos se leen de
                         processed/<code>_prepared_out/pockets/pocket<N>_atm.pdb.
                         La numeración de residuos de <code>_prepared.pdb es
                         idéntica a la del raw de AF (verificado en las 32
                         estructuras), así que los residuos del bolsillo
                         mapean directo sobre el pLDDT por residuo.

Los JSON bajados se cachean al lado del PDB, en
<paths.preparation_dir>/<code>/raw/, con el nombre original de AFDB. Una
segunda corrida no vuelve a bajar nada.

Banderas rojas (ajustables por CLI o editando las constantes de abajo)
----------------------------------------------------------------------
  1. pLDDT promedio del bolsillo  < 70        (--plddt-pocket-min)
  2. (pLDDT global - pLDDT bolsillo) > 15 pts (--diff-max)
  3. PAE promedio intra-bolsillo  > 5 A       (--pae-max)

Clasificación:
  OK        : ninguna bandera.
  REVISAR   : exactamente una bandera, y no es la del pLDDT del bolsillo.
  DESCARTAR : se disparó la bandera 1 (el bolsillo en sí es poco confiable),
              o se dispararon 2 o más banderas a la vez.

Uso
---
    python src/07_qc_plddt_pocket.py

    # umbrales distintos
    python src/07_qc_plddt_pocket.py --plddt-pocket-min 75 --diff-max 10

    # sin red (usa lo cacheado + B-factor; el PAE va a quedar pendiente)
    python src/07_qc_plddt_pocket.py --no-download

Salidas (en <paths.analysis_dir>/)
---------------------------------
    qc_plddt_pocket.csv            tabla principal, peor -> mejor por diff
    qc_plddt_pocket_pendientes.csv proteínas sin JSON o sin bolsillo definido
    qc_plddt_pocket_residuos.csv   detalle residuo por residuo del bolsillo
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import urllib.request
from pathlib import Path

from utils import config

# --------------------------------------------------------------------------
# Banderas rojas -- constantes ajustables. Son criterios de trabajo, no
# verdades fijas: cambiarlos acá (o por CLI) y volver a correr es barato,
# el script no recalcula nada de fpocket ni vuelve a bajar los JSON.
# --------------------------------------------------------------------------
PLDDT_POCKET_MIN = 70.0    # pLDDT promedio del bolsillo por debajo de esto -> rojo
DIFF_GLOBAL_POCKET_MAX = 15.0   # global - bolsillo por encima de esto -> rojo
PAE_POCKET_MAX = 5.0       # PAE promedio intra-bolsillo por encima de esto -> rojo

# Umbral informativo, NO clasifica: residuo de bolsillo por debajo de esto se
# menciona en el comentario (pLDDT < 50 es la franja "muy baja" de AlphaFold).
PLDDT_RESIDUE_VERYLOW = 50.0

# Criterio de elección de bolsillo -- el mismo que 05_run_pocketvec_batch.
# POCKET_SELECTION_CRITERION para los AF de cruzi (que nunca traen ligando).
POCKET_SELECTION_CRITERION = "druggability"

AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
USER_AGENT = "Mozilla/5.0"   # mismo header que usa pipeline_steps._query_afdb_api


# --------------------------------------------------------------------------
# Lectura de artefactos locales
# --------------------------------------------------------------------------
def read_fpocket_info(path: Path) -> dict[int, dict[str, float]]:
    """
    Mismo parseo que utils.pocketvec_utils.read_fpocket_scores, reescrito acá
    para no arrastrar la dependencia de scipy (pocketvec_utils la importa en
    el tope del módulo y este script no necesita nada de scipy).
    """
    pockets: dict[int, dict[str, float]] = {}
    current = None
    for line in path.read_text().splitlines():
        if "Pocket " in line:
            current = int(line.split(":")[0].split("Pocket")[1].strip())
            pockets[current] = {}
        elif line.strip() and current is not None:
            key, _, value = line.partition(":")
            try:
                pockets[current][key.strip()] = float(value.strip())
            except ValueError:
                continue
    return pockets


def select_best_pocket(pockets: dict[int, dict[str, float]]) -> int:
    """
    Igual que pipeline_steps.select_best_pocket_centroid(criterion=
    "druggability"): el de mayor Druggability Score. Empate -> el de menor id
    (fpocket ya los entrega ordenados por su propio Score).
    """
    return max(pockets, key=lambda i: (pockets[i].get("Druggability Score", 0.0), -i))


def read_pocket_residues(path: Path) -> list[tuple[str, int]]:
    """Residuos (cadena, resseq) tocados por el bolsillo, según pocketN_atm.pdb."""
    residues: list[tuple[str, int]] = []
    seen = set()
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], int(line[22:26]))
            if key not in seen:
                seen.add(key)
                residues.append(key)
    return residues


def read_plddt_from_pdb(path: Path) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], str]]:
    """
    pLDDT por residuo desde la columna B-factor del PDB de AlphaFold.
    AlphaFold escribe el mismo valor en todos los átomos del residuo; se
    promedia igual por robustez. Devuelve además el resname por residuo.
    """
    acc: dict[tuple[str, int], list[float]] = {}
    names: dict[tuple[str, int], str] = {}
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            key = (line[21], int(line[22:26]))
            acc.setdefault(key, []).append(float(line[60:66]))
            names.setdefault(key, line[17:20].strip())
    return {k: sum(v) / len(v) for k, v in acc.items()}, names


# --------------------------------------------------------------------------
# AlphaFold DB: resolución de URLs y descarga cacheada
# --------------------------------------------------------------------------
def query_afdb(uniprot: str) -> dict | None:
    """
    Le pregunta a AFDB por la entrada vigente, igual que
    pipeline_steps._query_afdb_api() -- así no hay que adivinar el número de
    versión (v3, v4, v6...) a mano.
    """
    req = urllib.request.Request(AFDB_API.format(uniprot=uniprot),
                                 headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        print(f"    [warn] API de AFDB falló para {uniprot}: {exc}")
        return None
    return data[0] if data else None


def fetch_json_cached(url: str, dest: Path, allow_download: bool) -> dict | list | None:
    """Baja url a dest si no está cacheado. Devuelve el JSON parseado o None."""
    if dest.exists():
        try:
            return json.loads(dest.read_text())
        except Exception:
            dest.unlink()   # cache corrupto (descarga cortada): se rehace
    if not allow_download:
        return None
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
    except Exception as exc:
        print(f"    [warn] no pude bajar {url}: {exc}")
        return None
    dest.write_bytes(payload)
    return json.loads(payload.decode())


def parse_confidence_json(doc) -> list[float] | None:
    """pLDDT por residuo del JSON de confianza de AFDB."""
    if isinstance(doc, list):
        doc = doc[0] if doc else {}
    for key in ("confidenceScore", "plddt", "pLDDT"):
        if key in doc:
            return [float(x) for x in doc[key]]
    return None


def parse_pae_json(doc) -> list[list[float]] | None:
    """Matriz de PAE del JSON de AFDB (formato v4+: predicted_aligned_error)."""
    if isinstance(doc, list):
        doc = doc[0] if doc else {}
    if "predicted_aligned_error" in doc:
        return doc["predicted_aligned_error"]
    # Formato viejo (v1/v2): listas planas residue1/residue2/distance.
    if {"residue1", "residue2", "distance"} <= set(doc):
        n = max(doc["residue1"])
        mat = [[0.0] * n for _ in range(n)]
        for i, j, d in zip(doc["residue1"], doc["residue2"], doc["distance"]):
            mat[i - 1][j - 1] = float(d)
        return mat
    return None


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def mean_intra_pocket_pae(pae: list[list[float]], indices: list[int]) -> float | None:
    """
    PAE promedio entre todos los pares ORDENADOS i != j de residuos del
    bolsillo. La matriz de PAE no es simétrica (PAE[i][j] = error esperado en
    la posición de i cuando se alinea por j), así que se promedian las dos
    direcciones de cada par, que es justamente lo que se quiere acá: qué tan
    confiable es la posición relativa entre las partes del bolsillo.
    """
    vals = [pae[i][j] for i in indices for j in indices if i != j]
    return sum(vals) / len(vals) if vals else None


def classify(plddt_pocket: float, diff: float, pae_pocket: float | None,
             plddt_min: float, plddt_pocket_min: float, diff_max: float,
             pae_max: float) -> tuple[str, list[str]]:
    """Devuelve (flag, lista de motivos). Ver docstring del módulo."""
    reasons: list[str] = []
    low_pocket = plddt_pocket < plddt_pocket_min
    if low_pocket:
        reasons.append(f"pLDDT bolsillo {plddt_pocket:.1f} < {plddt_pocket_min:g}")
    if diff > diff_max:
        reasons.append(f"caida global-bolsillo {diff:.1f} > {diff_max:g} pts")
    if pae_pocket is not None and pae_pocket > pae_max:
        reasons.append(f"PAE intra-bolsillo {pae_pocket:.2f} A > {pae_max:g} A")

    if not reasons:
        flag = "OK"
    elif low_pocket or len(reasons) >= 2:
        flag = "DESCARTAR"
    else:
        flag = "REVISAR"
    return flag, reasons


# --------------------------------------------------------------------------
# Corrida por proteína
# --------------------------------------------------------------------------
def process_row(row: dict, preparation_dir: Path, allow_download: bool,
                thresholds: dict) -> tuple[dict | None, dict | None, list[dict]]:
    """
    Devuelve (fila_resultado, fila_pendiente, filas_residuo).
    fila_pendiente no es None cuando falta algo que impide completar la fila
    (no se omite en silencio: o sale en la tabla, o sale en pendientes).
    """
    code = row["cruzi_af_code"].strip()
    uniprot = row["cruzi_uniprot"].strip()
    pair_id = row.get("pair_id", "").strip()
    name = row.get("cruzi_name", "").strip()

    prot_dir = preparation_dir / code
    raw_pdb = prot_dir / "raw" / f"{code}.pdb"
    fp_out = prot_dir / "processed" / f"{code}_prepared_out"
    info_txt = fp_out / f"{code}_prepared_info.txt"

    def pendiente(motivo: str) -> dict:
        return {"protein_id": code, "pair_id": pair_id, "cruzi_uniprot": uniprot,
                "cruzi_name": name, "motivo": motivo}

    if not raw_pdb.exists():
        return None, pendiente(f"falta el modelo AF: {raw_pdb.relative_to(preparation_dir)}"), []
    if not info_txt.exists():
        return None, pendiente(f"falta la salida de fpocket: {fp_out.relative_to(preparation_dir)}"), []

    pockets = read_fpocket_info(info_txt)
    if not pockets:
        return None, pendiente(f"fpocket no reportó ningún pocket en {info_txt.name}"), []
    best_id = select_best_pocket(pockets)
    drugg = pockets[best_id].get("Druggability Score")
    atm = fp_out / "pockets" / f"pocket{best_id}_atm.pdb"
    if not atm.exists():
        return None, pendiente(f"falta la definición de residuos del bolsillo: {atm.name}"), []

    pocket_res = read_pocket_residues(atm)
    if not pocket_res:
        return None, pendiente(f"{atm.name} no tiene residuos"), []

    # --- pLDDT: primero el JSON de AFDB, con el B-factor del PDB como respaldo
    plddt_bfac, resnames = read_plddt_from_pdb(raw_pdb)
    chain_res = sorted(plddt_bfac)            # orden de la cadena, 1..N
    n_res = len(chain_res)

    notas: list[str] = []
    plddt_source = "B-factor del PDB de AF"
    plddt_by_res = dict(plddt_bfac)
    entry = None
    raw_dir = prot_dir / "raw"

    # AF-0000000066513361 es una predicción propia (AlphaFold Server), no una
    # entrada de AFDB: el JSON de AFDB para su UniProt es OTRA corrida. Se baja
    # igual (sirve de referencia) pero no se usa para el pLDDT, y el PAE queda
    # marcado como proveniente de un modelo distinto.
    es_modelo_propio = not code.startswith(f"AF-{uniprot}")

    if uniprot:
        entry = query_afdb(uniprot) if allow_download else None
        conf_url = (entry or {}).get("plddtDocUrl")
        if conf_url:
            doc = fetch_json_cached(conf_url, raw_dir / Path(conf_url).name, allow_download)
            plddt_list = parse_confidence_json(doc) if doc else None
            if plddt_list and len(plddt_list) == n_res:
                from_json = {k: plddt_list[i] for i, k in enumerate(chain_res)}
                delta = max(abs(from_json[k] - plddt_bfac[k]) for k in chain_res)
                if es_modelo_propio:
                    notas.append(f"modelo local es predicción propia; se usa el B-factor "
                                 f"(JSON de AFDB para {uniprot} difiere hasta {delta:.1f} pts)")
                else:
                    plddt_by_res = from_json
                    plddt_source = f"JSON AFDB {Path(conf_url).name}"
                    if delta > 1.0:
                        notas.append(f"JSON de AFDB y B-factor difieren hasta {delta:.1f} pts")
            elif plddt_list:
                notas.append(f"JSON de confianza de AFDB tiene {len(plddt_list)} residuos "
                             f"y el modelo local {n_res}: se usa el B-factor")

    if plddt_source.startswith("B-factor") and not notas:
        notas.append("sin JSON de confianza de AFDB; pLDDT leído del B-factor del PDB de AF")

    # --- PAE
    pae_pocket = None
    pae_source = ""
    pae_pendiente = None
    pae_url = (entry or {}).get("paeDocUrl")
    if pae_url:
        doc = fetch_json_cached(pae_url, raw_dir / Path(pae_url).name, allow_download)
        mat = parse_pae_json(doc) if doc else None
        if mat is None:
            pae_pendiente = f"no pude leer el JSON de PAE ({Path(pae_url).name})"
        elif len(mat) != n_res:
            pae_pendiente = (f"matriz de PAE de {len(mat)} residuos vs {n_res} del modelo "
                             f"local: no se puede mapear el bolsillo")
        else:
            idx = [chain_res.index(r) for r in pocket_res if r in plddt_by_res]
            pae_pocket = mean_intra_pocket_pae(mat, idx)
            pae_source = Path(pae_url).name
            if es_modelo_propio:
                notas.append(f"PAE proviene de AF-{uniprot}-F1 (AFDB), no del modelo local: "
                             f"misma longitud pero es otra corrida de predicción")
    else:
        pae_pendiente = ("sin acceso al PAE de AFDB"
                         + ("" if allow_download else " (--no-download)"))

    # --- Métricas
    pocket_plddts = [plddt_by_res[r] for r in pocket_res if r in plddt_by_res]
    faltantes = len(pocket_res) - len(pocket_plddts)
    if not pocket_plddts:
        return None, pendiente("ningún residuo del bolsillo mapeó contra el modelo"), []
    if faltantes:
        notas.append(f"{faltantes} residuo(s) del bolsillo sin pLDDT en el modelo")

    plddt_global = statistics.fmean(plddt_by_res.values())
    plddt_pocket = statistics.fmean(pocket_plddts)
    plddt_min = min(pocket_plddts)
    diff = plddt_global - plddt_pocket

    flag, reasons = classify(plddt_pocket, diff, pae_pocket, plddt_min,
                             thresholds["plddt_pocket_min"], thresholds["diff_max"],
                             thresholds["pae_max"])

    n_verylow = sum(1 for v in pocket_plddts if v < PLDDT_RESIDUE_VERYLOW)
    if n_verylow:
        notas.append(f"{n_verylow}/{len(pocket_plddts)} residuos del bolsillo "
                     f"con pLDDT < {PLDDT_RESIDUE_VERYLOW:g}")
    if pae_pendiente:
        notas.append(pae_pendiente)

    comentario = "; ".join(reasons + notas) if (reasons or notas) else "sin observaciones"

    resultado = {
        "protein_id": code,
        "pLDDT_global": round(plddt_global, 2),
        "pLDDT_bolsillo": round(plddt_pocket, 2),
        "pLDDT_min_bolsillo": round(plddt_min, 2),
        "diff": round(diff, 2),
        "PAE_bolsillo": "" if pae_pocket is None else round(pae_pocket, 3),
        "flag": flag,
        "comentario": comentario,
        # contexto, después de las columnas pedidas
        "pair_id": pair_id,
        "target_name": row.get("target_name", "").strip(),
        "cruzi_uniprot": uniprot,
        "cruzi_name": name,
        "n_res_modelo": n_res,
        "pocket_id_fpocket": best_id,
        "pocket_druggability": drugg,
        "n_res_bolsillo": len(pocket_plddts),
        "plddt_fuente": plddt_source,
        "pae_fuente": pae_source,
    }

    residuo_rows = [
        {"protein_id": code, "pair_id": pair_id, "chain": ch, "resseq": num,
         "resname": resnames.get((ch, num), ""), "plddt": round(plddt_by_res[(ch, num)], 2)}
        for (ch, num) in pocket_res if (ch, num) in plddt_by_res
    ]
    return resultado, None, residuo_rows


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", default=config.QC_TARGETS_CSV,
                        help="CSV con la lista de proteínas (default: paths.qc_targets_csv).")
    parser.add_argument("--preparation-dir", default=config.PREPARATION_DIR,
                        help="Per-protein preparation folder (default: paths.preparation_dir).")
    parser.add_argument("--out-dir", default=config.ANALYSIS_DIR,
                        help="Carpeta de salida (default: paths.analysis_dir).")
    parser.add_argument("--plddt-pocket-min", type=float, default=PLDDT_POCKET_MIN)
    parser.add_argument("--diff-max", type=float, default=DIFF_GLOBAL_POCKET_MAX)
    parser.add_argument("--pae-max", type=float, default=PAE_POCKET_MAX)
    parser.add_argument("--no-download", action="store_true",
                        help="No tocar la red: usa sólo JSON ya cacheados y el B-factor.")
    args = parser.parse_args()

    preparation_dir = Path(args.preparation_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets_csv = Path(args.targets).expanduser().resolve()

    thresholds = {"plddt_pocket_min": args.plddt_pocket_min,
                  "diff_max": args.diff_max, "pae_max": args.pae_max}

    print("QC de pLDDT/PAE en residuos de bolsillo -- modelos AF de T. cruzi")
    print(f"Prep dir:  {preparation_dir}")
    print(f"Lista:     {targets_csv}")
    print(f"Bolsillo:  pocket de fpocket con mayor Druggability Score "
          f"(criterio '{POCKET_SELECTION_CRITERION}', igual que 05_run_pocketvec_batch)")
    print(f"Umbrales:  pLDDT bolsillo < {args.plddt_pocket_min:g} | "
          f"diff > {args.diff_max:g} pts | PAE > {args.pae_max:g} A")
    print(f"Red:       {'deshabilitada (--no-download)' if args.no_download else 'habilitada'}\n")

    if not targets_csv.exists():
        sys.exit(f"[error] no encuentro {targets_csv}")

    with targets_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Proteínas en la lista: {len(rows)}\n")

    resultados, pendientes, residuos = [], [], []
    for i, row in enumerate(rows, 1):
        code = row["cruzi_af_code"].strip()
        print(f"[{i:2d}/{len(rows)}] {code} ({row.get('pair_id','')})")
        res, pend, res_rows = process_row(row, preparation_dir, not args.no_download, thresholds)
        if res:
            resultados.append(res)
            residuos.extend(res_rows)
            print(f"    global {res['pLDDT_global']:.1f} | bolsillo {res['pLDDT_bolsillo']:.1f} "
                  f"(min {res['pLDDT_min_bolsillo']:.1f}, n={res['n_res_bolsillo']}) | "
                  f"diff {res['diff']:+.1f} | PAE {res['PAE_bolsillo'] or 'n/d'} -> {res['flag']}")
        else:
            pendientes.append(pend)
            print(f"    [pendiente] {pend['motivo']}")

    # Peor -> mejor por caída global-bolsillo.
    resultados.sort(key=lambda r: r["diff"], reverse=True)

    cols = ["protein_id", "pLDDT_global", "pLDDT_bolsillo", "pLDDT_min_bolsillo",
            "diff", "PAE_bolsillo", "flag", "comentario",
            "pair_id", "target_name", "cruzi_uniprot", "cruzi_name", "n_res_modelo",
            "pocket_id_fpocket", "pocket_druggability", "n_res_bolsillo",
            "plddt_fuente", "pae_fuente"]
    main_csv = out_dir / "qc_plddt_pocket.csv"
    with main_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(resultados)
    print(f"\n[ok] {main_csv}")

    pend_csv = out_dir / "qc_plddt_pocket_pendientes.csv"
    with pend_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["protein_id", "pair_id", "cruzi_uniprot",
                                           "cruzi_name", "motivo"])
        w.writeheader()
        w.writerows(pendientes)
    print(f"[ok] {pend_csv} ({len(pendientes)} pendiente(s))")

    res_csv = out_dir / "qc_plddt_pocket_residuos.csv"
    with res_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["protein_id", "pair_id", "chain", "resseq",
                                           "resname", "plddt"])
        w.writeheader()
        w.writerows(residuos)
    print(f"[ok] {res_csv} ({len(residuos)} residuos de bolsillo)")

    # ---- Tabla en consola
    print("\n" + "=" * 118)
    print("TABLA (peor -> mejor por caída global - bolsillo)")
    print("=" * 118)
    head = (f"{'protein_id':<22}{'global':>8}{'bolsillo':>10}{'min':>8}"
            f"{'diff':>8}{'PAE':>8}  {'flag':<10}comentario")
    print(head)
    print("-" * 118)
    for r in resultados:
        pae = f"{r['PAE_bolsillo']:.2f}" if r["PAE_bolsillo"] != "" else "n/d"
        com = r["comentario"]
        print(f"{r['protein_id']:<22}{r['pLDDT_global']:>8.1f}{r['pLDDT_bolsillo']:>10.1f}"
              f"{r['pLDDT_min_bolsillo']:>8.1f}{r['diff']:>8.1f}{pae:>8}  {r['flag']:<10}"
              f"{com[:44]}")

    if pendientes:
        print("\nPENDIENTES (no entraron en la tabla):")
        for p in pendientes:
            print(f"  - {p['protein_id']} ({p['pair_id']}): {p['motivo']}")

    sin_pae = [r for r in resultados if r["PAE_bolsillo"] == ""]
    if sin_pae:
        print(f"\nSin PAE disponible ({len(sin_pae)}): "
              + ", ".join(r["protein_id"] for r in sin_pae))

    # ---- Resumen
    n_ok = sum(1 for r in resultados if r["flag"] == "OK")
    n_rev = sum(1 for r in resultados if r["flag"] == "REVISAR")
    n_des = sum(1 for r in resultados if r["flag"] == "DESCARTAR")
    print("\n" + "=" * 118)
    print("RESUMEN")
    print("=" * 118)
    print(f"{len(resultados)} de {len(rows)} proteínas evaluadas "
          f"({len(pendientes)} pendiente(s)): {n_ok} OK, {n_rev} a REVISAR, {n_des} a DESCARTAR.")
    print(f"Umbrales usados: pLDDT bolsillo < {args.plddt_pocket_min:g}, "
          f"caída global-bolsillo > {args.diff_max:g} pts, "
          f"PAE intra-bolsillo > {args.pae_max:g} A.")
    if resultados:
        peor_diff = resultados[0]
        peor_pocket = min(resultados, key=lambda r: r["pLDDT_bolsillo"])
        print(f"Mayor caída global->bolsillo: {peor_diff['protein_id']} "
              f"({peor_diff['cruzi_name'] or peor_diff['target_name']}), "
              f"{peor_diff['pLDDT_global']:.1f} -> {peor_diff['pLDDT_bolsillo']:.1f} "
              f"({peor_diff['diff']:+.1f} pts) [{peor_diff['flag']}].")
        print(f"Bolsillo menos confiable en absoluto: {peor_pocket['protein_id']} "
              f"({peor_pocket['cruzi_name'] or peor_pocket['target_name']}), "
              f"pLDDT bolsillo {peor_pocket['pLDDT_bolsillo']:.1f} "
              f"(mín {peor_pocket['pLDDT_min_bolsillo']:.1f}) [{peor_pocket['flag']}].")


if __name__ == "__main__":
    main()
