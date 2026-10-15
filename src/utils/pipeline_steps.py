"""
Funciones reutilizables de preparación (ex 1_preparation.py), caracterización
de ligandos (ex Caracterizacion_conlig.py) y PocketVec (ex 2_runpocketvec.py),
pensadas para ser llamadas desde 05_run_pocketvec_batch.py sobre cada fila de
targets.csv.

Pipeline stages implemented here (per protein):
  1. Structure retrieval: fetch_rcsb_structure / fetch_alphafold_structure.
  3. Pocket detection: characterize_ligands (ligand-defined pocket) or
     run_pocket_detection (fpocket + P2Rank rescoring).
  4. PocketVec fingerprint: run_pocketvec / finalize_pocketvec_fingerprint.

REGLA: estas funciones NO atajan errores para "seguir andando". Si algo
falla (falta un binario, un centroide, un archivo esperado), tiran la
excepción con un mensaje concreto de qué faltó y en qué proteína. Decidir
"loguear y seguir con la próxima" es responsabilidad de 05_run_pocketvec_batch.py.

ESTRATEGIA DE SELECCIÓN DE CENTROIDE (automática, no depende de lo que
escribiste a mano en targets.csv):
  1. Si la proteína tiene ligando(s) -> se corren los criterios de
     characterize_holo_pdb_ligand_pockets (n_C>6, solvent_access<0.4,
     buriedness>15). Si al menos uno pasa, se usa el de mejor
     docking_suitability_score como centroide.
  2. Si ningún ligando pasa (o la proteína no tiene ligando -- SIEMPRE el
     caso de los AF de cruzi) -> fpocket+PRANK sobre toda la proteína,
     y se elige el pocket de mayor fpocket_druggability_score.
"""

import json
import logging
import os
import pickle
import shutil
import subprocess
import urllib.request

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

logger = logging.getLogger("pocketvec")

from utils.pocketvec_utils import (
    prepare_pdb, select_first_model, remove_ligands,
    remove_hydrogens, select_occupancies, calculate_centroid,
    create_fpocket_ds, read_fpocket_centers, read_fpocket_scores,
    read_prank_scores, environmental_variables, create_parameter_file,
    create_file_scores, read_rDock_scores, create_cavity, run_rDock,
    rank_fp, raw_fp, extract_ligand_coords, create_pocket_centroid,
)
from utils.config import (
    get_protein_paths, LOCAL_ALPHAFOLD_SEARCH_DIRS,
    PATH_TO_CHECK_STRUCTURE, PATH_TO_FPOCKET, PATH_TO_PRANK, OBABEL_BIN,
    PATH_TO_RDOCK, PATH_TO_LIB, PATH_TO_LIB_ORDER, POCKETVEC_RADIUS, POCKETVEC_NRUNS,
    MOEBATCH_BIN, MOE_FUNCTIONS_SVL,
)
from utils.ligand_characterization import (
    characterize_holo_pdb_ligand_pockets, STANDARD_AA, COMMON_AA_MODS, WATERS,
)

# Rutas a herramientas externas y parámetros de PocketVec (RADIUS, NRUNS)
# viven en utils/config.py (config/config.yaml) -- único lugar.
#
# LOCAL_ALPHAFOLD_SEARCH_DIRS (config: paths.local_alphafold_dirs): carpetas
# donde buscar modelos de AlphaFold ya descargados/predichos a mano (ej.
# corridos en AlphaFold Server cuando no estaban en AFDB), antes de intentar
# bajarlos de AlphaFold DB por UniProt ID.


class ChainSelect(Select):
    def __init__(self, chains):
        self.chains = chains

    def accept_chain(self, chain_obj):
        return chain_obj.get_id() in self.chains


# --------------------------------------------------------------------
# Lectura de config desde una fila de targets.csv
# --------------------------------------------------------------------
def _split_list(value, seps=(";", ",")):
    """Separa por ';' o ',' -- tu CSV usa los dos segun la columna."""
    if not value:
        return []
    v = value
    for sep in seps[1:]:
        v = v.replace(sep, seps[0])
    return [x.strip() for x in v.split(seps[0]) if x.strip()]


def parse_row_cfg(row, side="human"):
    """
    side: "human" o "cruzi". Nombres de columna alineados a tu targets.csv real:
    target_id,target_name,pdb_id_human,human_uniprot,human_pdb_code,human_chain,
    ligand_for_centroid__DEPRECATED_unused,ligand_chains,ligands_to_remove,
    bioactivity_source,cruzi_uniprot,cruzi_af_status,cruzi_af_code,cruzi_name,
    pair_id,pocket_druggability_HUMAN,pocket_druggability_CRUZI,
    pocket_simple_flag,selected_for_deep_dive,notes

    NOTA: ligand_for_centroid__DEPRECATED_unused y ligands_to_remove ya NO se
    usan para decidir el centroide (eso ahora es automático, ver módulo).
    ligands_to_remove se sigue usando para limpiar el receptor antes del
    docking -- si la dejás vacía, se autodetecta (ver autodetect_ligand_resnames).
    """
    if side == "human":
        return {
            "code": row.get("human_pdb_code", "").strip(),
            "chain": (row.get("human_chain") or "A").strip(),
            "ligands_to_remove": _split_list(row.get("ligands_to_remove", "")),
            "ligand_chains": _split_list(row.get("ligand_chains", "")),
            "source": "rcsb",
            "uniprot_id": row.get("human_uniprot", "").strip() or None,
        }
    else:
        # "code" = identificador de carpeta (examples/1_preparation/<code>/) y
        # clave de deduplicación entre filas. Usamos cruzi_af_code (no
        # cruzi_name) por dos motivos vistos en los 33 targets de homólogos:
        # 1. cruzi_name puede traer "/" (ej. "Serine/threonine-protein kinase
        #    RIO1") -- como path component crearía subcarpetas espurias.
        # 2. Dos pair_id distintos pueden compartir la misma estructura AF
        #    (mismo cruzi_af_code) con cruzi_name distinto -- ej. pair_001
        #    "MAGL_TCRUZI_AF_V6" y pair_117 "Monoglyceride lipase, putative"
        #    son ambos AF-Q4CSM0-F1. Con cruzi_af_code como code, el batch
        #    los detecta como la misma proteína y no recalcula fpocket/rDock
        #    dos veces (ver cache en run_side de 05_run_pocketvec_batch.py).
        af_code = row.get("cruzi_af_code", "").strip()
        return {
            "code": af_code or (row.get("cruzi_name") or "").strip(),
            "af_accession": af_code,  # se usa solo para buscar/bajar el archivo
            "chain": "A",
            "ligands_to_remove": [],   # los AF no traen ligando
            "ligand_chains": [],
            "source": "alphafold",
            "uniprot_id": row.get("cruzi_uniprot", "").strip() or None,
        }


# --------------------------------------------------------------------
# Descarga / localización de estructura
# --------------------------------------------------------------------
def fetch_rcsb_structure(pdb_code, raw_path):
    urllib.request.urlretrieve(f"http://files.rcsb.org/download/{pdb_code}.pdb", raw_path)


def _find_local_alphafold_file(code, uniprot_id):
    """
    Busca un archivo ya existente para este modelo de AF -- cubre el caso de
    predicciones hechas a mano (ej. AF-0000000066513361, que no viene de
    AlphaFold DB por UniProt) o descargas previas.
    """
    candidates = [code, uniprot_id] if uniprot_id else [code]
    for base_dir in LOCAL_ALPHAFOLD_SEARCH_DIRS:
        if not os.path.isdir(base_dir):
            continue
        for fname in os.listdir(base_dir):
            if not fname.endswith(".pdb"):
                continue
            for c in candidates:
                if c and c in fname:
                    return os.path.join(base_dir, fname)
    return None


def _query_afdb_api(uniprot_id):
    """
    Le pregunta a AlphaFold DB directamente cuál es la URL del modelo vigente
    para este uniprot_id, en vez de adivinar el número de versión (v3, v4,
    v6...) a mano -- así no se rompe cada vez que Google DeepMind sube una
    versión nueva de la base.
    """
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
    except Exception:
        return None

    if not data:
        return None
    return data[0].get("pdbUrl")


def fetch_alphafold_structure(code, raw_path, uniprot_id=None):
    """
    1. Busca un archivo ya presente localmente para este código/uniprot
       (predicciones manuales tipo AlphaFold Server que no están en AFDB).
    2. Si no encuentra, le pregunta a la API de AlphaFold DB cuál es la
       versión vigente para ese uniprot_id y la descarga.
    3. Como último recurso, prueba adivinando v6 hacia atrás hasta v1 (por
       si la API está caída pero el archivo existe).
    Si nada de esto funciona, tira error explícito -- este es el caso que
    marcaste como "las tengo que hacer yo": no se puede automatizar más.
    """
    local = _find_local_alphafold_file(code, uniprot_id)
    if local:
        shutil.copyfile(local, raw_path)
        return f"local:{local}"

    if not uniprot_id:
        raise RuntimeError(
            f"{code}: no encontré archivo local y no hay uniprot_id para intentar "
            f"AlphaFold DB. Este target probablemente necesita predicción propia."
        )

    api_url = _query_afdb_api(uniprot_id)
    urls_to_try = [api_url] if api_url else []
    urls_to_try += [
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{v}.pdb"
        for v in (6, 5, 4, 3, 2, 1)
    ]

    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status == 200:
                    with open(raw_path, "wb") as fh:
                        fh.write(response.read())
                    return url
        except Exception:
            continue

    raise RuntimeError(
        f"{code}: no está en AlphaFold DB (uniprot={uniprot_id}, API y versiones v1-v6 "
        f"probadas) ni encontré un archivo local. Necesita predicción propia "
        f"(ESMFold/AF server) -- marcar como 'needs_prediction' en targets.csv y "
        f"seguir con el resto."
    )


# --------------------------------------------------------------------
# Autodetección de ligands_to_remove (si no la completaste a mano)
# --------------------------------------------------------------------
def autodetect_ligand_resnames(pdb_path):
    """Devuelve todos los resnames HETATM no-agua/no-AA presentes en el pdb."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("st", pdb_path)
    resnames = set()
    for residue in structure[0].get_residues():
        resname = residue.get_resname().strip()
        hetflag = residue.id[0].strip()
        if hetflag and resname not in STANDARD_AA and resname not in COMMON_AA_MODS:
            resnames.add(resname)
    resnames |= WATERS  # remover aguas siempre, aparezcan o no como HETATM "W"
    return sorted(resnames)


# --------------------------------------------------------------------
# Caracterización de ligandos -> selección de centroide (Capa 1: ligando)
# --------------------------------------------------------------------
def characterize_ligands(protein_code, paths, chain_pdb_path):
    """
    Corre characterize_holo_pdb_ligand_pockets sobre el pdb CON el/los
    ligando(s) todavía presentes (antes de remove_ligands). Devuelve
    (df_passing, df_all) -- df_passing puede estar vacío si ninguno pasa
    los 3 criterios.
    """
    df_all = characterize_holo_pdb_ligand_pockets(
        pdb_code_or_path=chain_pdb_path,
        outpath=paths["root"],           # escribe ligand_centroids/ y *_out acá
        path_to_fpocket=PATH_TO_FPOCKET,
        run_fpocket=True,
        overwrite_fpocket=False,
        min_carbons=6,
        accessibility_cutoff=0.4,
        buriedness_cutoff=15,
        buriedness_mode="contacts",
        write_centroid_sd=True,
        return_all=True,      # queremos ver también los descartados, para auditar
        verbose=False,
    )

    audit_csv = os.path.join(paths["root"], f"{protein_code}_ligand_characterization.csv")
    df_all.to_csv(audit_csv, index=False)

    df_passing = df_all[df_all["passes_all_filters"]] if not df_all.empty else df_all
    return df_passing, df_all


def merge_multichain_ligand_centroid(chain_pdb_path, resname, chains):
    """
    Combina las coordenadas de todas las copias de `resname` presentes en
    `chains` en un solo centroide -- replica exactamente lo que hacías a
    mano en 1_preparation.py para MAGL (coords_A + coords_B -> vstack ->
    calculate_centroid). Devuelve None si el ligando no aparece en 2+ de
    esas cadenas (en cuyo caso no hay nada que fusionar).
    """
    coords_per_chain = []
    for chain_id in chains:
        coords = extract_ligand_coords(chain_pdb_path, resname, chain_id)
        if coords is not None and len(coords) > 0:
            coords_per_chain.append(coords)

    if len(coords_per_chain) < 2:
        return None

    coords_total = np.vstack(coords_per_chain)
    return calculate_centroid(coords_total)


def promote_ligand_centroid(protein_code, paths, df_passing, chain_pdb_path=None, ligand_chains=None):
    """
    Toma la primera fila de df_passing (ya viene ordenada, mejor primero).

    Si ese ligando (mismo resname) aparece en 2+ de las cadenas listadas en
    ligand_chains (ej. MAGL con el ligando en A y B), el centroide se
    recalcula fusionando las coordenadas de todas las copias -- igual que
    hacías a mano. Si aparece en una sola cadena, se usa el .sd que ya
    generó characterize_holo_pdb_ligand_pockets sin modificar.

    LIMITACIÓN CONOCIDA: los filtros (solvent_accessibility, buriedness,
    etc.) que decidieron que este ligando "pasa" se calcularon sobre una
    sola copia, no sobre el sitio combinado. Para MAGL-like (2 copias
    simétricas del mismo ligando) esto en la práctica no cambia la
    decisión, pero si en algún momento un ligando pasa en una cadena y no
    en la otra, avisame para revisarlo con más cuidado.
    """
    best = df_passing.iloc[0]
    dst = os.path.join(paths["ligand_centroids"], f"{protein_code}_CTR_LIG.sd")
    resname = best["resname"]

    merged_centroid = None
    if chain_pdb_path and ligand_chains and len(ligand_chains) > 1:
        merged_centroid = merge_multichain_ligand_centroid(chain_pdb_path, resname, ligand_chains)

    best = best.copy()
    if merged_centroid is not None:
        create_pocket_centroid(merged_centroid, dst, obabel_bin=OBABEL_BIN)
        best["centroid_source"] = f"merged across chains {ligand_chains}"
    else:
        src = best["centroid_sd_path"]
        if src and os.path.exists(src) and src != dst:
            shutil.copyfile(src, dst)
        best["centroid_source"] = "single chain"

    return dst, best.to_dict()


# --------------------------------------------------------------------
# Preparación del receptor para docking (ex 1_preparation.py)
# --------------------------------------------------------------------
def prepare_protein(cfg):
    """
    Corre el pipeline de preparación para UNA proteína: descarga, primer
    modelo, selección de cadena, caracterización de ligando (si aplica),
    limpieza y generación de mol2. Devuelve (paths, ligand_result) donde
    ligand_result es None si no hay ligando válido (cae a pocket-detection).
    """
    protein_code = cfg["code"]
    if not protein_code:
        raise ValueError("cfg['code'] vacío -- revisar la fila de targets.csv")

    paths = get_protein_paths(protein_code)
    raw_pdb = os.path.join(paths["raw"], protein_code + ".pdb")

    if not os.path.exists(raw_pdb):
        if cfg["source"] == "rcsb":
            fetch_rcsb_structure(protein_code, raw_pdb)
        elif cfg["source"] == "alphafold":
            search_code = cfg.get("af_accession") or protein_code
            fetch_alphafold_structure(search_code, raw_pdb, uniprot_id=cfg["uniprot_id"])
        else:
            raise ValueError(f"{protein_code}: source desconocido '{cfg['source']}'")

    # 1. Primer modelo
    model_pdb = os.path.join(paths["processed"], protein_code + "_model.pdb")
    select_first_model(raw_pdb, model_pdb)

    # 2. Cadena(s)
    chains_to_keep = sorted(set([cfg["chain"]] + cfg["ligand_chains"]))
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", model_pdb)
    chain_pdb = os.path.join(paths["processed"], f"{protein_code}_{'_'.join(chains_to_keep)}.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(chain_pdb, ChainSelect(chains_to_keep))

    # 3. Caracterizar ligando(s) ANTES de removerlos (solo tiene sentido si
    #    la proteína puede tener ligando -- los AF nunca lo tienen).
    ligand_result = None
    if cfg["source"] == "rcsb":
        df_passing, _df_all = characterize_ligands(protein_code, paths, chain_pdb)
        if not df_passing.empty:
            _dst, ligand_result = promote_ligand_centroid(
                protein_code, paths, df_passing,
                chain_pdb_path=chain_pdb, ligand_chains=cfg["ligand_chains"],
            )

    # 4. Remover ligandos para el receptor de docking (SIEMPRE se remueven
    #    todos, incluido el que se usó como centroide -- rDock dockea contra
    #    la cavidad vacía, no contra el ligando cristalizado).
    ligands_to_remove = cfg["ligands_to_remove"] or autodetect_ligand_resnames(chain_pdb)
    no_ligands_pdb = os.path.join(paths["processed"], protein_code + "_ligands.pdb")
    remove_ligands(chain_pdb, no_ligands_pdb, ligands_to_remove)

    # 5-6. Hidrógenos + ocupancias. remove_hydrogens/select_occupancies invocan
    # check_structure como "python <PATH_TO_CHECK_STRUCTURE> ..." (ver
    # pocketvec_utils.py) -- el "python" plano tiene que resolver al de ESTE
    # env (el que tiene biobb_structure_checking instalado), no al que ande
    # resolviendo el PATH de la shell en ese momento.
    _check_structure_env_bin = os.path.dirname(PATH_TO_CHECK_STRUCTURE)
    if _check_structure_env_bin not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = _check_structure_env_bin + ":" + os.environ.get("PATH", "")
    no_h_pdb = os.path.join(paths["processed"], protein_code + "_hydrogens.pdb")
    remove_hydrogens(PATH_TO_CHECK_STRUCTURE, no_ligands_pdb, no_h_pdb)

    alt_pdb = os.path.join(paths["processed"], protein_code + "_alt.pdb")
    select_occupancies(PATH_TO_CHECK_STRUCTURE, no_h_pdb, alt_pdb)

    # Preparar mol2. Primero se intenta vía MOE (prepare_pdb); si MOE no está
    # disponible en esta máquina (caso normal -- moebatch no está instalado),
    # se cae a un obabel simple. El input del fallback tiene que ser SIEMPRE
    # alt_pdb (ya sin ligandos/aguas/H, ocupancias resueltas) -- si se usa
    # raw_pdb por error, el receptor final queda con el ligando cristalizado
    # todavía adentro y rbcavity encuentra una cavidad espuria mucho más chica
    # (este fue exactamente el bug del volumen de MAGL: 368 vs 1796.5 A^3).
    prepared_mol2 = os.path.join(paths["processed"], protein_code + "_prepared.mol2")
    prep_log = os.path.join(paths["root"], "prep_logs.log")
    moe_ok = prepare_pdb(alt_pdb, prepared_mol2, prep_log,
                         moebatch=MOEBATCH_BIN, moefunct=MOE_FUNCTIONS_SVL)

    if moe_ok:
        logger.info("%s: receptor preparado con MOE (%s)", protein_code, prepared_mol2)
    else:
        logger.info(
            "%s: MOE no disponible, fallback a obabel sobre alt_pdb (%s)",
            protein_code, alt_pdb,
        )
        command = f"{OBABEL_BIN} -i pdb {alt_pdb} -o mol2 -O {prepared_mol2}"
        if os.system(command) != 0:
            raise RuntimeError(f"{protein_code}: obabel (pdb->mol2) falló, revisar {prep_log}")

    return paths, ligand_result


# --------------------------------------------------------------------
# Pocket-detection ciega (Capa 2: sin ligando válido -- ej. todos los AF)
# --------------------------------------------------------------------
def run_pocket_detection(protein_code, paths, overwrite=False):
    """
    IMPORTANTE sobre rutas: create_fpocket_ds() (utilidad heredada del
    notebook original) escribe en el .ds las rutas RELATIVAS a la carpeta
    donde vive el propio .ds, asumiendo que el .pdb preparado y la carpeta
    *_out de fpocket están justo al lado. Por eso el .ds, el prepared.pdb
    y la carpeta *_out de fpocket TIENEN que vivir todos en la misma
    carpeta (paths["processed"]) -- no se separan en fpocket_out/ como en
    otras partes del pipeline. Si en algún momento movés esto, PRANK va a
    fallar con "protein file doesn't exist" igual que pasó acá.
    """
    fpocket_out_dir = os.path.join(paths["processed"], f"{protein_code}_prepared_out")
    info_path = os.path.join(fpocket_out_dir, protein_code + "_prepared_info.txt")
    pqr_path = os.path.join(fpocket_out_dir, protein_code + "_prepared_pockets.pqr")
    already_done = os.path.exists(info_path) and os.path.exists(pqr_path) and not overwrite

    if not already_done:
        prepared_mol2 = os.path.join(paths["processed"], protein_code + "_prepared.mol2")
        prepared_pdb = os.path.join(paths["processed"], protein_code + "_prepared.pdb")

        command = f"{OBABEL_BIN} -i mol2 {prepared_mol2} -o pdb -O {prepared_pdb}"
        if os.system(command) != 0:
            raise RuntimeError(f"{protein_code}: obabel (mol2->pdb) falló")

        if os.path.exists(fpocket_out_dir):
            shutil.rmtree(fpocket_out_dir)

        cwd = os.getcwd()
        os.chdir(paths["processed"])
        ret = os.system(f"{PATH_TO_FPOCKET} -f {protein_code}_prepared.pdb")
        os.chdir(cwd)
        if ret != 0:
            raise RuntimeError(f"{protein_code}: fpocket devolvió error (código {ret})")

    dsfile = os.path.join(paths["processed"], "fpocket3.ds")
    create_fpocket_ds(protein_code, dsfile)

    result = subprocess.run(
        [PATH_TO_PRANK, "rescore", dsfile, "-o", os.path.join(paths["root"], "prank")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{protein_code}: PRANK falló -- {result.stderr}")

    fpockets = read_fpocket_scores(info_path)
    centers = read_fpocket_centers(pqr_path)
    centers = {i: calculate_centroid(centers[i]) for i in sorted(centers)}
    pranks = read_prank_scores(os.path.join(paths["root"], "prank", protein_code + "_prepared.pdb_rescored.csv"))

    for pocket_id, centroid in centers.items():
        outfile = os.path.join(paths["pockets_PD"], f"{protein_code}_CTR_PD_{pocket_id}.sd")
        create_pocket_centroid(centroid, outfile, obabel_bin=OBABEL_BIN)

    return {"fpockets": fpockets, "pranks": pranks, "centers": centers}


def select_best_pocket_centroid(protein_code, paths, pocket_scores, criterion="druggability"):
    """
    criterion="druggability" (default, y el que pediste para el caso sin
    ligando): mayor Druggability Score de fpocket.
    criterion="prank": mayor rescore de PRANK, alternativa si alguna vez
    querés priorizar eso en cambio.
    """
    fpockets = pocket_scores["fpockets"]
    pranks = pocket_scores["pranks"]

    if criterion == "druggability":
        best_id = max(fpockets, key=lambda i: fpockets[i]["Druggability Score"])
    elif criterion == "prank":
        best_id = max(pranks, key=lambda i: pranks[i])
    else:
        raise ValueError(f"criterion desconocido: {criterion}")

    return os.path.join(paths["pockets_PD"], f"{protein_code}_CTR_PD_{best_id}.sd"), fpockets[best_id]


# --------------------------------------------------------------------
# PocketVec (ex 2_runpocketvec.py)
# --------------------------------------------------------------------
def run_pocketvec(protein_code, paths, pocket_centroid_path, radius=POCKETVEC_RADIUS,
                   nruns=POCKETVEC_NRUNS, run_docking=True):
    """
    run_docking=False corta después de rbcavity (define la cavidad, escribe
    cavity_log.log/cavity.grd) y no corre rbdock -- útil para validar
    geometría/volumen de cavidad rápido (segundos) sin pagar el costo de
    dockear las 128 moléculas. Devuelve None en ese caso en vez del path al
    fingerprint.
    """
    receptor = f"{protein_code}_prepared.mol2"
    path_to_st = os.path.join(paths["processed"], receptor)

    os.makedirs(paths["pocketvec_out"], exist_ok=True)
    rdock_out = os.path.join(paths["pocketvec_out"], "rDock_results")
    os.makedirs(rdock_out, exist_ok=True)

    outfile = os.path.join(rdock_out, "st_parameters.prm")
    create_parameter_file(outfile, path_to_st, pocket_centroid_path, radius=str(radius))

    environmental_variables(PATH_TO_RDOCK)

    path_to_log = os.path.join(rdock_out, "cavity_log.log")
    path_to_cavity = os.path.join(rdock_out, "cavity.grd")
    create_cavity(rdock_out, path_to_log, path_to_cavity, "rbcavity")

    if not run_docking:
        return None

    shutil.copyfile(os.path.join(PATH_TO_RDOCK, "dock.prm"), os.path.join(rdock_out, "dock.prm"))

    run_rDock(rdock_out, PATH_TO_LIB, nruns=nruns)

    return finalize_pocketvec_fingerprint(protein_code, paths)


def finalize_pocketvec_fingerprint(protein_code, paths):
    """
    Calcula scores.tsv + el fingerprint PocketVec (.pkl) a partir de un
    results.sd de rDock YA generado (separado de run_pocketvec() para poder
    recuperar una corrida cuyo docking terminó bien pero falló en este último
    paso -- p.ej. por una ruta rota -- sin tener que re-dockear las 128
    moléculas de nuevo, que es la parte cara).
    """
    rdock_out = os.path.join(paths["pocketvec_out"], "rDock_results")
    path_to_scores = os.path.join(paths["pocketvec_out"], f"{protein_code}_scores.tsv")
    path_to_results = os.path.join(rdock_out, "results.sd")
    create_file_scores(path_to_scores, path_to_results)

    path_to_sorted_lib = PATH_TO_LIB_ORDER
    scores = read_rDock_scores(path_to_scores)
    raw = raw_fp(scores, path_to_sorted_lib)
    rank = rank_fp(raw)

    fp_path = os.path.join(paths["pocketvec_out"], f"{protein_code}_PocketVec_fp.pkl")
    pickle.dump(rank, open(fp_path, "wb"))

    return fp_path
