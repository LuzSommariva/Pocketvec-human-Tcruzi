"""
Orquestador batch para el screening de targets (humanos + homólogos T. cruzi).

Runs pipeline stages 1 and 3-5 for every human–T. cruzi pair (structure
retrieval, pocket detection/selection, PocketVec fingerprint) and the
pairwise cosine distance between fingerprints. Paths come from
config/config.yaml; ``--targets-csv``/``--results-csv`` override them.

Lee targets.csv, corre prep + caracterización + pocketvec para cada lado
disponible, y calcula la distancia coseno del par cuando ambos lados están
listos. Al final escribe targets_results.csv con las columnas de
druggability completadas (no pisa tu targets.csv original).

Un target fallido NO frena el batch -- se loguea en batch_run_log.csv y se
sigue con el próximo. Nada se reintenta solo.

Uso:
    python src/05_run_pocketvec_batch.py                        # corre todas las filas
    python src/05_run_pocketvec_batch.py --target-ids 1          # solo target_id=1 (MAGL)
    python src/05_run_pocketvec_batch.py --target-ids 1,3        # varios, separados por coma
    python src/05_run_pocketvec_batch.py --target-names MAGL     # por nombre en vez de id
"""

import argparse
import csv
import os
import pickle
import traceback
from datetime import datetime

from scipy.spatial import distance

import utils.pipeline_steps as ps
from utils import config

# Todo sale de config/config.yaml -- no depende de desde dónde corras el script.
TARGETS_CSV = config.TARGETS_CSV
RESULTS_CSV = config.RESULTS_CSV
LOG_CSV = os.path.join(config.ANALYSIS_DIR, "batch_run_log.csv")
PAIR_DISTANCES_CSV = os.path.join(config.ANALYSIS_DIR, "pair_distances.csv")

# Criterio para elegir el mejor pocket cuando NO hay ligando válido
# (siempre el caso de los AF de cruzi, y a veces también el de humanas
# cuyo ligando no pasó los 3 filtros).
POCKET_SELECTION_CRITERION = "druggability"


def load_targets(path=TARGETS_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_targets_with_prior_results():
    """
    targets.csv sigue siendo la fuente de verdad para la CONFIG de cada fila
    (human_pdb_code, cruzi_af_code, etc.). Pero si ya existe un
    targets_results.csv de una corrida anterior (ej. --target-ids por
    tandas), le superponemos los resultados YA calculados (druggability,
    pair_cosine_distance) antes de correr esta tanda -- si no, cada corrida
    parcial pisaba targets_results.csv entero con solo sus propias filas y
    se perdía el resultado de las tandas anteriores en el resumen (los
    archivos por proteína no se pierden nunca, esto es solo
    sobre el CSV de resumen).
    """
    targets = load_targets(TARGETS_CSV)
    if not os.path.exists(RESULTS_CSV):
        return targets

    prior_by_id = {row["target_id"]: row for row in load_targets(RESULTS_CSV)}
    result_cols = ["pocket_druggability_HUMAN", "pocket_druggability_CRUZI", "pair_cosine_distance"]
    for row in targets:
        prior = prior_by_id.get(row["target_id"])
        if not prior:
            continue
        for col in result_cols:
            if prior.get(col):
                row[col] = prior[col]
    return targets


def _append_csv_rows(path, fieldnames, new_rows, dedupe_key=None):
    """
    Escribe new_rows en path preservando lo que ya había de tandas
    anteriores, en vez de pisarlo (ver load_targets_with_prior_results).
    Con dedupe_key, una fila nueva con la misma key reemplaza a la vieja
    (ej. volviste a correr un target y se recalculó su distancia); sin
    dedupe_key se van acumulando todas (ej. el log histórico de pasos).
    """
    existing_rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    if dedupe_key:
        by_key = {r[dedupe_key]: r for r in existing_rows}
        for r in new_rows:
            by_key[r[dedupe_key]] = r
        all_rows = list(by_key.values())
    else:
        all_rows = existing_rows + new_rows

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def log_result(rows, target_id, step, status, detail=""):
    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target_id": target_id,
        "step": step,
        "status": status,
        "detail": detail,
    })
    marker = {"ok": "OK", "failed": "FAIL", "skipped": "SKIP"}.get(status, status)
    print(f"  [{marker}] {step}" + (f" -- {detail}" if detail else ""))


def run_side(cfg, target_id, side_label, log_rows, done_cache=None):
    """
    Corre prep -> (ligando válido, o si no fpocket+PRANK) -> pocketvec.
    Devuelve (fp_path, druggability) -- ambos None si falló en algún punto.

    done_cache: dict opcional {code: (fp_path, druggability)} compartido entre
    filas de todo el batch. Dos motivos para esto:
    1. Dedup real -- ej. pair_001/pair_117 (humano: mismo human_pdb_code 3PE6;
       cruzi: mismo cruzi_af_code AF-Q4CSM0-F1, ver parse_row_cfg) son la
       misma proteína física. Sin este cache, batch_pipeline volvería a
       correr fpocket+rDock (25 runs x 128 moléculas) una segunda vez para
       nada.
    2. Resume-safety -- si el fingerprint .pkl ya existe en disco (corrida
       anterior interrumpida), no se vuelve a dockear.
    """
    if not cfg["code"]:
        log_result(log_rows, target_id, f"{side_label}_prep", "skipped", "sin código de proteína")
        return None, None

    code = cfg["code"]
    if done_cache is not None and code in done_cache:
        fp_path, druggability = done_cache[code]
        log_result(log_rows, target_id, f"{side_label}_pocketvec", "skipped",
                   f"reusando resultado ya calculado para '{code}' (mismo pair físico)")
        return fp_path, druggability

    paths_probe = ps.get_protein_paths(code)
    existing_fp = os.path.join(paths_probe["pocketvec_out"], f"{code}_PocketVec_fp.pkl")
    if os.path.exists(existing_fp):
        log_result(log_rows, target_id, f"{side_label}_pocketvec", "skipped",
                   f"fingerprint ya existe en disco para '{code}' -- no se re-dockea")
        if done_cache is not None:
            done_cache[code] = (existing_fp, None)
        return existing_fp, None

    try:
        paths, ligand_result = ps.prepare_protein(cfg)
        log_result(log_rows, target_id, f"{side_label}_prep", "ok")
    except Exception as e:
        log_result(log_rows, target_id, f"{side_label}_prep", "failed",
                   f"{e}\n{traceback.format_exc(limit=2)}")
        return None, None

    try:
        if ligand_result is not None:
            ctr_path = os.path.join(paths["ligand_centroids"], f"{cfg['code']}_CTR_LIG.sd")
            druggability = ligand_result.get("fpocket_druggability_score")
            log_result(log_rows, target_id, f"{side_label}_pocket", "ok",
                       f"centroide por ligando: {ligand_result.get('ligand_id')} "
                       f"({ligand_result.get('centroid_source')})")
        else:
            pocket_scores = ps.run_pocket_detection(cfg["code"], paths)
            ctr_path, best_pocket = ps.select_best_pocket_centroid(
                cfg["code"], paths, pocket_scores, criterion=POCKET_SELECTION_CRITERION
            )
            druggability = best_pocket.get("Druggability Score")
            log_result(log_rows, target_id, f"{side_label}_pocket", "ok",
                       "centroide por pocket-detection (sin ligando válido)")
    except Exception as e:
        log_result(log_rows, target_id, f"{side_label}_pocket", "failed", str(e))
        return None, None

    try:
        fp_path = ps.run_pocketvec(cfg["code"], paths, ctr_path)
        log_result(log_rows, target_id, f"{side_label}_pocketvec", "ok")
        if done_cache is not None:
            done_cache[code] = (fp_path, druggability)
        return fp_path, druggability
    except Exception as e:
        log_result(log_rows, target_id, f"{side_label}_pocketvec", "failed", str(e))
        return None, druggability


def compare_pair(target_id, human_fp_path, cruzi_fp_path, pair_rows):
    try:
        fp_human = pickle.load(open(human_fp_path, "rb"))
        fp_cruzi = pickle.load(open(cruzi_fp_path, "rb"))
        dist = round(distance.cosine(fp_human, fp_cruzi), 4)
        pair_rows.append({"target_id": target_id, "cosine_distance": dist})
        print(f"  [OK] distancia coseno humano-cruzi: {dist}")
        return dist
    except Exception as e:
        print(f"  [FAIL] no se pudo comparar el par -- {e}")
        return None


def main():
    global TARGETS_CSV, RESULTS_CSV
    parser = argparse.ArgumentParser(description="Batch de screening humano-cruzi")
    parser.add_argument("--target-ids", help="target_id separados por coma (ej: 1,3). Default: todos.")
    parser.add_argument("--target-names", help="target_name separados por coma (ej: MAGL,DPP8). Default: todos.")
    parser.add_argument("--targets-csv", default=TARGETS_CSV,
                        help="Input targets CSV (default: paths.targets_csv in config.yaml).")
    parser.add_argument("--results-csv", default=RESULTS_CSV,
                        help="Output results CSV (default: paths.results_csv in config.yaml).")
    args = parser.parse_args()
    TARGETS_CSV, RESULTS_CSV = args.targets_csv, args.results_csv

    selected_ids = {x.strip() for x in args.target_ids.split(",")} if args.target_ids else None
    selected_names = {x.strip() for x in args.target_names.split(",")} if args.target_names else None

    def is_selected(row):
        if selected_ids is None and selected_names is None:
            return True
        if selected_ids and row["target_id"] in selected_ids:
            return True
        if selected_names and row.get("target_name", "") in selected_names:
            return True
        return False

    targets = load_targets_with_prior_results()
    n_selected = sum(1 for row in targets if is_selected(row))
    log_rows = []
    pair_rows = []
    human_cache = {}
    cruzi_cache = {}

    print(f"targets.csv tiene {len(targets)} filas -- corriendo {n_selected}.\n")

    for row in targets:
        target_id = row["target_id"]
        if not is_selected(row):
            continue

        print(f"--- Target {target_id}: {row.get('target_name') or '(sin nombre)'} ---")

        human_cfg = ps.parse_row_cfg(row, side="human")
        cruzi_cfg = ps.parse_row_cfg(row, side="cruzi")

        human_fp, human_drug = run_side(human_cfg, target_id, "human", log_rows, done_cache=human_cache)
        row["pocket_druggability_HUMAN"] = human_drug if human_drug is not None else row.get("pocket_druggability_HUMAN", "")

        cruzi_status = row.get("cruzi_af_status", "").strip()
        if cruzi_status == "found":
            cruzi_fp, cruzi_drug = run_side(cruzi_cfg, target_id, "cruzi", log_rows, done_cache=cruzi_cache)
            row["pocket_druggability_CRUZI"] = cruzi_drug if cruzi_drug is not None else row.get("pocket_druggability_CRUZI", "")
        else:
            log_result(log_rows, target_id, "cruzi_prep", "skipped",
                       f"cruzi_af_status='{cruzi_status}' -- resolver antes de correr el batch")
            cruzi_fp = None

        if human_fp and cruzi_fp:
            dist = compare_pair(target_id, human_fp, cruzi_fp, pair_rows)
            row["pair_cosine_distance"] = dist

        print()

    fieldnames = list(targets[0].keys())
    if "pair_cosine_distance" not in fieldnames:
        fieldnames.append("pair_cosine_distance")

    # targets ya trae los resultados de tandas anteriores superpuestos (ver
    # load_targets_with_prior_results) -- este write es la vista COMPLETA
    # acumulada hasta ahora, no solo la de esta tanda.
    os.makedirs(os.path.dirname(RESULTS_CSV) or ".", exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(targets)

    _append_csv_rows(LOG_CSV, ["timestamp", "target_id", "step", "status", "detail"], log_rows)

    if pair_rows:
        _append_csv_rows(PAIR_DISTANCES_CSV, ["target_id", "cosine_distance"], pair_rows, dedupe_key="target_id")

    n_failed = sum(1 for r in log_rows if r["status"] == "failed")
    n_skipped = sum(1 for r in log_rows if r["status"] == "skipped")
    print(f"\nListo. {n_failed} pasos fallidos, {n_skipped} saltados.")
    print(f"Resultados: {RESULTS_CSV}")
    print(f"Log completo: {LOG_CSV}")
    if pair_rows:
        print(f"Distancias de pares: {PAIR_DISTANCES_CSV}")


if __name__ == "__main__":
    main()
