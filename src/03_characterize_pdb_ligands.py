"""
Fase 1 del screening masivo: por cada PDB candidato (unico, de-duplicado en
Fase 0) -> descarga + characterize_holo_pdb_ligand_pockets() (sin modificar
su logica) + clasificacion nueva de 4 niveles (utils/screening_classify.py).

Diseno:
  - Resumable: si ya existe el marker de una estructura, se saltea.
  - Paralelizable a nivel de estructura (ProcessPoolExecutor, 16 workers;
    adjust N_WORKERS to the machine you run on).
  - Un PDB que falla (descarga, corrupto, fpocket crashea) se loguea y sigue,
    no tira abajo el batch.
  - fpocket UNA sola vez por estructura apo (ya lo hace characterize_holo_pdb_
    ligand_pockets internamente).
  - Si ningun ligando pasa los 4 criterios: NO fallback a fpocket ciego (eso es
    solo para los AF de cruzi sin ligando, caso distinto) -- se marca "sin
    ligando valido" y queda fuera del screening amplio, documentado igual.

Salidas:
  chembl_screening_run/pdb_characterization_raw.csv  -- una fila por (pdb, ligando
    candidato), TODOS (pasen o no los 4 criterios), para poder auditar.
  chembl_screening_run/screening_run_log.csv          -- un log por PDB procesado.
"""

import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import config
from utils.chembl_target_characterization import fetch_rcsb_structure, characterize_holo, STRUCT_DIR
from utils.screening_classify import classify_docking_suitability_screening

CANDIDATES_CSV = Path(config.CHEMBL_CANDIDATES_CSV)

RUN_DIR = Path(config.CHEMBL_SCREENING_RUN_DIR)
STRUCT_DIR_SCREENING = RUN_DIR / "structures"
RAW_CSV = RUN_DIR / "pdb_characterization_raw.csv"
LOG_CSV = RUN_DIR / "screening_run_log.csv"

N_WORKERS = 16

REPORT_COLUMNS = [
    "target_chembl_id", "pdb_code", "ligand_id",
    "n_carbons", "solvent_accessibility", "buriedness", "n_residues_8A_centroid",
    "fpocket_nearest_pocket", "fpocket_volume_A3", "fpocket_score",
    "fpocket_druggability_score", "fpocket_dist_to_ligand_centroid_A",
    "fpocket_residue_overlap_jaccard_5A",
    "hydrophobic_fraction_5A", "polar_fraction_5A", "charged_fraction_5A", "aromatic_fraction_5A",
    "docking_suitability",
]


def _marker_path(pdb_code):
    return STRUCT_DIR_SCREENING / pdb_code / "_DONE.csv"


def process_one_pdb(pdb_code):
    """
    Corre en un worker separado. Devuelve dict:
      {"pdb_code", "status": "ok"|"failed"|"skipped", "detail", "rows": [...]}
    rows: lista de dicts, una por ligando candidato evaluado (return_all=True),
    SIN target_chembl_id todavia (eso se pega despues via merge, un PDB puede
    servir a >1 target).
    """
    marker = _marker_path(pdb_code)
    if marker.exists():
        try:
            df = pd.read_csv(marker)
            return {"pdb_code": pdb_code, "status": "skipped", "detail": "ya procesado (checkpoint)",
                    "rows": df.to_dict("records")}
        except Exception:
            pass  # marker corrupto -> reprocesar

    workdir = STRUCT_DIR_SCREENING / pdb_code
    try:
        pdb_path = fetch_rcsb_structure(pdb_code, workdir)
        if pdb_path is None:
            return {"pdb_code": pdb_code, "status": "failed", "detail": "no se pudo descargar de RCSB", "rows": []}

        char_df = characterize_holo(pdb_code, workdir, return_all=True)  # todos, para poder auditar descartados
        rows = char_df.to_dict("records")

        # docking_suitability que trae characterize_holo_pdb_ligand_pockets() es el gate
        # binario historico del pipeline PocketVec (ver docstring de chembl_screening_
        # classify.py) -- se conserva para auditar, y se agrega la clasificacion nueva
        # de 4 niveles (seccion 5 del prompt) en 'docking_suitability' (la que importa
        # para este screening).
        for r in rows:
            r["docking_suitability_legacy_binary_gate"] = r.get("docking_suitability")
            r["docking_suitability"] = classify_docking_suitability_screening(r)

        marker.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(marker, index=False)

        return {"pdb_code": pdb_code, "status": "ok", "detail": f"{len(rows)} ligandos candidatos", "rows": rows}
    except Exception as e:
        return {"pdb_code": pdb_code, "status": "failed",
                "detail": f"{e}\n{traceback.format_exc(limit=2)}", "rows": []}


def run_fase1(n_workers=N_WORKERS):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_DIR_SCREENING.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATES_CSV)
    unique_pdbs = sorted(candidates["pdb_code"].unique())
    print(f"[Fase 1] {len(unique_pdbs)} PDB unicos a procesar, {n_workers} workers.")

    log_rows = []
    all_char_rows = []
    t0 = time.time()
    n_done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_one_pdb, pdb): pdb for pdb in unique_pdbs}
        for future in as_completed(futures):
            pdb = futures[future]
            n_done += 1
            try:
                result = future.result()
            except Exception as e:
                result = {"pdb_code": pdb, "status": "failed", "detail": f"excepcion en worker: {e}", "rows": []}

            log_rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pdb_code": result["pdb_code"], "status": result["status"], "detail": result["detail"],
            })
            for r in result["rows"]:
                r["pdb_code"] = result["pdb_code"]
                all_char_rows.append(r)

            if n_done % 100 == 0 or n_done == len(unique_pdbs):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(unique_pdbs) - n_done) / rate if rate > 0 else float("nan")
                n_failed = sum(1 for r in log_rows if r["status"] == "failed")
                print(f"[Fase 1] {n_done}/{len(unique_pdbs)} ({n_failed} fallidos) -- "
                      f"{elapsed:.0f}s transcurridos, ETA {eta:.0f}s", flush=True)

    raw_df = pd.DataFrame(all_char_rows)
    raw_df.to_csv(RAW_CSV, index=False)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_CSV, index=False)

    n_failed = (log_df["status"] == "failed").sum()
    n_ok = (log_df["status"] == "ok").sum()
    n_skipped = (log_df["status"] == "skipped").sum()
    print(f"\n[Fase 1] Listo. ok={n_ok} failed={n_failed} skipped={n_skipped}")
    print(f"[Fase 1] {len(raw_df)} filas (pdb, ligando candidato) -> {RAW_CSV}")
    print(f"[Fase 1] Log -> {LOG_CSV}")

    return raw_df, log_df


if __name__ == "__main__":
    run_fase1()
