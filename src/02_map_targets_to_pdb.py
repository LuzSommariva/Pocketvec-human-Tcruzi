"""
Fase 0 del screening masivo ChEMBL -> PDB.

Todo bulk, CERO llamadas API por-target:
  1. Universo de targets: ya viene resuelto en data_chembl_37_filtrada_31_08_26.pkl
     (bioactividad ya filtrada) -- no se rehace ese filtro.
  2. ChEMBL target_id -> UniProt: consulta LOCAL a chembl_37.db (target_dictionary/
     target_components/component_sequences), reusando chembl_targets_to_uniprot()
     de chembl_target_characterization.py -- una sola query para todos los targets,
     no target-por-target. NOTA: se usa esto en vez del chembl_downloader que
     sugiere el prompt porque ya tenemos el dump completo de ChEMBL 37 en SQLite
     local con las mismas tablas -- es la misma fuente bulk, sin instalar nada
     nuevo ni pegarle a ninguna API.
  3. UniProt -> PDB + cadena: mapeo bulk SIFTS (pdb_chain_uniprot.csv.gz, EBI),
     un solo archivo para todo el PDB.
  4. PDB -> resolucion + metodo experimental: entries.idx (wwPDB derived_data
     index), un solo archivo para todo el PDB.
  5. Filtro liviano (metodo, resolucion) + top 2-3 PDB por target por resolucion.

Salida: chembl_pdb_candidates.csv + funnel impreso en stdout.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from utils import config
from utils.chembl_target_characterization import chembl_targets_to_uniprot, CHEMBL_DB

PKL_PATH = Path(config.CHEMBL_FILTERED_PKL)
SIFTS_PATH = Path(config.SIFTS_PATH)
ENTRIES_IDX_PATH = Path(config.ENTRIES_IDX_PATH)

OUT_CANDIDATES_CSV = Path(config.CHEMBL_CANDIDATES_CSV)

# Config de filtro liviano (Section 2.1) -- pensados para ajustar mirando el funnel real.
MAX_RESOLUTION_A = 3.0
ACCEPTED_METHODS = {"X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"}
MAX_PDBS_PER_TARGET = 3


def load_target_universe(pkl_path=PKL_PATH, organism="Homo sapiens"):
    """
    Universo de targets desde el .pkl ya filtrado. Se restringe a Homo sapiens
    porque el objetivo final (por el resto del pipeline PocketVec) es una
    shortlist de targets HUMANOS -- el pkl trae 8021 targets en total pero solo
    4638 son Homo sapiens (el resto son rata, raton, patogenos, etc.)
    """
    df = pd.read_pickle(pkl_path)
    per_target = df[["tid", "target", "pref_name", "organism", "target_type"]].drop_duplicates()
    n_bioactivities = df.groupby("target").size().rename("n_bioactivities")

    human = per_target[per_target["organism"] == organism].copy()
    human = human.merge(n_bioactivities, left_on="target", right_index=True, how="left")
    return per_target, human


def load_sifts_mapping(path=SIFTS_PATH):
    """PDB, CHAIN, SP_PRIMARY (UniProt) -- bulk, todo el PDB de una."""
    df = pd.read_csv(path, comment="#", compression="gzip")
    df = df.rename(columns={"SP_PRIMARY": "uniprot_accession", "PDB": "pdb_code", "CHAIN": "chain"})
    df["pdb_code"] = df["pdb_code"].str.upper()
    return df[["pdb_code", "chain", "uniprot_accession"]].drop_duplicates()


def load_entries_index(path=ENTRIES_IDX_PATH):
    """
    entries.idx (wwPDB): IDCODE, HEADER, ACCESSION DATE, COMPOUND, SOURCE,
    AUTHOR LIST, RESOLUTION, EXPERIMENT TYPE (vacio == X-RAY DIFFRACTION segun
    el propio header del archivo).
    """
    cols = ["pdb_code", "header", "accession_date", "compound", "source",
            "author_list", "resolution_raw", "experiment_type_raw"]
    df = pd.read_csv(
        path, sep="\t", skiprows=2, names=cols, engine="python",
        quoting=3, on_bad_lines="skip", dtype=str,
    )
    df = df.dropna(subset=["pdb_code"])
    df["pdb_code"] = df["pdb_code"].str.strip().str.upper()

    df["experiment_type"] = df["experiment_type_raw"].fillna("").str.strip()
    df.loc[df["experiment_type"] == "", "experiment_type"] = "X-RAY DIFFRACTION"

    df["resolution"] = pd.to_numeric(df["resolution_raw"], errors="coerce")

    return df[["pdb_code", "resolution", "experiment_type"]].drop_duplicates(subset=["pdb_code"])


def run_fase0():
    funnel = {}

    per_target_all, human_targets = load_target_universe()
    funnel["targets_totales_en_pkl"] = per_target_all["target"].nunique()
    funnel["targets_homo_sapiens_single_protein"] = human_targets["target"].nunique()

    print(f"[Fase 0] Universo: {funnel['targets_totales_en_pkl']} targets totales en el pkl, "
          f"{funnel['targets_homo_sapiens_single_protein']} Homo sapiens SINGLE PROTEIN.")

    print("[Fase 0] Mapeando ChEMBL target_id -> UniProt (query local a chembl_37.db, "
          "una sola consulta para todos los targets)...")
    map_df = chembl_targets_to_uniprot(human_targets["target"].tolist(), chembl_db=CHEMBL_DB)
    map_df = map_df[
        (map_df["component_organism"] == "Homo sapiens") & (map_df["component_type"] == "PROTEIN")
    ]
    if (map_df["homologue"] == 0).any():
        map_df = map_df[map_df["homologue"] == 0]

    targets_with_uniprot = map_df["target_id"].nunique()
    funnel["targets_con_uniprot"] = targets_with_uniprot
    print(f"[Fase 0] {targets_with_uniprot}/{funnel['targets_homo_sapiens_single_protein']} "
          f"targets con >=1 UniProt humano resuelto (local, sin API).")

    print("[Fase 0] Cargando mapeo bulk SIFTS (PDB<->UniProt<->cadena)...")
    sifts = load_sifts_mapping()
    print(f"[Fase 0] SIFTS: {len(sifts)} filas (PDB, cadena, UniProt) para todo el PDB.")

    print("[Fase 0] Cargando indice bulk de resolucion/metodo experimental (entries.idx)...")
    entries = load_entries_index()
    print(f"[Fase 0] entries.idx: {len(entries)} entradas PDB con resolucion/metodo.")

    merged = map_df.merge(sifts, on="uniprot_accession", how="left")
    targets_with_any_pdb = merged.dropna(subset=["pdb_code"])["target_id"].nunique()
    funnel["targets_con_al_menos_1_pdb"] = targets_with_any_pdb
    print(f"[Fase 0] {targets_with_any_pdb}/{targets_with_uniprot} targets con >=1 PDB mapeado via SIFTS.")

    merged = merged.merge(entries, on="pdb_code", how="left")

    accepted = merged[
        merged["experiment_type"].isin(ACCEPTED_METHODS)
        & merged["resolution"].notna()
        & (merged["resolution"] <= MAX_RESOLUTION_A)
    ].copy()

    targets_with_acceptable_pdb = accepted["target_id"].nunique()
    funnel["targets_con_pdb_resolucion_aceptable"] = targets_with_acceptable_pdb
    print(f"[Fase 0] {targets_with_acceptable_pdb}/{targets_with_any_pdb} targets con >=1 PDB de "
          f"metodo/resolucion aceptable (metodo in {sorted(ACCEPTED_METHODS)}, resolucion <= {MAX_RESOLUTION_A} A).")

    accepted = accepted.sort_values(["target_id", "resolution"])
    accepted["rank_within_target"] = accepted.groupby("target_id").cumcount() + 1
    top_candidates = accepted[accepted["rank_within_target"] <= MAX_PDBS_PER_TARGET].copy()

    n_bio_map = human_targets.set_index("target")["n_bioactivities"]
    top_candidates["n_bioactivities"] = top_candidates["target_id"].map(n_bio_map)

    out = top_candidates.rename(columns={
        "target_id": "target_chembl_id", "target_name": "target_name",
        "uniprot_accession": "uniprot", "pdb_code": "pdb_code",
    })[[
        "target_chembl_id", "target_name", "uniprot", "pdb_code", "chain",
        "resolution", "experiment_type", "n_bioactivities",
    ]].drop_duplicates(subset=["target_chembl_id", "pdb_code", "chain"])

    OUT_CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CANDIDATES_CSV, index=False)

    funnel["filas_finales_candidatas_pdb"] = len(out)
    funnel["targets_finales_con_candidato"] = out["target_chembl_id"].nunique()

    print("\n=== FUNNEL FASE 0 ===")
    for k, v in funnel.items():
        print(f"  {k}: {v}")
    print(f"\nSalida: {OUT_CANDIDATES_CSV} ({len(out)} filas, "
          f"{out['target_chembl_id'].nunique()} targets distintos)")

    return out, funnel


if __name__ == "__main__":
    run_fase0()
