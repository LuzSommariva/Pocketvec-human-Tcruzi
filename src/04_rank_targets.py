"""
Fase 2 (seccion 6 del prompt): agregacion a nivel target + ranking + shortlist.

Un target puede tener varios (pdb, ligando) validos -- nos quedamos con el de
mejor nivel de docking_suitability (Favorable > Intermedio > Dificil >
Descartar) y, dentro del mismo nivel, mayor composite_score.

FORMULA DE SCORE COMPUESTO (propuesta, constantes ajustables abajo):
  composite_score = 0.40 * druggability_score
                   + 0.25 * jaccard_5A
                   + 0.20 * norm_dist_to_centroid
                   + 0.15 * norm_volume

  norm_dist_to_centroid = clip(1 - dist_A / 10, 0, 1)
    -- 1.0 en dist=0A, 0.0 en dist>=10A (mismo corte que "Dificil" en la
    clasificacion de la seccion 5, para que el score sea consistente con los
    niveles).

  norm_volume = 1.0 si VOLUME_SWEET_SPOT_MIN <= vol <= VOLUME_SWEET_SPOT_MAX
              = vol / MIN si vol < MIN (penaliza cavidades muy chicas)
              = max(0, 1 - (vol-MAX)/MAX) si vol > MAX (penaliza cavidades gigantes,
                muchas veces artefacto de fpocket uniendo varias sub-cavidades)

Pesos elegidos para que druggability (la metrica mas validada de fpocket) domine,
pero sin ignorar concordancia geometrica con el ligando real (dist+jaccard, que
juntas pesan 0.45, casi lo mismo que druggability solo) ni volumen razonable.
AJUSTAR estos pesos mirando la distribucion real es esperable -- quedan como
constantes al toque, no la logica.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils import config

SCREENING_RUN_DIR = Path(config.CHEMBL_SCREENING_RUN_DIR)
CANDIDATES_CSV = Path(config.CHEMBL_CANDIDATES_CSV)
RAW_CSV = SCREENING_RUN_DIR / "pdb_characterization_raw.csv"

RANKED_CSV = Path(config.CHEMBL_RANKED_CSV)
TOP30_CSV = Path(config.CHEMBL_TOP_CSV)
FUNNEL_TXT = SCREENING_RUN_DIR / "funnel_summary.txt"

TOP_N = 30

TIER_ORDER = {
    "Favorable para docking": 0,
    "Intermedio: revisar": 1,
    "Dificil: revisar estructura/protocolo": 2,
    "Descartar/revisar filtros": 3,
}

W_DRUGGABILITY = 0.40
W_JACCARD = 0.25
W_DIST = 0.20
W_VOLUME = 0.15

DIST_CUTOFF_A = 10.0  # mismo corte que "Dificil" en la clasificacion (seccion 5)
VOLUME_SWEET_SPOT_MIN = 300.0
VOLUME_SWEET_SPOT_MAX = 5000.0


def _norm_dist(dist):
    if pd.isna(dist):
        return 0.0
    return float(np.clip(1 - dist / DIST_CUTOFF_A, 0, 1))


def _norm_volume(vol):
    if pd.isna(vol) or vol <= 0:
        return 0.0
    if VOLUME_SWEET_SPOT_MIN <= vol <= VOLUME_SWEET_SPOT_MAX:
        return 1.0
    if vol < VOLUME_SWEET_SPOT_MIN:
        return vol / VOLUME_SWEET_SPOT_MIN
    return float(max(0.0, 1 - (vol - VOLUME_SWEET_SPOT_MAX) / VOLUME_SWEET_SPOT_MAX))


def composite_score(row):
    drug = row.get("fpocket_druggability_score")
    if pd.isna(drug):
        return 0.0
    jac = row.get("fpocket_residue_overlap_jaccard_5A")
    jac = 0.0 if pd.isna(jac) else jac
    dist = row.get("fpocket_dist_to_ligand_centroid_A")
    vol = row.get("fpocket_volume_A3")

    return (
        W_DRUGGABILITY * drug
        + W_JACCARD * jac
        + W_DIST * _norm_dist(dist)
        + W_VOLUME * _norm_volume(vol)
    )


LOG_CSV = SCREENING_RUN_DIR / "screening_run_log.csv"


def run_fase2():
    candidates = pd.read_csv(CANDIDATES_CSV)
    raw = pd.read_csv(RAW_CSV)
    log = pd.read_csv(LOG_CSV)

    # Join (pdb, ligando) -> target_chembl_id. Un mismo PDB puede servir a >1
    # target -- pero OJO con complejos multi-subunidad (ej. 9I2B, complejo
    # Arp2/3): ahi cada subunidad es un target_chembl_id DISTINTO mapeado a
    # cadenas SIFTS distintas del MISMO pdb_code. Sin filtrar por cadena, un
    # ligando encontrado en la cadena de una subunidad (ej. ADP en cadena S)
    # terminaba atribuido a las 6 subunidades del complejo por igual -- 6
    # entradas "distintas" en el top30 que en realidad son la misma evidencia
    # estructural, y ademas mal atribuida (verificado a mano: la cadena S del
    # ligando ADP en 9I2B no coincide con NINGUNA de las cadenas SIFTS de esas
    # 6 subunidades). Fix: exigir que la cadena del ligando (columna 'chain'
    # que devuelve characterize_holo_pdb_ligand_pockets) este entre las
    # cadenas que SIFTS le asigno a ESE target especifico en ESE pdb.
    candidate_chains = (
        candidates[["target_chembl_id", "target_name", "uniprot", "pdb_code", "chain"]]
        .dropna(subset=["chain"])
        .drop_duplicates()
    )
    joined = raw.merge(
        candidate_chains, on="pdb_code", how="left", suffixes=("", "_sifts"),
    )
    n_before_chain_filter = joined["target_chembl_id"].notna().sum()
    joined = joined[joined["chain"] == joined["chain_sifts"]].drop(columns=["chain_sifts"])
    print(f"[Fase 2] Filtro por cadena SIFTS: {n_before_chain_filter} filas (pdb,ligando,target) antes -> "
          f"{len(joined)} despues (descarta atribuciones cruzadas en complejos multi-cadena/multi-subunidad).")

    joined["tier_rank"] = joined["docking_suitability"].map(TIER_ORDER).fillna(9)
    joined["composite_score"] = joined.apply(composite_score, axis=1)

    joined_sorted = joined.sort_values(
        ["target_chembl_id", "tier_rank", "composite_score"],
        ascending=[True, True, False],
    )
    best_per_target = joined_sorted.groupby("target_chembl_id", as_index=False).first()

    ranked = best_per_target.sort_values(
        ["tier_rank", "composite_score"], ascending=[True, False]
    ).reset_index(drop=True)

    report_cols = [
        "target_chembl_id", "target_name", "uniprot", "pdb_code", "ligand_id",
        "docking_suitability", "docking_suitability_legacy_binary_gate", "composite_score",
        "fpocket_druggability_score", "fpocket_dist_to_ligand_centroid_A",
        "fpocket_residue_overlap_jaccard_5A", "fpocket_volume_A3",
        "n_carbons", "solvent_accessibility", "buriedness", "n_residues_8A_centroid",
        "hydrophobic_fraction_5A", "polar_fraction_5A", "charged_fraction_5A", "aromatic_fraction_5A",
    ]
    ranked_out = ranked[[c for c in report_cols if c in ranked.columns]]
    RANKED_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranked_out.to_csv(RANKED_CSV, index=False)

    top30 = ranked_out.head(TOP_N)
    top30.to_csv(TOP30_CSV, index=False)

    # --- funnel completo para la tesis ---
    n_pkl_total = 8021
    n_human = 4638
    n_with_uniprot = 4638
    n_with_pdb = len(candidates["target_chembl_id"].unique())  # ya filtrado en fase0/1 por resolucion aceptable
    n_pdb_total = len(log)
    n_pdb_ok = (log["status"] == "ok").sum()
    n_pdb_failed = (log["status"] == "failed").sum()
    n_pdb_with_any_candidate_row = raw["pdb_code"].nunique()
    n_targets_any_valid_ligand = joined[joined["passes_all_filters"] == True]["target_chembl_id"].nunique()
    n_targets_favorable = (ranked_out["docking_suitability"] == "Favorable para docking").sum()
    n_targets_intermedio = (ranked_out["docking_suitability"] == "Intermedio: revisar").sum()

    funnel_lines = [
        "=== FUNNEL COMPLETO: ChEMBL -> top 20-30 targets ===",
        f"1. Targets totales en el pkl filtrado (ChEMBL 37, bioactividad ya filtrada): {n_pkl_total}",
        f"2. Homo sapiens, SINGLE PROTEIN:                                              4638",
        f"3. Con >=1 UniProt resuelto (local, sin API):                                 4638 (100%)",
        f"4. Con >=1 PDB candidato de resolucion/metodo aceptable (Fase 0):             {n_with_pdb}",
        f"   -> {n_pdb_total} PDB unicos candidatos; {n_pdb_ok} descargados y caracterizados OK, "
        f"{n_pdb_failed} fallaron (todos 404 en RCSB -- probablemente obsoletos/supersedidos);",
        f"      de los {n_pdb_ok} OK, {n_pdb_with_any_candidate_row} tuvieron >=1 residuo HETATM candidato "
        f"(los demas no tenian ningun HETATM con >6 carbonos -- apo de facto)",
        f"5. Targets con >=1 ligando VALIDO (paso los 4 criterios seccion 3.2):          {n_targets_any_valid_ligand}",
        f"6. De esos, clasificados 'Favorable para docking' (nueva clasif. 4 niveles):   {n_targets_favorable}",
        f"   clasificados 'Intermedio: revisar':                                        {n_targets_intermedio}",
        f"7. Shortlist final (top {TOP_N} por composite_score, ya ordenado por tier):        {len(top30)}",
        "",
        f"Formula de composite_score: {W_DRUGGABILITY}*druggability + {W_JACCARD}*jaccard_5A + "
        f"{W_DIST}*norm_dist(cutoff={DIST_CUTOFF_A}A) + {W_VOLUME}*norm_volume(sweet_spot="
        f"{VOLUME_SWEET_SPOT_MIN}-{VOLUME_SWEET_SPOT_MAX} A3)",
    ]
    funnel_txt = "\n".join(funnel_lines)
    FUNNEL_TXT.write_text(funnel_txt)
    print(funnel_txt)
    print(f"\ntargets_ranked.csv: {len(ranked_out)} filas -> {RANKED_CSV}")
    print(f"targets_top30.csv: {len(top30)} filas -> {TOP30_CSV}")

    return ranked_out, top30


if __name__ == "__main__":
    run_fase2()
