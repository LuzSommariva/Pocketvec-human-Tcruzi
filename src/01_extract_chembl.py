"""
Reemplazo del query MySQL (INTO OUTFILE) del notebook data_filtering_pipline_chembl34_02_25.ipynb
para ChEMBL37, ejecutado contra el .db de SQLite con sqlite3/pandas.

Stage 0a of target selection. DB_PATH and OUT_DIR come from config/config.yaml
(chembl.db, chembl.raw_dir). The resulting CSV is filtered in
notebooks/chembl37_bioactivity_filtering.ipynb.

Diferencias de esquema respecto a ChEMBL34 (confirmadas contra schema_documentation.txt
de EBI para chembl_34 y chembl_37):
  - compound_properties YA NO tiene: cx_most_apka, cx_most_bpka, cx_logp, cx_logd,
    molecular_species, mw_monoisotopic, hba_lipinski, hbd_lipinski,
    num_lipinski_ro5_violations (9 columnas eliminadas, todas dependientes de ChemAxon
    o derivadas de Lipinski "originales"). Ninguna de estas se usa en la lógica de
    filtrado/limpieza del notebook (units, bioactivity, comp_id, target, relation,
    type, organism, assay_type, smiles) así que se pueden dropear sin romper el
    pipeline downstream.
  - activities, assays, molecule_dictionary tuvieron cambios de columnas (MODALITY
    agregada, CURATED_BY removida, CHEBI_PAR_ID/INDICATION_CLASS removidas, etc.)
    pero NINGUNA de las columnas usadas por este query fue afectada.
  - target_dictionary, compound_structures, target_components, component_sequences:
    sin cambios de columnas relevantes al query.
"""

import sqlite3
import pandas as pd

from utils import config

DB_PATH = config.CHEMBL_DB
OUT_DIR = config.CHEMBL_RAW_DIR
OUT_PATH_CSV = f"{OUT_DIR}/data_chembl37_full.csv"
CHUNKSIZE = 200_000

QUERY = """
SELECT act.molregno,
    md.chembl_id AS comp_id,
    ass.tid AS tid,
    ass.assay_id AS assay_id,
    trgd.chembl_id AS target,
    act.standard_relation AS relation,
    act.standard_value AS bioactivity,
    act.standard_units AS units,
    act.standard_type AS type,
    act.potential_duplicate,
    act.pchembl_value,
    trgd.organism,
    trgd.target_type,
    trgd.pref_name,
    ass.assay_type,
    cmpstc.canonical_smiles AS smiles,
    cmpstc.standard_inchi_key AS inchi_key,
    cmpprop.mw_freebase,
    cmpprop.alogp,
    cmpprop.hba,
    cmpprop.hbd,
    cmpprop.psa,
    cmpprop.rtb,
    cmpprop.ro3_pass,
    cmpprop.num_ro5_violations,
    cmpprop.full_mwt,
    cmpprop.aromatic_rings,
    cmpprop.heavy_atoms,
    cmpprop.qed_weighted,
    cmpprop.full_molformula,
    cmpprop.np_likeness_score,
    cs.sequence AS sequence,
    cs.organism AS sequence_organism
FROM activities AS act
JOIN assays AS ass ON act.assay_id = ass.assay_id
JOIN target_dictionary AS trgd ON ass.tid = trgd.tid
JOIN compound_structures AS cmpstc ON act.molregno = cmpstc.molregno
JOIN molecule_dictionary AS md ON act.molregno = md.molregno
JOIN compound_properties AS cmpprop ON md.molregno = cmpprop.molregno
JOIN target_components AS tc ON ass.tid = tc.tid
JOIN component_sequences AS cs ON tc.component_id = cs.component_id
WHERE trgd.target_type = 'SINGLE PROTEIN'
AND ass.assay_type = 'B'
AND cmpprop.heavy_atoms > 3
AND act.standard_value IS NOT NULL
AND trgd.organism IS NOT NULL
"""


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    con = sqlite3.connect(DB_PATH)

    first_chunk = True
    total_rows = 0
    for chunk in pd.read_sql_query(QUERY, con, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        chunk.to_csv(
            OUT_PATH_CSV,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        print(f"acumulado: {total_rows} filas")

    con.close()
    print(f"listo. total filas: {total_rows}. escrito en {OUT_PATH_CSV}")


if __name__ == "__main__":
    main()
