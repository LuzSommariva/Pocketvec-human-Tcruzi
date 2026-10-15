"""
Clasificacion de docking_suitability (4 niveles) para el screening masivo.
Ver seccion 5 del prompt -- cutoffs como constantes configurables, no
hardcodeados en el medio de la logica, porque se van a ajustar mirando la
distribucion real de resultados de la Fase 1.

OJO (ver seccion 3.2 del prompt): esto es DISTINTO del docking_suitability que
ya devuelve characterize_holo_pdb_ligand_pockets() en ligand_characterization.py
-- ese es un gate binario ("paso los 4 criterios de ligando valido" si/no,
mezclado con un score 0-1 de heuristicas propias) y por eso puede marcar
"Favorable para docking" con fpocket_druggability_score=0.0 (ver filas NAG:C:1
y QXQ:*:* del fixture). Esta clasificacion nueva combina druggability_score,
distancia al centroide y jaccard de fpocket -- especifica para rankear
candidatos del screening amplio, no para el pipeline PocketVec existente.
"""

import pandas as pd

DRUGGABILITY_FAVORABLE = 0.5
DRUGGABILITY_INTERMEDIATE_MIN = 0.3

DIST_FAVORABLE_A = 6.0
DIST_INTERMEDIATE_MAX_A = 10.0

JACCARD_FAVORABLE = 0.3
JACCARD_INTERMEDIATE_MIN = 0.15


def classify_docking_suitability_screening(row):
    """
    row: dict-like con al menos fpocket_druggability_score,
    fpocket_dist_to_ligand_centroid_A, fpocket_residue_overlap_jaccard_5A,
    passes_all_filters (de characterize_holo_pdb_ligand_pockets).

    Devuelve el nivel (str). "Descartar/revisar filtros" si no paso los 4
    criterios de ligando valido de la seccion 3.2 -- en ese caso no llega a
    tener fpocket asociado en este esquema.
    """
    if not row.get("passes_all_filters", False):
        return "Descartar/revisar filtros"

    drug = row.get("fpocket_druggability_score")
    dist = row.get("fpocket_dist_to_ligand_centroid_A")
    jac = row.get("fpocket_residue_overlap_jaccard_5A")

    if pd.isna(drug) or pd.isna(dist) or pd.isna(jac):
        return "Descartar/revisar filtros"

    if drug >= DRUGGABILITY_FAVORABLE and dist <= DIST_FAVORABLE_A and jac >= JACCARD_FAVORABLE:
        return "Favorable para docking"

    if drug < DRUGGABILITY_INTERMEDIATE_MIN or dist > DIST_INTERMEDIATE_MAX_A or jac < JACCARD_INTERMEDIATE_MIN:
        return "Dificil: revisar estructura/protocolo"

    return "Intermedio: revisar"


FIXTURE_ROWS = [
    {"source_pdb": "1Q5K", "ligand_id": "TMU:B:902", "n_carbons": 12, "solvent_accessibility": 0.182970, "buriedness": 49, "n_residues_8A_centroid": 18, "fpocket_nearest_pocket": 35, "fpocket_volume_A3": 717.212, "fpocket_score": -0.095, "fpocket_druggability_score": 0.460, "fpocket_dist_to_ligand_centroid_A": 2.413590, "fpocket_residue_overlap_jaccard_5A": 0.727273, "hydrophobic_fraction_5A": 0.529412, "polar_fraction_5A": 0.235294, "charged_fraction_5A": 0.235294, "aromatic_fraction_5A": 0.058824},
    {"source_pdb": "1Q5K", "ligand_id": "TMU:A:901", "n_carbons": 12, "solvent_accessibility": 0.250902, "buriedness": 44, "n_residues_8A_centroid": 18, "fpocket_nearest_pocket": 54, "fpocket_volume_A3": 1276.414, "fpocket_score": -0.337, "fpocket_druggability_score": 0.350, "fpocket_dist_to_ligand_centroid_A": 6.311600, "fpocket_residue_overlap_jaccard_5A": 0.307692, "hydrophobic_fraction_5A": 0.562500, "polar_fraction_5A": 0.187500, "charged_fraction_5A": 0.250000, "aromatic_fraction_5A": 0.062500},
    {"source_pdb": "3PE6", "ligand_id": "ZYH:A:304", "n_carbons": 25, "solvent_accessibility": 0.028805, "buriedness": 72, "n_residues_8A_centroid": 21, "fpocket_nearest_pocket": 1, "fpocket_volume_A3": 1197.276, "fpocket_score": 0.152, "fpocket_druggability_score": 0.975, "fpocket_dist_to_ligand_centroid_A": 1.045674, "fpocket_residue_overlap_jaccard_5A": 0.454545, "hydrophobic_fraction_5A": 0.629630, "polar_fraction_5A": 0.259259, "charged_fraction_5A": 0.185185, "aromatic_fraction_5A": 0.148148},
    {"source_pdb": "4OC6", "ligand_id": "2S1:B:402", "n_carbons": 44, "solvent_accessibility": 0.178912, "buriedness": 90, "n_residues_8A_centroid": 16, "fpocket_nearest_pocket": 15, "fpocket_volume_A3": 1435.034, "fpocket_score": -0.187, "fpocket_druggability_score": 0.072, "fpocket_dist_to_ligand_centroid_A": 3.293251, "fpocket_residue_overlap_jaccard_5A": 0.375000, "hydrophobic_fraction_5A": 0.548387, "polar_fraction_5A": 0.354839, "charged_fraction_5A": 0.129032, "aromatic_fraction_5A": 0.161290},
    {"source_pdb": "4OC6", "ligand_id": "NAG:C:1", "n_carbons": 8, "solvent_accessibility": 0.388344, "buriedness": 18, "n_residues_8A_centroid": 10, "fpocket_nearest_pocket": 9, "fpocket_volume_A3": 204.573, "fpocket_score": -0.050, "fpocket_druggability_score": 0.000, "fpocket_dist_to_ligand_centroid_A": 3.936212, "fpocket_residue_overlap_jaccard_5A": 0.466667, "hydrophobic_fraction_5A": 0.375000, "polar_fraction_5A": 0.500000, "charged_fraction_5A": 0.125000, "aromatic_fraction_5A": 0.000000},
    {"source_pdb": "7SXF", "ligand_id": "6VL:A:501", "n_carbons": 20, "solvent_accessibility": 0.207965, "buriedness": 45, "n_residues_8A_centroid": 16, "fpocket_nearest_pocket": 8, "fpocket_volume_A3": 880.103, "fpocket_score": 0.116, "fpocket_druggability_score": 0.614, "fpocket_dist_to_ligand_centroid_A": 4.039495, "fpocket_residue_overlap_jaccard_5A": 0.520000, "hydrophobic_fraction_5A": 0.388889, "polar_fraction_5A": 0.388889, "charged_fraction_5A": 0.222222, "aromatic_fraction_5A": 0.111111},
    {"source_pdb": "7A3G", "ligand_id": "QXQ:B:902", "n_carbons": 24, "solvent_accessibility": 0.139041, "buriedness": 62, "n_residues_8A_centroid": 17, "fpocket_nearest_pocket": 90, "fpocket_volume_A3": 246.293, "fpocket_score": -0.038, "fpocket_druggability_score": 0.000, "fpocket_dist_to_ligand_centroid_A": 5.894550, "fpocket_residue_overlap_jaccard_5A": 0.433333, "hydrophobic_fraction_5A": 0.333333, "polar_fraction_5A": 0.428571, "charged_fraction_5A": 0.333333, "aromatic_fraction_5A": 0.428571},
    {"source_pdb": "7A3G", "ligand_id": "QXT:B:910", "n_carbons": 24, "solvent_accessibility": 0.167900, "buriedness": 40, "n_residues_8A_centroid": 14, "fpocket_nearest_pocket": 88, "fpocket_volume_A3": 451.747, "fpocket_score": -0.029, "fpocket_druggability_score": 0.044, "fpocket_dist_to_ligand_centroid_A": 5.605441, "fpocket_residue_overlap_jaccard_5A": 0.166667, "hydrophobic_fraction_5A": 0.562500, "polar_fraction_5A": 0.250000, "charged_fraction_5A": 0.250000, "aromatic_fraction_5A": 0.375000},
    {"source_pdb": "7A3G", "ligand_id": "QXQ:A:1002", "n_carbons": 24, "solvent_accessibility": 0.147604, "buriedness": 61, "n_residues_8A_centroid": 17, "fpocket_nearest_pocket": 98, "fpocket_volume_A3": 240.398, "fpocket_score": -0.052, "fpocket_druggability_score": 0.000, "fpocket_dist_to_ligand_centroid_A": 5.972977, "fpocket_residue_overlap_jaccard_5A": 0.433333, "hydrophobic_fraction_5A": 0.333333, "polar_fraction_5A": 0.428571, "charged_fraction_5A": 0.333333, "aromatic_fraction_5A": 0.428571},
    {"source_pdb": "7A3G", "ligand_id": "QXT:A:1010", "n_carbons": 24, "solvent_accessibility": 0.141045, "buriedness": 44, "n_residues_8A_centroid": 13, "fpocket_nearest_pocket": 10, "fpocket_volume_A3": 496.182, "fpocket_score": 0.151, "fpocket_druggability_score": 0.625, "fpocket_dist_to_ligand_centroid_A": 5.907890, "fpocket_residue_overlap_jaccard_5A": 0.523810, "hydrophobic_fraction_5A": 0.533333, "polar_fraction_5A": 0.266667, "charged_fraction_5A": 0.266667, "aromatic_fraction_5A": 0.400000},
    {"source_pdb": "1A52", "ligand_id": "EST:A:1", "n_carbons": 18, "solvent_accessibility": 0.019345, "buriedness": 51, "n_residues_8A_centroid": 19, "fpocket_nearest_pocket": 1, "fpocket_volume_A3": 1206.586, "fpocket_score": 0.276, "fpocket_druggability_score": 0.995, "fpocket_dist_to_ligand_centroid_A": 5.757919, "fpocket_residue_overlap_jaccard_5A": 0.321429, "hydrophobic_fraction_5A": 0.789474, "polar_fraction_5A": 0.105263, "charged_fraction_5A": 0.157895, "aromatic_fraction_5A": 0.105263},
    {"source_pdb": "1A52", "ligand_id": "EST:B:2", "n_carbons": 18, "solvent_accessibility": 0.013671, "buriedness": 49, "n_residues_8A_centroid": 19, "fpocket_nearest_pocket": 4, "fpocket_volume_A3": 766.079, "fpocket_score": 0.059, "fpocket_druggability_score": 0.918, "fpocket_dist_to_ligand_centroid_A": 1.754510, "fpocket_residue_overlap_jaccard_5A": 0.727273, "hydrophobic_fraction_5A": 0.777778, "polar_fraction_5A": 0.111111, "charged_fraction_5A": 0.166667, "aromatic_fraction_5A": 0.111111},
]


def validate_against_fixture():
    df = pd.DataFrame(FIXTURE_ROWS)
    df["passes_all_filters"] = True  # el fixture ya son ligandos que pasaron los 4 criterios
    df["docking_suitability_screening"] = df.apply(classify_docking_suitability_screening, axis=1)
    cols = ["source_pdb", "ligand_id", "fpocket_druggability_score",
            "fpocket_dist_to_ligand_centroid_A", "fpocket_residue_overlap_jaccard_5A",
            "docking_suitability_screening"]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    validate_against_fixture()
