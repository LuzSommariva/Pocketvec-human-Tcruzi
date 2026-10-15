"""
superpose_magl.py  (formerly Superposicion_MAGL.py)

Alineamiento estructural MAGL humana (3PE6) vs MAGL de T. cruzi (AF-Q4CSM0-F1),
anclado en la triada catalitica -- reemplaza, para este par, el metodo de
Superposicion.ipynb (legacy notebook, not included in this repository;
alineamiento de secuencia global de Biopython +
Bio.PDB.Superimposer sobre TODO el backbone, sin foco en el sitio activo y
sin RMSD impreso en ningun lado).

Hallazgo de la auditoria previa: Superposicion.ipynb, tal como esta hoy, NO
corre sobre MAGL -- tiene un base_dir de otra maquina/usuario y carga datos
de GSK3alpha (7SXF / AF_GSKalpha_TCruzi). Este script es nuevo y standalone,
no lo reemplaza ni lo modifica (ver restricciones del pedido).

No se integra a batch_pipeline.py / pipeline_steps.py -- ninguno de esos dos
archivos se toca. Las rutas a las estructuras "prepared" reales se resuelven
con utils.config.get_protein_paths(), no se hardcodean a mano.

METODO DE IDENTIFICACION DE LA TRIADA (generico, pensado para reusar con
otros pares humano-cruzi, no solo MAGL):
  1. Serina nucleofilica: se busca el motivo "nucleophile elbow" GXSXG por
     regex en la secuencia de cruzi, usando como ancla la columna del
     alineamiento pairwise (Biopython) que corresponde a la Ser humana
     conocida (ver HUMAN_KNOWN_TRIAD). Si hay mas de un match de GXSXG en la
     proteina (pasa en MAGL: hay un GXSXG "decoy" mas temprano en la
     secuencia, no catalitico), se elige el mas cercano a esa columna, no el
     primero.
  2. Histidina catalitica: el HIS de la cadena cuyo atomo mas cercano
     (heavy-atom) al OG de la Ser nucleofilica es minimo, con corte en
     HIS_MAX_DIST. Validado contra 3PE6: recupera His269 a 2.8 A (el
     siguiente candidato, His121, esta a 4.4 A -- separacion clara).
  3. Acido catalitico (Asp/Glu): NO se busca por cercania a la Ser
     directamente -- se probo y falla (en 3PE6, Asp239 esta a 6.9 A de la
     Ser122-OG, fuera de un corte razonable de 6 A, y Glu53 -- que no es
     catalitico -- esta casi igual de cerca). La geometria real de la
     triada es un relevo Ser-His-Asp (H-bonds Ser-Og...Ne2-His y
     Nd1-His...Od-Asp), asi que el Asp/Glu catalitico se busca por cercania
     al ND1 de la His YA identificada en el paso 2. Validado contra 3PE6:
     recupera Asp239 a 2.8 A del ND1 de His269 (siguiente candidato, Asp243,
     a 6.8 A -- separacion clara).
Los tres residuos humanos (Ser122/Asp239/His269) se usan SOLO como ancla del
alineamiento pairwise y como chequeo de sanity (se verifica contra la
estructura real de 3PE6, no se asumen ciegamente) -- el algoritmo para cruzi
no los hardcodea.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1

# src/ (three levels up) must be on sys.path to import utils.* as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import get_protein_paths, OBABEL_BIN, CASE_STUDY_DIR

import pymol  # noqa: E402  (tiene que importarse despues de fijar pymol_argv)
pymol.pymol_argv = ["pymol", "-qc"]
pymol.finish_launching()
from pymol import cmd  # noqa: E402


# ---------------------------------------------------------------------
# Configuracion del par MAGL. Todo lo que sigue despues de esta seccion
# es generico (no depende de estos valores salvo a traves de las variables).
# ---------------------------------------------------------------------
HUMAN_CODE = "3PE6"
HUMAN_CHAIN = "A"
CRUZI_CODE = "AF-Q4CSM0-F1"
CRUZI_CHAIN = "A"

# Triada humana conocida de literatura -- ancla del alineamiento + sanity
# check, no input del algoritmo para el lado cruzi.
HUMAN_KNOWN_TRIAD = {"ser": ("SER", 122), "asp": ("ASP", 239), "his": ("HIS", 269)}

MOTIF_REGEX = re.compile(r"G.S.G")
MOTIF_SEARCH_WINDOW = 20        # +- residuos alrededor de la columna alineada (asuncion)
HIS_MAX_DIST = 6.0              # A, corte Ser-OG -> His (validado: 2.8 A en 3PE6)
ACID_MAX_DIST = 6.0             # A, corte His-ND1 -> Asp/Glu (validado: 2.8 A en 3PE6)
POCKET_CUTOFF_A = 8.0           # A, mismo criterio que "pocket_residues_8A_centroid" del CSV humano

# Si el RMSD de la triada post-cealign es peor que esto, se corre un
# refinamiento local con pair_fit (triada + vecinos de bolsillo). Umbral
# elegido a ojo (no hay literatura que fije un numero "correcto" aca): mitad
# de un angstrom de margen sobre lo que se ve como una buena superposicion
# de sitio activo en la practica, o notablemente peor que el global.
TRIAD_RMSD_REFINE_ABS = 2.0     # A
TRIAD_RMSD_REFINE_FACTOR = 1.2  # * RMSD global de cealign

OUT_DIR = CASE_STUDY_DIR  # config: paths.case_study_dir
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------
# 0. Resolver el PDB "prepared" real (el que efectivamente usa PocketVec),
#    via utils.config -- no hardcodeado a mano.
# ---------------------------------------------------------------------
def resolve_prepared_pdb(protein_code):
    """
    prepare_protein() (pipeline_steps.py) siempre termina en un
    '<code>_prepared.mol2'. El '<code>_prepared.pdb' recien se genera
    despues, dentro de run_pocket_detection() (conversion mol2->pdb via
    obabel), y esa funcion SOLO corre para proteinas sin centroide de
    ligando valido (ej. AF-Q4CSM0-F1: nunca tiene ligando). Para 3PE6 el
    centroide salio de ZYH (ver ligand_characterization.csv), asi que
    run_pocket_detection() nunca corrio y no existe 3PE6_prepared.pdb en
    disco -- solo el .mol2.

    Para no tocar processed/ (carpeta del pipeline), si falta el .pdb lo
    generamos aca con el MISMO comando que usa run_pocket_detection
    (obabel mol2->pdb), pero escribiendo el resultado en OUT_DIR.
    """
    paths = get_protein_paths(protein_code)
    prepared_pdb = os.path.join(paths["processed"], f"{protein_code}_prepared.pdb")
    if os.path.exists(prepared_pdb):
        return prepared_pdb, paths

    prepared_mol2 = os.path.join(paths["processed"], f"{protein_code}_prepared.mol2")
    if not os.path.exists(prepared_mol2):
        raise FileNotFoundError(
            f"{protein_code}: no encontre '_prepared.pdb' ni '_prepared.mol2' en "
            f"{paths['processed']} -- correr prepare_protein() para esta proteina primero."
        )

    out_pdb = os.path.join(OUT_DIR, f"{protein_code}_prepared_from_mol2.pdb")
    command = f"{OBABEL_BIN} -i mol2 {prepared_mol2} -o pdb -O {out_pdb}"
    if os.system(command) != 0 or not os.path.exists(out_pdb):
        raise RuntimeError(f"{protein_code}: obabel mol2->pdb fallo ({command})")
    log(f"[{protein_code}] no tenia _prepared.pdb (nunca corrio pocket-detection ciega "
        f"porque el centroide salio de un ligando) -- generado desde _prepared.mol2 en {out_pdb}")
    return out_pdb, paths


# ---------------------------------------------------------------------
# 1. Secuencia + numeracion real de residuos, a partir del PDB "prepared".
# ---------------------------------------------------------------------
_parser = PDBParser(QUIET=True)


def get_chain_sequence(pdb_path, chain_id):
    structure = _parser.get_structure("st", pdb_path)
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " " and r.has_id("CA")]
    seq = "".join(seq1(r.get_resname()) for r in residues)
    resnums = [r.id[1] for r in residues]
    return seq, resnums, structure


def verify_known_residue(structure, chain_id, resnum, expected_resname, label):
    chain = structure[0][chain_id]
    try:
        res = chain[(" ", resnum, " ")]
    except KeyError:
        raise ValueError(
            f"Sanity check fallo: no encontre el residuo {resnum} en cadena {chain_id} "
            f"(esperaba {expected_resname} como {label})."
        )
    if res.get_resname() != expected_resname:
        raise ValueError(
            f"Sanity check fallo: residuo {resnum} en cadena {chain_id} es "
            f"{res.get_resname()}, esperaba {expected_resname} como {label}."
        )
    log(f"  sanity check OK: {label} humano = {expected_resname}{resnum}")


# ---------------------------------------------------------------------
# 2. Alineamiento pairwise de secuencia (Biopython, BLOSUM62) -- se usa
#    SOLO para ubicar la columna homologa a la Ser humana y anclar la
#    busqueda del motivo GXSXG en cruzi. No es el metodo de superposicion
#    estructural (eso lo hace PyMOL cealign mas abajo).
# ---------------------------------------------------------------------
def pairwise_align(seq_a, seq_b):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(seq_a, seq_b)
    return alignments[0]


def build_index_map(alignment):
    """Mapea indice (0-based) en seq_a -> indice en seq_b, solo columnas sin gap."""
    index_map = {}
    for seg_a, seg_b in zip(alignment.aligned[0], alignment.aligned[1]):
        a0, a1 = seg_a
        b0, b1 = seg_b
        for i in range(a1 - a0):
            index_map[a0 + i] = b0 + i
    return index_map


# ---------------------------------------------------------------------
# 3. Ubicar la Ser nucleofilica de cruzi: motivo GXSXG + columna alineada.
# ---------------------------------------------------------------------
def find_nucleophile_serine(cruzi_seq, cruzi_resnums, anchor_idx_in_cruzi):
    """
    Busca todos los matches de GXSXG en cruzi_seq. Si hay mas de uno (pasa
    en MAGL), se queda con el mas cercano a `anchor_idx_in_cruzi` (la
    columna que el alineamiento pairwise mapeo desde la Ser humana). Primero
    intenta restringido a MOTIF_SEARCH_WINDOW alrededor del ancla; si no
    encuentra nada ahi, cae a buscar en toda la secuencia (mas lento pero
    mas robusto si el alineamiento global no fue preciso en esa zona).
    """
    matches = list(MOTIF_REGEX.finditer(cruzi_seq))
    if not matches:
        raise ValueError("No encontre ningun motivo GXSXG en la secuencia de cruzi.")

    def ser_index(m):
        return m.start() + 2  # G(0) X(1) S(2) X(3) G(4)

    window_matches = [
        m for m in matches
        if abs(ser_index(m) - anchor_idx_in_cruzi) <= MOTIF_SEARCH_WINDOW
    ]
    candidates = window_matches or matches
    best = min(candidates, key=lambda m: abs(ser_index(m) - anchor_idx_in_cruzi))

    all_evidence = [
        (cruzi_resnums[ser_index(m)], m.group(), abs(ser_index(m) - anchor_idx_in_cruzi))
        for m in matches
    ]
    return cruzi_resnums[ser_index(best)], best.group(), all_evidence


# ---------------------------------------------------------------------
# 4. Ubicar His y Asp/Glu cataliticos por proximidad 3D (ver docstring del
#    modulo para la justificacion del esquema en dos pasos Ser->His->Asp).
# ---------------------------------------------------------------------
def _min_heavy_atom_dist(atom_ref, residue):
    return min(
        np.linalg.norm(atom.get_coord() - atom_ref.get_coord())
        for atom in residue
        if atom.element != "H"
    )


def find_catalytic_his(structure, chain_id, ser_resnum, max_dist=HIS_MAX_DIST):
    chain = structure[0][chain_id]
    ser_res = chain[(" ", ser_resnum, " ")]
    ref_atom = ser_res["OG"] if ser_res.has_id("OG") else ser_res["CA"]

    candidates = []
    for res in chain:
        if res.id[0] != " " or res.get_resname() != "HIS" or res.id[1] == ser_resnum:
            continue
        candidates.append((res.id[1], _min_heavy_atom_dist(ref_atom, res)))
    candidates.sort(key=lambda x: x[1])

    if not candidates or candidates[0][1] > max_dist:
        raise ValueError(
            f"No encontre ninguna HIS a <= {max_dist} A de Ser{ser_resnum}-OG "
            f"(candidatos mas cercanos: {candidates[:3]})."
        )
    return candidates[0][0], candidates


def find_catalytic_acid(structure, chain_id, his_resnum, max_dist=ACID_MAX_DIST):
    chain = structure[0][chain_id]
    his_res = chain[(" ", his_resnum, " ")]
    ref_atom = his_res["ND1"] if his_res.has_id("ND1") else his_res["CA"]

    candidates = []
    for res in chain:
        if res.id[0] != " " or res.get_resname() not in ("ASP", "GLU"):
            continue
        candidates.append((res.get_resname(), res.id[1], _min_heavy_atom_dist(ref_atom, res)))
    candidates.sort(key=lambda x: x[2])

    if not candidates or candidates[0][2] > max_dist:
        raise ValueError(
            f"No encontre ningun ASP/GLU a <= {max_dist} A del ND1 de His{his_resnum} "
            f"(candidatos mas cercanos: {candidates[:3]})."
        )
    return candidates[0][1], candidates[0][0], candidates


# ---------------------------------------------------------------------
# 5. Residuos de bolsillo (para mostrar en la sesion de PyMOL / extender
#    el refinamiento local si hace falta).
# ---------------------------------------------------------------------
def human_pocket_resnums(paths, protein_code, chain_id):
    """
    Reusa el CSV que ya escribe characterize_ligands() (pipeline_steps.py)
    -- columna pocket_residues_8A_centroid, ya calculada sobre el mismo
    centroide que PocketVec usa para dockear. Puramente lectura, no vuelve
    a correr nada.
    """
    csv_path = os.path.join(paths["root"], f"{protein_code}_ligand_characterization.csv")
    df = pd.read_csv(csv_path)
    best = df.sort_values("docking_suitability_score_0_1", ascending=False).iloc[0]
    # formato de cada entrada: "<chain>:<RESNAME><RESNUM>", ej. "A:ASP239"
    entries = str(best["pocket_residues_8A_centroid"]).split(";")
    resnums = set()
    for e in entries:
        chain, resname_resnum = e.split(":")
        if chain == chain_id:
            m = re.search(r"\d+$", resname_resnum)
            if m:
                resnums.add(int(m.group()))
    return resnums


def _parse_fpocket_druggability(info_path):
    """Parser minimo, solo-lectura, de <code>_info.txt de fpocket (evita
    depender de utils/pocketvec_utils.py, que carga scipy/tqdm/openbabel-pybel
    -- dependencias pesadas que no hacen falta para este script)."""
    scores = {}
    current = None
    with open(info_path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"Pocket\s+(\d+)\s*:", line)
            if m:
                current = int(m.group(1))
                continue
            if current is not None and "Druggability Score" in line:
                scores[current] = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
    return scores


def cruzi_best_pocket_centroid(paths, protein_code):
    """Mismo criterio que select_best_pocket_centroid(criterion='druggability')
    de pipeline_steps.py, aplicado por lectura directa de los archivos ya
    generados por el pipeline (no vuelve a correr fpocket/PRANK)."""
    fpocket_dir = os.path.join(paths["processed"], f"{protein_code}_prepared_out")
    info_path = os.path.join(fpocket_dir, f"{protein_code}_prepared_info.txt")
    scores = _parse_fpocket_druggability(info_path)
    if not scores:
        raise FileNotFoundError(f"No pude leer druggability scores de {info_path}")
    best_id = max(scores, key=scores.get)

    # formato MDL molfile V2000: linea 0=titulo, 1=programa, 2=comentario,
    # 3=counts line, 4=primer (unico) atomo con las coordenadas.
    sd_path = os.path.join(paths["pockets_PD"], f"{protein_code}_CTR_PD_{best_id}.sd")
    with open(sd_path) as fh:
        lines = fh.readlines()
    atom_line = lines[4]
    x, y, z = (float(v) for v in atom_line.split()[:3])
    log(f"  [{protein_code}] mejor pocket fpocket = #{best_id} "
        f"(Druggability Score = {scores[best_id]:.3f}), centroide = ({x:.2f}, {y:.2f}, {z:.2f})")
    return np.array([x, y, z]), best_id, scores[best_id]


def pocket_resnums_near_point(structure, chain_id, point, cutoff=POCKET_CUTOFF_A):
    chain = structure[0][chain_id]
    resnums = set()
    for res in chain:
        if res.id[0] != " ":
            continue
        if any(np.linalg.norm(atom.get_coord() - point) <= cutoff
               for atom in res if atom.element != "H"):
            resnums.add(res.id[1])
    return resnums


# ---------------------------------------------------------------------
# 6. Metodo VIEJO (reproducido tal cual estaba en Superposicion.ipynb):
#    PairwiseAligner default (sin matriz de sustitucion) + Bio.PDB.Superimposer
#    (Kabsch) sobre TODOS los CA alineables. Se corre sobre este mismo par
#    MAGL humano/cruzi para poder comparar numeros con el metodo nuevo.
# ---------------------------------------------------------------------
def old_method_backbone_superimpose(human_struct, human_chain_id, cruzi_struct, cruzi_chain_id):
    chain_h = human_struct[0][human_chain_id]
    chain_c = cruzi_struct[0][cruzi_chain_id]
    res_h = [r for r in chain_h if r.has_id("CA")]
    res_c = [r for r in chain_c if r.has_id("CA")]
    seq_h = "".join(seq1(r.get_resname()) for r in res_h)
    seq_c = "".join(seq1(r.get_resname()) for r in res_c)

    aligner = PairwiseAligner()
    aligner.mode = "global"  # tal cual el notebook viejo: sin substitution_matrix (scoring por identidad)
    alignment = aligner.align(seq_h, seq_c)[0]

    pairs = []
    for seg_h, seg_c in zip(alignment.aligned[0], alignment.aligned[1]):
        h0, h1 = seg_h
        c0, c1 = seg_c
        for i in range(h1 - h0):
            pairs.append((res_h[h0 + i]["CA"], res_c[c0 + i]["CA"]))

    si = Superimposer()
    si.set_atoms([a for a, _ in pairs], [b for _, b in pairs])
    return si, len(pairs)


def apply_rotran(coords, rot, tran):
    return coords @ rot + tran


# ---------------------------------------------------------------------
# 7. RMSD manual (numpy) sobre coordenadas ya posicionadas -- se evita
#    cmd.rms_cur() de PyMOL: en pruebas devolvio valores absurdos
#    (>1e5 A) en este build de pymol-open-source 3.1.0 para selecciones de
#    pocos atomos, mientras que el calculo manual desde cmd.get_model()
#    reprodujo el numero esperado. Se prefiere un calculo propio,
#    auditable, antes que confiar en un comando que se mostro poco fiable.
# ---------------------------------------------------------------------
def manual_rmsd(coords_a, coords_b):
    diff = np.asarray(coords_a) - np.asarray(coords_b)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def get_ca_coords(sel):
    model = cmd.get_model(sel)
    return np.array([a.coord for a in model.atom]), [(a.resn, a.resi) for a in model.atom]


def main():
    log("=" * 70)
    log("1. Resolviendo estructuras 'prepared' reales via utils.config")
    log("=" * 70)
    human_pdb, human_paths = resolve_prepared_pdb(HUMAN_CODE)
    cruzi_pdb, cruzi_paths = resolve_prepared_pdb(CRUZI_CODE)
    log(f"  humano ({HUMAN_CODE}): {human_pdb}")
    log(f"  cruzi  ({CRUZI_CODE}): {cruzi_pdb}")

    log("")
    log("=" * 70)
    log("2. Secuencias + sanity check de la triada humana conocida")
    log("=" * 70)
    human_seq, human_resnums, human_struct = get_chain_sequence(human_pdb, HUMAN_CHAIN)
    cruzi_seq, cruzi_resnums, cruzi_struct = get_chain_sequence(cruzi_pdb, CRUZI_CHAIN)
    log(f"  humano: {len(human_seq)} residuos con CA (cadena {HUMAN_CHAIN})")
    log(f"  cruzi:  {len(cruzi_seq)} residuos con CA (cadena {CRUZI_CHAIN})")

    for label, (resname, resnum) in HUMAN_KNOWN_TRIAD.items():
        verify_known_residue(human_struct, HUMAN_CHAIN, resnum, resname, label)

    log("")
    log("=" * 70)
    log("3. Identificacion dinamica de la triada catalitica de cruzi")
    log("=" * 70)
    alignment = pairwise_align(human_seq, cruzi_seq)
    index_map = build_index_map(alignment)

    human_ser_idx = human_resnums.index(HUMAN_KNOWN_TRIAD["ser"][1])
    anchor_idx_in_cruzi = index_map.get(human_ser_idx)
    if anchor_idx_in_cruzi is None:
        raise ValueError(
            "La columna del alineamiento correspondiente a la Ser humana cayo en un "
            "gap -- no puedo anclar la busqueda del motivo GXSXG en cruzi."
        )
    log(f"  Ser{HUMAN_KNOWN_TRIAD['ser'][1]} humana (indice {human_ser_idx}) -> "
        f"columna alineada de cruzi = indice {anchor_idx_in_cruzi} "
        f"(residuo {cruzi_seq[anchor_idx_in_cruzi]}{cruzi_resnums[anchor_idx_in_cruzi]})")

    cruzi_ser_resnum, motif_matched, motif_evidence = find_nucleophile_serine(
        cruzi_seq, cruzi_resnums, anchor_idx_in_cruzi
    )
    log(f"  Motivo(s) GXSXG encontrados en cruzi: {motif_evidence}")
    log(f"  -> Ser nucleofilica elegida: Ser{cruzi_ser_resnum} (motivo '{motif_matched}', "
        f"la mas cercana a la columna alineada de la Ser humana)")

    cruzi_his_resnum, his_candidates = find_catalytic_his(cruzi_struct, CRUZI_CHAIN, cruzi_ser_resnum)
    log(f"  Candidatos HIS por distancia a Ser{cruzi_ser_resnum}-OG (A): {his_candidates[:4]}")
    log(f"  -> His catalitica elegida: His{cruzi_his_resnum}")

    cruzi_asp_resnum, cruzi_asp_resname, acid_candidates = find_catalytic_acid(
        cruzi_struct, CRUZI_CHAIN, cruzi_his_resnum
    )
    log(f"  Candidatos ASP/GLU por distancia al ND1 de His{cruzi_his_resnum} (A): {acid_candidates[:4]}")
    log(f"  -> Acido catalitico elegido: {cruzi_asp_resname}{cruzi_asp_resnum}")

    log("")
    log(f"  TRIADA CRUZI IDENTIFICADA: Ser{cruzi_ser_resnum} / "
        f"{cruzi_asp_resname}{cruzi_asp_resnum} / His{cruzi_his_resnum}")
    log(f"  TRIADA HUMANA (literatura, verificada): "
        f"Ser{HUMAN_KNOWN_TRIAD['ser'][1]} / Asp{HUMAN_KNOWN_TRIAD['asp'][1]} / "
        f"His{HUMAN_KNOWN_TRIAD['his'][1]}")

    human_triad_resnums = [HUMAN_KNOWN_TRIAD["ser"][1], HUMAN_KNOWN_TRIAD["asp"][1], HUMAN_KNOWN_TRIAD["his"][1]]
    cruzi_triad_resnums = [cruzi_ser_resnum, cruzi_asp_resnum, cruzi_his_resnum]

    log("")
    log("=" * 70)
    log("4. Residuos de bolsillo (para visualizacion / posible refinamiento local)")
    log("=" * 70)
    human_pocket = human_pocket_resnums(human_paths, HUMAN_CODE, HUMAN_CHAIN)
    log(f"  humano: {len(human_pocket)} residuos a <= {POCKET_CUTOFF_A} A del centroide de "
        f"ligando (pocket_residues_8A_centroid, ya calculado por pipeline_steps.py)")
    cruzi_centroid, cruzi_best_pocket_id, cruzi_drug_score = cruzi_best_pocket_centroid(cruzi_paths, CRUZI_CODE)
    cruzi_pocket = pocket_resnums_near_point(cruzi_struct, CRUZI_CHAIN, cruzi_centroid)
    log(f"  cruzi: {len(cruzi_pocket)} residuos a <= {POCKET_CUTOFF_A} A del centroide del "
        f"pocket #{cruzi_best_pocket_id} (mayor Druggability Score = {cruzi_drug_score:.3f})")

    log("")
    log("=" * 70)
    log("5. Metodo NUEVO: PyMOL cealign (backbone completo)")
    log("=" * 70)
    cmd.load(human_pdb, "human")
    cmd.load(cruzi_pdb, "cruzi")
    cealign_result = cmd.cealign("human", "cruzi")  # target=human (fijo), mobile=cruzi (se mueve)
    global_rmsd_new = cealign_result["RMSD"]
    aligned_len_new = cealign_result["alignment_length"]
    log(f"  cealign: RMSD global = {global_rmsd_new:.3f} A sobre {aligned_len_new} posiciones")

    human_triad_sel = "human and resi " + "+".join(str(r) for r in human_triad_resnums) + " and name CA"
    cruzi_triad_sel = "cruzi and resi " + "+".join(str(r) for r in cruzi_triad_resnums) + " and name CA"
    coords_h_triad, order_h = get_ca_coords(human_triad_sel)
    coords_c_triad, order_c = get_ca_coords(cruzi_triad_sel)
    assert [o[0] for o in order_h] == [o[0] for o in order_c] == ["SER", "ASP", "HIS"], (
        f"Orden de la triada no coincide entre humano y cruzi -- humano={order_h}, cruzi={order_c}"
    )
    triad_rmsd_new = manual_rmsd(coords_h_triad, coords_c_triad)
    log(f"  RMSD triada catalitica (post-cealign, sin re-fit) = {triad_rmsd_new:.3f} A "
        f"sobre {len(coords_h_triad)} atomos CA (Ser/Asp/His)")

    refine_threshold = max(TRIAD_RMSD_REFINE_ABS, TRIAD_RMSD_REFINE_FACTOR * global_rmsd_new)
    needs_refine = triad_rmsd_new > refine_threshold
    triad_rmsd_refined = None
    if needs_refine:
        log(f"  RMSD de triada ({triad_rmsd_new:.3f} A) > umbral de refinamiento "
            f"({refine_threshold:.3f} A) -- corriendo pair_fit local (triada + vecinos de bolsillo)")

        human_ext = set(human_triad_resnums) | human_pocket
        mapped_cruzi_pocket = {
            cruzi_resnums[index_map[human_resnums.index(hr)]]
            for hr in human_ext
            if hr in human_resnums and human_resnums.index(hr) in index_map
        }
        cruzi_ext = set(cruzi_triad_resnums) | (mapped_cruzi_pocket & cruzi_pocket)
        cruzi_ext |= set(cruzi_triad_resnums)  # la triada siempre entra, tenga o no match de bolsillo

        # pares explicitos (mobile, target) atomo-a-atomo -- pair_fit necesita
        # las selecciones en orden 1:1; construimos los pares "a mano" via el
        # index_map de secuencia en vez de confiar en que ambos lados queden
        # ordenados igual por numero de residuo.
        cmd.create("cruzi_refined", "cruzi")
        pair_args = []
        for hr in human_ext:
            if hr not in human_resnums:
                continue
            h_idx = human_resnums.index(hr)
            c_idx = index_map.get(h_idx)
            if c_idx is None:
                continue
            cr = cruzi_resnums[c_idx]
            if cr not in cruzi_ext:
                continue
            h_sel = f"human and resi {hr} and name CA"
            c_sel = f"cruzi_refined and resi {cr} and name CA"
            if cmd.count_atoms(h_sel) == 1 and cmd.count_atoms(c_sel) == 1:
                pair_args += [c_sel, h_sel]  # (mobile, target)

        if len(pair_args) < 6:  # menos de 3 pares -- no alcanza para un fit estable
            log("  No junte suficientes pares mapeados para un pair_fit estable -- se omite el refinamiento.")
            cmd.delete("cruzi_refined")
        else:
            cmd.pair_fit(*pair_args)
            coords_c_refined, _ = get_ca_coords(
                "cruzi_refined and resi " + "+".join(str(r) for r in cruzi_triad_resnums) + " and name CA"
            )
            triad_rmsd_refined = manual_rmsd(coords_h_triad, coords_c_refined)
            log(f"  RMSD triada tras pair_fit local ({len(pair_args)//2} pares) = "
                f"{triad_rmsd_refined:.3f} A")
    else:
        log(f"  RMSD de triada ({triad_rmsd_new:.3f} A) <= umbral de refinamiento "
            f"({refine_threshold:.3f} A) -- el ajuste global de cealign ya superpone bien "
            f"el sitio activo, no hace falta refinamiento local.")

    log("")
    log("=" * 70)
    log("6. Metodo VIEJO (reproducido): Biopython pairwise (sin matriz) + Kabsch, backbone completo")
    log("=" * 70)
    si_old, n_pairs_old = old_method_backbone_superimpose(human_struct, HUMAN_CHAIN, cruzi_struct, CRUZI_CHAIN)
    global_rmsd_old = si_old.rms
    log(f"  Superimposer (Kabsch): RMSD global = {global_rmsd_old:.3f} A sobre {n_pairs_old} pares de CA")

    rot, tran = si_old.rotran
    chain_c_old = cruzi_struct[0][CRUZI_CHAIN]
    cruzi_triad_coords_raw = np.array([chain_c_old[(" ", r, " ")]["CA"].get_coord() for r in cruzi_triad_resnums])
    cruzi_triad_coords_old = apply_rotran(cruzi_triad_coords_raw, rot, tran)
    human_triad_coords_raw = np.array([human_struct[0][HUMAN_CHAIN][(" ", r, " ")]["CA"].get_coord()
                                        for r in human_triad_resnums])
    triad_rmsd_old = manual_rmsd(human_triad_coords_raw, cruzi_triad_coords_old)
    log(f"  RMSD triada catalitica (post-Kabsch viejo, mismos 3 residuos) = {triad_rmsd_old:.3f} A")

    log("")
    log("=" * 70)
    log("7. Tabla comparativa VIEJO vs NUEVO")
    log("=" * 70)
    rows = [
        ("Viejo (secuencia+Kabsch, backbone completo)", global_rmsd_old, triad_rmsd_old, n_pairs_old),
        ("Nuevo (cealign, backbone completo)", global_rmsd_new, triad_rmsd_new, aligned_len_new),
    ]
    if triad_rmsd_refined is not None:
        rows.append(("Nuevo + refinamiento local (pair_fit triada+bolsillo)",
                      global_rmsd_new, triad_rmsd_refined, aligned_len_new))

    log(f"  {'Metodo':<55} {'RMSD global (A)':>16} {'RMSD triada (A)':>17} {'N atomos':>9}")
    for name, g, t, n in rows:
        log(f"  {name:<55} {g:>16.3f} {t:>17.3f} {n:>9}")

    log("")
    log("=" * 70)
    log("8. Sesion de PyMOL (.pse)")
    log("=" * 70)
    cmd.hide("everything")
    cmd.show("cartoon", "human or cruzi")
    cmd.color("gray70", "human")
    cmd.color("skyblue", "cruzi")
    cmd.set("cartoon_transparency", 0.35, "human or cruzi")

    human_triad_all = "human and resi " + "+".join(str(r) for r in human_triad_resnums)
    cruzi_triad_all = "cruzi and resi " + "+".join(str(r) for r in cruzi_triad_resnums)
    cmd.show("sticks", human_triad_all + " or " + cruzi_triad_all)
    cmd.color("yellow", human_triad_all + " and elem C")
    cmd.color("orange", cruzi_triad_all + " and elem C")
    cmd.set("cartoon_transparency", 0.0, human_triad_all + " or " + cruzi_triad_all)

    human_pocket_sel = "human and resi " + "+".join(str(r) for r in human_pocket) if human_pocket else None
    cruzi_pocket_sel = "cruzi and resi " + "+".join(str(r) for r in cruzi_pocket) if cruzi_pocket else None
    if human_pocket_sel:
        cmd.show("lines", human_pocket_sel)
    if cruzi_pocket_sel:
        cmd.show("lines", cruzi_pocket_sel)

    if triad_rmsd_refined is not None and cmd.count_atoms("cruzi_refined") > 0:
        cmd.show("cartoon", "cruzi_refined")
        cmd.color("purple", "cruzi_refined")
        cmd.set("cartoon_transparency", 0.6, "cruzi_refined")
        refined_triad_sel = "cruzi_refined and resi " + "+".join(str(r) for r in cruzi_triad_resnums)
        cmd.show("sticks", refined_triad_sel)
        cmd.color("magenta", refined_triad_sel + " and elem C")

    cmd.bg_color("white")
    cmd.zoom(human_triad_all + " or " + cruzi_triad_all, 10)
    cmd.set("ray_opaque_background", 0)

    pse_path = os.path.join(OUT_DIR, "MAGL_human_vs_cruzi_cealign.pse")
    cmd.save(pse_path)
    log(f"  Sesion guardada en: {pse_path}")

    log("")
    log("=" * 70)
    log("9. Export de PDBs alineados + resumen JSON (para el notebook interactivo)")
    log("=" * 70)
    log(
        "  Se exporta la MISMA transformacion ya calculada por cealign (objeto 'cruzi' "
        "en memoria, transformado en la seccion 5) -- no se recalcula ningun alineamiento aca."
    )

    human_full_pdb = os.path.join(OUT_DIR, f"{HUMAN_CODE}_fixed_full.pdb")
    human_pocket_pdb = os.path.join(OUT_DIR, f"{HUMAN_CODE}_fixed_pocket.pdb")
    cruzi_full_pdb = os.path.join(OUT_DIR, f"{CRUZI_CODE}_mobile_aligned_full.pdb")
    cruzi_pocket_pdb = os.path.join(OUT_DIR, f"{CRUZI_CODE}_mobile_aligned_pocket.pdb")

    cmd.save(human_full_pdb, "human")
    cmd.save(cruzi_full_pdb, "cruzi")
    if human_pocket_sel:
        cmd.save(human_pocket_pdb, human_pocket_sel)
    if cruzi_pocket_sel:
        cmd.save(cruzi_pocket_pdb, cruzi_pocket_sel)
    log(f"  humano completo (fijo)        -> {human_full_pdb}")
    log(f"  humano bolsillo (fijo)        -> {human_pocket_pdb}")
    log(f"  cruzi completo (post-cealign) -> {cruzi_full_pdb}")
    log(f"  cruzi bolsillo (post-cealign) -> {cruzi_pocket_pdb}")

    human_ligand_row = pd.read_csv(
        os.path.join(human_paths["root"], f"{HUMAN_CODE}_ligand_characterization.csv")
    ).sort_values("docking_suitability_score_0_1", ascending=False).iloc[0]

    summary = {
        "human": {
            "code": HUMAN_CODE,
            "chain": HUMAN_CHAIN,
            "centroid_source": f"ligando {human_ligand_row['ligand_id']}",
            "fpocket_druggability_score": float(human_ligand_row["fpocket_druggability_score"]),
            "docking_suitability_score_0_1": float(human_ligand_row["docking_suitability_score_0_1"]),
            "pocket_resnums": sorted(human_pocket),
            "triad": {"ser": human_triad_resnums[0], "asp": human_triad_resnums[1], "his": human_triad_resnums[2]},
            "full_pdb": human_full_pdb,
            "pocket_pdb": human_pocket_pdb if human_pocket_sel else None,
        },
        "cruzi": {
            "code": CRUZI_CODE,
            "chain": CRUZI_CHAIN,
            "centroid_source": f"fpocket pocket-detection ciega, pocket #{cruzi_best_pocket_id}",
            "fpocket_druggability_score": float(cruzi_drug_score),
            "pocket_resnums": sorted(cruzi_pocket),
            "triad": {"ser": cruzi_triad_resnums[0], "asp": cruzi_triad_resnums[1], "his": cruzi_triad_resnums[2],
                      "asp_resname": cruzi_asp_resname},
            "full_pdb": cruzi_full_pdb,
            "pocket_pdb": cruzi_pocket_pdb if cruzi_pocket_sel else None,
        },
        "rmsd": {
            "old_method": {
                "name": "Biopython pairwise (sin matriz de sustitucion) + Kabsch, backbone completo",
                "global_A": global_rmsd_old, "triad_A": triad_rmsd_old, "n_atoms": n_pairs_old,
            },
            "new_method": {
                "name": "PyMOL cealign, backbone completo",
                "global_A": global_rmsd_new, "triad_A": triad_rmsd_new, "n_atoms": aligned_len_new,
            },
            "new_method_refined": (
                {"name": "PyMOL cealign + pair_fit local (triada + vecinos de bolsillo)",
                 "global_A": global_rmsd_new, "triad_A": triad_rmsd_refined, "n_atoms": aligned_len_new}
                if triad_rmsd_refined is not None else None
            ),
        },
        "pse_path": pse_path,
    }
    summary_path = os.path.join(OUT_DIR, "alignment_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"  resumen JSON -> {summary_path}")

    log("")
    log("=" * 70)
    log("10. Conclusion")
    log("=" * 70)
    delta_global = global_rmsd_old - global_rmsd_new
    delta_triad = triad_rmsd_old - triad_rmsd_new
    log(
        f"  El metodo viejo (secuencia global sin matriz de sustitucion + Kabsch) daba "
        f"RMSD global = {global_rmsd_old:.3f} A y RMSD de triada = {triad_rmsd_old:.3f} A.\n"
        f"  El metodo nuevo (cealign) da RMSD global = {global_rmsd_new:.3f} A y RMSD de "
        f"triada = {triad_rmsd_new:.3f} A.\n"
        f"  Diferencia: {'mejora' if delta_global > 0 else 'empeora'} el RMSD global en "
        f"{abs(delta_global):.3f} A, y {'mejora' if delta_triad > 0 else 'empeora'} el RMSD "
        f"de la triada catalitica en {abs(delta_triad):.3f} A.\n"
    )
    if triad_rmsd_new <= global_rmsd_new:
        log(
            "  El sitio activo quedo, con cealign, MEJOR superpuesto que el promedio del "
            "resto de la proteina (RMSD triada <= RMSD global) -- no hizo falta refinamiento "
            "local adicional para este par. Esto es consistente con que la geometria del "
            "sitio catalitico esta mas conservada evolutivamente que el fold completo."
        )
    else:
        log(
            "  El sitio activo quedo PEOR superpuesto que el promedio del resto de la "
            "proteina (RMSD triada > RMSD global) -- revisar el refinamiento local de la "
            "seccion 5 antes de confiar en la superposicion global para analisis de bolsillo."
        )
    log(
        "  El notebook viejo (Superposicion.ipynb) nunca llego a correr sobre MAGL -- estaba "
        "apuntando a datos de GSK3alpha con un path de otra maquina -- asi que la observacion "
        "original de 'humano corrido hacia la derecha' no se pudo re-verificar directamente "
        "sobre esos mismos datos; los numeros de esta seccion son la primera comparacion "
        "real disponible para el par MAGL humano/cruzi."
    )


if __name__ == "__main__":
    main()
