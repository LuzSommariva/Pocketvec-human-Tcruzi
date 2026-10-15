"""
Caracterizacion de ligandos y bolsillos - extraido de Caracterizacion_conlig.py
para poder importarlo desde pipeline_steps.py en vez de tenerlo pegado en un notebook.

Contiene characterize_holo_pdb_ligand_pockets(), que implementa tus criterios:
  - anotado como HETATM, no AA/modificacion de AA
  - n_carbons > 6
  - solvent_accessibility < 0.4 (SASA en complejo / SASA aislado)
  - buriedness > 15 (atomos pesados de la proteina a 4.5 A del ligando)
  - n_residues_8A_centroid calculado
y devuelve un DataFrame ordenado por passes_all_filters, docking_suitability_score
y n_carbons -- la primera fila es el mejor ligando candidato.
"""

# === Ligand/pocket characterizer for PocketVec + fpocket ===
# Pegar esta celda despues de tus imports de PocketVec/Biopython, o ejecutarla sola.

import os
import copy
import shlex
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

try:
    from Bio.PDB.SASA import ShrakeRupley
except Exception:
    ShrakeRupley = None

# PocketVec helpers, si estan disponibles en tu entorno.
try:
    from utils.pocketvec_utils import create_pocket_centroid as _pv_create_pocket_centroid
except Exception:
    _pv_create_pocket_centroid = globals().get("create_pocket_centroid", None)

# obabel explicito (ver comentario en utils/config.py sobre un 'obabel' del
# PATH roto) -- si el modulo no esta disponible (uso standalone
# fuera del pipeline batch), se cae al 'obabel' del PATH como antes.
try:
    from utils.config import OBABEL_BIN as _OBABEL_BIN
except Exception:
    _OBABEL_BIN = "obabel"

try:
    from utils.pocketvec_utils import read_fpocket_scores as _pv_read_fpocket_scores
except Exception:
    _pv_read_fpocket_scores = globals().get("read_fpocket_scores", None)

try:
    from utils.pocketvec_utils import read_fpocket_centers as _pv_read_fpocket_centers
except Exception:
    _pv_read_fpocket_centers = globals().get("read_fpocket_centers", None)


# -----------------------------
# Parametros quimicos basicos
# -----------------------------
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Modificaciones/caps frecuentes que conviene tratar como parte de la proteina, no como ligandos dockeables.
COMMON_AA_MODS = {
    "MSE", "SEC", "PYL", "SEP", "TPO", "PTR", "CSO", "CSD", "CME", "CSE", "FME",
    "HYP", "PCA", "KCX", "LLP", "MLY", "MLZ", "M3L", "DAL", "DAR", "DSG", "DGN",
    "DGL", "DLY", "DPR", "DSN", "DTH", "DTR", "DTY", "DVA", "ORN", "SAR",
    "ACE", "NME", "NH2", "FOR"
}

WATERS = {"HOH", "WAT", "H2O", "DOD"}

# No todos son necesarios porque el filtro de carbonos ya elimina muchos, pero ayuda a limpiar PDBs reales.
COMMON_CRYSTAL_ADDITIVES = {
    "GOL", "EDO", "PEG", "PGE", "PG4", "PG6", "1PE", "MPD", "DMS", "DTT", "BME",
    "SO4", "PO4", "ACT", "ACY", "FMT", "NO3", "CL", "BR", "IOD", "TRS", "MES", "EPE",
    "HEP", "PIP", "MPO", "IPA", "EOH", "MOH", "TLA", "BU3", "TBU", "IMD"
}

COMMON_IONS = {
    "NA", "K", "CL", "CA", "MG", "MN", "ZN", "FE", "CU", "CO", "NI", "CD", "HG", "SR", "CS",
    "BA", "LI", "AL", "F", "BR", "IOD", "RB", "YB"
}

DEFAULT_EXCLUDE_RESNAMES = STANDARD_AA | COMMON_AA_MODS | WATERS | COMMON_CRYSTAL_ADDITIVES | COMMON_IONS

HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"}
POLAR_RES = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS"}
POSITIVE_RES = {"LYS", "ARG", "HIS"}
NEGATIVE_RES = {"ASP", "GLU"}
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}


# -----------------------------
# Helpers generales
# -----------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _atom_element(atom):
    """Elemento robusto para atomos PDB. Evita contar Ca/Zn como carbono."""
    el = getattr(atom, "element", "")
    el = (el or "").strip().upper()
    if el and el != "X":
        return el

    name = atom.get_name().strip().upper()
    letters = "".join([c for c in name if c.isalpha()])
    if not letters:
        return ""

    two_letter = {
        "CL", "BR", "NA", "CA", "MG", "MN", "ZN", "FE", "CU", "CO", "NI", "CD", "HG", "SR",
        "CS", "BA", "LI", "AL", "SI", "SE", "AS"
    }
    if len(letters) >= 2 and letters[:2] in two_letter:
        return letters[:2]
    return letters[0]


def _is_heavy_atom(atom):
    return _atom_element(atom) not in {"H", "D", ""}


def _residue_key(residue):
    chain = residue.get_parent()
    model = chain.get_parent()
    hetflag, resseq, icode = residue.id
    return (
        model.id,
        chain.id,
        residue.get_resname().strip(),
        hetflag.strip(),
        int(resseq),
        icode.strip() if icode.strip() else ""
    )


def _residue_label(residue_or_key):
    if isinstance(residue_or_key, tuple):
        _, chain_id, resname, _, resseq, icode = residue_or_key
    else:
        _, chain_id, resname, _, resseq, icode = _residue_key(residue_or_key)
    return f"{chain_id}:{resname}{resseq}{icode}"


def _ligand_label(residue):
    model_id, chain_id, resname, hetflag, resseq, icode = _residue_key(residue)
    return f"{resname}:{chain_id}:{resseq}{icode}"


def _is_water(residue):
    return residue.get_resname().strip() in WATERS or residue.id[0].strip() == "W"


def _is_protein_residue(residue):
    # standard=False incluye MSE y modificaciones aminoacidicas como parte de la cadena proteica.
    return (not _is_water(residue)) and is_aa(residue, standard=False)


def _ligand_atoms(residue, heavy_only=True):
    atoms = list(residue.get_atoms())
    if heavy_only:
        atoms = [a for a in atoms if _is_heavy_atom(a)]
    return atoms


def _count_carbons(residue):
    return sum(1 for atom in _ligand_atoms(residue, heavy_only=False) if _atom_element(atom) == "C")


def _count_atoms(residue, heavy_only=False):
    return len(_ligand_atoms(residue, heavy_only=heavy_only))


def _centroid_from_atoms(atoms):
    coords = np.array([a.get_coord() for a in atoms], dtype=float)
    if coords.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return coords.mean(axis=0)


def _first_model(structure):
    return next(structure.get_models())


def _protein_heavy_atoms(structure):
    atoms = []
    for residue in _first_model(structure).get_residues():
        if _is_protein_residue(residue):
            atoms.extend([a for a in residue.get_atoms() if _is_heavy_atom(a)])
    return atoms


def _candidate_ligand_residues(structure, exclude_resnames, min_carbons):
    """Devuelve residuos HETATM candidatos antes de aplicar SASA/buriedness."""
    candidates = []
    for residue in _first_model(structure).get_residues():
        resname = residue.get_resname().strip()
        hetflag = residue.id[0].strip()

        annotated_as_hetatm = bool(hetflag) and hetflag != "W"
        not_aa = not _is_protein_residue(residue)
        not_excluded = resname not in exclude_resnames
        n_carbons = _count_carbons(residue)
        enough_carbons = n_carbons > min_carbons

        if annotated_as_hetatm and not_aa and not_excluded and enough_carbons:
            candidates.append(residue)

    return candidates


def _copy_structure_without_waters(structure):
    st = copy.deepcopy(structure)
    for model in list(st):
        for chain in list(model):
            for residue in list(chain):
                if _is_water(residue):
                    chain.detach_child(residue.id)
    return st


def _sasa_complex_by_residue(structure, sasa_n_points=100):
    """SASA por residuo en el complejo, excluyendo aguas cristalograficas como bloqueadores."""
    if ShrakeRupley is None:
        raise ImportError(
            "No pude importar Bio.PDB.SASA.ShrakeRupley. "
            "Actualiza Biopython o instala una version reciente."
        )

    st = _copy_structure_without_waters(structure)
    sr = ShrakeRupley(n_points=sasa_n_points)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr.compute(st, level="A")

    out = {}
    for residue in _first_model(st).get_residues():
        sasa = 0.0
        for atom in residue.get_atoms():
            sasa += float(getattr(atom, "sasa", 0.0))
        out[_residue_key(residue)] = sasa

    return out


def _isolated_residue_sasa(residue, sasa_n_points=100):
    """SASA del ligando aislado. Se usa para accesibilidad relativa = SASA_complex / SASA_free."""
    if ShrakeRupley is None:
        raise ImportError(
            "No pude importar Bio.PDB.SASA.ShrakeRupley. "
            "Actualiza Biopython o instala una version reciente."
        )

    st = Structure("isolated_ligand")
    model = Model(0)
    chain = Chain("L")
    res_copy = copy.deepcopy(residue)

    chain.add(res_copy)
    model.add(chain)
    st.add(model)

    sr = ShrakeRupley(n_points=sasa_n_points)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr.compute(st, level="A")

    return sum(float(getattr(atom, "sasa", 0.0)) for atom in res_copy.get_atoms())


def _residues_near_point(ns, point, radius):
    residues = ns.search(np.array(point, dtype=float), radius, level="R")
    return {_residue_key(r): r for r in residues if _is_protein_residue(r)}


def _residues_near_ligand_atoms(ns, ligand_atoms, radius):
    residue_map = {}
    for atom in ligand_atoms:
        for residue in ns.search(atom.get_coord(), radius, level="R"):
            if _is_protein_residue(residue):
                residue_map[_residue_key(residue)] = residue
    return residue_map


def _protein_atoms_near_ligand_atoms(ns, ligand_atoms, radius):
    atom_ids = set()
    residues = {}

    for atom in ligand_atoms:
        for near_atom in ns.search(atom.get_coord(), radius, level="A"):
            parent_res = near_atom.get_parent()
            if _is_protein_residue(parent_res) and _is_heavy_atom(near_atom):
                atom_ids.add(near_atom.get_full_id())
                residues[_residue_key(parent_res)] = parent_res

    return atom_ids, residues


def _pocket_composition(residue_map):
    n = len(residue_map)

    if n == 0:
        return {
            "n_hydrophobic_residues_5A": 0,
            "n_polar_residues_5A": 0,
            "n_charged_residues_5A": 0,
            "n_aromatic_residues_5A": 0,
            "hydrophobic_fraction_5A": np.nan,
            "polar_fraction_5A": np.nan,
            "charged_fraction_5A": np.nan,
            "aromatic_fraction_5A": np.nan,
        }

    resnames = [r.get_resname().strip() for r in residue_map.values()]

    n_hydro = sum(r in HYDROPHOBIC_RES for r in resnames)
    n_polar = sum(r in POLAR_RES for r in resnames)
    n_charged = sum(r in POSITIVE_RES or r in NEGATIVE_RES for r in resnames)
    n_arom = sum(r in AROMATIC_RES for r in resnames)

    return {
        "n_hydrophobic_residues_5A": n_hydro,
        "n_polar_residues_5A": n_polar,
        "n_charged_residues_5A": n_charged,
        "n_aromatic_residues_5A": n_arom,
        "hydrophobic_fraction_5A": n_hydro / n,
        "polar_fraction_5A": n_polar / n,
        "charged_fraction_5A": n_charged / n,
        "aromatic_fraction_5A": n_arom / n,
    }


# -----------------------------
# fpocket helpers
# -----------------------------
class _ProteinOnlySelect(Select):
    def __init__(self, keep_het_resnames=None):
        self.keep_het_resnames = set(keep_het_resnames or [])

    def accept_residue(self, residue):
        if _is_protein_residue(residue):
            return 1
        if residue.get_resname().strip() in self.keep_het_resnames:
            return 1
        return 0


def _write_protein_only_pdb(structure, outfile, keep_het_resnames=None):
    io = PDBIO()
    io.set_structure(_first_model(structure))
    io.save(str(outfile), _ProteinOnlySelect(keep_het_resnames=keep_het_resnames))
    return str(outfile)


def _fallback_read_fpocket_scores(path_to_file):
    fpocket = {}
    pocket = None

    with open(path_to_file, "r") as f:
        for line in f:
            s = line.strip()

            if not s:
                continue

            if s.startswith("Pocket ") and ":" in s:
                try:
                    pocket = int(s.split(":", 1)[0].replace("Pocket", "").strip())
                    fpocket[pocket] = {}
                except Exception:
                    pocket = None

            elif pocket is not None and ":" in s:
                key, val = s.split(":", 1)
                fpocket[pocket][key.strip()] = _safe_float(val.strip())

    return fpocket


def _fallback_read_fpocket_centers(path_to_file):
    centers = {}

    with open(path_to_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    # En el PQR de fpocket, el numero de residuo suele ser el ID de pocket.
                    pocket_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except Exception:
                    parts = line.split()

                    if len(parts) < 8:
                        continue

                    try:
                        pocket_id = int(parts[4])
                        x, y, z = map(float, parts[5:8])
                    except Exception:
                        continue

                centers.setdefault(pocket_id, []).append([x, y, z])

    return {k: np.array(v, dtype=float) for k, v in centers.items()}


def _read_fpocket_scores_safe(info_path):
    if _pv_read_fpocket_scores is not None:
        try:
            return _pv_read_fpocket_scores(info_path)
        except Exception:
            pass

    return _fallback_read_fpocket_scores(info_path)


def _read_fpocket_centers_safe(pqr_path):
    if _pv_read_fpocket_centers is not None:
        try:
            return _pv_read_fpocket_centers(pqr_path)
        except Exception:
            pass

    return _fallback_read_fpocket_centers(pqr_path)


def _run_fpocket(receptor_pdb, path_to_fpocket, overwrite=False):
    receptor_pdb = Path(receptor_pdb).resolve()
    workdir = receptor_pdb.parent
    stem = receptor_pdb.stem

    outdir = workdir / f"{stem}_out"
    info_path = outdir / f"{stem}_info.txt"
    pqr_path = outdir / f"{stem}_pockets.pqr"

    if info_path.exists() and pqr_path.exists() and not overwrite:
        return str(info_path), str(pqr_path), str(outdir), 0

    cwd = os.getcwd()
    os.chdir(str(workdir))

    try:
        command = f"{shlex.quote(str(path_to_fpocket))} -f {shlex.quote(receptor_pdb.name)}"
        exit_code = os.system(command)
    finally:
        os.chdir(cwd)

    return str(info_path), str(pqr_path), str(outdir), exit_code


def _metric_from_fpocket_dict(d, possible_names):
    if not isinstance(d, dict):
        return np.nan

    # exact match primero
    for name in possible_names:
        if name in d:
            return d[name]

    # match flexible despues
    lower = {str(k).lower(): k for k in d.keys()}

    for name in possible_names:
        target = name.lower()

        for low_key, original_key in lower.items():
            if target == low_key or target in low_key:
                return d[original_key]

    return np.nan


def _load_fpocket_results(fpocket_info_path, fpocket_pqr_path):
    if not (fpocket_info_path and fpocket_pqr_path):
        return {}, {}

    if not (os.path.exists(fpocket_info_path) and os.path.exists(fpocket_pqr_path)):
        return {}, {}

    scores = _read_fpocket_scores_safe(fpocket_info_path)
    raw_centers = _read_fpocket_centers_safe(fpocket_pqr_path)

    centers = {}
    for pocket_id, coords in raw_centers.items():
        coords = np.array(coords, dtype=float)

        if coords.shape[0] > 0:
            centers[int(pocket_id)] = coords.mean(axis=0)

    return scores, centers


def _match_nearest_fpocket_pocket(
    ligand_centroid,
    fpocket_scores,
    fpocket_centers,
    ns,
    ligand_residue_keys_5A,
    fpocket_residue_radius=8.0
):
    empty = {
        "fpocket_nearest_pocket": np.nan,
        "fpocket_center_x": np.nan,
        "fpocket_center_y": np.nan,
        "fpocket_center_z": np.nan,
        "fpocket_dist_to_ligand_centroid_A": np.nan,
        "fpocket_score": np.nan,
        "fpocket_druggability_score": np.nan,
        "fpocket_volume_A3": np.nan,
        "fpocket_n_alpha_spheres": np.nan,
        "fpocket_residue_overlap_jaccard_5A": np.nan,
    }

    if not fpocket_centers:
        return empty

    distances = {
        pid: float(np.linalg.norm(np.array(center, dtype=float) - np.array(ligand_centroid, dtype=float)))
        for pid, center in fpocket_centers.items()
    }

    nearest_id = min(distances, key=distances.get)
    center = np.array(fpocket_centers[nearest_id], dtype=float)
    pocket_dict = fpocket_scores.get(nearest_id, {})

    fpocket_residues = _residues_near_point(ns, center, fpocket_residue_radius)
    pred_keys = set(fpocket_residues.keys())
    lig_keys = set(ligand_residue_keys_5A)
    union = pred_keys | lig_keys
    jaccard = len(pred_keys & lig_keys) / len(union) if union else np.nan

    return {
        "fpocket_nearest_pocket": int(nearest_id),
        "fpocket_center_x": float(center[0]),
        "fpocket_center_y": float(center[1]),
        "fpocket_center_z": float(center[2]),
        "fpocket_dist_to_ligand_centroid_A": distances[nearest_id],
        "fpocket_score": _metric_from_fpocket_dict(pocket_dict, ["Score"]),
        "fpocket_druggability_score": _metric_from_fpocket_dict(pocket_dict, ["Druggability Score", "Drug Score"]),
        "fpocket_volume_A3": _metric_from_fpocket_dict(pocket_dict, ["Volume", "Pocket volume", "Pocket Volume"]),
        "fpocket_n_alpha_spheres": _metric_from_fpocket_dict(
            pocket_dict,
            ["Number of Alpha Spheres", "Nb AS", "Alpha spheres"]
        ),
        "fpocket_residue_overlap_jaccard_5A": jaccard,
    }


# -----------------------------
# Clasificacion heuristica
# -----------------------------
def _classify_docking_suitability(row, accessibility_cutoff, buriedness_cutoff):
    """
    Clasificacion practica, no reemplaza redocking.
    Usa solo senales estructurales/quimicas y fpocket si esta disponible.
    """
    flags = []
    score = 0
    total = 0

    def add_check(condition, bad_flag):
        nonlocal score, total
        total += 1

        if bool(condition):
            score += 1
        else:
            flags.append(bad_flag)

    add_check(row["solvent_accessibility"] < accessibility_cutoff, "ligando_muy_expuesto")
    add_check(row["buriedness"] > buriedness_cutoff, "pocos_contactos_proteina_ligando")
    add_check(row["n_residues_8A_centroid"] >= 10, "pocos_residuos_en_entorno_8A")

    if pd.notna(row.get("fpocket_dist_to_ligand_centroid_A", np.nan)):
        add_check(row["fpocket_dist_to_ligand_centroid_A"] <= 6.0, "fpocket_no_coincide_con_ligando")

    if pd.notna(row.get("fpocket_volume_A3", np.nan)):
        # Rango amplio para small molecules; ligandos grandes o pockets alostericos pueden salir de este rango.
        add_check(100.0 <= row["fpocket_volume_A3"] <= 1500.0, "volumen_fpocket_extremo")

    if pd.notna(row.get("charged_fraction_5A", np.nan)):
        add_check(row["charged_fraction_5A"] <= 0.50, "entorno_muy_cargado")

    frac = score / total if total else np.nan

    if not row["passes_all_filters"]:
        label = "Descartar/revisar filtros"
    elif frac >= 0.80:
        label = "Favorable para docking"
    elif frac >= 0.60:
        label = "Intermedio: revisar"
    else:
        label = "Dificil: revisar estructura/protocolo"

    return label, frac, ";".join(flags) if flags else "OK"


# -----------------------------
# Funcion principal
# -----------------------------
def characterize_holo_pdb_ligand_pockets(
    pdb_code_or_path,
    outpath="examples/1_preparation/",
    path_to_fpocket="fpocket",
    fpocket_receptor_pdb=None,
    run_fpocket=True,
    overwrite_fpocket=False,
    keep_het_resnames_for_fpocket=None,
    min_carbons=6,
    accessibility_cutoff=0.4,
    buriedness_cutoff=15,
    buriedness_mode="contacts",       # "contacts" o "delta_sasa"
    contact_cutoff_A=4.5,
    ligand_pocket_cutoff_A=5.0,
    centroid_environment_cutoff_A=8.0,
    fpocket_residue_radius_A=8.0,
    sasa_n_points=100,
    exclude_resnames=None,
    write_centroid_sd=False,
    return_all=False,
    verbose=True,
):
    """
    Identifica, filtra y caracteriza ligandos validos en una estructura Holo-PDB.

    Criterios estrictos implementados:
    1) HETATM / hetero-residue anotado como ligando.
    2) No aminoacido estandar ni modificacion comun de cadena proteica.
    3) Mas de min_carbons atomos de carbono. Por defecto min_carbons=6 equivale a n_C > 6.
    4) Solvent accessibility = SASA_ligando_en_complejo / SASA_ligando_aislado < accessibility_cutoff.
    5) Buriedness > buriedness_cutoff.
       - buriedness_mode="contacts": numero de atomos pesados proteicos a contact_cutoff_A
         de cualquier atomo pesado del ligando.
       - buriedness_mode="delta_sasa": SASA_aislado - SASA_en_complejo, en A^2.
    6) Calcula n_residues_8A_centroid: residuos proteicos a menos de 8 A del centroide del ligando.

    Devuelve por defecto solo los ligandos que pasan filtros. Usa return_all=True para ver descartados.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    # Resolver entrada: codigo PDB o ruta directa.
    if os.path.isfile(str(pdb_code_or_path)):
        pdb_path = Path(pdb_code_or_path).resolve()
        pdb_code = pdb_path.stem
    else:
        pdb_code = str(pdb_code_or_path)
        pdb_path = (outpath / f"{pdb_code}.pdb").resolve()

    if not pdb_path.exists():
        raise FileNotFoundError(f"No encontre el PDB holo: {pdb_path}")

    exclude_resnames = set(DEFAULT_EXCLUDE_RESNAMES if exclude_resnames is None else exclude_resnames)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_code, str(pdb_path))

    protein_atoms = _protein_heavy_atoms(structure)
    if len(protein_atoms) == 0:
        raise ValueError("No encontre atomos proteicos. Revisa si el PDB tiene registros ATOM/proteina.")

    ns = NeighborSearch(protein_atoms)

    candidates = _candidate_ligand_residues(
        structure,
        exclude_resnames=exclude_resnames,
        min_carbons=min_carbons
    )

    if verbose:
        print(f"PDB holo: {pdb_path}")
        print(f"Ligandos candidatos tras HETATM/no-AA/n_C>{min_carbons}: {len(candidates)}")

    # SASA complejo para todos los residuos una sola vez.
    sasa_by_residue = _sasa_complex_by_residue(structure, sasa_n_points=sasa_n_points)

    # fpocket: correr sobre receptor sin ligandos, salvo que pases un receptor ya preparado.
    fpocket_info_path = None
    fpocket_pqr_path = None
    fpocket_outdir = None
    fpocket_exit_code = np.nan
    fpocket_scores = {}
    fpocket_centers = {}

    if run_fpocket:
        if fpocket_receptor_pdb is None:
            fpocket_receptor_pdb = outpath / f"{pdb_code}_protein_only_for_fpocket.pdb"
            _write_protein_only_pdb(
                structure,
                fpocket_receptor_pdb,
                keep_het_resnames=keep_het_resnames_for_fpocket
            )
        else:
            fpocket_receptor_pdb = Path(fpocket_receptor_pdb).resolve()

            if not fpocket_receptor_pdb.exists():
                raise FileNotFoundError(f"No encontre fpocket_receptor_pdb: {fpocket_receptor_pdb}")

        fpocket_info_path, fpocket_pqr_path, fpocket_outdir, fpocket_exit_code = _run_fpocket(
            fpocket_receptor_pdb,
            path_to_fpocket=path_to_fpocket,
            overwrite=overwrite_fpocket
        )

        if fpocket_exit_code != 0:
            warnings.warn(
                f"fpocket devolvio codigo {fpocket_exit_code}. "
                f"Revisar instalacion/ruta: {path_to_fpocket}"
            )

        fpocket_scores, fpocket_centers = _load_fpocket_results(fpocket_info_path, fpocket_pqr_path)

        if verbose:
            print(f"fpocket outdir: {fpocket_outdir}")
            print(f"Pockets fpocket leidos: {len(fpocket_centers)}")

    rows = []
    centroid_dir = outpath / "ligand_centroids"

    if write_centroid_sd:
        centroid_dir.mkdir(parents=True, exist_ok=True)

    for residue in candidates:
        key = _residue_key(residue)
        ligand_id = _ligand_label(residue)
        resname = residue.get_resname().strip()
        model_id, chain_id, _, hetflag, resseq, icode = key

        lig_atoms_heavy = _ligand_atoms(residue, heavy_only=True)
        lig_centroid = _centroid_from_atoms(lig_atoms_heavy)

        n_carbons = _count_carbons(residue)
        n_atoms = _count_atoms(residue, heavy_only=False)
        n_heavy_atoms = _count_atoms(residue, heavy_only=True)

        # Entorno del sitio activo.
        residues_8A_centroid = _residues_near_point(ns, lig_centroid, centroid_environment_cutoff_A)
        residues_5A_ligand = _residues_near_ligand_atoms(ns, lig_atoms_heavy, ligand_pocket_cutoff_A)
        contact_atom_ids, contact_residues = _protein_atoms_near_ligand_atoms(ns, lig_atoms_heavy, contact_cutoff_A)

        # Accesibilidad relativa y buried SASA.
        sasa_complex = sasa_by_residue.get(key, np.nan)

        try:
            sasa_free = _isolated_residue_sasa(residue, sasa_n_points=sasa_n_points)
        except Exception as e:
            warnings.warn(f"No pude calcular SASA aislado para {ligand_id}: {e}")
            sasa_free = np.nan

        solvent_accessibility = sasa_complex / sasa_free if pd.notna(sasa_free) and sasa_free > 0 else np.nan
        buried_sasa_A2 = sasa_free - sasa_complex if pd.notna(sasa_free) and pd.notna(sasa_complex) else np.nan
        buried_fraction = 1.0 - solvent_accessibility if pd.notna(solvent_accessibility) else np.nan

        buriedness_contacts = len(contact_atom_ids)
        buriedness_residues = len(contact_residues)

        if buriedness_mode == "contacts":
            buriedness = buriedness_contacts
        elif buriedness_mode == "delta_sasa":
            buriedness = buried_sasa_A2
        else:
            raise ValueError("buriedness_mode debe ser 'contacts' o 'delta_sasa'")

        # Composicion quimica del bolsillo experimental: residuos a 5 A de atomos del ligando.
        comp = _pocket_composition(residues_5A_ligand)

        # fpocket: pocket geometrico mas cercano al centroide del ligando.
        fpocket_metrics = _match_nearest_fpocket_pocket(
            lig_centroid,
            fpocket_scores=fpocket_scores,
            fpocket_centers=fpocket_centers,
            ns=ns,
            ligand_residue_keys_5A=set(residues_5A_ligand.keys()),
            fpocket_residue_radius=fpocket_residue_radius_A
        )

        centroid_sd_path = ""

        if write_centroid_sd:
            centroid_sd_path = str(centroid_dir / f"CTR_LIG_{resname}_{chain_id}_{resseq}{icode}.sd")

            if _pv_create_pocket_centroid is not None:
                try:
                    _pv_create_pocket_centroid(lig_centroid, centroid_sd_path, obabel_bin=_OBABEL_BIN)
                except Exception as e:
                    warnings.warn(f"No pude escribir centroide SD para {ligand_id}: {e}")
            else:
                warnings.warn("create_pocket_centroid no esta disponible; no escribo archivo SD.")

        filters = {
            "annotated_HETATM": bool(hetflag) and hetflag != "W",
            "not_standard_or_modified_AA": (
                not _is_protein_residue(residue)
            ) and (resname not in STANDARD_AA) and (resname not in COMMON_AA_MODS),
            f"n_carbons_gt_{min_carbons}": n_carbons > min_carbons,
            f"solvent_accessibility_lt_{accessibility_cutoff}": (
                pd.notna(solvent_accessibility) and solvent_accessibility < accessibility_cutoff
            ),
            f"buriedness_gt_{buriedness_cutoff}": (
                pd.notna(buriedness) and buriedness > buriedness_cutoff
            ),
            "n_residues_8A_centroid_calculated": (
                pd.notna(len(residues_8A_centroid)) and len(residues_8A_centroid) > 0
            ),
        }

        passes_all = all(filters.values())
        failed = [name for name, ok in filters.items() if not ok]

        row = {
            "pdb_code": pdb_code,
            "pdb_path": str(pdb_path),
            "ligand_id": ligand_id,
            "model": model_id,
            "chain": chain_id,
            "resname": resname,
            "resseq": resseq,
            "icode": icode,
            "hetflag": hetflag,
            "n_atoms": n_atoms,
            "n_heavy_atoms": n_heavy_atoms,
            "n_carbons": n_carbons,
            "centroid_x": float(lig_centroid[0]),
            "centroid_y": float(lig_centroid[1]),
            "centroid_z": float(lig_centroid[2]),
            "sasa_complex_A2": sasa_complex,
            "sasa_free_A2": sasa_free,
            "solvent_accessibility": solvent_accessibility,
            "buried_sasa_A2": buried_sasa_A2,
            "buried_fraction": buried_fraction,
            "buriedness": buriedness,
            "buriedness_mode": buriedness_mode,
            "buriedness_contact_atoms_4p5A": buriedness_contacts,
            "buriedness_contact_residues_4p5A": buriedness_residues,
            "n_residues_8A_centroid": len(residues_8A_centroid),
            "n_residues_5A_ligand_atoms": len(residues_5A_ligand),
            "pocket_residues_5A": ";".join(sorted(_residue_label(k) for k in residues_5A_ligand.keys())),
            "pocket_residues_8A_centroid": ";".join(sorted(_residue_label(k) for k in residues_8A_centroid.keys())),
            "centroid_sd_path": centroid_sd_path,
            "fpocket_info_path": fpocket_info_path or "",
            "fpocket_pqr_path": fpocket_pqr_path or "",
            "fpocket_outdir": fpocket_outdir or "",
            "fpocket_exit_code": fpocket_exit_code,
            "passes_all_filters": passes_all,
            "failed_filters": ";".join(failed) if failed else "OK",
        }

        row.update(comp)
        row.update(fpocket_metrics)

        # Clasificacion heuristica final para reportar dockability sin hacer redocking.
        label, score_frac, flags = _classify_docking_suitability(
            pd.Series(row),
            accessibility_cutoff=accessibility_cutoff,
            buriedness_cutoff=buriedness_cutoff,
        )

        row["docking_suitability"] = label
        row["docking_suitability_score_0_1"] = score_frac
        row["docking_flags"] = flags

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=[
            "pdb_code", "ligand_id", "chain", "resname", "resseq", "n_carbons",
            "solvent_accessibility", "buriedness", "n_residues_8A_centroid",
            "fpocket_nearest_pocket", "fpocket_volume_A3", "docking_suitability",
            "passes_all_filters", "failed_filters"
        ])
        return df

    preferred_cols = [
        "pdb_code", "ligand_id", "chain", "resname", "resseq", "n_carbons", "n_heavy_atoms",
        "centroid_x", "centroid_y", "centroid_z",
        "solvent_accessibility", "buriedness", "buriedness_mode", "buried_sasa_A2", "buried_fraction",
        "n_residues_8A_centroid", "n_residues_5A_ligand_atoms",
        "hydrophobic_fraction_5A", "polar_fraction_5A", "charged_fraction_5A", "aromatic_fraction_5A",
        "fpocket_nearest_pocket", "fpocket_dist_to_ligand_centroid_A", "fpocket_volume_A3",
        "fpocket_score", "fpocket_druggability_score", "fpocket_residue_overlap_jaccard_5A",
        "docking_suitability", "docking_suitability_score_0_1", "docking_flags",
        "passes_all_filters", "failed_filters",
        "pocket_residues_5A", "pocket_residues_8A_centroid",
        "centroid_sd_path", "fpocket_info_path", "fpocket_pqr_path", "pdb_path"
    ]

    ordered = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]

    df = df[ordered].sort_values(
        by=["passes_all_filters", "docking_suitability_score_0_1", "n_carbons"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    if not return_all:
        df = df[df["passes_all_filters"]].reset_index(drop=True)

    return df
