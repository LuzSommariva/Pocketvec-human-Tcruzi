"""
Configuración centralizada para el pipeline PocketVec.

All paths and tool locations are read from ``config/config.yaml`` (relative
paths are resolved against the repository root, or against
``$POCKETVEC_BASE_DIR`` if set). An additional YAML file given in
``$POCKETVEC_CONFIG`` is merged on top of it (see ``config/example.yaml``).
The individual ``POCKETVEC_*`` environment variables documented below still
override the YAML values.

Idea: en vez de escribir el pdb_code, la cadena, el ligando, etc. a mano
en cada celda de cada notebook, los completás UNA vez acá por proteína,
y todos los scripts lo leen desde este archivo.
"""

import os
import shutil

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def _deep_merge(base, override):
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path=DEFAULT_CONFIG_PATH, override_path=None):
    """Load the base YAML config and optionally merge an override file on top."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    override_path = override_path or os.environ.get("POCKETVEC_CONFIG")
    if override_path:
        if not os.path.isabs(override_path):
            override_path = os.path.join(PROJECT_ROOT, override_path)
        with open(override_path) as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    return cfg


CONFIG = load_config()

# Única ruta "base" del proyecto: todas las rutas relativas de config.yaml
# se resuelven contra ella.
BASE_DIR = os.environ.get("POCKETVEC_BASE_DIR", PROJECT_ROOT)


def resolve_path(value):
    """Resolve a config path against BASE_DIR (absolute paths are kept)."""
    if value is None or value == "":
        return value
    value = os.path.expanduser(str(value))
    return value if os.path.isabs(value) else os.path.join(BASE_DIR, value)


def resolve_tool(value):
    """
    Resolve an external tool: a bare command name is looked up in PATH
    (absolute path returned if found), anything containing a path separator
    is resolved like a regular config path.
    """
    if not value:
        return value
    value = str(value)
    if os.sep not in value:
        return shutil.which(value) or value
    return resolve_path(value)


_PATHS = CONFIG.get("paths", {})
_TOOLS = CONFIG.get("tools", {})
_POCKETVEC = CONFIG.get("pocketvec", {})
_CHEMBL = CONFIG.get("chembl", {})

# Pipeline inputs / outputs
TARGETS_CSV = resolve_path(_PATHS.get("targets_csv"))
RESULTS_CSV = resolve_path(_PATHS.get("results_csv"))
QC_TARGETS_CSV = resolve_path(_PATHS.get("qc_targets_csv"))
PREPARATION_DIR = resolve_path(_PATHS.get("preparation_dir"))
POCKETVEC_DIR = resolve_path(_PATHS.get("pocketvec_dir"))
ANALYSIS_DIR = resolve_path(_PATHS.get("analysis_dir"))
CASE_STUDY_DIR = resolve_path(_PATHS.get("case_study_dir"))
LOCAL_ALPHAFOLD_SEARCH_DIRS = [resolve_path(p) for p in _PATHS.get("local_alphafold_dirs", [])]

# ChEMBL target-selection stage
CHEMBL_DB = resolve_path(_CHEMBL.get("db"))
CHEMBL_RAW_DIR = resolve_path(_CHEMBL.get("raw_dir"))
CHEMBL_FILTERED_PKL = resolve_path(_CHEMBL.get("filtered_pkl"))
SIFTS_PATH = resolve_path(_CHEMBL.get("sifts"))
ENTRIES_IDX_PATH = resolve_path(_CHEMBL.get("entries_idx"))
CHEMBL_CANDIDATES_CSV = resolve_path(_CHEMBL.get("candidates_csv"))
CHEMBL_SCREENING_RUN_DIR = resolve_path(_CHEMBL.get("screening_run_dir"))
CHEMBL_TARGET_RUN_DIR = resolve_path(_CHEMBL.get("target_run_dir"))
CHEMBL_RANKED_CSV = resolve_path(_CHEMBL.get("ranked_csv"))
CHEMBL_TOP_CSV = resolve_path(_CHEMBL.get("top_csv"))

# --------------------------------------------------------------------------
# Rutas a herramientas externas y parámetros de PocketVec -- ÚNICO lugar
# donde deberían estar definidas. pipeline_steps.py y pocketvec_utils.py las
# importan de acá en vez de tener cada uno su propia copia.
#
# Todas overrideables por variable de entorno. Los paths a binarios se
# resuelven a rutas ABSOLUTAS (no relativas al cwd desde donde se invoque el
# script) para que environmental_variables()/rbcavity/rbdock no dependan de
# desde dónde se corra el batch.
#
# IMPORTANTE sobre obabel: NO uses el 'obabel' plano que resuelve el PATH de
# la shell si puede haber instalaciones rotas (openbabel-wheel sin el módulo
# openbabel) que se cuelan antes que la del env de conda. Usá siempre
# OBABEL_BIN acá definido, explícito.
# --------------------------------------------------------------------------
PATH_TO_CHECK_STRUCTURE = resolve_tool(
    os.environ.get("POCKETVEC_CHECK_STRUCTURE", _TOOLS.get("check_structure", "check_structure"))
)
PATH_TO_FPOCKET = resolve_tool(os.environ.get("POCKETVEC_FPOCKET", _TOOLS.get("fpocket", "fpocket")))
PATH_TO_PRANK = resolve_tool(os.environ.get("POCKETVEC_PRANK", _TOOLS.get("prank", "prank")))
OBABEL_BIN = resolve_tool(os.environ.get("POCKETVEC_OBABEL", _TOOLS.get("obabel", "obabel")))
# rDock installation directory (must contain bin/, lib/ and dock.prm).
PATH_TO_RDOCK = resolve_path(_TOOLS.get("rdock_root")).rstrip("/") + "/"
# Optional MOE protein preparation (falls back to obabel when unavailable).
MOEBATCH_BIN = resolve_tool(_TOOLS.get("moebatch") or "moebatch")
MOE_FUNCTIONS_SVL = resolve_path(_TOOLS.get("moe_functions")) or ""

PATH_TO_LIB = resolve_path(_POCKETVEC.get("ligand_library"))
PATH_TO_LIB_ORDER = resolve_path(_POCKETVEC.get("ligand_library_order"))

# Parámetros de PocketVec. RADIUS es el radio (en A) de la esfera de
# búsqueda de cavidad alrededor del centroide (rbcavity); NRUNS es el
# número de corridas de docking por molécula (rbdock). El volumen de la
# cavidad NO depende de NRUNS -- solo de RADIUS y del receptor/centroide.
POCKETVEC_RADIUS = str(os.environ.get("POCKETVEC_RADIUS", _POCKETVEC.get("radius", "12.0")))
POCKETVEC_NRUNS = int(os.environ.get("POCKETVEC_NRUNS", _POCKETVEC.get("nruns", 25)))

# --------------------------------------------------------------------------
# Config por proteína.
# --------------------------------------------------------------------------
# NOTA: ya no se usa este dict como fuente de verdad -- la config por
# proteína ahora vive en targets.csv (una fila por target, con columnas
# human_* y cruzi_*). Ver pipeline_steps.parse_row_cfg().
# Se deja get_protein_config() por compatibilidad si algún notebook viejo
# todavía la llama directo.
PROTEINS = {}


def get_protein_config(pdb_code):
    """Devuelve el dict de config para pdb_code, o avisa si falta agregarlo."""
    if pdb_code not in PROTEINS:
        raise KeyError(
            f"'{pdb_code}' no está en PROTEINS (dict legacy, vacío). "
            f"La config ahora sale de targets.csv -- usá "
            f"pipeline_steps.parse_row_cfg(row) en vez de esta función."
        )
    return PROTEINS[pdb_code]


def get_protein_paths(pdb_code, base_dir=None):
    """
    Devuelve (y crea si no existen) las rutas estándar para una proteína,
    ya con la carpeta propia por proteína. Reemplaza el 'outpath' compartido
    que causaba que pockets_PD se pisara entre proteínas.

    ``base_dir`` (optional) replaces BASE_DIR as the parent of the default
    ``1_preparation``/``2_pocketvec`` folders; by default the configured
    ``preparation_dir``/``pocketvec_dir`` are used.
    """
    preparation_dir = os.path.join(base_dir, "1_preparation") if base_dir else PREPARATION_DIR
    pocketvec_dir = os.path.join(base_dir, "2_pocketvec") if base_dir else POCKETVEC_DIR
    root = os.path.join(preparation_dir, pdb_code)

    paths = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fpocket_out": os.path.join(root, "fpocket_out"),
        "pockets_PD": os.path.join(root, "pockets_PD"),
        "ligand_centroids": os.path.join(root, "ligand_centroids"),
        # Salida de PocketVec (rDock + fingerprint) -- separada de
        # 1_preparation porque es una etapa distinta del pipeline.
        "pocketvec_out": os.path.join(pocketvec_dir, pdb_code),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def set_active_protein(pdb_code):
    """
    Guarda el pdb_code activo en un archivo compartido (.active_protein en la
    raíz del repo) para que otros scripts/notebooks lo lean solos,
    sin que tengas que volver a escribirlo.
    """
    with open(os.path.join(BASE_DIR, ".active_protein"), "w") as f:
        f.write(pdb_code.strip())


def get_active_protein():
    """Lee el pdb_code activo guardado por set_active_protein()."""
    path = os.path.join(BASE_DIR, ".active_protein")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No hay proteína activa guardada. Corré set_active_protein('XXXX') "
            "o definí pdb_code a mano en este notebook."
        )
    with open(path) as f:
        return f.read().strip()
