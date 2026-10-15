"""
tunnel_magl.py  (formerly Tunel_MAGL.py)

Analisis del tunel de acceso al sustrato de MAGL humana (3PE6) vs. MAGL de
T. cruzi (AF-Q4CSM0-F1), con punto de partida en el OG de la Ser catalitica
de cada lado (Ser122 humano / Ser139 cruzi).

*** POR QUE NO USA CAVER ***
CAVER (plugin de PyMOL) NO esta instalado en este entorno y no hay una copia
local en el servidor. Su distribucion oficial (caver.cz) requiere completar
un formulario de registro (nombre/email/institucion) para descargar el
.zip -- un paso que no puedo completar en nombre del usuario sin su
intervencion. En su lugar, este script implementa un metodo geometrico
propio, conceptualmente equivalente al nucleo de CAVER (busqueda del
"camino mas ancho" / maximum bottleneck path a traves de una grilla de
huecos, en vez de la teselacion de Voronoi + Dijkstra que usa CAVER):

  1. Grilla 3D (spacing configurable) centrada en el atomo de partida.
  2. Para cada punto de la grilla, radio libre = distancia al atomo pesado
     de proteina mas cercano menos su radio de van der Waals (KDTree para
     que sea rapido).
  3. Dijkstra de costo minimo, donde el costo de entrar a un voxel penaliza
     fuerte los radios libres chicos (costo = paso * max(1, R_REF/radio)^exp)
     -- el mismo principio que la ponderacion de aristas de CAVER sobre su
     malla de Voronoi (favorece huecos anchos, pero ademas, a diferencia de
     un camino de "bottleneck maximo" puro, penaliza la distancia recorrida,
     asi que no se desvia sin necesidad). Termina en el primer voxel
     "exterior" (radio libre > EXTERIOR_RADIUS_A, ya en solvente masivo) que
     Dijkstra extrae de la cola -- por construccion es el mas barato
     alcanzable, i.e. la salida "natural" mas corta/ancha. El bottleneck
     reportado es el minimo radio libre a lo largo de ese camino.
  4. Longitud del tunel = longitud del camino reconstruido (suma de pasos
     euclidianos en la grilla).
  5. Residuos que revisten el tunel = cualquier residuo con un atomo
     pesado a <= LINING_CUTOFF_A del eje del tunel (los puntos del camino).

Esto NO reemplaza una corrida real de CAVER (que ademas soporta trayectorias
de MD, deteccion de multiples tuneles alternativos, y esta validado en la
literatura) -- se documenta explicitamente como aproximacion para poder
responder longitud / bottleneck / composicion de forma cuantitativa mientras
no se instale la herramienta real. Si el usuario decide registrarse en
caver.cz y descargar el plugin, este script puede ser reemplazado sin tocar
el resto del pipeline (opera sobre archivos PDB independientes).
"""

import heapq
import json
import os
import sys

import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

# src/ (three levels up) must be on sys.path to import utils.* as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import CASE_STUDY_DIR, PREPARATION_DIR

VDW_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.10}
DEFAULT_VDW = 1.70

HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY", "CYS", "TYR"}
POLAR = {"SER", "THR", "ASN", "GLN", "HIS"}
CHARGED = {"ASP", "GLU", "LYS", "ARG"}

OUT_DIR = CASE_STUDY_DIR  # config: paths.case_study_dir
SUB_DIR = os.path.join(OUT_DIR, "subregiones")
os.makedirs(SUB_DIR, exist_ok=True)

# Parametros de la grilla / busqueda
BOX_HALF_WIDTH_A = 26.0    # A, medio-ancho de la caja cubica centrada en el atomo de partida
GRID_SPACING_A = 1.0       # A
EXTERIOR_RADIUS_A = 5.5    # A, radio libre a partir del cual se considera "solvente masivo"
LINING_CUTOFF_A = 4.5      # A, distancia atomo-de-residuo <-> eje del tunel para contar como "revestidor"
PROBE_MIN_RADIUS_A = 0.6   # A, radio libre minimo para que un voxel sea transitable (permite cuellos angostos)
R_REF_A = 2.2              # A, radio "comodo" de referencia para la penalizacion de costo
PENALTY_EXPONENT = 3.0     # cuanto mas alto, mas fuerte castiga los cuellos angostos


def load_protein_only_chain(pdb_path, chain_id, exclude_resnames=()):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    chain = structure[0][chain_id]
    atoms = []
    residues = []
    for res in chain:
        if res.id[0] != " ":
            continue
        if res.get_resname() in exclude_resnames:
            continue
        residues.append(res)
        for atom in res:
            if atom.element == "H":
                continue
            atoms.append((atom.get_coord(), atom.element, res.id[1], res.get_resname()))
    return atoms, residues


def build_grid(start_point, half_width=BOX_HALF_WIDTH_A, spacing=GRID_SPACING_A):
    n = int(round(2 * half_width / spacing)) + 1
    axis = (np.arange(n) - n // 2) * spacing
    origin = start_point
    coords_1d = axis
    return origin, coords_1d, n


def widest_path_tunnel(atoms, start_point, half_width=BOX_HALF_WIDTH_A, spacing=GRID_SPACING_A):
    coords = np.array([a[0] for a in atoms])
    elements = [a[1] for a in atoms]
    vdw = np.array([VDW_RADIUS.get(e, DEFAULT_VDW) for e in elements])
    tree = cKDTree(coords)

    origin, axis, n = build_grid(start_point, half_width, spacing)
    # grilla de puntos: origin + axis[i,j,k]
    xs = origin[0] + axis
    ys = origin[1] + axis
    zs = origin[2] + axis
    grid_points = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    dist_to_atom, nearest_idx = tree.query(grid_points, k=1)
    free_radius = dist_to_atom - vdw[nearest_idx]
    free_radius = free_radius.reshape(n, n, n)

    def to_idx(i, j, k):
        return (i * n + j) * n + k

    # voxel de partida = el mas cercano al punto de partida real (centro de la grilla)
    start_i = start_j = start_k = n // 2
    start_idx = to_idx(start_i, start_j, start_k)

    N = n * n * n
    best_cost = np.full(N, np.inf)
    best_cost[start_idx] = 0.0
    min_radius_on_path = np.full(N, np.inf)
    min_radius_on_path[start_idx] = free_radius[start_i, start_j, start_k]
    visited = np.zeros(N, dtype=bool)
    prev = np.full(N, -1, dtype=np.int64)

    heap = [(0.0, start_idx)]

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]
    offset_len = {o: spacing * np.linalg.norm(o) for o in offsets}

    flat_free = free_radius.reshape(-1)

    end_idx = None
    while heap:
        cost_u, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True

        if flat_free[u] > EXTERIOR_RADIUS_A and u != start_idx:
            end_idx = u
            break

        ui = u // (n * n)
        urem = u % (n * n)
        uj = urem // n
        uk = urem % n

        for off in offsets:
            dx, dy, dz = off
            vi, vj, vk = ui + dx, uj + dy, uk + dz
            if not (0 <= vi < n and 0 <= vj < n and 0 <= vk < n):
                continue
            v = to_idx(vi, vj, vk)
            if visited[v]:
                continue
            fr_v = flat_free[v]
            if fr_v < PROBE_MIN_RADIUS_A:
                continue
            penalty = max(1.0, R_REF_A / fr_v) ** PENALTY_EXPONENT
            step_cost = offset_len[off] * penalty
            cand_cost = cost_u + step_cost
            if cand_cost < best_cost[v]:
                best_cost[v] = cand_cost
                min_radius_on_path[v] = min(min_radius_on_path[u], fr_v)
                prev[v] = u
                heapq.heappush(heap, (cand_cost, v))

    if end_idx is None:
        raise RuntimeError("No se encontro salida al solvente dentro de la caja de busqueda "
                            "-- probar con BOX_HALF_WIDTH_A mas grande.")

    bottleneck_radius = float(min_radius_on_path[end_idx])

    # reconstruir camino
    path_idx = []
    cur = end_idx
    while cur != -1:
        path_idx.append(cur)
        if cur == start_idx:
            break
        cur = prev[cur]
    path_idx.reverse()

    path_coords = []
    for idx in path_idx:
        i = idx // (n * n)
        rem = idx % (n * n)
        j = rem // n
        k = rem % n
        path_coords.append(np.array([xs[i], ys[j], zs[k]]))
    path_coords = np.array(path_coords)

    length = float(np.sum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1)))

    return {
        "bottleneck_radius_A": float(bottleneck_radius),
        "length_A": length,
        "n_path_points": len(path_coords),
        "path_coords": path_coords,
        "start_free_radius_A": float(free_radius[start_i, start_j, start_k]),
    }


def lining_residues(path_coords, residues, cutoff=LINING_CUTOFF_A):
    hits = {}
    for res in residues:
        resnum = res.id[1]
        resname = res.get_resname()
        min_d = None
        for atom in res:
            if atom.element == "H":
                continue
            d = np.min(np.linalg.norm(path_coords - atom.get_coord(), axis=1))
            if min_d is None or d < min_d:
                min_d = d
        if min_d is not None and min_d <= cutoff:
            hits[resnum] = {"resname": resname, "min_dist_to_axis_A": round(float(min_d), 2)}
    return hits


def hydrophobicity_summary(lining):
    counts = {"hidrofobico": 0, "polar": 0, "cargado": 0, "otro": 0}
    for info in lining.values():
        rn = info["resname"]
        if rn in HYDROPHOBIC:
            counts["hidrofobico"] += 1
        elif rn in POLAR:
            counts["polar"] += 1
        elif rn in CHARGED:
            counts["cargado"] += 1
        else:
            counts["otro"] += 1
    total = sum(counts.values())
    frac_hydrophobic = counts["hidrofobico"] / total if total else 0.0
    return counts, frac_hydrophobic


def analyze_side(label, pdb_path, chain_id, ser_resnum, exclude_resnames):
    print(f"\n=== TUNEL {label} (inicio: Ser{ser_resnum}-OG) ===")
    atoms, residues = load_protein_only_chain(pdb_path, chain_id, exclude_resnames)
    res_by_num = {r.id[1]: r for r in residues}
    ser = res_by_num[ser_resnum]
    start_point = ser["OG"].get_coord() if ser.has_id("OG") else ser["CA"].get_coord()

    # La Ser catalitica no puede ser "pared" de su propio hueco de union al
    # sustrato -- si se la deja en el set de obstaculos, el KDTree devuelve
    # distancia ~0 (el propio OG) justo en el punto de partida y el algoritmo
    # de bottleneck maximo queda trabado en radio ~0 para siempre (el "ancho"
    # de un camino nunca puede superar su valor inicial). Se excluye solo
    # esta residuo del campo de radio libre; sigue siendo obstaculo real para
    # todo lo demas via las residuos vecinas.
    atoms_for_field = [a for a in atoms if a[2] != ser_resnum]

    result = widest_path_tunnel(atoms_for_field, start_point)
    lining = lining_residues(result["path_coords"], residues)
    counts, frac_hydro = hydrophobicity_summary(lining)

    print(f"  Longitud del tunel principal: {result['length_A']:.2f} A "
          f"({result['n_path_points']} puntos de grilla)")
    print(f"  Radio del bottleneck: {result['bottleneck_radius_A']:.2f} A")
    print(f"  Residuos que revisten el tunel (<= {LINING_CUTOFF_A} A del eje): {len(lining)}")
    print(f"    hidrofobicos={counts['hidrofobico']}  polares={counts['polar']}  "
          f"cargados={counts['cargado']}  otros={counts['otro']}  "
          f"(fraccion hidrofobica = {frac_hydro:.2f})")
    ordered = sorted(lining.items(), key=lambda x: x[1]["min_dist_to_axis_A"])
    for resnum, info in ordered:
        print(f"    {info['resname']}{resnum}  dist_al_eje={info['min_dist_to_axis_A']:.2f} A")

    return {
        "label": label,
        "start_resnum": ser_resnum,
        "length_A": round(result["length_A"], 2),
        "bottleneck_radius_A": round(result["bottleneck_radius_A"], 2),
        "n_lining_residues": len(lining),
        "lining_residues": {str(k): v for k, v in sorted(lining.items())},
        "hydrophobicity_counts": counts,
        "hydrophobic_fraction": round(frac_hydro, 3),
        "path_coords": result["path_coords"].tolist(),
    }


def main():
    human_result = analyze_side(
        "HUMANO (3PE6)",
        os.path.join(PREPARATION_DIR, "3PE6", "raw", "3PE6.pdb"),
        "A", 122, exclude_resnames={"ZYH", "HOH"},
    )
    cruzi_result = analyze_side(
        "CRUZI (AF-Q4CSM0-F1)",
        os.path.join(PREPARATION_DIR, "AF-Q4CSM0-F1", "processed", "AF-Q4CSM0-F1_prepared.pdb"),
        "A", 139, exclude_resnames=set(),
    )

    out_path = os.path.join(SUB_DIR, "tunel_resultados.json")
    with open(out_path, "w") as fh:
        json.dump({"human": human_result, "cruzi": cruzi_result}, fh, indent=2)
    print(f"\nResultados de tunel -> {out_path}")

    print("\n=== COMPARACION RAPIDA ===")
    print(f"  Longitud:   humano {human_result['length_A']:.2f} A  vs  cruzi {cruzi_result['length_A']:.2f} A")
    print(f"  Bottleneck: humano {human_result['bottleneck_radius_A']:.2f} A  vs  "
          f"cruzi {cruzi_result['bottleneck_radius_A']:.2f} A")
    print(f"  Fraccion hidrofobica: humano {human_result['hydrophobic_fraction']:.2f}  vs  "
          f"cruzi {cruzi_result['hydrophobic_fraction']:.2f}")


if __name__ == "__main__":
    main()
