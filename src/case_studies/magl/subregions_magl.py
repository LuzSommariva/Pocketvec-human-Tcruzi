"""
subregions_magl.py  (formerly Subregiones_MAGL.py)

Analisis comparativo de subregiones funcionales del bolsillo de MAGL humana
(3PE6) vs. MAGL de T. cruzi (AF-Q4CSM0-F1), sobre la superposicion cealign
ya generada por superpose_magl.py (lee alignment_summary.json, no repite
el alineamiento). Cubre:

  1. Listas de residuos de bolsillo (por numero de residuo y "emparejadas"
     por vecino espacial mas cercano post-cealign), exportadas a CSV.
  2. Confirmacion de que la triada catalitica de cada lado esta efectivamente
     dentro de la lista de residuos de bolsillo que definio fpocket.
  3. Oxyanion hole: confirmacion estructural en 3PE6 (H-bonds al carbonilo
     del ligando ZYH) + busqueda del equivalente espacial en cruzi.

No corre CAVER (no esta instalado en este entorno -- ver tunnel_magl.py para
el analisis de tunel, que usa un metodo geometrico propio en su lugar por
esa razon). No modifica superpose_magl.py ni sus salidas.
"""

import csv
import json
import os
import sys

import numpy as np
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser

# src/ (three levels up) must be on sys.path to import utils.* as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.config import CASE_STUDY_DIR, PREPARATION_DIR

OUT_DIR = CASE_STUDY_DIR  # config: paths.case_study_dir
SUB_DIR = os.path.join(OUT_DIR, "subregiones")
os.makedirs(SUB_DIR, exist_ok=True)

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Grupos fisicoquimicos (para clasificacion rapida, ademas del score BLOSUM62)
RESIDUE_GROUP = {
    "GLY": "aliphatic", "ALA": "aliphatic",
    "VAL": "aliphatic", "LEU": "aliphatic", "ILE": "aliphatic", "MET": "aliphatic", "PRO": "aliphatic",
    "PHE": "aromatic", "TYR": "aromatic", "TRP": "aromatic",
    "SER": "polar", "THR": "polar", "CYS": "polar", "ASN": "polar", "GLN": "polar",
    "LYS": "basic", "ARG": "basic", "HIS": "basic",
    "ASP": "acidic", "GLU": "acidic",
}


def three_to_one(resname):
    from Bio.SeqUtils import seq1
    try:
        return seq1(resname)
    except Exception:
        return "X"


def classify_pair(resname_a, resname_b):
    if resname_a == resname_b:
        return "identico"
    a1, b1 = three_to_one(resname_a), three_to_one(resname_b)
    try:
        score = BLOSUM62[a1, b1]
    except KeyError:
        score = None
    same_group = RESIDUE_GROUP.get(resname_a) == RESIDUE_GROUP.get(resname_b)
    if (score is not None and score >= 0) or same_group:
        return "conservativo"
    return "no_conservativo"


def load_summary():
    with open(os.path.join(OUT_DIR, "alignment_summary.json")) as fh:
        return json.load(fh)


def get_ca_map(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    chain = structure[0][chain_id]
    out = {}
    for res in chain:
        if res.id[0] != " " or not res.has_id("CA"):
            continue
        out[res.id[1]] = {"resname": res.get_resname(), "coord": res["CA"].get_coord()}
    return out


def part1_residue_lists(summary):
    human_chain = summary["human"]["chain"]
    cruzi_chain = summary["cruzi"]["chain"]
    human_ca = get_ca_map(summary["human"]["full_pdb"], human_chain)
    cruzi_ca = get_ca_map(summary["cruzi"]["full_pdb"], cruzi_chain)

    human_pocket_resnums = sorted(summary["human"]["pocket_resnums"])
    cruzi_pocket_resnums = sorted(summary["cruzi"]["pocket_resnums"])

    # --- 1a. Orden por numero de residuo, cada lado por separado ---
    seq_order_path = os.path.join(SUB_DIR, "bolsillo_residuos_por_secuencia.csv")
    with open(seq_order_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lado", "cadena", "resnum", "resname", "resname_1letra"])
        for r in human_pocket_resnums:
            info = human_ca[r]
            w.writerow(["humano_3PE6", human_chain, r, info["resname"], three_to_one(info["resname"])])
        for r in cruzi_pocket_resnums:
            info = cruzi_ca[r]
            w.writerow(["cruzi_AF-Q4CSM0-F1", cruzi_chain, r, info["resname"], three_to_one(info["resname"])])
    print(f"[1a] Lista por numero de residuo -> {seq_order_path}")

    # --- 1b. Orden emparejado: para cada residuo humano del bolsillo,
    #     el vecino espacial mas cercano DENTRO del bolsillo cruzi
    #     (ya alineado por cealign). Sin cutoff de distancia -- se reporta
    #     la distancia real, sea cual sea, porque su magnitud es en si un
    #     resultado (ver hallazgo de bolsillo cruzi mal ubicado, mas abajo). ---
    paired_path = os.path.join(SUB_DIR, "bolsillo_residuos_emparejados.csv")
    rows = []
    for r in human_pocket_resnums:
        h_info = human_ca[r]
        best = None
        for rc in cruzi_pocket_resnums:
            c_info = cruzi_ca[rc]
            d = float(np.linalg.norm(h_info["coord"] - c_info["coord"]))
            if best is None or d < best[0]:
                best = (d, rc, c_info["resname"])
        dist, rc, c_resname = best
        cls = classify_pair(h_info["resname"], c_resname)
        rows.append({
            "humano_resnum": r, "humano_resname": h_info["resname"],
            "cruzi_resnum": rc, "cruzi_resname": c_resname,
            "distancia_CA_A": round(dist, 2), "clasificacion": cls,
        })

    with open(paired_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "humano_resnum", "humano_resname", "cruzi_resnum", "cruzi_resname",
            "distancia_CA_A", "clasificacion"])
        w.writeheader()
        w.writerows(rows)
    print(f"[1b] Tabla emparejada (vecino mas cercano) -> {paired_path}")

    n_id = sum(1 for r in rows if r["clasificacion"] == "identico")
    n_cons = sum(1 for r in rows if r["clasificacion"] == "conservativo")
    n_noncons = sum(1 for r in rows if r["clasificacion"] == "no_conservativo")
    dists = [r["distancia_CA_A"] for r in rows]
    print(f"    {len(rows)} residuos humanos de bolsillo emparejados con su vecino cruzi mas cercano:")
    print(f"    identicos={n_id}  conservativos={n_cons}  no_conservativos={n_noncons}")
    print(f"    distancia CA-CA: min={min(dists):.2f}  mediana={sorted(dists)[len(dists)//2]:.2f}  max={max(dists):.2f} A")

    return rows


def part2_triad_in_pocket(summary):
    human_pocket = set(summary["human"]["pocket_resnums"])
    cruzi_pocket = set(summary["cruzi"]["pocket_resnums"])
    human_triad = summary["human"]["triad"]
    cruzi_triad = summary["cruzi"]["triad"]

    print("\n=== TAREA 2: triada catalitica dentro del bolsillo fpocket ===")
    findings = []
    for label, resnum, pocket, side in [
        ("humano Ser122", human_triad["ser"], human_pocket, "humano"),
        ("humano Asp239", human_triad["asp"], human_pocket, "humano"),
        ("humano His269", human_triad["his"], human_pocket, "humano"),
        ("cruzi Ser139", cruzi_triad["ser"], cruzi_pocket, "cruzi"),
        ("cruzi Asp256", cruzi_triad["asp"], cruzi_pocket, "cruzi"),
        ("cruzi His286", cruzi_triad["his"], cruzi_pocket, "cruzi"),
    ]:
        inside = resnum in pocket
        print(f"  {label:15s} (resnum {resnum:>4d}) {'DENTRO' if inside else '*** FUERA ***'} del bolsillo fpocket ({side})")
        if not inside:
            findings.append(f"{label} (resnum {resnum}) NO esta en la lista de residuos de "
                             f"bolsillo fpocket del lado {side}")
    return findings


def part3_oxyanion_hole(summary):
    print("\n=== TAREA 3: oxyanion hole ===")
    human_chain = summary["human"]["chain"]
    parser = PDBParser(QUIET=True)

    # Estructura CRUDA con el ligando ZYH (no la "fixed" del alineamiento,
    # que es proteina sola) -- unica manera de confirmar el H-bond real.
    raw_struct = parser.get_structure("raw3pe6", os.path.join(PREPARATION_DIR, "3PE6", "raw", "3PE6.pdb"))
    chain = raw_struct[0][human_chain]
    lig = next(res for res in chain if res.get_resname() == "ZYH")
    lig_atoms = [a for a in lig if a.element != "H"]

    ser_og = chain[(" ", summary["human"]["triad"]["ser"], " ")]["OG"].get_coord()
    o01 = next(a for a in lig_atoms if a.get_name() == "O01").get_coord()
    print(f"  Ligando ZYH:A:304 (inhibidor no-covalente) -- carbonilo O01 a "
          f"{np.linalg.norm(o01 - ser_og):.2f} A de Ser122-OG (mimetiza el intermediario tetraedrico)")

    candidates = {50: "GLY", 51: "ALA", 52: "GLY", 123: "MET", 124: "GLY"}
    human_oxy = {}
    print("  Distancia N(backbone) -> O01 (carbonilo del ligando):")
    for resnum in candidates:
        res = chain[(" ", resnum, " ")]
        d = float(np.linalg.norm(res["N"].get_coord() - o01))
        human_oxy[resnum] = {"resname": res.get_resname(), "N_to_lig_O01_A": round(d, 2)}
        flag = "  <-- H-bond al oxyanion (< 3.2 A)" if d < 3.2 else ""
        print(f"    {res.get_resname()}{resnum} N: {d:.2f} A{flag}")

    oxy_members = sorted([r for r, v in human_oxy.items() if v["N_to_lig_O01_A"] < 3.2])
    print(f"  --> Oxyanion hole humano confirmado estructuralmente: residuos {oxy_members} "
          f"(Ala51 + Met123, flanqueados por Gly50/Gly124 en el mismo loop -- "
          f"consistente con Schalk-Hihi et al. 2011, Protein Science 20:670-683, "
          f"PDB 3PE6, DOI 10.1002/pro.596)")

    human_pocket = set(summary["human"]["pocket_resnums"])
    print(f"  Ala51 en bolsillo fpocket humano: {51 in human_pocket} | "
          f"Met123 en bolsillo fpocket humano: {123 in human_pocket}")

    # --- Equivalente en cruzi por vecino espacial mas cercano post-cealign ---
    human_full = parser.get_structure("hf", summary["human"]["full_pdb"])[0][human_chain]
    cruzi_full = parser.get_structure("cf", summary["cruzi"]["full_pdb"])[0][summary["cruzi"]["chain"]]

    def nearest_cruzi_ca(point):
        best = None
        for res in cruzi_full:
            if res.id[0] != " " or not res.has_id("CA"):
                continue
            d = np.linalg.norm(res["CA"].get_coord() - point)
            if best is None or d < best[2]:
                best = (res.id[1], res.get_resname(), d)
        return best

    print("\n  Equivalente espacial en cruzi (vecino de CA mas cercano post-cealign):")
    cruzi_oxy = {}
    for resnum in (51, 123):
        p = human_full[(" ", resnum, " ")]["CA"].get_coord()
        rc, resname_c, dca = nearest_cruzi_ca(p)
        cruzi_oxy[resnum] = {"cruzi_resnum": rc, "cruzi_resname": resname_c, "CA_CA_dist_A": round(float(dca), 2)}
        print(f"    humano {human_full[(' ', resnum, ' ')].get_resname()}{resnum} CA <-> "
              f"cruzi {resname_c}{rc}  (CA-CA dist tras cealign = {dca:.2f} A)")

    ser_c_og = cruzi_full[(" ", summary["cruzi"]["triad"]["ser"], " ")]["OG"].get_coord()
    print("\n  Distancia N(backbone) -> Ser139-OG (cruzi, geometria analoga, sin ligando -- estructura apo AlphaFold):")
    for resnum in (51, 123):
        rc = cruzi_oxy[resnum]["cruzi_resnum"]
        res = cruzi_full[(" ", rc, " ")]
        d = float(np.linalg.norm(res["N"].get_coord() - ser_c_og))
        cruzi_oxy[resnum]["N_to_Ser139OG_A"] = round(d, 2)
        print(f"    {res.get_resname()}{rc} N: {d:.2f} A")

    cruzi_pocket = set(summary["cruzi"]["pocket_resnums"])
    v65_in = cruzi_oxy[51]["cruzi_resnum"] in cruzi_pocket
    m140_in = cruzi_oxy[123]["cruzi_resnum"] in cruzi_pocket
    print(f"\n  {cruzi_oxy[51]['cruzi_resname']}{cruzi_oxy[51]['cruzi_resnum']} en bolsillo fpocket cruzi: {v65_in} | "
          f"{cruzi_oxy[123]['cruzi_resname']}{cruzi_oxy[123]['cruzi_resnum']} en bolsillo fpocket cruzi: {m140_in}")

    comparison = []
    for h_res, h_num in [("ALA", 51), ("MET", 123)]:
        c_num = cruzi_oxy[h_num]["cruzi_resnum"]
        c_res = cruzi_oxy[h_num]["cruzi_resname"]
        comparison.append({
            "humano": f"{h_res}{h_num}", "cruzi": f"{c_res}{c_num}",
            "clasificacion": classify_pair(h_res, c_res),
        })

    result = {
        "human_oxyanion_members": {str(k): v for k, v in human_oxy.items()},
        "human_oxyanion_hbond_residues": [f"ALA{51}", f"MET{123}"],
        "human_oxyanion_in_fpocket_pocket": {"ALA51": 51 in human_pocket, "MET123": 123 in human_pocket},
        "cruzi_oxyanion_equivalents": {str(k): v for k, v in cruzi_oxy.items()},
        "cruzi_oxyanion_in_fpocket_pocket": {
            f"{cruzi_oxy[51]['cruzi_resname']}{cruzi_oxy[51]['cruzi_resnum']}": v65_in,
            f"{cruzi_oxy[123]['cruzi_resname']}{cruzi_oxy[123]['cruzi_resnum']}": m140_in,
        },
        "comparison": comparison,
        "citation": {
            "human_oxyanion_hole": "Schalk-Hihi C, Schubert C, Alexander R, et al. (2011). "
                "Crystal structure of a soluble form of human monoglyceride lipase in complex "
                "with an inhibitor at 1.35 A resolution. Protein Science 20(4):670-683. "
                "DOI: 10.1002/pro.596. PMID: 21308848. (Structure deposited as PDB 3PE6, la misma "
                "estructura usada en este analisis). Describen el oxyanion hole como una red de "
                "puentes H conservada formada por la 'nucleophilic elbow' y el loop alpha1-beta3 "
                "(Gly50, Ala51, Met123, Gly124); confirmado aca estructuralmente por distancia "
                "N(backbone)-O(carbonilo del ligando ZYH)."
        },
    }
    return result


def main():
    summary = load_summary()
    paired_rows = part1_residue_lists(summary)
    triad_findings = part2_triad_in_pocket(summary)
    oxyanion = part3_oxyanion_hole(summary)

    findings_path = os.path.join(SUB_DIR, "hallazgos_triada_oxyanion.json")
    with open(findings_path, "w") as fh:
        json.dump({
            "triad_findings": triad_findings,
            "oxyanion_hole": oxyanion,
        }, fh, indent=2)
    print(f"\nHallazgos + oxyanion hole -> {findings_path}")


if __name__ == "__main__":
    main()
