"""
render_magl_views.py  (formerly render_superposicion_magl_views.py)

Genera PNGs (ray-traced, headless) a partir de la sesion ya guardada por
superpose_magl.py (MAGL_human_vs_cruzi_cealign.pse). No vuelve a correr
cealign ni fpocket/rDock -- la sesion ya tiene las coordenadas superpuestas y
las representaciones (cartoon, sticks de triada, lineas de bolsillo).

Los numeros de residuo de la triada/bolsillo NO se hardcodean aca: se
reimportan las funciones de superpose_magl.py (motivo GXSXG + proximidad
3D, lectura de CSV/fpocket) para recalcularlos -- es rapido (no involucra
PyMOL ni docking, solo Biopython/numpy) y evita que este script quede
desincronizado si el de arriba cambia de resultado en otro par.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import superpose_magl as sm  # esto ya lanza pymol (finish_launching) al importarse
from pymol import cmd

RENDER_DIR = os.path.join(sm.OUT_DIR, "renders")
os.makedirs(RENDER_DIR, exist_ok=True)

WIDTH, HEIGHT = 1600, 1200


def log(msg):
    print(msg, flush=True)


def recompute_key_resnums():
    """Repite (barato, sin PyMOL/fpocket) los pasos 1-4 de superpose_magl.py
    para recuperar que residuos son la triada y el bolsillo de cada lado,
    sin hardcodearlos aca."""
    human_pdb, human_paths = sm.resolve_prepared_pdb(sm.HUMAN_CODE)
    cruzi_pdb, cruzi_paths = sm.resolve_prepared_pdb(sm.CRUZI_CODE)

    human_seq, human_resnums, human_struct = sm.get_chain_sequence(human_pdb, sm.HUMAN_CHAIN)
    cruzi_seq, cruzi_resnums, cruzi_struct = sm.get_chain_sequence(cruzi_pdb, sm.CRUZI_CHAIN)

    alignment = sm.pairwise_align(human_seq, cruzi_seq)
    index_map = sm.build_index_map(alignment)

    human_ser_idx = human_resnums.index(sm.HUMAN_KNOWN_TRIAD["ser"][1])
    anchor_idx_in_cruzi = index_map[human_ser_idx]
    cruzi_ser, _, _ = sm.find_nucleophile_serine(cruzi_seq, cruzi_resnums, anchor_idx_in_cruzi)
    cruzi_his, _ = sm.find_catalytic_his(cruzi_struct, sm.CRUZI_CHAIN, cruzi_ser)
    cruzi_asp, _, _ = sm.find_catalytic_acid(cruzi_struct, sm.CRUZI_CHAIN, cruzi_his)

    human_triad = [sm.HUMAN_KNOWN_TRIAD["ser"][1], sm.HUMAN_KNOWN_TRIAD["asp"][1], sm.HUMAN_KNOWN_TRIAD["his"][1]]
    cruzi_triad = [cruzi_ser, cruzi_asp, cruzi_his]

    human_pocket = sm.human_pocket_resnums(human_paths, sm.HUMAN_CODE, sm.HUMAN_CHAIN)
    cruzi_centroid, best_id, drug_score = sm.cruzi_best_pocket_centroid(cruzi_paths, sm.CRUZI_CODE)
    cruzi_pocket = sm.pocket_resnums_near_point(cruzi_struct, sm.CRUZI_CHAIN, cruzi_centroid)

    return human_triad, cruzi_triad, human_pocket, cruzi_pocket


def resi_sel(obj, resnums):
    return f"{obj} and resi " + "+".join(str(r) for r in resnums)


def setup_render_style():
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)  # evita distorsion de perspectiva al comparar paneles
    cmd.set("ray_trace_fog", 0)
    cmd.set("specular", 0.15)
    cmd.set("label_size", 18)
    cmd.set("label_color", "black")
    cmd.set("label_font_id", 7)


def main():
    pse_path = os.path.join(sm.OUT_DIR, "MAGL_human_vs_cruzi_cealign.pse")
    if not os.path.exists(pse_path):
        raise FileNotFoundError(
            f"No encontre {pse_path} -- correr primero superpose_magl.py."
        )

    log(f"Recalculando triada/bolsillo (sin PyMOL) para reconstruir selecciones...")
    human_triad, cruzi_triad, human_pocket, cruzi_pocket = recompute_key_resnums()
    log(f"  triada humana: {human_triad} | triada cruzi: {cruzi_triad}")
    log(f"  bolsillo humano: {len(human_pocket)} residuos | bolsillo cruzi: {len(cruzi_pocket)} residuos")

    cmd.reinitialize()
    cmd.load(pse_path)
    setup_render_style()

    human_triad_sel = resi_sel("human", human_triad)
    cruzi_triad_sel = resi_sel("cruzi", cruzi_triad)
    human_pocket_sel = resi_sel("human", human_pocket) if human_pocket else "none"
    cruzi_pocket_sel = resi_sel("cruzi", cruzi_pocket) if cruzi_pocket else "none"
    triad_both = f"({human_triad_sel}) or ({cruzi_triad_sel})"
    pocket_both = f"({human_pocket_sel}) or ({cruzi_pocket_sel})"

    png_paths = {}

    # -----------------------------------------------------------------
    # 1. overview.png -- ambas estructuras completas, cartoon
    # -----------------------------------------------------------------
    log("Render 1/4: overview.png")
    cmd.hide("everything")
    cmd.show("cartoon", "human or cruzi")
    cmd.set("cartoon_transparency", 0.0, "human or cruzi")
    cmd.color("gray60", "human")
    cmd.color("skyblue", "cruzi")
    cmd.show("sticks", triad_both)
    cmd.color("yellow", human_triad_sel + " and elem C")
    cmd.color("orange", cruzi_triad_sel + " and elem C")
    cmd.orient("human or cruzi")
    cmd.zoom("human or cruzi", 3)
    cmd.ray(WIDTH, HEIGHT)
    out = os.path.join(RENDER_DIR, "overview.png")
    cmd.png(out, dpi=300)
    png_paths["overview"] = out

    # -----------------------------------------------------------------
    # 2. triada_zoom.png -- zoom cerrado, sticks + labels de residuo
    # -----------------------------------------------------------------
    log("Render 2/4: triada_zoom.png")
    cmd.hide("everything")
    cmd.show("cartoon", "human or cruzi")
    cmd.set("cartoon_transparency", 0.6, "human or cruzi")
    cmd.show("sticks", triad_both)
    cmd.util.cnc(triad_both)  # color-by-element (carbonos conservan el color del objeto)
    cmd.color("yellow", human_triad_sel + " and elem C")
    cmd.color("orange", cruzi_triad_sel + " and elem C")
    # Los residuos de la triada quedan casi exactamente superpuestos (esa es
    # justamente la conclusion del script principal) -- sin offset, las
    # etiquetas de humano y cruzi caen una arriba de la otra y se vuelven
    # ilegibles. Se separan con un offset opuesto en cada lado (asuncion:
    # +/-2.2 A en cada eje, suficiente a este nivel de zoom).
    cmd.set("label_position", (2.2, 1.2, 0.5), human_triad_sel)
    cmd.set("label_position", (-2.2, -1.2, -0.5), cruzi_triad_sel)
    cmd.label(f"({human_triad_sel}) and name CA", '"%s%s (human)" % (resn, resi)')
    cmd.label(f"({cruzi_triad_sel}) and name CA", '"%s%s (cruzi)" % (resn, resi)')
    cmd.zoom(triad_both, 4)
    cmd.ray(WIDTH, HEIGHT)
    out = os.path.join(RENDER_DIR, "triada_zoom.png")
    cmd.png(out, dpi=300)
    png_paths["triada_zoom"] = out
    cmd.hide("labels")
    cmd.label(triad_both, "")

    # -----------------------------------------------------------------
    # 3. bolsillo_zoom.png -- region de bolsillo completa, triada destacada
    # -----------------------------------------------------------------
    log("Render 3/4: bolsillo_zoom.png")
    cmd.hide("everything")
    cmd.show("cartoon", "human or cruzi")
    cmd.set("cartoon_transparency", 0.75, "human or cruzi")
    cmd.show("lines", pocket_both)
    cmd.set("line_width", 1.5, pocket_both)
    cmd.color("gray40", human_pocket_sel + " and elem C")
    cmd.color("deepteal", cruzi_pocket_sel + " and elem C")
    cmd.show("sticks", triad_both)
    cmd.set("stick_radius", 0.25, triad_both)
    cmd.color("yellow", human_triad_sel + " and elem C")
    cmd.color("orange", cruzi_triad_sel + " and elem C")
    zoom_region = f"({pocket_both}) or ({triad_both})"
    cmd.zoom(zoom_region, 3)
    cmd.ray(WIDTH, HEIGHT)
    out = os.path.join(RENDER_DIR, "bolsillo_zoom.png")
    cmd.png(out, dpi=300)
    png_paths["bolsillo_zoom"] = out

    # -----------------------------------------------------------------
    # 4. lados_a_lado.png -- grid_mode: mismo eje de camara, sin superponer
    # -----------------------------------------------------------------
    log("Render 4/4: lados_a_lado.png")
    cmd.orient(zoom_region)  # fija una orientacion de camara comun antes de separar en paneles
    cmd.set("grid_mode", 1)
    cmd.zoom(zoom_region, 3)
    cmd.ray(WIDTH, HEIGHT)
    out = os.path.join(RENDER_DIR, "lados_a_lado.png")
    cmd.png(out, dpi=300)
    png_paths["lados_a_lado"] = out
    cmd.set("grid_mode", 0)

    log("")
    log("PNGs generados:")
    for name, path in png_paths.items():
        size_kb = os.path.getsize(path) / 1024
        log(f"  {name:<15} -> {path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
