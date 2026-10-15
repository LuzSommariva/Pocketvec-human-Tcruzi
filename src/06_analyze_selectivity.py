#!/usr/bin/env python3
"""
06_analyze_selectivity.py
=========================

Script de análisis y visualización para el pipeline
"PocketVec — Selectividad Humano vs. T. cruzi".

Lee (default paths from config/config.yaml):

    - <paths.results_csv>                          (targets_results.csv)
    - <paths.analysis_dir>/pair_distances.csv      (distancias coseno por par)
    - <paths.pocketvec_dir>/<código>/<código>_PocketVec_fp.pkl   (opcional, para el
      heatmap cruzado completo)

No modifica ninguno de esos archivos (todos son "fuente de verdad" regenerada
por `05_run_pocketvec_batch.py`); solo lee y grafica. Todo lo que genera queda en
`<paths.analysis_dir>/plots/` (se crea si no existe) más un CSV de resumen.

Uso típico:

    python src/06_analyze_selectivity.py

Opciones:

    python src/06_analyze_selectivity.py \
        --threshold 0.17 \
        --gray-zone 0.20 \
        --out results/3_analysis/plots

Qué genera:

    1. bar_pair_distances.png
       Gráfico de barras horizontales, un par por fila, ordenado por distancia
       coseno ascendente. Rojo = conservado (riesgo de pegarle también al target
       humano), amarillo = zona gris, verde = selectivo. Línea de referencia en
       el umbral de PocketVec (0.17 por defecto) y otra en 0.20 (punto donde el
       paper de PocketVec ve caer a ~0 los compuestos compartidos entre pockets).

    2. hist_pair_distances.png
       Histograma de la distribución de distancias coseno de los pares
       calculados, con las mismas líneas de referencia.

    3. heatmap_pairs_strip.png
       "Heatmap" de una columna (todos los pares apilados), coloreado por
       distancia, con el valor anotado en cada celda. Es la versión heatmap
       de la tabla par-a-par (lo que en tu guía es el cierre del "Informe 1").

    4. heatmap_full_cross.png   (solo si hay .pkl de fingerprints disponibles)
       Heatmap NxN de TODAS las proteínas con fingerprint calculado (humanas +
       cruzi juntas), no solo los pares emparejados. Esto es opcional pero muy
       informativo para tu tesis: te deja ver si algún bolsillo de cruzi se
       parece a un bolsillo HUMANO CON EL QUE NO ESTABA EMPAREJADO (posible
       off-target cruzado que `pair_distances.csv` no muestra porque ahí solo
       comparás cada humano contra SU PROPIO ortólogo).

    5. summary_selectivity.csv
       Tabla consolidada: por cada par, distancia, druggability humano/cruzi,
       y una clasificación en 3 franjas (conservado / zona gris / selectivo).
       Pensada para pegar directo en la tesis o filtrar candidatos.

Todas las figuras se guardan también en PDF vectorial (mismo nombre, .pdf) por
si las querés meter directo en LaTeX/Word sin perder calidad.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils import config

try:
    import matplotlib
    matplotlib.use("Agg")  # sin display, por si corre en el cluster
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    sys.exit(
        "Falta matplotlib. Instalalo en tu entorno (pocketvec_env) con:\n"
        "    conda install matplotlib\n"
        "o:\n"
        "    pip install matplotlib"
    )


# --------------------------------------------------------------------------- #
# Detección flexible de columnas
# --------------------------------------------------------------------------- #
# targets_results.csv trae tanto "target_id" (entero, 1..N) como "pair_id"
# (string "pair_001", "pair_002", ...), y pair_distances.csv solo trae
# "target_id" (el entero). Por eso target_id va primero en ID_CANDIDATES: si
# "pair_id" fuera el primer candidato, el merge terminaría cruzando la columna
# "target_id" de pair_distances.csv (enteros) contra la columna "pair_id" de
# targets_results.csv (strings "pair_00N"), que nunca matchean, y toda la
# druggability saldría NaN en silencio.
ID_CANDIDATES = ["target_id", "pair_id", "id"]
NAME_CANDIDATES = ["target_name", "name", "pair_name", "protein_name"]
HUMAN_CODE_CANDIDATES = ["human_pdb_code", "human_pdb", "human_code", "human_uniprot"]
CRUZI_CODE_CANDIDATES = ["cruzi_af_code", "cruzi_code", "cruzi_uniprot", "cruzi_name"]
DISTANCE_CANDIDATES = [
    "pair_cosine_distance",
    "cosine_distance",
    "distance",
    "cosine_dist",
    "pocketvec_distance",
]
DRUG_HUMAN_CANDIDATES = [
    "pocket_druggability_HUMAN",
    "pocket_druggability_human",
    "druggability_human",
    "human_druggability_score",
    "fpocket_druggability_score_human",
]
DRUG_CRUZI_CANDIDATES = [
    "pocket_druggability_CRUZI",
    "pocket_druggability_cruzi",
    "druggability_cruzi",
    "cruzi_druggability_score",
    "fpocket_druggability_score_cruzi",
]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _require_column(df: pd.DataFrame, candidates: list[str], what: str, source: str) -> str:
    col = _find_column(df, candidates)
    if col is None:
        raise SystemExit(
            f"\nNo encontré una columna de '{what}' en {source}.\n"
            f"Busqué (sin importar mayúsculas): {candidates}\n"
            f"Columnas reales encontradas: {list(df.columns)}\n\n"
            f"Agregá el nombre real a la lista de candidatos correspondiente "
            f"al principio de este script y volvé a correrlo."
        )
    return col


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #

def load_pair_distances(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No encuentro {path}. ¿Corriste el batch al menos una vez?")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"{path} existe pero está vacío.")
    return df


def load_targets_results(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"No encontré {path}; sigo sin cruzar con druggability.")
        return None
    return pd.read_csv(path)


def build_master_table(pair_df: pd.DataFrame, results_df: pd.DataFrame | None) -> pd.DataFrame:
    """Arma una tabla única: id, nombre, distancia, druggability humano/cruzi."""
    id_col = _find_column(pair_df, ID_CANDIDATES)
    name_col = _find_column(pair_df, NAME_CANDIDATES)
    dist_col = _require_column(pair_df, DISTANCE_CANDIDATES, "distancia coseno", "pair_distances.csv")

    out = pd.DataFrame()
    out["pair_id"] = pair_df[id_col] if id_col else np.arange(len(pair_df))
    out["target_name"] = pair_df[name_col] if name_col else out["pair_id"].astype(str)
    out["cosine_distance"] = pd.to_numeric(pair_df[dist_col], errors="coerce")

    human_code_col = _find_column(pair_df, HUMAN_CODE_CANDIDATES)
    cruzi_code_col = _find_column(pair_df, CRUZI_CODE_CANDIDATES)
    out["human_code"] = pair_df[human_code_col] if human_code_col else np.nan
    out["cruzi_code"] = pair_df[cruzi_code_col] if cruzi_code_col else np.nan

    if results_df is not None:
        r_id_col = _find_column(results_df, ID_CANDIDATES)
        r_name_col = _find_column(results_df, NAME_CANDIDATES)
        drug_h_col = _find_column(results_df, DRUG_HUMAN_CANDIDATES)
        drug_c_col = _find_column(results_df, DRUG_CRUZI_CANDIDATES)
        if r_id_col and (drug_h_col or drug_c_col or (name_col is None and r_name_col)):
            merge_cols = [r_id_col] + [c for c in (drug_h_col, drug_c_col) if c]
            # pair_distances.csv no trae nombre de proteína propio: si no lo
            # encontramos ahí, lo traemos de targets_results.csv en el mismo merge.
            # Se renombra a una columna temporal porque "out" ya tiene una columna
            # "target_name" (el fallback numérico) y el merge la sufijaría en vez
            # de sobreescribirla si quedan con el mismo nombre.
            pull_name_from_results = name_col is None and r_name_col is not None
            rename_map = {}
            if pull_name_from_results and r_name_col not in merge_cols:
                merge_cols.append(r_name_col)
                rename_map[r_name_col] = "_target_name_from_results"
            merged = results_df[merge_cols].drop_duplicates(subset=r_id_col).rename(columns=rename_map)
            # merge por el id que realmente comparten ambas tablas (target_id).
            out = out.merge(
                merged, left_on="pair_id", right_on=r_id_col, how="left", suffixes=("", "_res")
            )
            if drug_h_col:
                out = out.rename(columns={drug_h_col: "druggability_human"})
            if drug_c_col:
                out = out.rename(columns={drug_c_col: "druggability_cruzi"})
            if pull_name_from_results:
                out["target_name"] = out["_target_name_from_results"].fillna(out["target_name"])
                out = out.drop(columns=["_target_name_from_results"])
        else:
            warnings.warn(
                "No pude cruzar targets_results.csv con pair_distances.csv "
                "(no encontré columnas de id/druggability reconocibles). "
                "Sigo solo con las distancias."
            )

    out = out.dropna(subset=["cosine_distance"]).sort_values("cosine_distance").reset_index(drop=True)
    return out


def classify(distance: float, threshold: float, gray_zone: float) -> str:
    if distance < threshold:
        return "conservado (riesgo off-target)"
    elif distance < gray_zone:
        return "zona gris"
    else:
        return "selectivo"


# --------------------------------------------------------------------------- #
# Paleta de colores (accesible, sin depender de seaborn)
# --------------------------------------------------------------------------- #

COLOR_CONSERVADO = "#D64550"   # rojo - riesgo de pegarle también al humano
COLOR_ZONA_GRIS = "#E8A33D"    # ámbar
COLOR_SELECTIVO = "#3E8F6F"    # verde - buen candidato de selectividad
CLASS_COLORS = {
    "conservado (riesgo off-target)": COLOR_CONSERVADO,
    "zona gris": COLOR_ZONA_GRIS,
    "selectivo": COLOR_SELECTIVO,
}


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #

def plot_bar_distances(df: pd.DataFrame, threshold: float, gray_zone: float, out_dir: Path) -> None:
    fig_h = max(4, 0.35 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    colors = [CLASS_COLORS[classify(d, threshold, gray_zone)] for d in df["cosine_distance"]]
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df["cosine_distance"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["target_name"], fontsize=9)
    ax.invert_yaxis()  # el más chico (más conservado) arriba
    ax.set_xlabel("Distancia coseno (PocketVec)")
    ax.set_title("Distancia coseno por par Humano vs. T. cruzi", fontsize=12, fontweight="bold")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.text(threshold, len(df) - 0.5, f" umbral PocketVec = {threshold}",
            rotation=90, va="bottom", ha="right", fontsize=8)

    if gray_zone and gray_zone != threshold:
        ax.axvline(gray_zone, color="gray", linestyle=":", linewidth=1)
        ax.text(gray_zone, len(df) - 0.5, f" ~{gray_zone} (paper: caen compuestos compartidos)",
                rotation=90, va="bottom", ha="right", fontsize=7, color="dimgray")

    for i, d in enumerate(df["cosine_distance"]):
        ax.text(d + 0.003, i, f"{d:.3f}", va="center", fontsize=7.5)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COLOR_CONSERVADO, label=f"conservado (< {threshold})"),
        Patch(facecolor=COLOR_ZONA_GRIS, label=f"zona gris ({threshold}–{gray_zone})"),
        Patch(facecolor=COLOR_SELECTIVO, label=f"selectivo (≥ {gray_zone})"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, framealpha=0.9)

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "bar_pair_distances")
    plt.close(fig)


def plot_histogram(df: pd.DataFrame, threshold: float, gray_zone: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_bins = min(15, max(5, len(df) // 2))
    ax.hist(df["cosine_distance"], bins=n_bins, color="#4C72B0", edgecolor="white")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"umbral PocketVec ({threshold})")
    if gray_zone and gray_zone != threshold:
        ax.axvline(gray_zone, color="gray", linestyle=":", linewidth=1,
                   label=f"~{gray_zone} (paper)")
    ax.set_xlabel("Distancia coseno")
    ax.set_ylabel("Cantidad de pares")
    ax.set_title("Distribución de distancias coseno entre pares", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "hist_pair_distances")
    plt.close(fig)


def plot_strip_heatmap(df: pd.DataFrame, threshold: float, gray_zone: float, out_dir: Path) -> None:
    """Heatmap de una sola columna (todos los pares), para "cerrar el Informe 1"."""
    fig_h = max(4, 0.4 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(4.5, fig_h))

    values = df["cosine_distance"].to_numpy().reshape(-1, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "riesgo_selectividad", [COLOR_CONSERVADO, COLOR_ZONA_GRIS, COLOR_SELECTIVO]
    )
    vmin, vmax = values.min(), values.max()
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["target_name"], fontsize=9)
    ax.set_xticks([])
    ax.set_title("Heatmap de distancias\npor par (Informe 1)", fontsize=11, fontweight="bold")

    for i, v in enumerate(df["cosine_distance"]):
        ax.text(0, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v - vmin) < (vmax - vmin) * 0.5 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.15)
    cbar.set_label("Distancia coseno")
    fig.tight_layout()
    _save(fig, out_dir / "heatmap_pairs_strip")
    plt.close(fig)


def _load_fingerprint(pkl_path: Path) -> np.ndarray | None:
    try:
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"No pude leer {pkl_path.name}: {exc}")
        return None

    # El formato exacto del pickle de PocketVec puede variar (array directo,
    # dict con 'fp'/'fingerprint', DataFrame de 1 fila, etc.) — cubrimos los
    # casos más comunes.
    if isinstance(obj, np.ndarray):
        return obj.flatten()
    if isinstance(obj, (list, tuple)):
        return np.array(obj).flatten()
    if isinstance(obj, dict):
        for key in ("fingerprint", "fp", "raw_fp", "vector", "values"):
            if key in obj:
                return np.array(obj[key]).flatten()
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy().flatten()
    if isinstance(obj, pd.Series):
        return obj.to_numpy().flatten()

    warnings.warn(f"Formato de pickle no reconocido en {pkl_path.name} ({type(obj)}), lo salteo.")
    return None


def plot_full_cross_heatmap(fp_dir: Path, out_dir: Path, threshold: float) -> None:
    """
    Heatmap NxN opcional con TODOS los fingerprints disponibles (humanos +
    cruzi), no solo los pares emparejados. Requiere los .pkl en
    <fp_dir>/<código>/<código>_PocketVec_fp.pkl.

    Sirve para detectar selectividad cruzada que pair_distances.csv no puede
    mostrar: p.ej. si el bolsillo de cruzi del target X se parece más al
    bolsillo HUMANO del target Y (uno con el que no fue emparejado) que al
    suyo propio.
    """
    pkl_paths = sorted(glob.glob(str(fp_dir / "*" / "*_PocketVec_fp.pkl")))
    if len(pkl_paths) < 3:
        print(f"[info] Encontré {len(pkl_paths)} fingerprints en {fp_dir}; "
              f"salteo el heatmap cruzado completo (necesito al menos 3).")
        return

    labels, vectors = [], []
    lengths = set()
    for p in pkl_paths:
        code = Path(p).parent.name
        vec = _load_fingerprint(Path(p))
        if vec is None:
            continue
        labels.append(code)
        vectors.append(vec)
        lengths.add(len(vec))

    if len(lengths) > 1:
        warnings.warn(
            f"Los fingerprints no tienen todos la misma longitud ({lengths}); "
            f"esto no debería pasar con PocketVec (128 fijo) — revisar antes de confiar "
            f"en el heatmap cruzado. Sigo igual truncando al mínimo común."
        )
        min_len = min(lengths)
        vectors = [v[:min_len] for v in vectors]

    if len(vectors) < 3:
        print("[info] Muy pocos fingerprints válidos, salteo el heatmap cruzado.")
        return

    matrix = np.vstack(vectors)
    # distancia coseno = 1 - similitud coseno
    norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    cosine_sim = norm @ norm.T
    cosine_dist = 1 - cosine_sim
    np.fill_diagonal(cosine_dist, 0.0)

    n = len(labels)
    fig_size = max(6, 0.35 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cosine_dist, cmap="RdYlGn", vmin=0, vmax=max(0.4, cosine_dist.max()))
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(
        f"Distancia coseno cruzada — TODOS los fingerprints ({n})\n"
        f"(no solo pares emparejados; útil para chequear cross-reactividad)",
        fontsize=10, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distancia coseno")
    fig.tight_layout()
    _save(fig, out_dir / "heatmap_full_cross")
    plt.close(fig)
    print(f"[ok] Heatmap cruzado completo con {n} proteínas (umbral de referencia {threshold}).")


def _save(fig, path_no_ext: Path) -> None:
    fig.savefig(f"{path_no_ext}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")
    print(f"[ok] {path_no_ext}.png / .pdf")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-distances", type=str,
                         default=os.path.join(config.ANALYSIS_DIR, "pair_distances.csv"),
                         help="pair_distances.csv written by 05_run_pocketvec_batch.py.")
    parser.add_argument("--results-csv", type=str, default=config.RESULTS_CSV,
                         help="targets_results.csv written by 05_run_pocketvec_batch.py.")
    parser.add_argument("--fingerprints-dir", type=str, default=config.POCKETVEC_DIR,
                         help="Folder with <código>/<código>_PocketVec_fp.pkl (cross heatmap).")
    parser.add_argument("--threshold", type=float, default=0.17,
                         help="Umbral de PocketVec para 'similar' (default 0.17, el que usás en la tesis).")
    parser.add_argument("--gray-zone", type=float, default=0.20,
                         help="Corte superior de 'zona gris' (default 0.20, según el paper de PocketVec: "
                              "por encima casi no hay compuestos compartidos entre pockets).")
    parser.add_argument("--out", type=str, default=os.path.join(config.ANALYSIS_DIR, "plots"),
                         help="Carpeta de salida (default <paths.analysis_dir>/plots).")
    parser.add_argument("--skip-cross-heatmap", action="store_true",
                         help="No intentar el heatmap NxN completo desde los .pkl (por si tarda o falla).")
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Salida:    {out_dir}")
    print(f"Umbral:    {args.threshold}  |  Zona gris hasta: {args.gray_zone}\n")

    pair_df = load_pair_distances(Path(args.pair_distances).expanduser())
    results_df = load_targets_results(Path(args.results_csv).expanduser())
    master = build_master_table(pair_df, results_df)

    print(f"Pares con distancia calculada: {len(master)}")
    n_below = (master["cosine_distance"] < args.threshold).sum()
    print(f"  -> {n_below} por debajo del umbral ({args.threshold}), es decir bolsillos conservados / "
          f"riesgo de pegarle también al target humano.\n")

    # Gráficos
    plot_bar_distances(master, args.threshold, args.gray_zone, out_dir)
    plot_histogram(master, args.threshold, args.gray_zone, out_dir)
    plot_strip_heatmap(master, args.threshold, args.gray_zone, out_dir)
    if not args.skip_cross_heatmap:
        plot_full_cross_heatmap(Path(args.fingerprints_dir).expanduser(), out_dir, args.threshold)

    # Tabla resumen
    master["clasificacion"] = master["cosine_distance"].apply(
        lambda d: classify(d, args.threshold, args.gray_zone)
    )
    summary_path = out_dir / "summary_selectivity.csv"
    master.to_csv(summary_path, index=False)
    print(f"[ok] {summary_path}")

    print("\nListo. Pares por categoría:")
    print(master["clasificacion"].value_counts().to_string())


if __name__ == "__main__":
    main()
