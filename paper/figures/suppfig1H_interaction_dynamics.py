"""
Supplementary Figure 1H — Epithelial interaction dynamics across pseudotime
============================================================================
Three heatmaps showing the fraction of ductal cells with ≥1 contact-neighbour
(within 30 µm) for each target cell type, across 10 pseudotime bins.

Panels:
  (a) Tier A: Fibroblasts, T cells, B lineage  (3 rows)
  (b) T cell subsets (Tier B)                  (7 rows)
  (c) B cell subsets (Tier B)                  (7 rows)

Rows sorted by peak-bin (argmax of fraction_source_with_target_neighbor).
Columns = pseudotime bins 0–9, labelled by median pseudotime.
Colour  = fraction (YlOrRd; white = 0, dark red = high).

Data sources:
  tier_a_interaction_bins.csv
  tier_b_t_cells_interaction_bins.csv
  tier_b_b_cells_interaction_bins.csv

Run:
    python notebooks/suppfig1H_interaction_dynamics.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig1H_interaction_dynamics.pdf (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
RESULT_DIR = ROOT  / "paper" / "notebooks" / "results" / "pseudotime_exp2"

TIER_A_CSV = RESULT_DIR / "tier_a_interaction_bins.csv"
T_CELL_CSV = RESULT_DIR / "tier_b_t_cells_interaction_bins.csv"
B_CELL_CSV = RESULT_DIR / "tier_b_b_cells_interaction_bins.csv"

VALUE_COL  = "fraction_source_with_target_neighbor"

# ── Cleaner display names ─────────────────────────────────────────────────────
TIER_A_NAMES = {
    "Fibroblasts": "Fibroblasts",
    "T cells":     "T cells",
    "B lineage":   "B lineage",
}
T_CELL_NAMES = {
    "activated CD4 T cells":  "Act. CD4 T",
    "activated CD8 T cells":  "Act. CD8 T",
    "CD8 T cells":             "CD8 T",
    "CD4 T cells":             "CD4 T",
    "Tregs":                   "Tregs",
    "Th2-like cells":          "Th2-like",
    "cytotoxic CD8 T cells":   "Cytotoxic CD8",
}
B_CELL_NAMES = {
    "naive B cells":      "Naive B",
    "memory B cells":     "Memory B",
    "plasmablasts":       "Plasmablasts",
    "plasmablasts-like":  "Plasmablasts-like",
    "GZMB+ B cells":      "GZMB+ B",
    "PDL1+ B cells":      "PDL1+ B",
    "APC-like B cells":   "APC-like B",
}

# ── rcParams ──────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "Arial",
    "font.size":       6,
    "axes.labelsize":  6,
    "axes.titlesize":  6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "axes.linewidth":  0.5,
    "pdf.fonttype":    42,
    "svg.fonttype":    "none",
})

MM2IN = 1 / 25.4


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_matrix(df: pd.DataFrame, name_map: dict) -> tuple[np.ndarray, list, list]:
    """
    Pivot interaction bins → (matrix, row_labels, col_labels).
    Rows = target phenotypes (sorted by peak bin), cols = pseudotime bins.
    """
    targets = [t for t in name_map if t in df["target_phenotype"].unique()]
    if not targets:
        return np.zeros((0, 0)), [], []

    pivot = (
        df[df["target_phenotype"].isin(targets)]
        .pivot_table(index="target_phenotype", columns="pseudotime_bin",
                     values=VALUE_COL, aggfunc="mean")
        .reindex(columns=sorted(df["pseudotime_bin"].unique()))
    )

    # Sort rows by peak bin (argmax)
    peak_bins = pivot.values.argmax(axis=1)
    order = np.argsort(peak_bins)
    pivot = pivot.iloc[order]

    # Column labels: median pseudotime per bin
    bin_medians = (
        df.groupby("pseudotime_bin")["pseudotime_median"]
        .mean()
        .reindex(pivot.columns)
    )
    col_labels = [f"{v:.1f}" for v in bin_medians]
    row_labels  = [name_map.get(t, t) for t in pivot.index]

    return pivot.values, row_labels, col_labels


def draw_heatmap(ax, mat, row_labels, col_labels, title,
                 vmax=None, cmap="YlOrRd", show_cbar=False, cbar_ax=None):
    """Draw one interaction heatmap on ax."""
    if mat.size == 0:
        ax.set_visible(False)
        return None

    vmax = vmax or float(np.nanpercentile(mat, 99))
    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax, interpolation="nearest")

    # Row labels
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.5)
    ax.yaxis.set_tick_params(length=0, pad=2)

    # Column labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       fontsize=5, va="top")
    ax.xaxis.set_tick_params(length=0, pad=1)
    ax.set_xlabel("Pseudotime (bin median)", fontsize=5.5, labelpad=2)

    # Grid
    for i in range(len(row_labels) + 1):
        ax.axhline(i - 0.5, color="white", lw=0.5)
    for j in range(len(col_labels) + 1):
        ax.axvline(j - 0.5, color="white", lw=0.5)

    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(title, fontsize=6, pad=3, fontweight="bold")

    if show_cbar and cbar_ax is not None:
        cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical")
        cbar.set_label("Fraction ductal cells\nwith ≥1 neighbour", fontsize=5, labelpad=2)
        cbar.ax.tick_params(labelsize=5, length=2, width=0.5)
        cbar.outline.set_linewidth(0.4)

    return im


# ── Main ──────────────────────────────────────────────────────────────────────
def make_figure():
    df_a = pd.read_csv(TIER_A_CSV)
    df_t = pd.read_csv(T_CELL_CSV)
    df_b = pd.read_csv(B_CELL_CSV)

    mat_a, rows_a, cols_a = build_matrix(df_a, TIER_A_NAMES)
    mat_t, rows_t, cols_t = build_matrix(df_t, T_CELL_NAMES)
    mat_b, rows_b, cols_b = build_matrix(df_b, B_CELL_NAMES)

    # Common vmax across all panels
    all_vals = np.concatenate([
        mat_a.ravel(), mat_t.ravel(), mat_b.ravel()
    ])
    vmax = float(np.nanpercentile(all_vals[all_vals > 0], 99))

    # ── Layout ──────────────────────────────────────────────────────────────
    # width proportional to n_rows in each panel; shared colorbar on far right
    n_a = max(len(rows_a), 1)
    n_t = max(len(rows_t), 1)
    n_b = max(len(rows_b), 1)
    # All panels have same number of columns (10 bins) so width ~ n_rows
    w_a = n_a / (n_a + n_t + n_b)
    w_t = n_t / (n_a + n_t + n_b)
    w_b = n_b / (n_a + n_t + n_b)

    fig_w = 170 * MM2IN
    # height: enough for the tallest panel (7 rows × 6mm + headers + x labels)
    n_max = max(n_a, n_t, n_b)
    fig_h = max(60, n_max * 6 + 22) * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = mgridspec.GridSpec(
        1, 4,
        width_ratios=[w_a, w_t, w_b, 0.04],
        left=0.06, right=0.97,
        top=0.93, bottom=0.20,
        wspace=0.35,
    )

    ax_a    = fig.add_subplot(gs[0, 0])
    ax_t    = fig.add_subplot(gs[0, 1])
    ax_b    = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3])

    draw_heatmap(ax_a, mat_a, rows_a, cols_a,
                 title="Tier A", vmax=vmax)
    draw_heatmap(ax_t, mat_t, rows_t, cols_t,
                 title="T cell subsets", vmax=vmax)
    last_im = draw_heatmap(ax_b, mat_b, rows_b, cols_b,
                           title="B cell subsets", vmax=vmax,
                           show_cbar=True, cbar_ax=ax_cbar)

    # Sub-panel labels i, ii, iii
    for lbl, ax in [("a", ax_a), ("b", ax_t), ("c", ax_b)]:
        ax.text(-0.22, 1.10, lbl,
                transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1H_interaction_dynamics.pdf"
    out_png = RESULT_DIR / "suppfig1H_interaction_dynamics.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
