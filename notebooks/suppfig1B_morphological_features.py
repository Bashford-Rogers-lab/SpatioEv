"""
Supplementary Figure 1B — Pseudotime input feature trends
==========================================================
Shows the non-module features that drove the pseudotime trajectory, visualised
as LOWESS trends vs quantile pseudotime — the same style as Figure 1H panels i/ii.

Panel i  (left):  Six representative cell-level morphological / texture features
                  (epithelial_state feature block).
Panel ii (right): Six niche-level architectural + microenvironment features
                  (architecture_topology + microenvironment + other blocks).

All features from pathology_feature_table_with_modules.csv joined with pseudotime.

Run:
    python notebooks/suppfig1B_morphological_features.py

Output: notebooks/results/pseudotime_exp2/suppfig1B_morphological_features.pdf  (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
RESULT_DIR = ROOT / "notebooks" / "results" / "pseudotime_exp2"

FEAT_CSV   = RESULT_DIR / "pathology_feature_table_with_modules.csv"
PT_CSV     = RESULT_DIR / "niche_pseudotime_results.csv"

NICHE_KEY  = "pancreatic ductal epithelium_mask_component"

# ── Feature definitions ───────────────────────────────────────────────────────
# Panel i — cell-level morphology & texture (epithelial_state block)
# (column, display label, colour)
CELL_FEATURES = [
    ("state__solidity__mean",            "Cell solidity",          "#1f78b4"),
    ("state__haralick_correlation__mean","Texture correlation",    "#33a02c"),
    ("state__haralick_energy__mean",     "Texture energy",         "#6a3d9a"),
    ("state__pcc_ck19_nak__mean",        "PCC CK19/NaK",           "#e31a1c"),
    ("state__polarity_score__std",       "Polarity heterogeneity", "#ff7f00"),
    ("state__entropy__std",              "Texture entropy (SD)",   "#b15928"),
]

# Panel ii — niche architecture + microenvironment
NICHE_FEATURES = [
    ("geometry__hull_circularity",               "Hull circularity",       "#1f78b4"),
    ("geometry__mean_nearest_neighbor_distance", "Mean cell spacing",      "#33a02c"),
    ("topology__skeleton_tortuosity",            "Duct tortuosity",        "#6a3d9a"),
    ("surround_prop__Fibroblasts",               "Fibroblast proportion",  "#e31a1c"),
    ("surround_prop__Vimentin_only_mesenchyme",  "Mesenchyme proportion",  "#ff7f00"),
    ("graph_boundary__boundary_fraction",        "Boundary fraction",      "#b15928"),
]

# ── Publication rcParams ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "Arial",
    "font.size":       6,
    "axes.labelsize":  6,
    "axes.titlesize":  6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "axes.linewidth":  0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "pdf.fonttype":    42,
    "svg.fonttype":    "none",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

MM2IN = 1 / 25.4


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    feat = pd.read_csv(FEAT_CSV)
    pt   = pd.read_csv(PT_CSV)

    all_feat_cols = list(
        set([f[0] for f in CELL_FEATURES] + [f[0] for f in NICHE_FEATURES])
        & set(feat.columns)
    )
    id_cols = [NICHE_KEY, "image_id"]

    df = feat[id_cols + all_feat_cols].merge(
        pt[id_cols + ["elpigraph_pseudotime_q"]],
        on=id_cols, how="inner",
    )
    print(f"  Loaded {len(df):,} niches")
    return df


# ── LOWESS ────────────────────────────────────────────────────────────────────
def _lowess(x, y, frac=0.35):
    import statsmodels.api as sm
    order  = np.argsort(x)
    result = sm.nonparametric.lowess(y[order], x[order], frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


# ── Generic 2×3 LOWESS facet block ───────────────────────────────────────────
def plot_facets(axes_grid, df, features, y_label="Value"):
    """
    features : list of (col, label, color) — exactly 6 items for a 2×3 grid.
    """
    pt_col  = "elpigraph_pseudotime_q"
    n_rows  = len(axes_grid)
    n_cols  = len(axes_grid[0])
    axes_flat = [ax for row in axes_grid for ax in row]

    for idx, (col, label, color) in enumerate(features):
        ax = axes_flat[idx]
        if col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[[pt_col, col]].dropna().sort_values(pt_col)
        if len(sub) < 30:
            ax.set_visible(False)
            continue

        x = sub[pt_col].to_numpy()
        y = sub[col].to_numpy()

        # Faint scatter
        rng  = np.random.default_rng(abs(hash(col)) % (2**31))
        samp = rng.choice(len(x), size=min(500, len(x)), replace=False)
        ax.scatter(x[samp], y[samp], s=0.6, color=color, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)

        # Mean reference line
        ax.axhline(float(np.mean(y)), color="#cccccc", lw=0.5, ls="--", zorder=0)

        # LOWESS
        lx, ly = _lowess(x, y, frac=0.35)
        ax.plot(lx, ly, color=color, lw=1.6, solid_capstyle="round", zorder=3)

        # Spearman ρ
        r, p = spearmanr(x, y)
        sig  = "*" if p < 0.05 else ""
        ax.text(0.97, 0.08, f"ρ = {r:+.2f}{sig}",
                transform=ax.transAxes, fontsize=5,
                ha="right", va="bottom", color=color)

        ax.set_title(label, fontsize=5, pad=2, color=color, fontweight="semibold")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(length=2, pad=1.5, labelsize=5.5)

        row_i = idx // n_cols
        col_i = idx %  n_cols

        if row_i == n_rows - 1:
            ax.set_xlabel("Pseudotime", fontsize=5.5, labelpad=1)
        else:
            ax.set_xticklabels([])

        if col_i == 0:
            ax.set_ylabel(y_label, fontsize=5.5, labelpad=2)
        else:
            ax.set_yticklabels([])

    for ax in axes_flat[len(features):]:
        ax.set_visible(False)


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure():
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    df = load_data()

    fig_w = 140 * MM2IN
    fig_h = 60  * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Two 2×3 blocks side by side with a gap
    gs_main = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 0.30, 1.0, 0.02],
        left=0.07, right=0.98,
        top=0.92, bottom=0.18,
        wspace=0.05,
    )

    # Panel i — cell features
    gs_cell = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_main[0],
        hspace=0.55, wspace=0.22,
    )
    axes_cell = [
        [fig.add_subplot(gs_cell[r, c]) for c in range(3)]
        for r in range(2)
    ]

    # Panel ii — niche features
    gs_niche = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_main[2],
        hspace=0.55, wspace=0.22,
    )
    axes_niche = [
        [fig.add_subplot(gs_niche[r, c]) for c in range(3)]
        for r in range(2)
    ]

    plot_facets(axes_cell,  df, CELL_FEATURES,  y_label="Value")
    plot_facets(axes_niche, df, NICHE_FEATURES, y_label="Value")

    # Sub-panel labels
    for lbl, ax, dx in [("i",  axes_cell[0][0],  -0.30),
                         ("ii", axes_niche[0][0], -0.30)]:
        ax.text(dx, 1.20, lbl,
                transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    # Section headers
    cell_x   = gs_main[0].get_position(fig).x0
    niche_x  = gs_main[2].get_position(fig).x0
    fig.text(cell_x,  0.975, "Cell morphological features",
             fontsize=6.5, fontweight="bold", va="top", color="#333333",
             transform=fig.transFigure)
    fig.text(niche_x, 0.975, "Niche architectural & microenvironment features",
             fontsize=6.5, fontweight="bold", va="top", color="#333333",
             transform=fig.transFigure)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1B_morphological_features.pdf"
    out_png = RESULT_DIR / "suppfig1B_morphological_features.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
