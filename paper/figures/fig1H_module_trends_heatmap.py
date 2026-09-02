"""
Figure 1H — PDAC module LOWESS + PanIN validation LOWESS + branch × module heatmap
====================================================================================
Panel i   (left):   Six PDAC pathology module LOWESS trends vs quantile pseudotime.
Panel ii  (middle): Six PanIN morphological validation LOWESS trends (CK19, NaKATPase,
                    Ki67, pcc_ck19_nak, solidity, circularity) with Spearman ρ.
Panel iii (right):  Branch × module z-enrichment heatmap.

Sized for a 6.69 × 8.86 inch canvas (Nature full-page): 124 mm × 76 mm.

Run:
    python notebooks/fig1H_module_trends_heatmap.py

Output: paper/notebooks/results/pseudotime_exp2/fig1H_module_trends_heatmap.pdf  (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
RESULT_DIR = ROOT  / "paper" / "notebooks" / "results" / "pseudotime_exp2"

FEAT_CSV   = RESULT_DIR / "pathology_feature_table_with_modules.csv"
PT_CSV     = RESULT_DIR / "niche_pseudotime_results.csv"
BRANCH_CSV = RESULT_DIR / "branch_summary.csv"

NICHE_KEY = "pancreatic ductal epithelium_mask_component"

# ── Module definitions ────────────────────────────────────────────────────────
# (column, display label, colour)
MODULES = [
    ("pdac_early_duct_anchor_score",     "Early duct anchor",      "#2166ac"),
    ("pdac_panin_like_dysplasia_score",  "PanIN-like dysplasia",   "#f4a582"),
    ("pdac_invasive_gland_forming_score","Invasive gland-forming", "#d6604d"),
    ("pdac_invasion_desmoplasia_axis",   "Invasion/desmoplasia",   "#b2182b"),
    ("pdac_proliferation_axis",          "Proliferation",          "#4dac26"),
    ("pdac_dedifferentiation_axis",      "Dedifferentiation",      "#7b3294"),
]
MOD_COLS   = [m[0] for m in MODULES]
MOD_LABELS = {m[0]: m[1] for m in MODULES}
MOD_COLORS = {m[0]: m[2] for m in MODULES}

# ── PanIN morphological validation features ───────────────────────────────────
# (column, display label, colour)
PANIN_FEATURES = [
    ("state__CK19_expr__mean",       "CK19 expression",    "#1f78b4"),
    ("state__NaKATPase_expr__mean",  "NaKATPase expr.",    "#00897b"),
    ("state__Ki67_expr__mean",       "Ki67 expression",    "#e53935"),
    ("state__pcc_ck19_nak__mean",    "PCC CK19/NaK",       "#fb8c00"),
    ("state__solidity__mean",        "Cell solidity",      "#8e24aa"),
    ("geometry__hull_circularity",   "Hull circularity",   "#43a047"),
]
PANIN_COLS   = [f[0] for f in PANIN_FEATURES]
PANIN_LABELS = {f[0]: f[1] for f in PANIN_FEATURES}
PANIN_COLORS = {f[0]: f[2] for f in PANIN_FEATURES}

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


# ── Load & merge data ─────────────────────────────────────────────────────────
def load_data():
    feat = pd.read_csv(FEAT_CSV)
    pt   = pd.read_csv(PT_CSV)
    bs   = pd.read_csv(BRANCH_CSV)

    all_feature_cols = list(set(MOD_COLS + PANIN_COLS) & set(feat.columns))
    id_cols = [NICHE_KEY, "image_id"]
    df = feat[id_cols + all_feature_cols].merge(
        pt[id_cols + ["elpigraph_pseudotime_q", "principal_tree_branch"]],
        on=id_cols, how="inner",
    )
    print(f"  Merged: {len(df):,} niches")
    return df, bs


# ── LOWESS helper ─────────────────────────────────────────────────────────────
def _lowess(x, y, frac=0.35):
    import statsmodels.api as sm
    order = np.argsort(x)
    result = sm.nonparametric.lowess(y[order], x[order], frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


# ── Generic LOWESS small-multiples plot ───────────────────────────────────────
def plot_lowess_facets(axes_grid, df, features, labels, colors,
                       y_label="Score", ref_line_at_mean=False):
    """
    Draw a 2×3 grid of small LOWESS panels.

    axes_grid : list of lists  [[ax00, ax01, ax02], [ax10, ax11, ax12]]
    features  : list of (col, label, color) triples
    ref_line_at_mean : if True, draw a horizontal line at the column mean
                       (useful for morphological features with real units)
    """
    pt_col = "elpigraph_pseudotime_q"
    n_rows = len(axes_grid)
    n_cols = len(axes_grid[0])
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
        samp = rng.choice(len(x), size=min(len(x), 500), replace=False)
        ax.scatter(x[samp], y[samp], s=0.6, color=color, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)

        # Reference line
        ref_y = float(np.mean(y)) if ref_line_at_mean else 0.0
        ax.axhline(ref_y, color="#cccccc", lw=0.5, ls="--", zorder=0)

        # LOWESS
        lx, ly = _lowess(x, y, frac=0.35)
        ax.plot(lx, ly, color=color, lw=1.6,
                solid_capstyle="round", zorder=3)

        # Spearman ρ annotation (upper-right)
        r, p = spearmanr(x, y)
        sig = "*" if p < 0.05 else ""
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

    # Hide unused panels
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)


# ── Panel iii: branch × module heatmap ───────────────────────────────────────
def plot_heatmap(ax, bs, cax):
    z_cols = [f"{c}__z_enrichment" for c in MOD_COLS if f"{c}__z_enrichment" in bs.columns]
    if not z_cols:
        ax.text(0.5, 0.5, "No z-enrichment columns found",
                ha="center", va="center", transform=ax.transAxes)
        return

    bs_sorted = bs.sort_values("median_pseudotime").reset_index(drop=True)
    hm = bs_sorted.set_index("branch")[z_cols].copy()

    row_labels = [
        f"{b}  ({int(bs_sorted.loc[bs_sorted['branch']==b, 'n_observations'].iloc[0])})"
        for b in hm.index
    ]
    col_labels = [MOD_LABELS.get(c.replace("__z_enrichment", ""), c) for c in hm.columns]

    mat  = hm.values.astype(float)
    vmax = max(2.5, float(np.nanmax(np.abs(mat))))

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=5.5, va="top")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5)

    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", lw=0.5)
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", lw=0.5)

    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("z-enrichment\n(branch vs. global)", fontsize=5.5, labelpad=3)
    cbar.ax.tick_params(labelsize=5, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)
    cbar.set_ticks([-2, -1, 0, 1, 2])


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure():
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    df, bs = load_data()

    mm2in = 1 / 25.4
    fig_w = 210 * mm2in   # wider than canvas — trimmed when placed in layout
    fig_h = 60  * mm2in

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Top-level columns:
    # [module LOWESS | gap | PanIN LOWESS | gap | heatmap | colorbar]
    gs_main = fig.add_gridspec(
        1, 6,
        width_ratios=[1.15, 0.32, 1.15, 0.38, 0.75, 0.055],
        left=0.055, right=0.985,
        top=0.93, bottom=0.17,
        wspace=0.05,
    )

    # 2×3 facets for module LOWESS (panel i)
    gs_mod = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_main[0],
        hspace=0.55, wspace=0.22,
    )
    axes_mod = [
        [fig.add_subplot(gs_mod[r, c]) for c in range(3)]
        for r in range(2)
    ]

    # 2×3 facets for PanIN validation (panel ii)
    gs_pan = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=gs_main[2],
        hspace=0.55, wspace=0.22,
    )
    axes_pan = [
        [fig.add_subplot(gs_pan[r, c]) for c in range(3)]
        for r in range(2)
    ]

    ax_heat = fig.add_subplot(gs_main[4])
    ax_cbar = fig.add_subplot(gs_main[5])

    # ── Draw panels ──────────────────────────────────────────────────────────
    plot_lowess_facets(axes_mod, df, MODULES, MOD_LABELS, MOD_COLORS,
                       y_label="Module score", ref_line_at_mean=False)

    plot_lowess_facets(axes_pan, df, PANIN_FEATURES, PANIN_LABELS, PANIN_COLORS,
                       y_label="Feature value", ref_line_at_mean=True)

    plot_heatmap(ax_heat, bs, ax_cbar)

    # ── Sub-panel labels ──────────────────────────────────────────────────────
    for label, ax, dx in [("i",   axes_mod[0][0], -0.25),
                           ("ii",  axes_pan[0][0], -0.25),
                           ("iii", ax_heat,         -0.18)]:
        ax.text(dx, 1.14, label,
                transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "fig1H_module_trends_heatmap.pdf"
    out_png = RESULT_DIR / "fig1H_module_trends_heatmap.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
