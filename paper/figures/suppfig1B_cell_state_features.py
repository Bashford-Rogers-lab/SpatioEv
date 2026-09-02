"""
Supplementary Figure 1B — Cell-state feature trends along pseudotime
=====================================================================
All 15 epithelial_state block features (per-niche aggregates of single-cell
morphology and texture) shown as LOWESS trends vs quantile pseudotime.

Layout: 3 rows × 5 columns, colour-coded by sub-category:
  Shape    (blues)   — solidity, perimeter variability, N/C ratio
  Texture  (greens)  — Haralick features, entropy
  Functional (oranges/reds) — polarity, inertia, lacunarity, PCC, intensity ratio

Run:
    python notebooks/suppfig1B_cell_state_features.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig1B_cell_state_features.pdf  (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
RESULT_DIR = ROOT  / "paper" / "notebooks" / "results" / "pseudotime_exp2"

FEAT_CSV  = RESULT_DIR / "pathology_feature_table_with_modules.csv"
PT_CSV    = RESULT_DIR / "niche_pseudotime_results.csv"
NICHE_KEY = "pancreatic ductal epithelium_mask_component"

# ── Feature definitions — (column, label, colour) ────────────────────────────
# Shape / morphology  →  blues
SHAPE_COLOR   = "#1f78b4"
TEXTURE_COLOR = "#33a02c"
FUNC_COLOR    = "#e6550d"

FEATURES = [
    # Row 0 — shape
    ("state__solidity__mean",        "Cell solidity",         "#1f78b4"),
    ("state__perimeter__std",        "Perimeter (SD)",        "#4393c3"),
    ("state__nc_ratio__std",         "N/C ratio (SD)",        "#6baed6"),
    ("state__nc_ratio__iqr",         "N/C ratio (IQR)",       "#9ecae1"),
    ("state__pcc_ck19_nak__mean",    "PCC CK19/NaK",          "#08519c"),
    # Row 1 — texture
    ("state__haralick_correlation__mean", "Haralick corr.",   "#33a02c"),
    ("state__haralick_energy__mean",      "Haralick energy",  "#74c476"),
    ("state__haralick_homogeneity__mean", "Haralick homog.",  "#a1d99b"),
    ("state__haralick_contrast__std",     "Haralick contrast (SD)", "#006d2c"),
    ("state__entropy__std",               "Entropy (SD)",     "#31a354"),
    # Row 2 — functional / heterogeneity
    ("state__polarity_score__std",   "Polarity (SD)",         "#e6550d"),
    ("state__inertia__std",          "Inertia (SD)",          "#fd8d3c"),
    ("state__lacunarity__std",       "Lacunarity (SD)",       "#fdae6b"),
    ("state__pcc_ck19_nak__std",     "PCC CK19/NaK (SD)",    "#d62728"),
    ("state__intensity_ratio__mean", "Intensity ratio",       "#9467bd"),
]

N_ROWS, N_COLS = 3, 5

CATEGORY_LEGEND = [
    mpatches.Patch(color="#1f78b4", label="Shape / morphology"),
    mpatches.Patch(color="#33a02c", label="Texture (Haralick / entropy)"),
    mpatches.Patch(color="#e6550d", label="Functional / heterogeneity"),
]

# ── rcParams ──────────────────────────────────────────────────────────────────
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


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    feat = pd.read_csv(FEAT_CSV)
    pt   = pd.read_csv(PT_CSV)
    cols = [f[0] for f in FEATURES if f[0] in feat.columns]
    id_c = [NICHE_KEY, "image_id"]
    df   = feat[id_c + cols].merge(
        pt[id_c + ["elpigraph_pseudotime_q"]], on=id_c, how="inner"
    )
    print(f"  {len(df):,} niches")
    return df


def _lowess(x, y, frac=0.35):
    import statsmodels.api as sm
    o = np.argsort(x)
    r = sm.nonparametric.lowess(y[o], x[o], frac=frac, return_sorted=True)
    return r[:, 0], r[:, 1]


# ── Draw ──────────────────────────────────────────────────────────────────────
def make_figure():
    df = load_data()

    fig_w = 170 * MM2IN
    fig_h = 82  * MM2IN

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.58, "wspace": 0.28},
    )
    fig.patch.set_facecolor("white")

    pt_col = "elpigraph_pseudotime_q"

    for idx, (col, label, color) in enumerate(FEATURES):
        r, c = divmod(idx, N_COLS)
        ax   = axes[r, c]

        if col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[[pt_col, col]].dropna().sort_values(pt_col)
        if len(sub) < 30:
            ax.set_visible(False)
            continue

        x = sub[pt_col].to_numpy()
        y = sub[col].to_numpy()

        # Scatter sample
        rng  = np.random.default_rng(abs(hash(col)) % (2**31))
        samp = rng.choice(len(x), size=min(500, len(x)), replace=False)
        ax.scatter(x[samp], y[samp], s=0.6, color=color, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)

        # Mean reference
        ax.axhline(float(np.mean(y)), color="#cccccc", lw=0.5, ls="--", zorder=0)

        # LOWESS
        lx, ly = _lowess(x, y)
        ax.plot(lx, ly, color=color, lw=1.6, solid_capstyle="round", zorder=3)

        # Spearman ρ
        rho, p = spearmanr(x, y)
        sig = "*" if p < 0.05 else ""
        ax.text(0.97, 0.08, f"ρ = {rho:+.2f}{sig}",
                transform=ax.transAxes, fontsize=4.8,
                ha="right", va="bottom", color=color)

        ax.set_title(label, fontsize=5, pad=2, color=color, fontweight="semibold")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(length=2, pad=1.5, labelsize=5.5)

        if r == N_ROWS - 1:
            ax.set_xlabel("Pseudotime", fontsize=5.5, labelpad=1)
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Value", fontsize=5.5, labelpad=2)
        else:
            ax.set_yticklabels([])

    # Hide unused
    for idx in range(len(FEATURES), N_ROWS * N_COLS):
        r, c = divmod(idx, N_COLS)
        axes[r, c].set_visible(False)

    # Category legend
    fig.legend(
        handles=CATEGORY_LEGEND, fontsize=5.5, ncol=3,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        frameon=False, handlelength=1.0, handleheight=0.9,
        borderpad=0.3, labelspacing=0.25, columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.14)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1B_cell_state_features.pdf"
    out_png = RESULT_DIR / "suppfig1B_cell_state_features.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
