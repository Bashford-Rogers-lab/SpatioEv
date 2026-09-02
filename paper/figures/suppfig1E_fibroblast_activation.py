"""
Supplementary Figure 1E — Fibroblast activation markers over pseudotime
========================================================================
Mean expression of activation markers (FAP, αSMA, PDPN, Thy1) in surrounding
Fibroblasts, shown as LOWESS trends vs quantile pseudotime.

Layout: 2 rows × 2 columns
All from pathology_feature_table_with_modules.csv → surround__Fibroblasts__*__mean

Run:
    python notebooks/suppfig1E_fibroblast_activation.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig1E_fibroblast_activation.pdf (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
FEATURES = [
    ("surround__Fibroblasts__FAP_expr__mean",   "FAP",   "#e41a1c"),
    ("surround__Fibroblasts__aSMA_expr__mean",  "αSMA",  "#377eb8"),
    ("surround__Fibroblasts__PDPN_expr__mean",  "PDPN",  "#4daf4a"),
    ("surround__Fibroblasts__Thy1_expr__mean",  "Thy1",  "#984ea3"),
]

N_ROWS, N_COLS = 2, 2

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
    print(f"  {len(df):,} niches loaded")
    missing = [f[0] for f in FEATURES if f[0] not in feat.columns]
    if missing:
        print(f"  Missing columns: {missing}")
    return df


def _lowess(x, y, frac=0.35):
    import statsmodels.api as sm
    o = np.argsort(x)
    r = sm.nonparametric.lowess(y[o], x[o], frac=frac, return_sorted=True)
    return r[:, 0], r[:, 1]


# ── Draw ──────────────────────────────────────────────────────────────────────
def make_figure():
    df = load_data()

    fig_w = 80 * MM2IN
    fig_h = 68 * MM2IN

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.58, "wspace": 0.32},
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

        ax.set_title(label, fontsize=5.5, pad=2, color=color, fontweight="semibold")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(length=2, pad=1.5, labelsize=5.5)

        if r == N_ROWS - 1:
            ax.set_xlabel("Pseudotime", fontsize=5.5, labelpad=1)
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Mean expr. (surround.)", fontsize=5.5, labelpad=2)
        else:
            ax.set_yticklabels([])

    # Figure title
    fig.text(0.5, 0.99, "Fibroblast activation markers in niche surroundings",
             ha="center", va="top", fontsize=6, fontweight="bold", color="#333333",
             transform=fig.transFigure)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.15)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1E_fibroblast_activation.pdf"
    out_png = RESULT_DIR / "suppfig1E_fibroblast_activation.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
