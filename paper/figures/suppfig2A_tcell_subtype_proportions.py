"""
Supplementary Figure 2A — T cell subtype composition over pseudotime
=====================================================================
Each T cell subtype shown as its conditional proportion within total
surrounding T cells (subtype / sum of all T subtypes per niche),
as LOWESS trends vs quantile pseudotime.

Only niches with total surrounding T cell proportion > 0.005 are used.

Layout: 2 rows × 4 columns (8 subtypes, full grid).
Data: surround_context_tier_b.csv

Run:
    python notebooks/suppfig2A_tcell_subtype_proportions.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig2A_tcell_subtype_proportions.pdf (.png)
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

TIERB_CSV = RESULT_DIR / "surround_context_tier_b.csv"
NICHE_KEY = "pancreatic ductal epithelium_mask_component"

# ── T cell subtype columns → (label, colour) ─────────────────────────────────
# column names are sanitized: spaces→_, +→_, -→_ in the CSV
T_SUBTYPES = [
    ("surround_prop__activated_CD4_T_cells",  "Act. CD4 T",     "#d62728"),
    ("surround_prop__activated_CD8_T_cells",  "Act. CD8 T",     "#e377c2"),
    ("surround_prop__CD8_T_cells",             "CD8 T",          "#1f77b4"),
    ("surround_prop__CD4_T_cells",             "CD4 T",          "#aec7e8"),
    ("surround_prop__Tregs",                   "Tregs",          "#ff7f0e"),
    ("surround_prop__Th2_like_cells",          "Th2-like",       "#2ca02c"),
    ("surround_prop__cytotoxic_CD8_T_cells",   "Cytotoxic CD8",  "#9467bd"),
    ("surround_prop__Tfh_like_cells",          "Tfh-like",       "#8c564b"),
]

# minimum total T cell surround proportion to include a niche
MIN_T_PROP = 0.005

N_ROWS, N_COLS = 2, 4

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
    df = pd.read_csv(TIERB_CSV)
    pt_col = "elpigraph_pseudotime_q"

    # Identify T subtype columns present
    t_cols = [c for c, *_ in T_SUBTYPES if c in df.columns]
    missing = [c for c, *_ in T_SUBTYPES if c not in df.columns]
    if missing:
        print(f"  Missing T columns: {missing}")

    # Total T proportion per niche
    df["_t_total"] = df[t_cols].sum(axis=1)

    # Keep only niches with enough T cells and pseudotime
    df = df[df["_t_total"] > MIN_T_PROP].copy()
    df = df[df[pt_col].notna()].copy()
    print(f"  {len(df):,} niches with T total > {MIN_T_PROP}")

    # Conditional proportion for each subtype
    for col in t_cols:
        df[f"_cond_{col}"] = df[col] / df["_t_total"]

    return df, pt_col


def _lowess(x, y, frac=0.40):
    import statsmodels.api as sm
    o = np.argsort(x)
    r = sm.nonparametric.lowess(y[o], x[o], frac=frac, return_sorted=True)
    return r[:, 0], r[:, 1]


# ── Draw ──────────────────────────────────────────────────────────────────────
def make_figure():
    df, pt_col = load_data()

    fig_w = 170 * MM2IN
    fig_h = 55  * MM2IN

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.60, "wspace": 0.28},
    )
    fig.patch.set_facecolor("white")

    for idx, (col, label, color) in enumerate(T_SUBTYPES):
        r, c = divmod(idx, N_COLS)
        ax   = axes[r, c]

        cond_col = f"_cond_{col}"
        if cond_col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[[pt_col, cond_col]].dropna().sort_values(pt_col)
        if len(sub) < 30:
            ax.set_visible(False)
            continue

        x = sub[pt_col].to_numpy()
        y = sub[cond_col].to_numpy()

        # Scatter sample
        rng  = np.random.default_rng(abs(hash(col)) % (2**31))
        samp = rng.choice(len(x), size=min(500, len(x)), replace=False)
        ax.scatter(x[samp], y[samp], s=0.6, color=color, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)

        # Mean reference
        ax.axhline(float(np.mean(y)), color="#cccccc", lw=0.5, ls="--", zorder=0)

        # LOWESS (wider frac for conditional noisy data)
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
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(length=2, pad=1.5, labelsize=5.5)

        if r == N_ROWS - 1:
            ax.set_xlabel("Pseudotime", fontsize=5.5, labelpad=1)
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Fraction within T cells", fontsize=5, labelpad=2)
        else:
            ax.set_yticklabels([])

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14)

    # Figure note
    fig.text(0.5, 0.005,
             f"Niches with ≥{MIN_T_PROP:.1%} surrounding T cells (total). "
             "Conditional proportion = subtype / sum(all T subtypes).",
             ha="center", va="bottom", fontsize=4.5, color="#666666",
             transform=fig.transFigure, style="italic")

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig2A_tcell_subtype_proportions.pdf"
    out_png = RESULT_DIR / "suppfig2A_tcell_subtype_proportions.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
