"""
Supplementary Figure 1G — B cell subtype proximity along pseudotime
====================================================================
Fraction of ductal cells with ≥1 Tier B B cell subset neighbour within 30 µm,
plotted across 10 pseudotime bins (from sv.compute_epithelial_centered_interaction_dynamics).

Each subtype shown as a separate facet with dot + line plot.
Layout: 2 rows × 4 columns (7 subtypes; last cell hidden).

Data source: tier_b_b_cells_interaction_bins.csv
  x = pseudotime_median (bin centre)
  y = fraction_source_with_target_neighbor

Run:
    python notebooks/suppfig1G_bcell_subtypes.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig1G_bcell_subtypes.pdf (.png)
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

BINS_CSV = RESULT_DIR / "tier_b_b_cells_interaction_bins.csv"

# ── Subtype definitions — (target_phenotype, label, colour) ──────────────────
SUBTYPES = [
    ("naive B cells",       "Naive B cells",        "#1f78b4"),
    ("memory B cells",      "Memory B cells",       "#33a02c"),
    ("plasmablasts",        "Plasmablasts",          "#e41a1c"),
    ("plasmablasts-like",   "Plasmablasts-like",     "#ff7f00"),
    ("GZMB+ B cells",       "GZMB+ B cells",         "#6a3d9a"),
    ("PDL1+ B cells",       "PDL1+ B cells",         "#b15928"),
    ("APC-like B cells",    "APC-like B cells",      "#666666"),
]

N_ROWS, N_COLS = 2, 4
VALUE_COL = "fraction_source_with_target_neighbor"

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
    df = pd.read_csv(BINS_CSV)
    print(f"  Loaded {len(df)} rows, subtypes: {sorted(df['target_phenotype'].unique())}")
    return df


# ── Draw ──────────────────────────────────────────────────────────────────────
def make_figure():
    df = load_data()

    fig_w = 120 * MM2IN
    fig_h = 60  * MM2IN

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.60, "wspace": 0.30},
    )
    fig.patch.set_facecolor("white")

    for idx, (phenotype, label, color) in enumerate(SUBTYPES):
        r, c = divmod(idx, N_COLS)
        ax   = axes[r, c]

        sub = df[df["target_phenotype"] == phenotype].sort_values("pseudotime_median")
        if sub.empty:
            ax.set_visible(False)
            continue

        x = sub["pseudotime_median"].to_numpy()
        y = sub[VALUE_COL].to_numpy()

        # Mean reference
        ax.axhline(float(np.mean(y)), color="#cccccc", lw=0.5, ls="--", zorder=0)

        # Line + dots
        ax.plot(x, y, color=color, lw=1.4, alpha=0.85,
                solid_capstyle="round", zorder=2)
        ax.scatter(x, y, s=8, color=color, edgecolors="white",
                   linewidths=0.5, zorder=3)

        # Spearman ρ across bins
        if len(x) >= 4:
            rho, p = spearmanr(np.arange(len(x)), y)
            sig = "*" if p < 0.05 else ""
            ax.text(0.97, 0.08, f"ρ = {rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=4.8,
                    ha="right", va="bottom", color=color)

        ax.set_title(label, fontsize=5, pad=2, color=color, fontweight="semibold")
        ax.tick_params(length=2, pad=1.5, labelsize=5.5)

        if r == N_ROWS - 1:
            ax.set_xlabel("Pseudotime", fontsize=5.5, labelpad=1)
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Fraction ductal cells\nwith ≥1 neighbour", fontsize=5, labelpad=2)
        else:
            ax.set_yticklabels([])

        # Light shading every other bin for readability
        for b in range(0, len(x), 2):
            if b < len(x):
                x0 = x[b] - (x[1] - x[0]) / 2 if b > 0 else x[b]
                x1 = x[b] + (x[1] - x[0]) / 2 if b < len(x) - 1 else x[b]
                ax.axvspan(x0, x1, color="#f5f5f5", zorder=0, lw=0)

    # Hide unused
    for idx in range(len(SUBTYPES), N_ROWS * N_COLS):
        r, c = divmod(idx, N_COLS)
        axes[r, c].set_visible(False)

    # Footnote
    fig.text(0.5, 0.005,
             "Proximity: fraction of ductal cells with ≥1 B cell subtype neighbour within 30 µm",
             ha="center", va="bottom", fontsize=4.5, color="#666666",
             transform=fig.transFigure, style="italic")

    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.12)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1G_bcell_subtypes.pdf"
    out_png = RESULT_DIR / "suppfig1G_bcell_subtypes.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
