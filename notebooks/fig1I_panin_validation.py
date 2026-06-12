"""
Figure 1I — PanIN morphological validation LOWESS
==================================================
Six niche-level features that validate PanIN progression along pseudotime,
plotted as small-multiple LOWESS facets — same style as Figure 1H panel i.

Features chosen for biological interpretability:
  • CK19 expression       (ductal epithelial marker)
  • NaKATPase expression  (basolateral polarity; lost in dysplasia)
  • Ki67 expression       (proliferation)
  • CK19–NaKATPase PCC   (co-expression / polarity coherence)
  • Mean cell solidity    (shape regularity; decreases with dysplasia)
  • Niche circularity     (organised round duct → irregular invasive gland)

Run:
    python notebooks/fig1I_panin_validation.py

Output: notebooks/results/pseudotime_exp2/fig1I_panin_validation.pdf  (.png)
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
RESULT_DIR = ROOT / "notebooks" / "results" / "pseudotime_exp2"

FEAT_CSV = RESULT_DIR / "pathology_feature_table_with_modules.csv"
PT_CSV   = RESULT_DIR / "niche_pseudotime_results.csv"

NICHE_KEY = "pancreatic ductal epithelium_mask_component"

# ── Feature definitions: (column, display label, colour) ─────────────────────
FEATURES = [
    ("state__CK19_expr__mean",      "CK19 expression",           "#1f78b4"),
    ("state__NaKATPase_expr__mean", "NaKATPase expression",      "#00897b"),
    ("state__Ki67_expr__mean",      "Ki67 expression",           "#e53935"),
    ("state__pcc_ck19_nak__mean",   "CK19–NaKATPase co-expr.",   "#fb8c00"),
    ("state__solidity__mean",       "Cell solidity",             "#8e24aa"),
    ("geometry__hull_circularity",  "Niche circularity",         "#43a047"),
]

# ── Publication rcParams (matches Fig 1H) ────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "Arial",
    "font.size":       7,
    "axes.labelsize":  7,
    "axes.titlesize":  7.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
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


# ── LOWESS helper ─────────────────────────────────────────────────────────────
def _lowess(x, y, frac=0.35):
    import statsmodels.api as sm
    order  = np.argsort(x)
    result = sm.nonparametric.lowess(y[order], x[order], frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


# ── Load & merge ──────────────────────────────────────────────────────────────
def load_data():
    feat_cols = [f[0] for f in FEATURES]
    feat = pd.read_csv(FEAT_CSV, usecols=[NICHE_KEY, "image_id"] + feat_cols)
    pt   = pd.read_csv(PT_CSV,   usecols=[NICHE_KEY, "image_id", "elpigraph_pseudotime_q"])
    df   = feat.merge(pt, on=[NICHE_KEY, "image_id"], how="inner")
    print(f"  {len(df):,} niches loaded")
    return df


# ── Plot 2 × 3 facets (identical structure to Fig 1H panel i) ────────────────
def plot_facets(axes_grid, df):
    pt_col  = "elpigraph_pseudotime_q"
    n_rows  = len(axes_grid)
    n_cols  = len(axes_grid[0])
    axes_flat = [ax for row in axes_grid for ax in row]

    for idx, (col, label, color) in enumerate(FEATURES):
        ax  = axes_flat[idx]
        sub = df[[pt_col, col]].dropna().sort_values(pt_col)
        if len(sub) < 30:
            ax.set_visible(False)
            continue

        x = sub[pt_col].to_numpy()
        y = sub[col].to_numpy()

        # Faint scatter cloud
        rng  = np.random.default_rng(abs(hash(col)) % (2**31))
        samp = rng.choice(len(x), size=min(len(x), 500), replace=False)
        ax.scatter(x[samp], y[samp], s=0.6, color=color, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)

        # Zero / mean reference line
        ax.axhline(np.nanmean(y), color="#cccccc", lw=0.5, ls="--", zorder=0)

        # LOWESS curve
        lx, ly = _lowess(x, y, frac=0.35)
        ax.plot(lx, ly, color=color, lw=1.6,
                solid_capstyle="round", zorder=3)

        # Spearman r annotation
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        sig  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        ax.text(0.97, 0.08, f"ρ = {r:+.2f}{sig}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.5, color=color)

        ax.set_title(label, fontsize=6.5, pad=3, color=color, fontweight="semibold")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(length=2, pad=1.5, labelsize=5.5)

        row_i = idx // n_cols
        col_i = idx %  n_cols

        if row_i == n_rows - 1:
            ax.set_xlabel("Pseudotime", fontsize=6, labelpad=1)
        else:
            ax.set_xticklabels([])

        if col_i == 0:
            ax.set_ylabel("Feature value", fontsize=6, labelpad=2)
        else:
            ax.set_yticklabels([])

    for ax in axes_flat[len(FEATURES):]:
        ax.set_visible(False)


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure():
    df = load_data()

    mm2in = 1 / 25.4
    # Same width as Fig 1H panel i block (left portion only — standalone figure)
    fig_w = 120 * mm2in
    fig_h = 80  * mm2in

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=fig.add_gridspec(1, 1, left=0.10, right=0.97,
                                      top=0.94, bottom=0.14)[0],
        hspace=0.52, wspace=0.30,
    )
    axes_grid = [
        [fig.add_subplot(gs[r, c]) for c in range(3)]
        for r in range(2)
    ]

    plot_facets(axes_grid, df)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_pdf = RESULT_DIR / "fig1I_panin_validation.pdf"
    out_png = RESULT_DIR / "fig1I_panin_validation.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
