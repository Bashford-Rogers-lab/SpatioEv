"""
Figure 2G — Niche morphological features consistently change along pseudotime
=============================================================================
Features selected by Spearman |ρ| vs pooled_pseudotime_q, requiring ≥80% of
non-trunk branches to show the same direction of change.

Six non-redundant features plotted as per-branch LOWESS + faint scatter +
Spearman ρ annotation.  Style matches Supplementary Figure 3B right panels.

Feature selection rationale:
  topology__mean_core_number       — cellular network density (↓: ducts lose
                                     their multi-layered, tightly-connected core)
  topology__skeleton_tortuosity    — ductal skeleton tortuosity (↓: glands
                                     become shorter / less convoluted)
  geometry__orientation_entropy    — cell orientation randomness (↓: cells adopt
                                     more uniform / anisotropic arrangement)
  features__area__edge_abs_diff_mean      — cell size divergence between
                                     neighbours (↑: increasing size heterogeneity)
  features__Ki67_expr__edge_abs_diff_mean — Ki67 spread between neighbours
                                     (↑: more heterogeneous proliferation)
  features__polarity_score__bin_assortativity — polarity coordination among
                                     neighbours (↓: loss of coordinated polarity)

Width: 85 mm   Height: 75 mm  (3 rows × 2 cols)

Run:
    python notebooks/fig2G_morphology_lowess.py

Output: notebooks/results/fig2/fig2G_morphology_lowess.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.lines as mlines
import statsmodels.api as sm

from fig2_shared_config import (
    CACHE_DIR, OUT_DIR, MM2IN, set_pub_rc,
    make_branch_palette, assign_branch_bio_names, MODULE_COLS,
)

set_pub_rc()

# ── Feature selection ──────────────────────────────────────────────────────────
FEATURES = [
    # (column, display_label, direction_note)
    ("topology__mean_core_number",
     "Niche core\ndensity",
     "↓ along PT"),
    ("topology__skeleton_tortuosity",
     "Ductal skeleton\ntortuosity",
     "↓ along PT"),
    ("geometry__orientation_entropy",
     "Cell orientation\nentropy",
     "↓ along PT"),
    ("features__area__edge_abs_diff_mean",
     "Cell size\nheterogeneity",
     "↑ along PT"),
    ("features__Ki67_expr__edge_abs_diff_mean",
     "Ki67 expression\nspread",
     "↑ along PT"),
    ("features__polarity_score__bin_assortativity",
     "Cell polarity\ncoordination",
     "↓ along PT"),
]
LAYOUT = (3, 2)   # 3 rows × 2 cols

BRANCH_ORDER = ["trunk", "branch 12", "branch 20", "branch 15",
                "branch 17", "branch 14.b", "branch 23.a"]

KEY = "pancreatic ductal epithelium_mask_component"


# ── LOWESS helper ──────────────────────────────────────────────────────────────
def _lowess(x, y, frac=0.25, delta=0.02):  # delta in quantile PT units (range 0–1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 30:
        return None, None
    order = np.argsort(x[valid])
    res = sm.nonparametric.lowess(
        y[valid][order], x[valid][order],
        frac=frac, delta=delta, return_sorted=True,
    )
    return res[:, 0], res[:, 1]


def _spearman(x, y):
    """Numpy-only Spearman ρ."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
    t = rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-12))
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return rho, p


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        nr = pickle.load(f)
    with open(CACHE_DIR / "pooled_pathology_feature_df.pkl", "rb") as f:
        pf = pickle.load(f)

    # Compute quantile-normalised pseudotime (0–1) if not already present
    if "pooled_pseudotime_q" not in nr.columns:
        from scipy.stats import rankdata
        mask = nr["pooled_pseudotime"].notna()
        nr.loc[mask, "pooled_pseudotime_q"] = (
            rankdata(nr.loc[mask, "pooled_pseudotime"].values) / mask.sum()
        )

    feat_cols = [col for col, _, _ in FEATURES]
    merged = (
        nr[[KEY, "pooled_pseudotime_q", "major_branch"]]
        .drop_duplicates(subset=[KEY])
        .merge(pf[[KEY] + feat_cols], on=KEY, how="inner")
    )
    return merged, nr


# ── Drawing ────────────────────────────────────────────────────────────────────
def draw_panel(ax, merged, col, label, direction,
               branch_present, branch_palette, bio_names, xlim_max=None):

    pt   = merged["pooled_pseudotime_q"].values.astype(float)
    y    = merged[col].values.astype(float)
    bc   = merged["major_branch"].values

    # Clip y to 1st–99th percentile to prevent outliers from compressing the axis
    y_lo, y_hi = np.nanpercentile(y[np.isfinite(y)], 1), np.nanpercentile(y[np.isfinite(y)], 99)
    y = np.clip(y, y_lo, y_hi)

    # Mean reference
    mean_y = np.nanmean(y)
    ax.axhline(mean_y, color="#cccccc", lw=0.6, ls="--", zorder=0)

    rng = np.random.default_rng(abs(hash(col)) % 2**31)

    for branch in branch_present:
        color = branch_palette.get(branch, "#888888")
        mask  = (bc == branch) & np.isfinite(pt) & np.isfinite(y)
        if mask.sum() < 20:
            continue

        bx, by = pt[mask], y[mask]

        # Clip to 5th–95th percentile of this branch's pseudotime
        lo, hi = np.percentile(bx, 5), np.percentile(bx, 95)

        # Faint scatter (downsampled to ≤300 pts)
        samp = rng.choice(mask.sum(), size=min(300, mask.sum()), replace=False)
        ax.scatter(bx[samp], by[samp],
                   s=0.8, color=color, alpha=0.10,
                   linewidths=0, rasterized=True, zorder=1)

        # LOWESS line clipped to branch range
        lx, ly = _lowess(bx, by)
        if lx is None:
            continue
        in_range = (lx >= lo) & (lx <= hi)
        if in_range.sum() > 5:
            ax.plot(lx[in_range], ly[in_range],
                    color=color, lw=1.4, alpha=0.90,
                    solid_capstyle="round", zorder=3)

    # Global Spearman ρ (all niches)
    valid = np.isfinite(pt) & np.isfinite(y)
    if valid.sum() > 30:
        rho, p = _spearman(pt[valid], y[valid])
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(0.97, 0.06, f"ρ = {rho:+.3f}{sig}",
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="bottom", color="#444444")

    ax.set_title(f"{label}  ({direction})", fontsize=4.0, pad=1,
                 fontweight="semibold", loc="left")
    if xlim_max is not None:
        ax.set_xlim(0, xlim_max)
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Main figure ────────────────────────────────────────────────────────────────
def make_figure():
    merged, nr = load_data()

    avail_mod      = [c for c in MODULE_COLS if c in nr.columns]
    branch_palette = make_branch_palette(BRANCH_ORDER)
    bio_names      = assign_branch_bio_names(nr, avail_mod)
    branch_present = [b for b in BRANCH_ORDER if b in nr["major_branch"].unique()]

    # x-axis: quantile PT is already 0–1
    xlim_max = 1.0

    # ── Layout ────────────────────────────────────────────────────────────────
    nrows, ncols = LAYOUT
    fig_w = 85 * MM2IN
    fig_h = 75 * MM2IN   # taller for 3 rows

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = mgridspec.GridSpec(
        nrows, ncols,
        left=0.12, right=0.98, top=0.86, bottom=0.18,
        hspace=0.55, wspace=0.40,
    )

    for idx, (col, label, direction) in enumerate(FEATURES):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        draw_panel(ax, merged, col, label, direction,
                   branch_present, branch_palette, bio_names,
                   xlim_max=xlim_max)

        if r == nrows - 1:
            ax.set_xlabel("Pseudotime", fontsize=5.0, labelpad=1)
        else:
            ax.set_xticklabels([])

        if c == 0:
            ax.set_ylabel("Feature value", fontsize=5.0, labelpad=2)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = []
    for b in branch_present:
        color = branch_palette.get(b, "#888")
        name  = "Trunk" if b == "trunk" else bio_names.get(b, b)
        legend_handles.append(
            mlines.Line2D([], [], color=color, lw=1.5, label=f"{name}  ({b})")
        )

    fig.legend(
        handles=legend_handles,
        fontsize=4.5, ncol=4,
        loc="lower center", bbox_to_anchor=(0.52, 0.01),
        frameon=False,
        handlelength=1.0, handleheight=0.7,
        borderpad=0.2, labelspacing=0.15, columnspacing=0.6,
    )

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.text(0.52, 0.97,
             "Niche morphological features change consistently across branches",
             ha="center", va="bottom", fontsize=7.5, fontweight="bold",
             transform=fig.transFigure)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2G_morphology_lowess.pdf"
    out_png = OUT_DIR / "fig2G_morphology_lowess.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
