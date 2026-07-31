"""
Figure 2G — Module score divergence from trunk (forest/dot plot)
================================================================
For each of the 6 canonical disease-progression module scores (y-axis),
shows the Δ z-score (branch mean − trunk mean) for every branch as
branch-coloured dots, plus a black diamond for the niche-weighted mean
across branches.  Rows sorted by mean |Δ| (most universal at top).

Reading guide:
  Dots all on same side of zero  → universal trunk→late change
  Dots on opposite sides         → branch-discriminating feature

Width: 80 mm   Height: 68 mm

Run:
    python notebooks/fig2G_trunk_divergence.py

Output: notebooks/results/fig2/fig2G_trunk_divergence.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fig2_shared_config import (
    CACHE_DIR, MM2IN, set_pub_rc,
    make_branch_palette, assign_branch_bio_names,
    MODULE_COLS, MODULE_LABELS,
)

set_pub_rc()

OUT_DIR    = Path("/Users/shihongwu/SpatioEv/notebooks/results/fig2")
BRANCH_ORDER = ["trunk", "branch 12", "branch 20", "branch 15",
                "branch 17", "branch 14.b", "branch 23.a"]

# Short single-line module labels for y-axis
MODULE_LABELS_SHORT = {
    "pdac_early_duct_anchor_score":      "Early-duct anchor",
    "pdac_panin_like_dysplasia_score":   "PanIN-like dysplasia",
    "pdac_invasive_gland_forming_score": "Invasive gland-forming",
    "pdac_invasion_desmoplasia_axis":    "Invasion–desmoplasia",
    "pdac_proliferation_axis":           "Proliferation",
    "pdac_dedifferentiation_axis":       "Dedifferentiation",
}


def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        nr = pickle.load(f)
    return nr


def compute_deltas(nr, avail_mod, branch_present):
    """
    For each module × non-trunk branch:
      Δ = branch_mean_z − trunk_mean_z
      where z-scores are computed globally across all niches.
    Returns a DataFrame with columns: module, branch, delta, n_niches.
    """
    records = []
    for col in avail_mod:
        mu  = nr[col].mean()
        sd  = nr[col].std()
        sd  = sd if sd > 0 else 1.0
        zs  = (nr[col] - mu) / sd

        trunk_mean = zs[nr["major_branch"] == "trunk"].mean()

        for branch in branch_present:
            if branch == "trunk":
                continue
            b_z = zs[nr["major_branch"] == branch]
            if len(b_z) < 5:
                continue
            records.append({
                "module":   col,
                "branch":   branch,
                "delta":    b_z.mean() - trunk_mean,
                "n_niches": len(b_z),
            })

    return pd.DataFrame(records)


def make_figure():
    nr = load_data()

    avail_mod      = [c for c in MODULE_COLS if c in nr.columns]
    branch_palette = make_branch_palette(BRANCH_ORDER)
    bio_names      = assign_branch_bio_names(nr, avail_mod)
    branch_present = [b for b in BRANCH_ORDER if b in nr["major_branch"].unique()]

    df = compute_deltas(nr, avail_mod, branch_present)

    # Weighted mean Δ per module (weight = n niches)
    wm = (df.groupby("module")
            .apply(lambda g: np.average(g["delta"], weights=g["n_niches"]))
            .rename("wm_delta"))
    abs_wm = wm.abs().rename("abs_wm")

    # Sort rows: largest |wm| at top (most universal / largest effect)
    sort_order = abs_wm.sort_values(ascending=True).index.tolist()  # ascending=True → top=large when yticks reversed

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_w = 88 * MM2IN
    fig_h = 72 * MM2IN
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # ── Plot ──────────────────────────────────────────────────────────────────
    non_trunk = [b for b in branch_present if b != "trunk"]

    for i, col in enumerate(sort_order):
        sub = df[df["module"] == col]

        # Individual branch dots
        for _, row in sub.iterrows():
            color = branch_palette.get(row["branch"], "#888888")
            ax.scatter(row["delta"], i,
                       color=color, s=22, alpha=0.9,
                       edgecolors="white", linewidths=0.4,
                       zorder=3, clip_on=False)

        # Weighted-mean diamond
        wm_val = wm[col]
        ax.scatter(wm_val, i,
                   marker="D", color="black", s=20,
                   edgecolors="none", zorder=4, clip_on=False)

    # Δ=0 reference
    ax.axvline(0, color="#bbbbbb", lw=0.8, ls="--", zorder=0)

    # Light horizontal band every other row for readability
    for i in range(0, len(sort_order), 2):
        ax.axhspan(i - 0.45, i + 0.45, color="#f5f5f5", zorder=0, lw=0)

    # Axes
    ax.set_yticks(range(len(sort_order)))
    ax.set_yticklabels(
        [MODULE_LABELS_SHORT.get(c, c) for c in sort_order],
        fontsize=5.2,
    )
    ax.set_xlabel("Δ z-score  (branch mean − trunk mean)", fontsize=5.5, labelpad=2)
    ax.set_xlim(None, None)   # auto
    ax.tick_params(axis="x", labelsize=5.0, length=2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title("Disease-progression module changes\nfrom trunk to branches",
                 fontsize=5.5, pad=4, fontweight="bold", loc="left")

    # ── Legend — branch palette ───────────────────────────────────────────────
    leg_patches = [
        mpatches.Patch(
            facecolor=branch_palette.get(b, "#888"),
            linewidth=0,
            label="Trunk" if b == "trunk" else bio_names.get(b, b),
        )
        for b in branch_present if b != "trunk"
    ]
    # Diamond for weighted mean
    leg_patches.append(
        plt.scatter([], [], marker="D", color="black", s=20, label="Weighted mean")
    )
    ax.legend(
        handles=leg_patches, fontsize=4.2, ncol=1,
        loc="lower right", frameon=False,
        handlelength=0.8, handleheight=0.7,
        borderpad=0.2, labelspacing=0.25,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2G_trunk_divergence.pdf"
    out_png = OUT_DIR / "fig2G_trunk_divergence.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
