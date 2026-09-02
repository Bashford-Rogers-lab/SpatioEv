"""
Supplementary Figure 3C — Trajectory clinical summary (2 panels)
=================================================================
  A (left):  Branch × module-score heatmap — each branch encodes a
             distinct and biologically coherent disease state.
  B (right): Per-sample branch proportion barplot — patient trajectory
             fingerprint enabling patient-level profiling.

Width: 170 mm   Height: 82 mm

Run:
    python notebooks/suppfig3C_trajectory_clinical.py

Output: notebooks/results/suppfig3/suppfig3C_trajectory_clinical.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches

from fig2_shared_config import (
    CACHE_DIR, MM2IN, set_pub_rc,
    SAMPLE_LABELS, SAMPLE_ORDER,
    make_branch_palette, assign_branch_bio_names, MODULE_COLS,
)

set_pub_rc()

OUT_SUPPFIG3 = Path("/Users/shihongwu/SpatioEv/paper/notebooks/results/suppfig3")

SCORE_COLS = [
    "pdac_early_duct_anchor_score",
    "pdac_panin_like_dysplasia_score",
    "pdac_invasive_gland_forming_score",
    "pdac_invasion_desmoplasia_score",
    "pdac_dedifferentiation_score",
    "pdac_proliferation_score",
]
SCORE_LABELS = {
    "pdac_early_duct_anchor_score":      "Early-duct",
    "pdac_panin_like_dysplasia_score":   "PanIN-like\ndysplasia",
    "pdac_invasive_gland_forming_score": "Invasive\ngland",
    "pdac_invasion_desmoplasia_score":   "Invasion–\ndesmoplasia",
    "pdac_dedifferentiation_score":      "Dediff.",
    "pdac_proliferation_score":          "Proliferation",
}

BRANCH_ORDER = ["trunk", "branch 12", "branch 20", "branch 15",
                "branch 17", "branch 14.b", "branch 23.a"]


def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        df = pickle.load(f)
    return df


def make_figure():
    df = load_data()

    avail_mod   = [c for c in MODULE_COLS if c in df.columns]
    branch_palette = make_branch_palette(BRANCH_ORDER)
    bio_names   = assign_branch_bio_names(df, avail_mod)
    branch_present = [b for b in BRANCH_ORDER if b in df["major_branch"].unique()]

    def blabel(b):
        nm = bio_names.get(b, b)
        return "Trunk" if b == "trunk" else f"{nm}\n({b})"

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_w = 170 * MM2IN
    fig_h = 82  * MM2IN          # taller to accommodate legend below panels

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = mgridspec.GridSpec(
        1, 2,
        width_ratios=[1.1, 1.0],
        left=0.09, right=0.98, top=0.93, bottom=0.26,   # extra bottom for legend
        wspace=0.48,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ═══════════════════════════════════════════════════════════════════════════
    # A — Branch × module score heatmap
    # ═══════════════════════════════════════════════════════════════════════════
    ax_a.set_title("Branch–module score profile", fontsize=5.5, pad=3,
                   loc="left", fontweight="bold")

    avail_scores = [c for c in SCORE_COLS if c in df.columns]
    hmap_mean = (
        df[df["major_branch"].isin(branch_present)]
        .groupby("major_branch", observed=True)[avail_scores]
        .mean()
        .reindex(branch_present)
    )
    hmap_z  = (hmap_mean - hmap_mean.mean()) / hmap_mean.std().replace(0, 1)
    mat     = hmap_z.values
    vmax    = np.nanmax(np.abs(mat))

    im = ax_a.imshow(mat, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label("Z-score", fontsize=5.5, labelpad=2)
    cbar.ax.tick_params(labelsize=5.0, length=1.5)

    ax_a.set_yticks(range(len(branch_present)))
    ax_a.set_yticklabels([blabel(b) for b in branch_present], fontsize=5.0)
    for tick_lbl, b in zip(ax_a.get_yticklabels(), branch_present):
        tick_lbl.set_color(branch_palette.get(b, "#333"))

    ax_a.set_xticks(range(len(avail_scores)))
    ax_a.set_xticklabels(
        [SCORE_LABELS.get(c, c) for c in avail_scores],
        fontsize=5.0, rotation=40, ha="right", rotation_mode="anchor",
    )
    ax_a.tick_params(length=0, pad=2)
    for x in np.arange(-0.5, len(avail_scores), 1):
        ax_a.axvline(x, color="white", lw=0.5)
    for y in np.arange(-0.5, len(branch_present), 1):
        ax_a.axhline(y, color="white", lw=0.5)
    for sp in ax_a.spines.values():
        sp.set_visible(False)

    # ═══════════════════════════════════════════════════════════════════════════
    # B — Per-sample branch proportion barplot
    # ═══════════════════════════════════════════════════════════════════════════
    ax_b.set_title("Patient trajectory fingerprint", fontsize=5.5, pad=3,
                   loc="left", fontweight="bold")

    sb = (
        df[df["major_branch"].isin(branch_present)]
        .groupby(["sample_id", "major_branch"], observed=True)
        .size().unstack(fill_value=0)
    )
    sb_pct = sb.div(sb.sum(axis=1), axis=0)
    samples_in_order = [s for s in SAMPLE_ORDER if s in sb_pct.index]
    sb_pct = sb_pct.reindex(samples_in_order)

    left_arr = np.zeros(len(samples_in_order))
    for b in branch_present:
        if b not in sb_pct.columns:
            continue
        vals = sb_pct[b].values
        ax_b.barh(range(len(samples_in_order)), vals, left=left_arr,
                  color=branch_palette.get(b, "#888"), height=0.65, linewidth=0)
        left_arr += vals

    ax_b.set_yticks(range(len(samples_in_order)))
    ax_b.set_yticklabels(
        [SAMPLE_LABELS.get(s, s) for s in samples_in_order], fontsize=5.5,
    )
    ax_b.yaxis.set_tick_params(length=0, pad=2)
    ax_b.set_xticks([0, 0.5, 1])
    ax_b.set_xticklabels(["0", "0.5", "1"], fontsize=5.5)
    ax_b.set_xlabel("Fraction of niches", fontsize=6.0, labelpad=2)
    ax_b.set_xlim(0, 1)
    ax_b.invert_yaxis()
    for sp in ["top", "right"]:
        ax_b.spines[sp].set_visible(False)

    leg_handles = [
        mpatches.Patch(facecolor=branch_palette.get(b, "#888"),
                       label=bio_names.get(b, b) if b != "trunk" else "Trunk",
                       linewidth=0)
        for b in branch_present
    ]
    # Legend placed below both panels — avoids overlapping the barplot
    fig.legend(
        handles=leg_handles, fontsize=5.0, ncol=4,
        loc="lower center", bbox_to_anchor=(0.54, 0.01),
        frameon=False,
        handlelength=0.9, handleheight=0.75,
        borderpad=0.2, labelspacing=0.25, columnspacing=0.9,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_SUPPFIG3.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_SUPPFIG3 / "suppfig3C_trajectory_clinical.pdf"
    out_png = OUT_SUPPFIG3 / "suppfig3C_trajectory_clinical.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
