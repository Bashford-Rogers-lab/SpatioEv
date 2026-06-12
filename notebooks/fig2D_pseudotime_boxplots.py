"""
Figure 2D — Pseudotime distribution by disease group and by sample
==================================================================
Two boxplots side by side:
  Left:  pooled_pseudotime_q by disease group (Normal vs PDAC)
  Right: pooled_pseudotime_q by sample, ordered Normal → PDAC

Width: 83 mm (half-page width)  Height: 42 mm

Run:
    python notebooks/fig2D_pseudotime_boxplots.py

Output: notebooks/results/fig2/fig2D_pseudotime_boxplots.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal

from fig2_shared_config import (
    CACHE_DIR, OUT_DIR, MM2IN, set_pub_rc,
    DISEASE_PALETTE, SAMPLE_PALETTE, SAMPLE_LABELS, SAMPLE_ORDER,
)

set_pub_rc()


def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        df = pickle.load(f)
    if "pooled_pseudotime_q" not in df.columns:
        from scipy.stats import rankdata
        mask = df["pooled_pseudotime"].notna()
        df.loc[mask, "pooled_pseudotime_q"] = (
            rankdata(df.loc[mask, "pooled_pseudotime"].values) / mask.sum()
        )
    return df


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def make_figure():
    df = load_data()
    pt_col = "pooled_pseudotime_q"

    fig_w = 83  * MM2IN   # half-page width
    fig_h = 42  * MM2IN

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.45, "width_ratios": [1, 1.8]},
    )
    fig.patch.set_facecolor("white")

    # ── Left: disease group ───────────────────────────────────────────────────
    ax = axes[0]
    groups  = [g for g in ["NormalPancreas", "PDAC"] if g in df["disease_group"].unique()]
    data    = [df.loc[df["disease_group"] == g, pt_col].dropna().values for g in groups]
    colors  = [DISEASE_PALETTE.get(g, "#888") for g in groups]
    xlabels = [g.replace("NormalPancreas", "Normal\npancreas") for g in groups]

    bp = ax.boxplot(
        data, positions=range(len(groups)), widths=0.42,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker=".", markersize=1.0, alpha=0.25, markeredgewidth=0),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.75, linewidth=0.6)

    if len(data) == 2 and all(len(d) > 0 for d in data):
        _, p = mannwhitneyu(data[0], data[1], alternative="two-sided")
        stars = sig_stars(p)
        y_top = max(np.percentile(d, 95) for d in data) + 0.04
        ax.plot([0, 1], [y_top, y_top], color="black", lw=0.6)
        ax.text(0.5, y_top + 0.01, stars, ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(xlabels, fontsize=5.0)
    ax.set_ylabel("Quantile pseudotime", fontsize=5.5, labelpad=2)
    ax.set_title("Disease group", fontsize=5.5, pad=3)
    ax.tick_params(length=2, pad=1.5, labelsize=5.0)

    # ── Right: by sample ─────────────────────────────────────────────────────
    ax = axes[1]
    sorder  = [s for s in SAMPLE_ORDER if s in df["sample_id"].unique()]
    sdata   = [df.loc[df["sample_id"] == s, pt_col].dropna().values for s in sorder]
    scolors = [SAMPLE_PALETTE.get(s, "#888") for s in sorder]
    slabels = [SAMPLE_LABELS.get(s, s) for s in sorder]

    bp2 = ax.boxplot(
        sdata, positions=range(len(sorder)), widths=0.42,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.2),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker=".", markersize=1.0, alpha=0.25, markeredgewidth=0),
    )
    for patch, c in zip(bp2["boxes"], scolors):
        patch.set(facecolor=c, alpha=0.75, linewidth=0.6)

    valid_sdata = [d for d in sdata if len(d) > 0]
    if len(valid_sdata) >= 2:
        _, p_k = kruskal(*valid_sdata)
        label = "p < 0.001" if p_k < 0.001 else f"p = {p_k:.2e}"
        ax.text(0.98, 0.03, f"Kruskal–Wallis {label}",
                transform=ax.transAxes, fontsize=4.2, ha="right", va="bottom",
                color="#555555", style="italic")

    ax.set_xticks(range(len(sorder)))
    ax.set_xticklabels(slabels, fontsize=4.8)
    ax.set_ylabel("Quantile pseudotime", fontsize=5.5, labelpad=2)
    ax.set_title("By sample", fontsize=5.5, pad=3)
    ax.tick_params(length=2, pad=1.5, labelsize=4.8)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.18)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2D_pseudotime_boxplots.pdf"
    out_png = OUT_DIR / "fig2D_pseudotime_boxplots.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
