"""
Supplementary Figure 3B — PanIN morphological validation
=========================================================
Two-panel row — consistent with Supp Fig 1 style (faint scatter + LOWESS +
Spearman ρ + mean reference line):

  Left  (1 panel):  Scatter of epithelial vs contextual pseudotime (quantile),
                    coloured by disease group.  Spearman ρ annotated.
  Right (4 panels): Each PanIN validation score vs epithelial pseudotime.
                    Per-disease LOWESS lines with faint scatter behind each.
                    Mean reference dashed line. Spearman ρ per disease.

Width: 170 mm  Height: 58 mm

Run:
    python notebooks/suppfig3B_panin_validation.py

Output: notebooks/results/suppfig3/suppfig3B_panin_validation.pdf (.png)
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
import statsmodels.api as sm
from scipy.stats import spearmanr

from fig2_shared_config import (
    CACHE_DIR, MM2IN, set_pub_rc,
    DISEASE_PALETTE, PANIN_VALIDATION_SCORE_COLS, PANIN_LABELS,
)

set_pub_rc()

OUT_SUPPFIG3 = Path("/Users/shihongwu/SpatioEv/paper/notebooks/results/suppfig3")


def load_data():
    with open(CACHE_DIR / "epithelial_intrinsic_pseudotime_result_df.pkl", "rb") as f:
        epi = pickle.load(f)
    return epi


def _lowess(x, y, frac=0.50, delta=0.01):
    """LOWESS with delta-acceleration (O(n) instead of O(n²) for large n).
    delta=0.01 means linear interpolation within 1% of the [0,1] x-range."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 30:
        return None, None
    o = np.argsort(x[valid])
    r = sm.nonparametric.lowess(
        y[valid][o], x[valid][o], frac=frac, delta=delta, return_sorted=True,
    )
    return r[:, 0], r[:, 1]


def make_figure():
    epi = load_data()
    avail_panin = [c for c in PANIN_VALIDATION_SCORE_COLS if c in epi.columns]

    fig_w = 170 * MM2IN
    fig_h = 58  * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = mgridspec.GridSpec(
        1, 2,
        width_ratios=[0.85, 2.0],
        left=0.07, right=0.98, top=0.90, bottom=0.16,
        wspace=0.28,
    )

    ax_scatter = fig.add_subplot(gs[0, 0])
    gs_right   = mgridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 1],
        hspace=0.55, wspace=0.28,
    )

    # ── Left: epi_q vs contextual_q ──────────────────────────────────────────
    for disease, sub in epi.groupby("disease_group", observed=True):
        c = DISEASE_PALETTE.get(disease, "#888")
        s = sub.sample(min(2000, len(sub)), random_state=0)
        ax_scatter.scatter(
            s["epithelial_pseudotime_q"], s["contextual_pseudotime_q"],
            s=0.5, alpha=0.20, c=c, linewidths=0, rasterized=True, zorder=2,
        )

    # Overall LOWESS
    mask = epi["epithelial_pseudotime_q"].notna() & epi["contextual_pseudotime_q"].notna()
    x_all = epi.loc[mask, "epithelial_pseudotime_q"].to_numpy()
    y_all = epi.loc[mask, "contextual_pseudotime_q"].to_numpy()
    lx, ly = _lowess(x_all, y_all, frac=0.50)
    if lx is not None:
        ax_scatter.plot(lx, ly, color="#333333", lw=1.2, zorder=4)

    # Mean reference
    ax_scatter.axhline(np.nanmean(y_all), color="#aaaaaa", lw=0.7, ls="--", zorder=1)

    # Spearman ρ
    rho, p = spearmanr(x_all, y_all)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax_scatter.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                    transform=ax_scatter.transAxes, fontsize=5.0,
                    ha="right", va="bottom", color="#333333")

    ax_scatter.set_xlabel("Epi. pseudotime (q)", fontsize=5.5, labelpad=1)
    ax_scatter.set_ylabel("Contextual pseudotime (q)", fontsize=5.5, labelpad=2)
    ax_scatter.set_title("Epi vs contextual\npseudotime", fontsize=5.5, pad=2)
    ax_scatter.tick_params(length=2, pad=1.5, labelsize=4.8)
    for sp in ["top", "right"]:
        ax_scatter.spines[sp].set_visible(False)

    handles_dis = [
        mpatches.Patch(facecolor=DISEASE_PALETTE.get(d, "#888"),
                       label=d.replace("NormalPancreas", "Normal"))
        for d in DISEASE_PALETTE if d in epi["disease_group"].unique()
    ]
    ax_scatter.legend(handles=handles_dis, fontsize=4.2, frameon=False,
                      loc="upper left", handlelength=0.8, handleheight=0.7,
                      borderpad=0.2, labelspacing=0.2)

    # ── Right: PanIN score trends — Supp Fig 1 style ──────────────────────────
    pt_col = "epithelial_pseudotime_q"
    diseases_in_data = epi["disease_group"].dropna().unique().tolist()
    disease_order    = [d for d in DISEASE_PALETTE if d in diseases_in_data]

    for idx, panin_col in enumerate(avail_panin[:4]):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs_right[r, c])

        # Mean reference
        all_vals = epi[panin_col].dropna()
        if len(all_vals):
            ax.axhline(all_vals.mean(), color="#aaaaaa", lw=0.7, ls="--", zorder=1)

        rho_texts = []
        for disease in disease_order:
            color = DISEASE_PALETTE.get(disease, "#888")
            sub   = epi.loc[epi["disease_group"] == disease, [pt_col, panin_col]].dropna()
            if len(sub) < 30:
                continue
            x = sub[pt_col].to_numpy()
            y = sub[panin_col].to_numpy()

            # Faint scatter (subsample, fixed seed for reproducibility)
            rng  = np.random.default_rng(idx * 100 + disease_order.index(disease))
            samp = rng.choice(len(x), size=min(500, len(x)), replace=False)
            ax.scatter(x[samp], y[samp], s=0.5, color=color, alpha=0.15,
                       linewidths=0, rasterized=True, zorder=2)

            # LOWESS
            lx, ly = _lowess(x, y)
            if lx is not None:
                ax.plot(lx, ly, color=color, lw=1.2, zorder=3)
                rho, p = spearmanr(x, y)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                short = disease.replace("NormalPancreas", "Nml").replace("PDAC", "PDAC")
                rho_texts.append((rho, sig, color, short))

        # Stack ρ annotations per disease
        for k, (rho, sig, color, short) in enumerate(rho_texts):
            ax.text(0.97, 0.04 + k * 0.13,
                    f"{short} ρ={rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=3.8,
                    ha="right", va="bottom", color=color)

        ax.set_title(PANIN_LABELS.get(panin_col, panin_col), fontsize=4.8, pad=2)
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(length=2, pad=1, labelsize=4.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        ax.set_xlabel("Epi. pseudotime (q)", fontsize=5.0, labelpad=1)
        ax.set_ylabel("Validation score", fontsize=4.8, labelpad=2)
        ax.tick_params(axis="y", labelsize=4.5)

    # Right-panel title
    fig.text(
        0.63, 0.94,
        "PanIN morphological validation scores vs epithelial pseudotime",
        ha="center", va="bottom", fontsize=5.5, fontweight="bold",
    )

    OUT_SUPPFIG3.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_SUPPFIG3 / "suppfig3B_panin_validation.pdf"
    out_png = OUT_SUPPFIG3 / "suppfig3B_panin_validation.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
