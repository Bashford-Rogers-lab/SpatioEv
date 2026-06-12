"""
Figure 2C — Spatial pseudotime maps, all 4 samples
====================================================
1 × 4 row: one panel per sample.
Ductal niche cells coloured by rank-normalised pooled pseudotime (viridis).
Non-ductal cells shown as faint gray background.
Scale bar on each panel (500 µm).

Width: 83 mm  Height: 54 mm  (auto-adjusts per sample aspect)

Run:
    python notebooks/fig2C_spatial_pseudotime.py

Output: notebooks/results/fig2/fig2C_spatial_pseudotime.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import rankdata

from fig2_shared_config import (
    CACHE_DIR, OUT_DIR, MM2IN, set_pub_rc,
    SAMPLE_ORDER, SAMPLE_LABELS, spatial_pkl,
)

set_pub_rc()

PIXEL_SIZE_UM = 0.325


def rank_norm(series: pd.Series) -> pd.Series:
    """Rank-normalise to [0, 1]; NaN-safe."""
    mask = series.notna()
    out  = series.copy().astype(float)
    out[mask] = rankdata(series[mask].values) / mask.sum()
    return out


def load_spatial(sample_id: str) -> pd.DataFrame:
    pkl_path = spatial_pkl(sample_id)
    with open(pkl_path, "rb") as f:
        sp = pickle.load(f)
    # Add quantile-normalised pseudotime if absent
    if "pooled_pseudotime_q" not in sp.columns:
        sp["pooled_pseudotime_q"] = rank_norm(sp["pooled_pseudotime"])
    return sp


def draw_sample(ax, sp: pd.DataFrame, sample_id: str, title: str,
                norm: Normalize, show_cbar: bool = False, cbar_ax=None):
    """Draw one spatial panel."""
    x_col, y_col = "x", "y"

    x_min, x_max = sp[x_col].min(), sp[x_col].max()
    y_min, y_max = sp[y_col].min(), sp[y_col].max()
    mx = (x_max - x_min) * 0.02
    my = (y_max - y_min) * 0.02

    # Background: non-niche cells
    bg = sp[~sp["has_pooled_niche"].astype(bool)]
    if len(bg) > 80_000:
        bg = bg.sample(80_000, random_state=1)
    ax.scatter(bg[x_col], bg[y_col], c="#cccccc", s=0.04,
               alpha=0.35, linewidths=0, rasterized=True, zorder=1)

    # Ductal niche cells coloured by pseudotime
    fg = sp[sp["has_pooled_niche"].astype(bool) & sp["pooled_pseudotime_q"].notna()]
    sc = ax.scatter(
        fg[x_col], fg[y_col],
        c=fg["pooled_pseudotime_q"].values,
        cmap="viridis", norm=norm,
        s=0.15, alpha=0.85, linewidths=0, rasterized=True, zorder=2,
    )

    # Scale bar (500 µm)
    scale_px = 500 / PIXEL_SIZE_UM
    sb_x0    = x_max - mx - scale_px
    sb_x1    = x_max - mx
    sb_y     = y_max - my
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y],
            color="black", lw=0.9, solid_capstyle="butt", zorder=10)
    ax.text((sb_x0 + sb_x1) / 2, sb_y - (y_max - y_min) * 0.02,
            "500 µm", ha="center", va="top", fontsize=4.0, color="black", zorder=10)

    ax.set_xlim(x_min - mx, x_max + mx)
    ax.set_ylim(y_max + my, y_min - my)   # inverted
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=5, pad=2)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)

    if show_cbar and cbar_ax is not None:
        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"),
                            cax=cbar_ax, orientation="vertical")
        cbar.set_label("Pseudotime\n(quantile)", fontsize=4.5, labelpad=2)
        cbar.ax.tick_params(labelsize=4.2, length=2, width=0.4)
        cbar.outline.set_linewidth(0.3)


def make_figure():
    n = len(SAMPLE_ORDER)

    fig_w = 83  * MM2IN
    fig_h = 54  * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = mgridspec.GridSpec(
        1, n + 1,
        width_ratios=[1] * n + [0.06],
        left=0.01, right=0.97, top=0.92, bottom=0.04,
        wspace=0.05,
    )

    norm = Normalize(vmin=0, vmax=1)

    for i, sample_id in enumerate(SAMPLE_ORDER):
        ax = fig.add_subplot(gs[0, i])
        try:
            sp = load_spatial(sample_id)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"No data\n{sample_id}",
                    ha="center", va="center", fontsize=5, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        title = SAMPLE_LABELS.get(sample_id, sample_id)
        is_last = (i == n - 1)
        cbar_ax = fig.add_subplot(gs[0, n]) if is_last else None
        draw_sample(ax, sp, sample_id, title, norm,
                    show_cbar=is_last, cbar_ax=cbar_ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2C_spatial_pseudotime.pdf"
    out_png = OUT_DIR / "fig2C_spatial_pseudotime.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
