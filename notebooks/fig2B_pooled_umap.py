"""
Figure 2B — Pooled UMAP diagnostics (disease group + early-duct anchor score)
==============================================================================
Two-panel UMAP:
  Left:  Points coloured by disease group (NormalPancreas vs PDAC)
  Right: Points coloured by pdac_early_duct_anchor_score (continuous, RdBu_r)

Both panels: MST tree skeleton (Prim's over node UMAP means), white node dots,
equal aspect, no tick labels — matches Figure 1F style.

Width: 83 mm  Height: 54 mm

Run:
    python notebooks/fig2B_pooled_umap.py

Output: notebooks/results/fig2/fig2B_pooled_umap.pdf (.png)
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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from fig2_shared_config import (
    CACHE_DIR, OUT_DIR, MM2IN, set_pub_rc,
    DISEASE_PALETTE,
)

set_pub_rc()


# ── MST tree skeleton helpers (matches fig1F) ─────────────────────────────────

def _prims_mst(dist):
    n = len(dist)
    in_mst = np.zeros(n, dtype=bool); in_mst[0] = True
    edges = []
    for _ in range(n - 1):
        d = dist[np.ix_(np.where(in_mst)[0], np.where(~in_mst)[0])]
        i_in, i_out = np.where(in_mst)[0], np.where(~in_mst)[0]
        r, c = np.unravel_index(d.argmin(), d.shape)
        edges.append((i_in[r], i_out[c]))
        in_mst[i_out[c]] = True
    return edges


def compute_tree_edges(df, node_col):
    """Prim's MST over per-node mean UMAP positions."""
    valid = df[[node_col, "UMAP1", "UMAP2"]].dropna()
    node_umap = valid.groupby(node_col, observed=True)[["UMAP1", "UMAP2"]].mean()
    if len(node_umap) < 2:
        return [], node_umap
    xy = node_umap.to_numpy()
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    pairs = _prims_mst(dist)
    tree_edges = [(xy[i], xy[j]) for i, j in pairs]
    return tree_edges, node_umap


def draw_mst(ax, tree_edges, node_umap):
    """Draw MST skeleton + white node dots (fig1F style)."""
    for (xy0, xy1) in tree_edges:
        ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]],
                color="#333333", lw=0.8, alpha=0.65,
                solid_capstyle="round", zorder=3)
    ax.scatter(node_umap["UMAP1"], node_umap["UMAP2"],
               color="white", s=14, edgecolors="#333333",
               linewidths=0.6, zorder=4)


def style_umap_ax(ax):
    """Remove tick labels/marks, equal aspect, clean spines."""
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        df = pickle.load(f)
    return df


def make_figure():
    df = load_data()

    # Determine node column
    node_col = None
    for c in ("simple_node_id", "elpigraph_node_id"):
        if c in df.columns:
            node_col = c
            break

    tree_edges, node_umap = [], pd.DataFrame()
    if node_col:
        tree_edges, node_umap = compute_tree_edges(df, node_col)

    fig_w = 83  * MM2IN
    fig_h = 54  * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = mgridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.07],
        left=0.03, right=0.97, top=0.88, bottom=0.18,
        wspace=0.10,
    )
    ax_dis  = fig.add_subplot(gs[0, 0])
    ax_cont = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # Shuffle for even overlap
    df_s = df.sample(frac=1, random_state=42)

    # ── Left: disease group ───────────────────────────────────────────────────
    for disease, sub in df_s.groupby("disease_group", observed=True):
        c = DISEASE_PALETTE.get(disease, "#888")
        ax_dis.scatter(sub["UMAP1"], sub["UMAP2"], c=c,
                       s=0.4, alpha=0.55, linewidths=0, rasterized=True, zorder=2)

    if len(tree_edges):
        draw_mst(ax_dis, tree_edges, node_umap)

    style_umap_ax(ax_dis)
    ax_dis.set_title("Disease group", fontsize=5.5, pad=2)

    handles = [mpatches.Patch(facecolor=DISEASE_PALETTE[d],
                               label=d.replace("NormalPancreas", "Normal"))
               for d in DISEASE_PALETTE if d in df["disease_group"].unique()]
    ax_dis.legend(handles=handles, fontsize=4.2, ncol=1,
                  loc="upper center", bbox_to_anchor=(0.5, -0.04),
                  frameon=False, handlelength=0.8, handleheight=0.7,
                  borderpad=0.2, labelspacing=0.2)

    # ── Right: early-duct anchor score (continuous) ───────────────────────────
    score_col = "pdac_early_duct_anchor_score"
    vals = df_s[score_col].values
    vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)

    ax_cont.scatter(
        df_s["UMAP1"], df_s["UMAP2"],
        c=vals, cmap="RdBu_r", norm=norm,
        s=0.4, alpha=0.7, linewidths=0, rasterized=True, zorder=2,
    )

    if len(tree_edges):
        draw_mst(ax_cont, tree_edges, node_umap)

    style_umap_ax(ax_cont)
    ax_cont.spines["left"].set_visible(False)
    ax_cont.set_title("Early-duct anchor score", fontsize=5.5, pad=2)

    # Colorbar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"),
                        cax=ax_cbar, orientation="vertical")
    cbar.set_label("Score", fontsize=4.5, labelpad=2)
    cbar.ax.tick_params(labelsize=4.2, length=2, width=0.4)
    cbar.outline.set_linewidth(0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2B_pooled_umap.pdf"
    out_png = OUT_DIR / "fig2B_pooled_umap.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
