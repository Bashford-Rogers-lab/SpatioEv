"""
Figure 2E — Branch structure: UMAP by branch + pseudotime
==========================================================
Two-panel row:
  Left  : UMAP coloured by major branch, with Prim's MST tree skeleton.
  Right : UMAP coloured by quantile-normalised pseudotime (viridis).

Width: ~113 mm  Height: ~35 mm  (2/3 of original 170×52 mm)

Run:
    python notebooks/fig2E_branch_structure.py

Output: paper/notebooks/results/fig2/fig2E_branch_structure.pdf (.png)
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
    CACHE_DIR, OUT_DIR, MM2IN, set_pub_rc,
    DISEASE_PALETTE, make_branch_palette, assign_branch_bio_names,
    MODULE_COLS,
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
    for (xy0, xy1) in tree_edges:
        ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]],
                color="#333333", lw=0.8, alpha=0.65,
                solid_capstyle="round", zorder=3)
    ax.scatter(node_umap["UMAP1"], node_umap["UMAP2"],
               color="white", s=14, edgecolors="#333333",
               linewidths=0.6, zorder=4)


def style_umap_ax(ax, title=""):
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=5.5, pad=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        df = pickle.load(f)
    return df


def make_figure():
    df = load_data()

    # ── Branch ordering: trunk first, then by niche count ────────────────────
    bc = df["major_branch"].value_counts()
    excl = {"unassigned"}
    branch_order = (
        (["trunk"] if "trunk" in bc.index else []) +
        [b for b in bc.index if b not in excl | {"trunk"}]
    )[:8]

    branch_palette = make_branch_palette(branch_order)
    avail_mod = [c for c in MODULE_COLS if c in df.columns]
    bio_names = assign_branch_bio_names(df, avail_mod)

    # ── Compute MST ───────────────────────────────────────────────────────────
    node_col = None
    for c in ("simple_node_id", "elpigraph_node_id"):
        if c in df.columns:
            node_col = c
            break

    tree_edges, node_umap = [], pd.DataFrame()
    if node_col:
        tree_edges, node_umap = compute_tree_edges(
            df[df["major_branch"].isin(branch_order)], node_col
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_w = round(170 * 2 / 3) * MM2IN   # ~113 mm
    fig_h = round(52  * 2 / 3) * MM2IN   # ~35 mm

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs  = mgridspec.GridSpec(
        1, 2, width_ratios=[1, 1],
        left=0.02, right=0.99, top=0.90, bottom=0.14,
        wspace=0.10,
    )
    ax_umap = fig.add_subplot(gs[0, 0])
    ax_pt   = fig.add_subplot(gs[0, 1])

    # ── UMAP coloured by branch ───────────────────────────────────────────────
    # Background: unassigned in light gray
    ua = df[df["major_branch"] == "unassigned"]
    if len(ua):
        samp = ua.sample(min(5000, len(ua)), random_state=1)
        ax_umap.scatter(samp["UMAP1"], samp["UMAP2"], c="#dddddd",
                        s=0.4, linewidths=0, rasterized=True, zorder=1, alpha=0.5)

    for branch in reversed(branch_order):
        sub = df[df["major_branch"] == branch]
        if not len(sub):
            continue
        color = branch_palette.get(branch, "#888")
        samp  = sub.sample(min(4000, len(sub)), random_state=42)
        ax_umap.scatter(samp["UMAP1"], samp["UMAP2"],
                        c=color, s=0.5, alpha=0.65, linewidths=0,
                        rasterized=True, zorder=2)

    # MST skeleton over all branches
    if len(tree_edges):
        draw_mst(ax_umap, tree_edges, node_umap)

    style_umap_ax(ax_umap, title="Pooled UMAP — major branches")

    # Legend below axes
    legend_handles = [
        mpatches.Patch(facecolor=branch_palette.get(b, "#888"),
                       label=f"{bio_names.get(b, b)}  ({b})", linewidth=0)
        for b in branch_order
    ]
    ax_umap.legend(
        handles=legend_handles, fontsize=4.0, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.03),
        frameon=False, handlelength=0.8, handleheight=0.7,
        borderpad=0.2, labelspacing=0.25, columnspacing=1.0,
    )

    # ── UMAP coloured by pseudotime ───────────────────────────────────────────
    # Use pooled_pseudotime_q (rank-quantile, 0–1) and viridis — same as fig2C
    pt_col_q = "pooled_pseudotime_q"
    df_pt = df[df[pt_col_q].notna()].copy() if pt_col_q in df.columns else df[df["pooled_pseudotime"].notna()].copy()
    if pt_col_q not in df.columns:
        pt_valid = df["pooled_pseudotime"].dropna()
        pt_lo = np.percentile(pt_valid, 2)
        pt_hi = np.percentile(pt_valid, 98)
        df_pt["pt_scaled"] = ((df_pt["pooled_pseudotime"] - pt_lo) / max(pt_hi - pt_lo, 1e-9)).clip(0, 1)
        pt_col_q = "pt_scaled"

    # Gray background for unassigned
    if len(ua):
        samp = ua.sample(min(5000, len(ua)), random_state=1)
        ax_pt.scatter(samp["UMAP1"], samp["UMAP2"], c="#dddddd",
                      s=0.4, linewidths=0, rasterized=True, zorder=1, alpha=0.5)

    # Plot assigned cells colored by pseudotime quantile — viridis matches fig2C
    plot_sub = df_pt[df_pt["major_branch"].isin(set(branch_order) - {"unassigned"})]
    plot_sub = plot_sub.sample(min(12000, len(plot_sub)), random_state=7)
    ax_pt.scatter(
        plot_sub["UMAP1"], plot_sub["UMAP2"],
        c=plot_sub[pt_col_q], cmap="viridis",
        s=0.5, alpha=0.75, linewidths=0, rasterized=True, zorder=2,
        vmin=0, vmax=1,
    )

    # MST skeleton
    if len(tree_edges):
        draw_mst(ax_pt, tree_edges, node_umap)

    style_umap_ax(ax_pt, title="Pooled UMAP — pseudotime")

    # Colorbar below — viridis, matches fig2C
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cax = ax_pt.inset_axes([0.05, -0.04, 0.90, 0.025])
    cb  = fig.colorbar(
        cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap="viridis"),
        cax=cax, orientation="horizontal",
    )
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Early", "Mid", "Late"], fontsize=4.2)
    cb.ax.tick_params(length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2E_branch_structure.pdf"
    out_png = OUT_DIR / "fig2E_branch_structure.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
