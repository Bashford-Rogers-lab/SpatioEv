"""
Figure 1F — PCA scree + UMAP (branch) + UMAP (pseudotime)
==========================================================
Loads pre-computed CSVs from the pseudotime notebook — no heavy recomputation.

Panels (left → right):
  F1  Scree plot — explained variance per PC
  F2  UMAP coloured by principal-tree branch
  F3  UMAP coloured by quantile-normalised pseudotime

Layout: 183 mm wide (Nature double column), ~65 mm tall, Arial 7 pt.

Run:
    python notebooks/fig1F_pca_umap.py

Output: paper/notebooks/results/pseudotime_exp2/fig1F_pca_umap.pdf  (and .png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
RESULT_DIR = ROOT / "notebooks" / "results" / "pseudotime_exp2"

PCA_CSV = RESULT_DIR / "pca_explained_variance.csv"
PT_CSV  = RESULT_DIR / "niche_pseudotime_results.csv"

# ── Publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "Arial",
    "font.size":       6,
    "axes.labelsize":  6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "axes.linewidth":  0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "pdf.fonttype":    42,
    "svg.fonttype":    "none",
})

# ── Branch colour palette (11 categories) ─────────────────────────────────────
# Ordered: trunk first, then branches by label, other/unassigned last
BRANCH_COLORS = {
    "trunk":    "#2166ac",   # deep blue  — the root / most organised
    "branch 1": "#4dac26",   # green
    "branch 2": "#d7191c",   # red
    "branch 3": "#f46d43",   # orange
    "branch 4": "#fdae61",   # amber
    "branch 5": "#abd9e9",   # pale blue
    "branch 6": "#74add1",   # mid blue
    "branch 7": "#9970ab",   # purple
    "branch 8": "#a6761d",   # brown
    "branch 9": "#e7298a",   # pink
    "other":    "#bdbdbd",   # grey
}

# Branch display order for the legend (trunk first, other last)
BRANCH_ORDER = ["trunk"] + [f"branch {i}" for i in range(1, 10)] + ["other"]

# ── Pseudotime colourmap ───────────────────────────────────────────────────────
PT_CMAP = "plasma"


# ── Tree-edge overlay (MST over per-node mean UMAP positions) ─────────────────
def _prims_mst(dist: np.ndarray) -> list[tuple[int, int]]:
    """Pure-numpy Prim's minimum spanning tree. Returns list of (i, j) pairs."""
    n = len(dist)
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    edges = []
    for _ in range(n - 1):
        d = dist[np.ix_(np.where(in_mst)[0], np.where(~in_mst)[0])]
        i_in  = np.where(in_mst)[0]
        i_out = np.where(~in_mst)[0]
        r, c  = np.unravel_index(d.argmin(), d.shape)
        edges.append((i_in[r], i_out[c]))
        in_mst[i_out[c]] = True
    return edges


def compute_tree_edges(df: pd.DataFrame) -> tuple:
    """
    Reconstruct approximate tree edges via MST over per-node mean UMAP positions.
    No need to re-run ElPiGraph.
    """
    node_umap = (
        df[["elpigraph_node_id", "UMAP1", "UMAP2"]]
        .dropna()
        .groupby("elpigraph_node_id")[["UMAP1", "UMAP2"]]
        .mean()
    )
    if len(node_umap) < 2:
        return [], node_umap
    xy   = node_umap.to_numpy()
    diff = xy[:, None, :] - xy[None, :, :]     # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(-1))         # (n, n)

    mst_pairs = _prims_mst(dist)
    edges = [(xy[i], xy[j]) for i, j in mst_pairs]
    return edges, node_umap


# ── Panel 1: Scree plot ───────────────────────────────────────────────────────
def plot_scree(ax: plt.Axes, pca: pd.DataFrame):
    n = len(pca)
    x = np.arange(1, n + 1)
    ev = pca["explained_variance_ratio"].to_numpy()
    cv = pca["cumulative_variance_ratio"].to_numpy()

    bar_color  = "#4393c3"
    line_color = "#e06c00"

    # Bars
    ax.bar(x, ev * 100, color=bar_color, width=0.65, zorder=2, linewidth=0)

    # Elbow line
    ax.plot(x, ev * 100, "o-", color=line_color, ms=3, lw=0.9, zorder=3)

    # 80 / 90 % cumulative thresholds as subtle horizontal references
    # (shown on a secondary twin axis for cumulative curve)
    ax2 = ax.twinx()
    ax2.plot(x, cv * 100, "s--", color="#888888", ms=2, lw=0.7, zorder=1, alpha=0.7)
    for thresh, label in [(80, "80 %"), (90, "90 %")]:
        idx = int(np.searchsorted(cv * 100, thresh))
        ax2.axhline(thresh, ls=":", lw=0.6, color="#aaaaaa")
        ax2.text(n + 0.15, thresh + 0.5, label, fontsize=4.5, color="#888888", va="bottom")
    ax2.set_ylim(0, 105)
    ax2.set_yticks([50, 80, 90, 100])
    ax2.set_ylabel("Cumulative variance (%)", fontsize=5.5, color="#888888", labelpad=2)
    ax2.tick_params(axis="y", labelsize=5, colors="#888888", length=2, width=0.5)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.4)
    ax2.spines["right"].set_color("#bbbbbb")

    ax.set_xlabel("Principal component", labelpad=2)
    ax.set_ylabel("Variance explained (%)", labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x], rotation=45, ha="right", fontsize=5)
    ax.set_ylim(0, max(ev * 100) * 1.18)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5)


# ── Panel 2: UMAP — branch ────────────────────────────────────────────────────
def plot_umap_branch(ax: plt.Axes, df: pd.DataFrame, tree_edges, node_umap):
    # Background: 'other' first so coloured branches are on top
    order = BRANCH_ORDER[::-1]  # draw 'other' first
    for branch in order:
        sub = df[df["principal_tree_branch"] == branch]
        if sub.empty:
            continue
        col = BRANCH_COLORS.get(branch, "#bdbdbd")
        alpha = 0.35 if branch == "other" else 0.75
        s     = 3.0  if branch == "other" else 5.5
        ax.scatter(
            sub["UMAP1"], sub["UMAP2"],
            color=col, s=s, alpha=alpha, linewidths=0,
            rasterized=True, zorder=2 if branch != "other" else 1,
        )

    # Tree edges
    for (xy0, xy1) in tree_edges:
        ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]],
                color="#333333", lw=0.8, alpha=0.65, solid_capstyle="round",
                zorder=3)

    # Node dots
    ax.scatter(
        node_umap["UMAP1"], node_umap["UMAP2"],
        color="white", s=14, edgecolors="#333333", linewidths=0.6,
        zorder=4,
    )

    # Legend — below the axes, 2 columns, so it doesn't intrude on the right panel
    patches = [
        mpatches.Patch(color=BRANCH_COLORS.get(b, "#bdbdbd"), label=b.capitalize())
        for b in BRANCH_ORDER
        if b in df["principal_tree_branch"].values
    ]
    ax.legend(
        handles=patches, fontsize=5, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        handlelength=1.0, handleheight=0.9,
        borderpad=0.3, labelspacing=0.25, columnspacing=0.8,
    )

    ax.set_xlabel("UMAP 1", labelpad=2)
    ax.set_ylabel("UMAP 2", labelpad=2)
    _style_umap_ax(ax)


# ── Panel 3: UMAP — pseudotime ────────────────────────────────────────────────
def plot_umap_pseudotime(ax: plt.Axes, df: pd.DataFrame, tree_edges, node_umap, fig: plt.Figure):
    # Grey 'other' / unassigned first
    na = df[df["elpigraph_pseudotime_q"].isna()]
    if not na.empty:
        ax.scatter(na["UMAP1"], na["UMAP2"],
                   color="#d3d3d3", s=3.0, alpha=0.4, linewidths=0,
                   rasterized=True, zorder=1)

    valid = df[df["elpigraph_pseudotime_q"].notna()]
    # Sort by pseudotime so early (dark) points don't overdraw late (bright)
    valid = valid.sort_values("elpigraph_pseudotime_q")
    sc = ax.scatter(
        valid["UMAP1"], valid["UMAP2"],
        c=valid["elpigraph_pseudotime_q"],
        cmap=PT_CMAP, vmin=0, vmax=1,
        s=5.5, alpha=0.85, linewidths=0,
        rasterized=True, zorder=2,
    )

    # Tree edges
    for (xy0, xy1) in tree_edges:
        ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]],
                color="#eeeeee", lw=0.8, alpha=0.8, solid_capstyle="round",
                zorder=3)

    # Node dots
    ax.scatter(
        node_umap["UMAP1"], node_umap["UMAP2"],
        color="white", s=14, edgecolors="#cccccc", linewidths=0.6,
        zorder=4,
    )

    # Colorbar — vertical, to the right of the axes (no height distortion)
    cbar = fig.colorbar(
        sc, ax=ax, orientation="vertical",
        pad=0.03, fraction=0.045, aspect=20,
    )
    cbar.set_label("Pseudotime", fontsize=5.5, labelpad=2)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)

    ax.set_xlabel("UMAP 1", labelpad=2)
    ax.set_ylabel("UMAP 2", labelpad=2)
    _style_umap_ax(ax)


def _style_umap_ax(ax: plt.Axes):
    ax.set_aspect("equal")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, width=0.5)
    # Remove tick labels — UMAP coordinates are arbitrary
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)


# ── Main ─────────────────────────────────────────────────────────────────────
def make_figure():
    pca = pd.read_csv(PCA_CSV)
    df  = pd.read_csv(PT_CSV)

    print(f"  Loaded {len(df):,} niches, {len(pca)} PCs")
    print(f"  Branch counts:\n{df['principal_tree_branch'].value_counts()}")

    tree_edges, node_umap = compute_tree_edges(df)
    print(f"  Tree: {len(node_umap)} nodes, {len(tree_edges)} MST edges")

    # ── Figure layout ─────────────────────────────────────────────────────────
    mm2in  = 1 / 25.4
    fig_w  = 134 * mm2in
    fig_h  = 47  * mm2in

    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.0, 1.3, 1.3], "wspace": 0.38},
    )
    fig.patch.set_facecolor("white")

    plot_scree(axes[0], pca)
    plot_umap_branch(axes[1], df, tree_edges, node_umap)
    plot_umap_pseudotime(axes[2], df, tree_edges, node_umap, fig)

    fig.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.24)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_pdf = RESULT_DIR / "fig1F_pca_umap.pdf"
    out_png = RESULT_DIR / "fig1F_pca_umap.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
