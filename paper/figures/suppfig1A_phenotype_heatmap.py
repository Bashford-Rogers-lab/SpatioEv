"""
Supplementary Figure 1A — Cell phenotype annotation heatmap (Tier A)
=====================================================================
Rows   = marker channels (from adata.var_names)
Columns = Tier A cell type labels
Values = mean expression, z-scored per marker (row-wise)

Clustered by marker correlation; cell types ordered by hierarchical clustering.
Excludes 'noise' and 'Unknown' labels.

Run:
    python notebooks/suppfig1A_phenotype_heatmap.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig1A_phenotype_heatmap.pdf  (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/shihongwu/SpatioEv")
DATA_DIR   = ROOT / "data" / "exp_2"
RESULT_DIR = ROOT  / "paper" / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH      = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH = DATA_DIR / "34434_1_annotation.csv"

# ── Cell types to exclude ─────────────────────────────────────────────────────
EXCLUDE_LABELS = {"noise", "Unknown"}

# ── Marker exclusion / rename ─────────────────────────────────────────────────
# Markers to drop entirely from the panel
EXCLUDE_MARKERS = {
    "CXCL13", "CD11b", "COL3", "CD56", "MERTK",
    "CXCR5", "DNA_1", "FOXP3", "CD8", "CD4",
    "PDGFRa", "RS6",
}
# Rename: keep the key on the right, display the value on the left
RENAME_MARKERS = {
    "FOXP3_2": "FOXP3",
    "CD8_2":   "CD8",
    "CD4_2":   "CD4",
}

# ── Tier A display order (epithelium first, then immune, then stroma) ─────────
# Will fall back to clustering order if custom order doesn't match data
PREFERRED_ORDER = [
    "pancreatic ductal epithelium",
    "pancreatic acinar epithelium",
    "Islets",
    "Duodenum epithelial",
    "Necrotic tumor",
    "T cells",
    "B lineage",
    "Endothelial cells",
    "Fibroblasts",
    "Vascular smooth muscle",
    "Vimentin only mesenchyme",
    "Muscularis mucosa",
    "Muscularis externa",
    "Nerves",
]

# ── Tier A colour palette ─────────────────────────────────────────────────────
TIER_A_COLORS = {
    "pancreatic ductal epithelium":  "#1f78b4",
    "pancreatic acinar epithelium":  "#a6cee3",
    "Islets":                        "#b2df8a",
    "Duodenum epithelial":           "#33a02c",
    "Necrotic tumor":                "#fb9a99",
    "T cells":                       "#e31a1c",
    "B lineage":                     "#fdbf6f",
    "Endothelial cells":             "#ff7f00",
    "Fibroblasts":                   "#cab2d6",
    "Vascular smooth muscle":        "#6a3d9a",
    "Vimentin only mesenchyme":      "#b15928",
    "Muscularis mucosa":             "#8dd3c7",
    "Muscularis externa":            "#ffffb3",
    "Nerves":                        "#bebada",
}

# ── Publication rcParams ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "Arial",
    "font.size":       6,
    "axes.labelsize":  6,
    "axes.titlesize":  7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "pdf.fonttype":   42,
    "svg.fonttype":   "none",
})

MM2IN = 1 / 25.4


# ── Load data ─────────────────────────────────────────────────────────────────
def load_mean_expression() -> pd.DataFrame:
    """
    Returns DataFrame  [markers × Tier_A]  — mean expression per cell type.
    """
    import anndata as ad

    print("Loading adata …")
    adata = ad.read_h5ad(ADATA_PATH)
    ann   = pd.read_csv(ANNOTATION_PATH, index_col=0)

    # Attach Tier_A
    adata.obs["Tier_A"] = ann["Tier_A"].reindex(adata.obs_names)

    # Exclude noise / Unknown
    keep = ~adata.obs["Tier_A"].isin(EXCLUDE_LABELS) & adata.obs["Tier_A"].notna()
    adata_f = adata[keep]

    print(f"  {keep.sum():,} cells across {adata_f.obs['Tier_A'].nunique()} Tier_A labels")

    # Drop excluded markers; keep renamed ones
    keep_vars = [v for v in adata_f.var_names if v not in EXCLUDE_MARKERS]
    adata_f   = adata_f[:, keep_vars]
    print(f"  {adata_f.n_vars} markers after filtering: {list(adata_f.var_names)}")

    # Mean expression per Tier_A
    X = adata_f.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)

    means = {}
    for label in sorted(adata_f.obs["Tier_A"].unique()):
        idx = (adata_f.obs["Tier_A"] == label).values
        means[label] = X[idx].mean(axis=0)

    df = pd.DataFrame(means, index=adata_f.var_names)   # markers × cell_types
    df.index = [RENAME_MARKERS.get(m, m) for m in df.index]
    return df


# ── Z-score per marker (row-wise) ─────────────────────────────────────────────
def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mu  = df.mean(axis=1)
    std = df.std(axis=1).replace(0, 1)
    return df.sub(mu, axis=0).div(std, axis=0)


# ── Hierarchical clustering order ─────────────────────────────────────────────
def cluster_order(mat: np.ndarray, metric="correlation") -> list[int]:
    """Return row indices sorted by hierarchical clustering."""
    if mat.shape[0] < 2:
        return list(range(mat.shape[0]))
    D  = pdist(mat, metric=metric)
    Z  = linkage(D, method="average")
    return list(leaves_list(Z))


# ── Main figure ───────────────────────────────────────────────────────────────
def make_figure():
    df_mean = load_mean_expression()
    df_z    = zscore_rows(df_mean)

    # ── Column order: preferred → then fill with whatever else is present ─────
    all_labels = list(df_z.columns)
    ordered_cols = [c for c in PREFERRED_ORDER if c in all_labels]
    remaining    = [c for c in all_labels if c not in ordered_cols]
    if remaining:
        # Cluster the remaining ones among themselves
        sub = df_z[remaining].values.T
        idx = cluster_order(sub)
        remaining = [remaining[i] for i in idx]
    final_cols = ordered_cols + remaining
    df_z = df_z[final_cols]

    # ── Row order: cluster markers ────────────────────────────────────────────
    row_idx = cluster_order(df_z.values)
    df_z    = df_z.iloc[row_idx]

    n_markers   = len(df_z)
    n_celltypes = len(df_z.columns)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # [color_strip | main_heatmap | colorbar]
    fig_w = min(120, max(80, n_celltypes * 6 + 12)) * MM2IN
    fig_h = max(50,  n_markers * 5.5 + 18) * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.04, 1.0],
        width_ratios=[1.0, 0.03, 0.03],
        left=0.14, right=0.93,
        top=0.88, bottom=0.22,
        hspace=0.02, wspace=0.04,
    )

    ax_strip = fig.add_subplot(gs[0, 0])   # Tier A colour strip (top)
    ax_hm    = fig.add_subplot(gs[1, 0])   # heatmap body
    ax_cbar  = fig.add_subplot(gs[1, 2])   # colorbar

    mat  = df_z.values
    vmax = min(3.0, float(np.nanpercentile(np.abs(mat), 99)))

    # ── Heatmap ───────────────────────────────────────────────────────────────
    im = ax_hm.imshow(
        mat, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )

    # Row labels (markers)
    ax_hm.set_yticks(range(n_markers))
    ax_hm.set_yticklabels(df_z.index, fontsize=6)
    ax_hm.yaxis.set_tick_params(length=0, pad=3)

    # Col labels (cell types)
    ax_hm.set_xticks(range(n_celltypes))
    ax_hm.set_xticklabels(df_z.columns, rotation=40, ha="right",
                           fontsize=6, va="top")
    ax_hm.xaxis.set_tick_params(length=0, pad=2)

    # Grid lines
    for i in range(n_markers + 1):
        ax_hm.axhline(i - 0.5, color="white", lw=0.35, zorder=3)
    for j in range(n_celltypes + 1):
        ax_hm.axvline(j - 0.5, color="white", lw=0.35, zorder=3)

    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    # ── Colour strip (Tier A category bar) ────────────────────────────────────
    strip_colors = [TIER_A_COLORS.get(c, "#aaaaaa") for c in df_z.columns]
    strip_mat    = np.array([[mcolors.to_rgba(c) for c in strip_colors]])
    ax_strip.imshow(strip_mat, aspect="auto")
    ax_strip.set_xlim(-0.5, n_celltypes - 0.5)
    ax_strip.axis("off")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="vertical")
    cbar.set_label("z-score", fontsize=6, labelpad=3)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)
    ticks = [-2, -1, 0, 1, 2]
    cbar.set_ticks([t for t in ticks if -vmax <= t <= vmax])

    # ── Legend (Tier A colour → label) ───────────────────────────────────────
    handles = [
        mpatches.Patch(
            facecolor=TIER_A_COLORS.get(c, "#aaaaaa"),
            label=c, linewidth=0
        )
        for c in df_z.columns
    ]
    fig.legend(
        handles=handles,
        fontsize=5, ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handlelength=1.0, handleheight=0.9,
        borderpad=0.3, labelspacing=0.25, columnspacing=0.8,
    )

    fig.text(0.01, 0.98, "A", fontsize=9, fontweight="bold", va="top", ha="left",
             transform=fig.transFigure)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1A_phenotype_heatmap.pdf"
    out_png = RESULT_DIR / "suppfig1A_phenotype_heatmap.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
