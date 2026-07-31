"""
Supplementary Figure 1 — Spatial cell-type scatter (Tier A)
============================================================
All cells plotted in whole-slide pixel space (X_centroid / Y_centroid),
coloured by Tier A cell-type label using the same palette as the
marker heatmap (suppfig1A).

Run:
    python notebooks/suppfig1_spatial_celltype.py

Output: paper/notebooks/results/pseudotime_exp2/suppfig1_spatial_celltype.pdf (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import anndata as ad

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path("/Users/shihongwu/SpatioEv")
DATA_DIR        = ROOT / "data" / "exp_2"
RESULT_DIR      = ROOT / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH      = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH = DATA_DIR / "34434_1_annotation.csv"

PIXEL_SIZE_UM   = 0.325          # µm per pixel

EXCLUDE_LABELS  = {"noise", "Unknown", "unknown"}

# ── Same palette as suppfig1A ─────────────────────────────────────────────────
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

# ── rcParams ──────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "Arial",
    "font.size":       6,
    "axes.labelsize":  6,
    "axes.titlesize":  6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "pdf.fonttype":    42,
    "svg.fonttype":    "none",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
})

MM2IN = 1 / 25.4


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading adata …")
    adata = ad.read_h5ad(ADATA_PATH)
    ann   = pd.read_csv(ANNOTATION_PATH, index_col=0)

    adata.obs["Tier_A"] = ann["Tier_A"].reindex(adata.obs_names)

    keep = ~adata.obs["Tier_A"].isin(EXCLUDE_LABELS) & adata.obs["Tier_A"].notna()
    obs  = adata.obs[keep][["X_centroid", "Y_centroid", "Tier_A"]].copy()
    obs["Tier_A"] = obs["Tier_A"].astype(str)
    print(f"  {len(obs):,} cells across {obs['Tier_A'].nunique()} Tier_A labels")
    return obs


# ── Draw ──────────────────────────────────────────────────────────────────────
def make_figure():
    obs = load_data()

    # Figure dimensions: auto-height from tissue aspect ratio, width ~80 mm
    x_range = obs["X_centroid"].max() - obs["X_centroid"].min()
    y_range = obs["Y_centroid"].max() - obs["Y_centroid"].min()
    aspect  = y_range / x_range          # typically ~1 or slightly >1

    target_w_mm = 80.0
    fig_w = target_w_mm * MM2IN
    # reserve ~12 mm bottom for legend; scale scatter area to aspect
    legend_h_mm = 14.0
    scatter_h_mm = target_w_mm * aspect
    fig_h = (scatter_h_mm + legend_h_mm) * MM2IN
    # cap total height at 68 mm to match 1E
    fig_h = min(fig_h, 68 * MM2IN)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    x_min = obs["X_centroid"].min()
    x_max = obs["X_centroid"].max()
    y_min = obs["Y_centroid"].min()
    y_max = obs["Y_centroid"].max()

    # Draw each cell type separately (ordered so rare types paint last)
    ct_counts = obs["Tier_A"].value_counts()
    ct_order  = ct_counts.index.tolist()  # most → least common

    rng = np.random.default_rng(42)

    for ct in reversed(ct_order):               # rare on top
        sub = obs[obs["Tier_A"] == ct]
        color = TIER_A_COLORS.get(ct, "#aaaaaa")

        # Downsample dense types
        max_pts = 30_000
        if len(sub) > max_pts:
            idx = rng.choice(len(sub), size=max_pts, replace=False)
            sub = sub.iloc[idx]

        ax.scatter(
            sub["X_centroid"].values,
            sub["Y_centroid"].values,
            c=color,
            s=0.10,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    ax.invert_yaxis()          # image convention: y=0 at top
    ax.set_aspect("equal")

    margin_x = (x_max - x_min) * 0.02
    margin_y = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_max + margin_y, y_min - margin_y)   # inverted limits

    ax.set_xticks([])
    ax.set_yticks([])

    # ── Scale bar (500 µm) ────────────────────────────────────────────────────
    scale_um  = 500
    scale_px  = scale_um / PIXEL_SIZE_UM
    sb_x0     = x_max - margin_x - scale_px
    sb_x1     = x_max - margin_x
    sb_y      = y_max - margin_y             # near bottom (note: y is inverted)
    ax.plot([sb_x0, sb_x1], [sb_y, sb_y],
            color="black", lw=1.2, solid_capstyle="butt", zorder=10,
            transform=ax.transData)
    ax.text((sb_x0 + sb_x1) / 2, sb_y - (y_max - y_min) * 0.015,
            "500 µm", ha="center", va="top", fontsize=5,
            color="black", zorder=10)

    # ── Legend (same order as TIER_A_COLORS, only present labels) ─────────────
    present = [ct for ct in TIER_A_COLORS if ct in obs["Tier_A"].unique()]
    handles = [
        mpatches.Patch(facecolor=TIER_A_COLORS[ct], edgecolor="none",
                       label=ct)
        for ct in present
    ]

    fig.legend(
        handles=handles,
        fontsize=4.5,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        handlelength=0.9,
        handleheight=0.8,
        borderpad=0.2,
        labelspacing=0.25,
        columnspacing=0.8,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.22)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig1_spatial_celltype.pdf"
    out_png = RESULT_DIR / "suppfig1_spatial_celltype.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
