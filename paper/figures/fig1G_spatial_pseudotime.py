"""
Figure 1G — Spatial pseudotime map
====================================
Plots all cells on the whole-slide coordinate system.
Non-ductal cells: subtle grey background  (identical style to Figure 1D).
Ductal niche cells: coloured by quantile-normalised pseudotime (plasma, 0 → 1)
  — same colourmap as Figure 1F right UMAP.

Run:
    python notebooks/fig1G_spatial_pseudotime.py

Output: paper/notebooks/results/pseudotime_exp2/fig1G_spatial_pseudotime.pdf  (and .png)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path("/Users/shihongwu/SpatioEv")
DATA_DIR    = ROOT / "data" / "exp_2"
RESULT_DIR  = ROOT  / "paper" / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH      = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH = DATA_DIR / "34434_1_annotation.csv"
SEG_DIR         = DATA_DIR / "segmentation"
PT_CSV          = RESULT_DIR / "niche_pseudotime_results.csv"

DUCT_NICHE_KEY = "pancreatic ductal epithelium_mask_component"
DUCT_LABEL     = "pancreatic ductal epithelium"
PIXEL_SIZE_UM  = 0.325

PT_CMAP = "plasma"    # matches Figure 1F right panel

# ── Publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":   "Arial",
    "font.size":      6,
    "axes.labelsize": 6,
    "pdf.fonttype":  42,
    "svg.fonttype":  "none",
})


# ── Load adata and attach pseudotime ──────────────────────────────────────────
def load_data():
    import anndata as ad
    import spatioev as sv

    print("Loading adata …")
    adata = ad.read_h5ad(ADATA_PATH)
    annotations = pd.read_csv(ANNOTATION_PATH, index_col=0)
    for col in ["Tier_A", "Tier_B"]:
        if col in annotations.columns:
            adata.obs[col] = annotations[col].reindex(adata.obs_names)

    print("Computing ductal niche components …")
    adata = sv.cluster_spatial_components_from_mask(
        adata,
        seg_dir=SEG_DIR,
        label_key="Tier_A",
        label_value=DUCT_LABEL,
        fov_key="fov",
        cell_label_key="label",
        connection_mode="label_adjacency",
        gap_tolerance=5,
        stitch_across_fovs=True,
        fov_grid_cols=2,
        stitch_gap_tolerance=5,
        connectivity=2,
        min_component_size=3,
        assign_singletons=True,
    )

    # Join per-niche quantile pseudotime onto per-cell obs
    pt = pd.read_csv(PT_CSV)[[DUCT_NICHE_KEY, "elpigraph_pseudotime_q"]].set_index(DUCT_NICHE_KEY)
    adata.obs["pseudotime_q"] = (
        adata.obs[DUCT_NICHE_KEY]
        .map(pt["elpigraph_pseudotime_q"])
        .astype(float)
    )

    n_mapped = adata.obs["pseudotime_q"].notna().sum()
    print(f"  Pseudotime mapped to {n_mapped:,} cells")
    return adata


# ── Build figure ─────────────────────────────────────────────────────────────
def make_figure(adata):
    obs = adata.obs.copy()

    # Split: non-ductal background | ductal with pseudotime
    non_duct = obs[obs["Tier_A"] != DUCT_LABEL]
    duct_pt  = obs[obs["pseudotime_q"].notna()].copy()

    print(f"  Non-ductal: {len(non_duct):,}  |  Ductal with pseudotime: {len(duct_pt):,}")

    # ── Figure sizing: 89 mm (single Nature column), auto-height ─────────────
    x_range = obs["X_centroid"].max() - obs["X_centroid"].min()
    y_range = obs["Y_centroid"].max() - obs["Y_centroid"].min()
    fig_w   = 46 / 25.4
    fig_h   = fig_w * (y_range / x_range) + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Background: non-ductal cells ─────────────────────────────────────────
    # Same grey RGBA as Figure 1D — explicit tuple avoids matplotlib string parsing
    gray_rgba = (0.78, 0.78, 0.78, 0.4)
    bg = non_duct.sample(frac=0.4, random_state=1) if len(non_duct) > 300_000 else non_duct
    ax.scatter(
        bg["X_centroid"].values,
        bg["Y_centroid"].values,
        color=gray_rgba,
        s=0.07,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # ── Ductal niche cells: plasma pseudotime ─────────────────────────────────
    # Sort so early (dark) points don't overdraw late (bright)
    duct_pt = duct_pt.sort_values("pseudotime_q")
    sc = ax.scatter(
        duct_pt["X_centroid"].values,
        duct_pt["Y_centroid"].values,
        c=duct_pt["pseudotime_q"].values,
        cmap=PT_CMAP,
        vmin=0, vmax=1,
        s=0.3,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # ── Colorbar ─────────────────────────────────────────────────────────────
    # Compact horizontal bar at the lower-left, matching figure style
    cbar_ax = fig.add_axes([0.08, 0.10, 0.28, 0.025])   # [x, y, w, h] in fig fraction
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Pseudotime", fontsize=5, labelpad=2)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)

    # ── Scale bar: 1 mm ───────────────────────────────────────────────────────
    scale_px = 1000.0 / PIXEL_SIZE_UM
    x_min    = obs["X_centroid"].min()
    x_max    = obs["X_centroid"].max()
    y_min    = obs["Y_centroid"].min()
    y_max    = obs["Y_centroid"].max()
    margin_x = (x_max - x_min) * 0.03
    margin_y = (y_max - y_min) * 0.03
    sb_x0    = x_max - margin_x - scale_px
    sb_x1    = x_max - margin_x
    sb_y     = y_min + margin_y

    ax.plot([sb_x0, sb_x1], [sb_y, sb_y],
            color="black", lw=1.5, solid_capstyle="butt", zorder=10)
    ax.text((sb_x0 + sb_x1) / 2, sb_y + (y_max - y_min) * 0.012,
            "1 mm", ha="center", va="bottom", fontsize=5, color="black", zorder=10)

    # ── Axes cleanup ─────────────────────────────────────────────────────────
    ax.set_xlim(x_min - margin_x * 2, x_max + margin_x * 2)
    ax.set_ylim(y_min - margin_y * 2, y_max + margin_y * 2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "fig1G_spatial_pseudotime.pdf"
    out_png = RESULT_DIR / "fig1G_spatial_pseudotime.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adata = load_data()
    make_figure(adata)
