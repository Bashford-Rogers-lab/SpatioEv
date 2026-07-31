"""
Figure 1E — Cell graph + niche graph schematic
===============================================
Selects a small tissue region containing 3-5 ductal niches,
then overlays:
  • Cell graph  — thin edges connecting cells within 30 µm (within each niche)
  • Niche graph — bold edges connecting spatially adjacent niche centroids

Run from the SpatioEv root:
    python notebooks/fig1E_cell_niche_graphs.py

Output: paper/notebooks/results/pseudotime_exp2/fig1E_cell_niche_graphs.pdf  (and .png)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT           = Path("/Users/shihongwu/SpatioEv")
DATA_DIR       = ROOT / "data" / "exp_2"
RESULT_DIR     = ROOT / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH     = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH = DATA_DIR / "34434_1_annotation.csv"
SEG_DIR        = DATA_DIR / "segmentation"

DUCT_NICHE_KEY  = "pancreatic ductal epithelium_mask_component"
DUCT_LABEL      = "pancreatic ductal epithelium"
PIXEL_SIZE_UM   = 0.325

# Graph parameters (matching the notebook)
CELL_GRAPH_RADIUS_UM   = 30.0
NICHE_GRAPH_RADIUS_UM  = 150.0   # connect niche centroids within this distance

CELL_GRAPH_RADIUS_PX   = CELL_GRAPH_RADIUS_UM  / PIXEL_SIZE_UM   # ≈  92 px
NICHE_GRAPH_RADIUS_PX  = NICHE_GRAPH_RADIUS_UM / PIXEL_SIZE_UM   # ≈ 462 px

# How many neighbouring niches to include around the seed
N_NICHES = 5

# Per-niche colour palette (distinct, accessible)
NICHE_PALETTE = [
    "#4393c3",   # blue
    "#d6604d",   # red-orange
    "#4dac26",   # green
    "#998ec3",   # purple
    "#f1a340",   # amber
    "#b35806",   # brown
]

# Publication style
matplotlib.rcParams.update({
    "font.family":  "Arial",
    "font.size":     6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ── Load adata ────────────────────────────────────────────────────────────────
def load_adata():
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
    print(f"  → {adata.obs[DUCT_NICHE_KEY].nunique()} components")
    return adata


# ── Select a region with N_NICHES adjacent niches ────────────────────────────
def select_region(obs: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """
    Returns (niche_ids, obs_crop) where obs_crop covers all selected niches
    plus a padding of NICHE_GRAPH_RADIUS_PX around them.
    """
    # Compute per-niche centroid (whole-slide coords)
    duct_obs = obs[obs["Tier_A"] == DUCT_LABEL].copy()
    niche_centroids = (
        duct_obs[duct_obs[DUCT_NICHE_KEY].notna()]
        .groupby(DUCT_NICHE_KEY)[["X_centroid", "Y_centroid"]]
        .mean()
        .rename(columns={"X_centroid": "cx", "Y_centroid": "cy"})
    )

    # Keep niches with ≥ 10 cells (skip singletons for the focal region)
    niche_sizes = (
        duct_obs[duct_obs[DUCT_NICHE_KEY].notna()]
        .groupby(DUCT_NICHE_KEY).size()
        .rename("n_cells")
    )
    niche_centroids = niche_centroids.join(niche_sizes)
    niche_centroids = niche_centroids[niche_centroids["n_cells"] >= 10]

    # Find a dense cluster: for each niche, count how many neighbours within
    # NICHE_GRAPH_RADIUS_PX * 2
    coords = niche_centroids[["cx", "cy"]].values
    tree = cKDTree(coords)
    counts = tree.query_ball_point(coords, r=NICHE_GRAPH_RADIUS_PX * 2)
    niche_centroids["neighbour_count"] = [len(c) for c in counts]

    # Seed = niche with most neighbours (preferring medium size)
    candidates = niche_centroids[
        (niche_centroids["n_cells"] >= 15) & (niche_centroids["n_cells"] <= 80)
    ]
    seed_id = candidates["neighbour_count"].idxmax()
    seed_cx, seed_cy = candidates.loc[seed_id, ["cx", "cy"]]
    print(f"  Seed niche: {seed_id}  ({candidates.loc[seed_id,'n_cells']:.0f} cells)")

    # Take the N_NICHES closest niches to the seed
    dists, idxs = tree.query(
        [seed_cx, seed_cy], k=min(N_NICHES, len(niche_centroids))
    )
    selected_ids = niche_centroids.index[idxs].tolist()
    print(f"  Selected niches: {selected_ids}")

    # Crop obs to the bounding box of selected niches + padding
    sel_cent = niche_centroids.loc[selected_ids]
    pad = NICHE_GRAPH_RADIUS_PX * 1.5
    x0 = sel_cent["cx"].min() - pad
    x1 = sel_cent["cx"].max() + pad
    y0 = sel_cent["cy"].min() - pad
    y1 = sel_cent["cy"].max() + pad

    obs_crop = obs[
        (obs["X_centroid"] >= x0) & (obs["X_centroid"] <= x1) &
        (obs["Y_centroid"] >= y0) & (obs["Y_centroid"] <= y1)
    ].copy()
    print(f"  Crop: {len(obs_crop):,} cells in region")
    return selected_ids, obs_crop, niche_centroids.loc[selected_ids]


# ── Build cell graph edges for a set of cells ────────────────────────────────
def cell_graph_edges(cells: pd.DataFrame, radius_px: float) -> list[tuple]:
    """Return list of (i, j) index pairs within radius_px."""
    xy = cells[["X_centroid", "Y_centroid"]].values
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=radius_px)
    return list(pairs)


# ── Main figure ───────────────────────────────────────────────────────────────
def make_figure(adata):
    obs = adata.obs.copy()
    selected_ids, obs_crop, sel_centroids = select_region(obs)

    # Assign one colour per selected niche
    niche_color = {nid: NICHE_PALETTE[i % len(NICHE_PALETTE)]
                   for i, nid in enumerate(selected_ids)}

    # ── Figure sizing: 89 mm wide, square-ish ────────────────────────────────
    mm2in = 1 / 25.4
    fig_w = 36 * mm2in
    x_range = obs_crop["X_centroid"].max() - obs_crop["X_centroid"].min()
    y_range = obs_crop["Y_centroid"].max() - obs_crop["Y_centroid"].min()
    fig_h = fig_w * (y_range / x_range) + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Background non-ductal cells ───────────────────────────────────────────
    bg = obs_crop[obs_crop["Tier_A"] != DUCT_LABEL]
    ax.scatter(
        bg["X_centroid"].values,
        bg["Y_centroid"].values,
        color=(0.82, 0.82, 0.82, 0.35),
        s=1.0,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # ── Cell graph edges (within each selected niche) ─────────────────────────
    for nid in selected_ids:
        niche_cells = obs_crop[obs_crop[DUCT_NICHE_KEY] == nid]
        if len(niche_cells) < 2:
            continue
        edges = cell_graph_edges(niche_cells, CELL_GRAPH_RADIUS_PX)
        xy = niche_cells[["X_centroid", "Y_centroid"]].values
        col = niche_color[nid]
        for i, j in edges:
            ax.plot(
                [xy[i, 0], xy[j, 0]],
                [xy[i, 1], xy[j, 1]],
                color=col, lw=0.35, alpha=0.45, solid_capstyle="round",
                zorder=2, rasterized=True,
            )

    # ── Ductal cells (non-selected niches: light, selected: coloured) ─────────
    other_duct = obs_crop[
        (obs_crop["Tier_A"] == DUCT_LABEL) &
        (~obs_crop[DUCT_NICHE_KEY].isin(selected_ids))
    ]
    ax.scatter(
        other_duct["X_centroid"].values,
        other_duct["Y_centroid"].values,
        color=(0.70, 0.70, 0.70, 0.5),
        s=3.0, linewidths=0, zorder=3, rasterized=True,
    )

    for nid in selected_ids:
        niche_cells = obs_crop[obs_crop[DUCT_NICHE_KEY] == nid]
        ax.scatter(
            niche_cells["X_centroid"].values,
            niche_cells["Y_centroid"].values,
            color=niche_color[nid],
            s=5.0, linewidths=0.3,
            edgecolors="white", zorder=4, rasterized=True,
        )

    # ── Niche graph edges (between selected niche centroids) ──────────────────
    cent_xy = sel_centroids[["cx", "cy"]].values
    tree = cKDTree(cent_xy)
    niche_pairs = tree.query_pairs(r=NICHE_GRAPH_RADIUS_PX)

    for i, j in niche_pairs:
        ax.plot(
            [cent_xy[i, 0], cent_xy[j, 0]],
            [cent_xy[i, 1], cent_xy[j, 1]],
            color="#333333", lw=1.2, alpha=0.75,
            solid_capstyle="round", zorder=5,
        )

    # Niche centroid markers
    for i, nid in enumerate(selected_ids):
        col = niche_color[nid]
        ax.scatter(
            cent_xy[i, 0], cent_xy[i, 1],
            s=12, color=col,
            edgecolors="#333333", linewidths=0.5,
            zorder=6,
        )

    # ── Annotations ───────────────────────────────────────────────────────────
    xmin = obs_crop["X_centroid"].min()
    ymin = obs_crop["Y_centroid"].min()
    xmax = obs_crop["X_centroid"].max()
    ymax = obs_crop["Y_centroid"].max()

    # Annotation: cell graph
    # Find a niche with edges for annotation target
    example_nid = selected_ids[0]
    ex_cells = obs_crop[obs_crop[DUCT_NICHE_KEY] == example_nid]
    if len(ex_cells) >= 2:
        xy_ex = ex_cells[["X_centroid", "Y_centroid"]].values
        mid_x = xy_ex[:2, 0].mean()
        mid_y = xy_ex[:2, 1].mean()
        ax.annotate(
            "Cell graph",
            xy=(mid_x, mid_y),
            xytext=(xmin + (xmax - xmin) * 0.05, ymin + (ymax - ymin) * 0.08),
            fontsize=5, color=niche_color[example_nid],
            arrowprops=dict(arrowstyle="-|>", color=niche_color[example_nid],
                            lw=0.7, shrinkA=0, shrinkB=2),
            zorder=10,
        )

    # Annotation: niche graph (point to one of the thick connecting edges)
    if len(niche_pairs) > 0:
        ni, nj = list(niche_pairs)[0]
        edge_mid = (cent_xy[ni] + cent_xy[nj]) / 2
        ax.annotate(
            "Niche graph",
            xy=(edge_mid[0], edge_mid[1]),
            xytext=(xmin + (xmax - xmin) * 0.60, ymin + (ymax - ymin) * 0.08),
            fontsize=5, color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#333333",
                            lw=0.7, shrinkA=0, shrinkB=2),
            zorder=10,
        )

    # ── Scale bar: 100 µm ─────────────────────────────────────────────────────
    sb_px  = 100.0 / PIXEL_SIZE_UM
    margin = (xmax - xmin) * 0.04
    sb_x0  = xmax - margin - sb_px
    sb_y   = ymin + (ymax - ymin) * 0.04
    ax.plot([sb_x0, sb_x0 + sb_px], [sb_y, sb_y],
            color="black", lw=1.2, solid_capstyle="butt", zorder=10)
    ax.text(sb_x0 + sb_px / 2, sb_y + (ymax - ymin) * 0.015,
            "100 µm", ha="center", va="bottom", fontsize=5, color="black", zorder=10)

    # ── Axes cleanup ─────────────────────────────────────────────────────────
    ax.set_xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.02)
    ax.set_ylim(ymin - (ymax - ymin) * 0.02, ymax + (ymax - ymin) * 0.02)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "fig1E_cell_niche_graphs.pdf"
    out_png = RESULT_DIR / "fig1E_cell_niche_graphs.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adata = load_adata()
    make_figure(adata)
