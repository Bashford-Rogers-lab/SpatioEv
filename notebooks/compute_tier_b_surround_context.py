"""
Compute and save Tier B surrounding context per ductal niche
============================================================
Replicates notebook 06 Cell 57: calls sv.summarize_niche_surrounding_context
with phenotype_key="Tier_B" to get per-niche surround proportions for all
Tier B subtype labels, then merges with pseudotime and saves to CSV.

Output: notebooks/results/pseudotime_exp2/surround_context_tier_b.csv

Run:
    python notebooks/compute_tier_b_surround_context.py
"""

from pathlib import Path

import pandas as pd
import anndata as ad
import spatioev as sv

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT            = Path("/Users/shihongwu/SpatioEv")
DATA_DIR        = ROOT / "data" / "exp_2"
RESULT_DIR      = ROOT / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH      = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH = DATA_DIR / "34434_1_annotation.csv"
SEG_DIR         = DATA_DIR / "segmentation"
PT_CSV          = RESULT_DIR / "niche_pseudotime_results.csv"

DUCT_NICHE_KEY  = "pancreatic ductal epithelium_mask_component"
DUCT_LABEL      = "pancreatic ductal epithelium"
PIXEL_SIZE_UM   = 0.325

# ── Load adata + annotations ───────────────────────────────────────────────────
print("Loading adata …")
adata = ad.read_h5ad(ADATA_PATH)
ann   = pd.read_csv(ANNOTATION_PATH, index_col=0)
for col in ["Tier_A", "Tier_B"]:
    if col in ann.columns:
        adata.obs[col] = ann[col].reindex(adata.obs_names)
print(f"  {adata.n_obs:,} cells, Tier_B labels: {adata.obs['Tier_B'].nunique()}")

# ── Compute ductal niche components (required for niche_key) ───────────────────
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
print(f"  → {adata.obs[DUCT_NICHE_KEY].nunique()} ductal niche components")

# ── Build cell graph (required by summarize_niche_surrounding_context) ────────
print("Building cell graph …")
CELL_GRAPH_FEATURES = [
    c for c in [
        "area", "eccentricity", "major_axis_length", "minor_axis_length", "perimeter",
        "convex_area", "equivalent_diameter", "orientation", "solidity", "feret_diameter_max",
        "major_minor_axis_ratio", "perim_square_over_area", "major_axis_equiv_diam_ratio",
        "convex_hull_resid", "centroid_dif", "num_concavities", "circularity",
        "fractal_dimension", "fractual_dimension", "boundary_irregularity", "nc_ratio",
        "polarity_score", "haralick_contrast", "haralick_correlation", "haralick_energy",
        "haralick_homogeneity", "entropy", "pcc_ck19_nak", "intensity_ratio", "inertia",
        "lacunarity", "CK19_expr", "Vimentin_expr", "NaKATPase_expr", "Ki67_expr",
    ] if c in adata.obs.columns
]
print(f"  {len(CELL_GRAPH_FEATURES)} cell graph features")
adata = sv.build_cell_graph(
    adata,
    feature_cols=CELL_GRAPH_FEATURES,
    phenotype_key="Tier_A",
    radius=30 / PIXEL_SIZE_UM,
    image_key="imageid",
    auto_log=True,
    compute_weights=True,
    sigma_space=None,
    sigma_feat=None,
    feature_obsm_key="cell_features",
    adjacency_key="cell_graph_connectivities",
    distance_key="cell_graph_distances",
    graph_obs_key="cell_graph_valid",
)
print(f"  cell_graph_connectivities: {adata.obsp['cell_graph_connectivities'].shape}, "
      f"{adata.obsp['cell_graph_connectivities'].nnz:,} edges")

# ── Tier B surround context ────────────────────────────────────────────────────
print("Computing Tier B surround context (surround_hops=5) …")
tier_b_labels = (
    adata.obs["Tier_B"]
    .value_counts()
    .loc[lambda v: ~v.index.isin(["noise", "Unknown"])]
    .index.tolist()
)
print(f"  Tier B labels ({len(tier_b_labels)}): {tier_b_labels}")

surround_context_df_B = sv.summarize_niche_surrounding_context(
    adata,
    niche_key=DUCT_NICHE_KEY,
    phenotype_key="Tier_B",
    phenotype_labels=tier_b_labels,
    surround_hops=5,
)
print(f"  surround_context_df_B: {surround_context_df_B.shape}")

# ── Merge pseudotime ───────────────────────────────────────────────────────────
print("Merging pseudotime …")
pt = pd.read_csv(PT_CSV)
surround_plot_df_B = surround_context_df_B.merge(
    pt[[DUCT_NICHE_KEY, "image_id",
        "elpigraph_pseudotime", "elpigraph_pseudotime_q",
        "elpigraph_edge_id", "principal_tree_branch"]],
    on=[DUCT_NICHE_KEY, "image_id"],
    how="left",
)
print(f"  merged: {surround_plot_df_B.shape}, pseudotime coverage: "
      f"{surround_plot_df_B['elpigraph_pseudotime_q'].notna().sum():,} niches")

# ── Save ───────────────────────────────────────────────────────────────────────
RESULT_DIR.mkdir(parents=True, exist_ok=True)
out = RESULT_DIR / "surround_context_tier_b.csv"
surround_plot_df_B.to_csv(out, index=False)
print(f"\nSaved: {out}")

# Quick summary of surround_prop columns saved
prop_cols = [c for c in surround_plot_df_B.columns if c.startswith("surround_prop__")]
print(f"surround_prop columns ({len(prop_cols)}): {prop_cols}")
