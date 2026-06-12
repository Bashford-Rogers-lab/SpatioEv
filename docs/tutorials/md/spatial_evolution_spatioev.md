# Mastering Spatial Evolution Analysis with SpatioEv

This tutorial mirrors the approachable style of spatial-biology package
tutorials while focusing on SpatioEv's own purpose: reconstructing spatial
evolution from static tissue images.

## 1. Set Up

```python
import spatioev as sv
```

Use the Scimap-like public namespaces:

```python
sv.pp  # preprocessing
sv.tl  # tools
sv.pl  # plotting
sv.hl  # helpers
sv.xe  # Xenium
```

## 2. Load and Prepare Data

```python
adata = sv.io.load_h5ad("data/exp_2/34434_1_adata.h5ad")
adata = sv.pp.run_segmentation_qc(adata)
adata = sv.pp.add_obs_from_var(adata, ["CK19", "Ki67"], zscore=True)
adata = sv.pp.add_zscore_obs_features(adata, ["area", "nc_ratio"])
```

## 3. Explore Spatial Organization

```python
density = sv.tl.compute_general_density(adata, tile_size=200)
phenotype_density = sv.tl.compute_phenotype_density(
    adata,
    phenotype_key="Tier_A",
    tile_size=200,
)
```

Use Ripley and Moran statistics to test spatial organization:

```python
ripley = sv.tl.cross_ripleys_k_by_phenotype(
    adata,
    phenotype_key="Tier_A",
    source_phenotype="pancreatic ductal epithelium",
    target_phenotype="Fibroblasts",
    radius=100,
)

moran = sv.tl.morans_i(
    adata.obs[["X_centroid", "Y_centroid"]],
    adata.obs["nc_ratio"],
)
```

## 4. Build Ductal Niche Graphs

```python
adata = sv.tl.cluster_spatial_components_from_mask(
    adata,
    seg_dir="data/exp_2/segmentation",
    label_key="Tier_A",
    label_value="pancreatic ductal epithelium",
)

adata = sv.tl.build_cell_graph(
    adata,
    feature_cols=["area", "nc_ratio", "CK19_expr", "Ki67_expr"],
    phenotype_key="Tier_A",
    radius=100,
)

niche_features = sv.tl.build_niche_feature_table_batched(
    adata,
    niche_key="pancreatic ductal epithelium_mask_component",
    phenotype_key="Tier_A",
)
```

## 5. Score PDAC Spatial Programs

```python
surroundings = sv.tl.summarize_niche_surrounding_context(
    adata,
    niche_key="pancreatic ductal epithelium_mask_component",
    phenotype_key="Tier_A",
)

feature_table = niche_features.merge(
    surroundings,
    on=["pancreatic ductal epithelium_mask_component", "image_id"],
    how="left",
)

modules = sv.tl.score_pdac_niche_pathology_modules(feature_table)
```

## 6. Infer Spatial Pseudotime

```python
feature_result = sv.tl.prepare_pseudotime_feature_matrix(
    feature_table,
    priority_features=[
        "pdac_early_duct_anchor_score",
        "pdac_dysplasia_score",
        "pdac_invasion_desmoplasia_score",
    ],
)
X_balanced = sv.tl.block_balance_feature_matrix(feature_result.matrix)
```

After fitting a principal tree, use the helper layer for branch interpretation:

```python
branch_labels, branch_metadata = sv.tl.infer_branch_labels(
    principal_tree,
    niche_results,
    source_node=root_node,
    node_col="elpigraph_node_id",
)
```

## 7. Interpret Dynamics

```python
trend_table = sv.tl.compute_feature_trend_table(
    feature_table,
    feature_result.selected_features,
    pseudotime_col="elpigraph_pseudotime",
)

interaction_cells = sv.tl.compute_epithelial_centered_interaction_dynamics(
    adata,
    pseudotime_key="elpigraph_pseudotime_pathology",
    phenotype_key="Tier_A",
    source_phenotype="pancreatic ductal epithelium",
    target_phenotypes=["Fibroblasts", "T cells", "B lineage"],
    radius_um=30,
    pixel_size_um=0.325,
)
interaction_summary = sv.tl.summarize_epithelial_interaction_dynamics(
    interaction_cells,
    pseudotime_key="elpigraph_pseudotime_pathology",
)
```

## 8. Analyze ECM-Cell Organization

```python
links = sv.tl.build_cell_fiber_links(adata, fiber_df)
fiber_df = sv.tl.fiber_density_near_cells(adata, fiber_df, links)
ecm_moran = sv.tl.cross_morans_i_ecm_cells(fiber_df, cell_feature_df)
```

## 9. Extend to Xenium

```python
marker_scores = sv.xe.compute_marker_set_scores(adata, marker_sets)
cluster_summary = sv.xe.summarize_cluster_marker_scores(
    marker_scores,
    adata.obs["cluster"],
)
histology_scores = sv.xe.score_xenium_histology_modules(niche_feature_table)
```

