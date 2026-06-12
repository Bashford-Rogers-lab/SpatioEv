# Quick Start

This example shows the intended public API style. It uses the new namespaces
while calling the same underlying SpatioEv implementation modules.

```python
import spatioev as sv

adata = sv.io.load_h5ad("data/exp_2/34434_1_adata.h5ad")

adata = sv.pp.run_segmentation_qc(adata)
adata = sv.pp.add_obs_from_var(adata, ["CK19", "Ki67"], zscore=True)
adata = sv.pp.add_zscore_obs_features(adata, ["area", "nc_ratio"])

density = sv.tl.compute_general_density(adata, tile_size=200)
ripley = sv.tl.cross_ripleys_k_by_phenotype(
    adata,
    phenotype_key="Tier_A",
    source_phenotype="pancreatic ductal epithelium",
    target_phenotype="Fibroblasts",
    radius=100,
)

adata = sv.tl.build_cell_graph(
    adata,
    feature_cols=["area", "nc_ratio", "CK19_expr"],
    phenotype_key="Tier_A",
    radius=100,
)
```

For trajectory analysis, the central workflow is:

```python
feature_result = sv.tl.prepare_pseudotime_feature_matrix(feature_table)
X_balanced = sv.tl.block_balance_feature_matrix(feature_result.matrix)
trend_table = sv.tl.compute_feature_trend_table(
    feature_table,
    feature_result.selected_features,
    pseudotime_col="elpigraph_pseudotime",
)
```

