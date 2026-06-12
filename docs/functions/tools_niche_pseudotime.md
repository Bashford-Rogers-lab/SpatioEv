# Niche Graphs & Pseudotime (`sv.tl.niche`, `sv.tl.pseudotime`)

These functions are the core of the SpatioEv spatial-evolution workflow.

---

## Niche Detection & Boundaries

::: spatioev.tl.niche.estimate_density_adaptive_dbscan_params

::: spatioev.tl.niche.estimate_spatial_component_params

::: spatioev.tl.niche.cluster_spatial_niches

::: spatioev.tl.niche.cluster_spatial_components

::: spatioev.tl.niche.cluster_spatial_components_hdbscan

::: spatioev.tl.niche.cluster_spatial_components_from_mask

::: spatioev.tl.niche.build_niche_boundaries

::: spatioev.tl.niche.buffer_niche_boundaries

::: spatioev.tl.niche.assign_cells_to_niche_regions

::: spatioev.tl.niche.summarize_niche_composition

::: spatioev.tl.niche.add_niche_regions_to_obs

---

## Cell Graph & Niche Graph Features

::: spatioev.tl.niche.build_cell_graph

::: spatioev.tl.niche.extract_niche_subgraph

::: spatioev.tl.niche.extract_all_niche_subgraphs

::: spatioev.tl.niche.summarize_niche_graph_features

::: spatioev.tl.niche.build_niche_feature_table

::: spatioev.tl.niche.build_niche_feature_table_batched

::: spatioev.tl.niche.summarize_niche_surrounding_context

::: spatioev.tl.niche.score_pdac_niche_pathology_modules

---

## Pseudotime Feature Engineering

::: spatioev.tl.pseudotime.prepare_pseudotime_feature_matrix

::: spatioev.tl.pseudotime.block_balance_feature_matrix

::: spatioev.tl.pseudotime.sample_center_feature_matrix

::: spatioev.tl.pseudotime.infer_branch_labels

::: spatioev.tl.pseudotime.summarize_branches

::: spatioev.tl.pseudotime.project_tree_nodes_to_embedding

---

## Pseudotime Dynamics & Trend Analysis

::: spatioev.tl.pseudotime.assign_pseudotime_bins

::: spatioev.tl.pseudotime.compute_epithelial_centered_interaction_dynamics

::: spatioev.tl.pseudotime.summarize_epithelial_interaction_dynamics

::: spatioev.tl.pseudotime.compute_feature_trend_table

::: spatioev.tl.pseudotime.add_branch_time_bins

::: spatioev.tl.pseudotime.branch_time_feature_matrix

::: spatioev.tl.pseudotime.find_branch_transition_features
