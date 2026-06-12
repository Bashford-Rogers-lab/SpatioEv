# Helper Functions (`sv.hl`)

Stable reusable building blocks available outside the main pseudotime workflow.

---

## Geometry & Spatial Helpers

::: spatioev.pp.spatial_prep.compute_convex_hull

::: spatioev.pp.spatial_prep.compute_convex_hull_area

::: spatioev.pp.spatial_prep.distance_to_convex_hull_boundary

---

## Pixel & Morphology Helpers

::: spatioev.pp.pixel.calculate_polarity_score

::: spatioev.pp.pixel.calculate_moment_of_inertia

::: spatioev.pp.pixel.calculate_haralick_features

::: spatioev.pp.pixel.calculate_entropy

::: spatioev.pp.pixel.calculate_lacunarity

::: spatioev.pp.pixel.calculate_channel_correlation

---

## Feature Matrix & Scoring Helpers

::: spatioev.tl.pseudotime.prepare_pseudotime_feature_matrix

::: spatioev.tl.pseudotime.block_balance_feature_matrix

::: spatioev.tl.pseudotime.sample_center_feature_matrix

---

## Tree & Branch Helpers

::: spatioev.tl.pseudotime.infer_branch_labels

::: spatioev.tl.pseudotime.summarize_branches

::: spatioev.tl.pseudotime.project_tree_nodes_to_embedding

---

## Pseudotime Table Helpers

::: spatioev.tl.pseudotime.add_branch_time_bins

::: spatioev.tl.pseudotime.branch_time_feature_matrix

::: spatioev.tl.pseudotime.find_branch_transition_features

::: spatioev.tl.pseudotime.compute_feature_trend_table
