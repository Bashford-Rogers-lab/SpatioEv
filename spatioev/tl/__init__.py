"""Analysis tools API for SpatioEv.

``spatioev.tl`` is the main public namespace for spatial biology analyses.
Functions are grouped into focused submodules that can be imported directly::

    from spatioev.tl.stats import morans_i, cross_ripleys_k_by_phenotype
    from spatioev.tl.niche import build_niche_feature_table
    from spatioev.tl.pseudotime import prepare_pseudotime_feature_matrix

Or accessed through the top-level namespace::

    import spatioev as sv
    sv.tl.morans_i(...)
    sv.tl.build_cell_graph(...)

Submodules
----------
stats       Ripley K and Moran I spatial statistics families
density     Tile, KDE, kNN, radius, and phenotype interaction density
niche       Niche boundary detection, cell graphs, graph feature tables
ecm         ECM–cell links, spatial stats, graph niches, neighborhoods
pseudotime  Feature matrix prep, branch annotation, trend analysis
phenotype   Clustering, subsetting, merging, refinement
ml          SVM classifier, feature construction, inspection
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Submodule → module path mapping (new clean locations)
_SUBMODULE_MAP = {
    "stats": "spatioev.tl.stats",
    "density": "spatioev.tl.density",
    "niche": "spatioev.tl.niche",
    "ecm": "spatioev.tl.ecm",
    "pseudotime": "spatioev.tl.pseudotime",
    "phenotype": "spatioev.tl.phenotype",
    "ml": "spatioev.tl.ml",
}

# Flat function → submodule path (new locations)
_EXPORTS = {
    # --- phenotype / ml ---
    "build_marker_features": "spatioev.tl.ml",
    "build_morphology_features": "spatioev.tl.ml",
    "build_feature_matrix": "spatioev.tl.ml",
    "train_svm_classifier": "spatioev.tl.ml",
    "predict_svm": "spatioev.tl.ml",
    "run_svm_phenotyping": "spatioev.tl.ml",
    "inspect_reassigned_cells": "spatioev.tl.ml",
    "inspect_disagreements": "spatioev.tl.ml",
    "cluster_cells": "spatioev.tl.phenotype",
    "scimap_napari_gater": "spatioev.tl.phenotype",
    "scimap_rescale": "spatioev.tl.phenotype",
    "scimap_phenotype_cells": "spatioev.tl.phenotype",
    "run_scimap_prior_knowledge_phenotyping": "spatioev.tl.phenotype",
    "subset_cells": "spatioev.tl.phenotype",
    "annotate_interactive": "spatioev.tl.phenotype",
    "annotate_from_csv": "spatioev.tl.phenotype",
    "merge_annotations": "spatioev.tl.phenotype",
    "merge_refinements": "spatioev.tl.phenotype",
    "refine_clusters": "spatioev.tl.phenotype",
    # --- density ---
    "assign_tiles": "spatioev.tl.density",
    "compute_general_density": "spatioev.tl.density",
    "compute_phenotype_density": "spatioev.tl.density",
    "phenotype_density_correlation": "spatioev.tl.density",
    "compute_kde_density": "spatioev.tl.density",
    "compute_local_density_by_phenotype": "spatioev.tl.density",
    "compute_local_density_all_cells": "spatioev.tl.density",
    "compute_radius_density": "spatioev.tl.density",
    "phenotype_interaction_density": "spatioev.tl.density",
    # --- spatial statistics ---
    "ripleys_k": "spatioev.tl.stats",
    "ripleys_curve": "spatioev.tl.stats",
    "ripley_envelope": "spatioev.tl.stats",
    "ripleys_k_by_image": "spatioev.tl.stats",
    "ripleys_k_by_phenotype": "spatioev.tl.stats",
    "cross_ripleys_k": "spatioev.tl.stats",
    "cross_ripleys_k_by_phenotype": "spatioev.tl.stats",
    "cross_ripleys_k_all_pairs": "spatioev.tl.stats",
    "cross_ripleys_curve": "spatioev.tl.stats",
    "cross_ripley_envelope": "spatioev.tl.stats",
    "cross_ripleys_curve_by_phenotype": "spatioev.tl.stats",
    "cross_ripley_envelope_by_phenotype": "spatioev.tl.stats",
    "cross_ripley_permutation_envelope": "spatioev.tl.stats",
    "ripley_local_counts_by_phenotype": "spatioev.tl.stats",
    "cross_ripley_local_counts": "spatioev.tl.stats",
    "ripley_interaction_scale": "spatioev.tl.stats",
    "ripley_spatial_scales": "spatioev.tl.stats",
    "morans_i": "spatioev.tl.stats",
    "morans_i_permutation_test": "spatioev.tl.stats",
    "morans_i_by_image": "spatioev.tl.stats",
    "morans_i_by_image_permutation_test": "spatioev.tl.stats",
    "local_morans_i": "spatioev.tl.stats",
    "add_local_morans_i": "spatioev.tl.stats",
    "classify_local_morans_i": "spatioev.tl.stats",
    "add_local_morans_i_quadrants": "spatioev.tl.stats",
    "cross_morans_i": "spatioev.tl.stats",
    "cross_morans_i_by_image": "spatioev.tl.stats",
    "cross_morans_i_permutation_test": "spatioev.tl.stats",
    "cross_morans_i_by_image_permutation_test": "spatioev.tl.stats",
    "local_cross_morans_i": "spatioev.tl.stats",
    "add_local_cross_morans_i": "spatioev.tl.stats",
    "classify_local_cross_morans_i": "spatioev.tl.stats",
    "add_local_cross_morans_i_quadrants": "spatioev.tl.stats",
    "summarize_target_features_around_source_cells": "spatioev.tl.stats",
    "cross_morans_i_feature_matrix": "spatioev.tl.stats",
    "add_local_cross_morans_i_between_phenotypes": "spatioev.tl.stats",
    # --- niche ---
    "estimate_density_adaptive_dbscan_params": "spatioev.tl.niche",
    "estimate_spatial_component_params": "spatioev.tl.niche",
    "cluster_spatial_niches": "spatioev.tl.niche",
    "cluster_spatial_components": "spatioev.tl.niche",
    "cluster_spatial_components_hdbscan": "spatioev.tl.niche",
    "cluster_spatial_components_from_mask": "spatioev.tl.niche",
    "build_niche_boundaries": "spatioev.tl.niche",
    "buffer_niche_boundaries": "spatioev.tl.niche",
    "assign_cells_to_niche_regions": "spatioev.tl.niche",
    "summarize_niche_composition": "spatioev.tl.niche",
    "add_niche_regions_to_obs": "spatioev.tl.niche",
    "build_cell_graph": "spatioev.tl.niche",
    "extract_niche_subgraph": "spatioev.tl.niche",
    "extract_all_niche_subgraphs": "spatioev.tl.niche",
    "summarize_niche_graph_features": "spatioev.tl.niche",
    "build_niche_feature_table": "spatioev.tl.niche",
    "build_niche_feature_table_batched": "spatioev.tl.niche",
    "summarize_niche_surrounding_context": "spatioev.tl.niche",
    "score_pdac_niche_pathology_modules": "spatioev.tl.niche",
    # --- pseudotime ---
    "prepare_pseudotime_feature_matrix": "spatioev.tl.pseudotime",
    "block_balance_feature_matrix": "spatioev.tl.pseudotime",
    "sample_center_feature_matrix": "spatioev.tl.pseudotime",
    "infer_branch_labels": "spatioev.tl.pseudotime",
    "summarize_branches": "spatioev.tl.pseudotime",
    "project_tree_nodes_to_embedding": "spatioev.tl.pseudotime",
    "assign_pseudotime_bins": "spatioev.tl.pseudotime",
    "compute_epithelial_centered_interaction_dynamics": "spatioev.tl.pseudotime",
    "summarize_epithelial_interaction_dynamics": "spatioev.tl.pseudotime",
    "compute_feature_trend_table": "spatioev.tl.pseudotime",
    "add_branch_time_bins": "spatioev.tl.pseudotime",
    "branch_time_feature_matrix": "spatioev.tl.pseudotime",
    "find_branch_transition_features": "spatioev.tl.pseudotime",
    "tree_edges": "spatioev.tl.pseudotime",
    "node_graph": "spatioev.tl.pseudotime",
    "zscore_series": "spatioev.tl.pseudotime",
    "minmax_scale": "spatioev.tl.pseudotime",
    "assign_feature_blocks": "spatioev.tl.pseudotime",
    "FeatureMatrixResult": "spatioev.tl.pseudotime",
    "benjamini_hochberg": "spatioev.tl.pseudotime",
    # --- ECM ---
    "build_cell_fiber_links": "spatioev.tl.ecm",
    "build_nearest_cell_fiber_map": "spatioev.tl.ecm",
    "cell_to_fiber_distance": "spatioev.tl.ecm",
    "fiber_density_near_cells": "spatioev.tl.ecm",
    "morans_i_fibers": "spatioev.tl.ecm",
    "local_morans_i_fibers": "spatioev.tl.ecm",
    "cross_morans_i_ecm_cells": "spatioev.tl.ecm",
    "cross_morans_i_ecm_cells_permutation_test": "spatioev.tl.ecm",
    "map_cells_to_fibers": "spatioev.tl.ecm",
    "map_cells_to_fibers_kernel": "spatioev.tl.ecm",
    "local_cross_morans_i_ecm_cells": "spatioev.tl.ecm",
    "spatial_linear_regression": "spatioev.tl.ecm",
    "spatial_mixed_model": "spatioev.tl.ecm",
    "spatial_enrichment_score": "spatioev.tl.ecm",
    "fiber_vectors": "spatioev.tl.ecm",
    "cell_fiber_alignment": "spatioev.tl.ecm",
    "ecm_cross_ripleys_k": ("spatioev.tl.ecm", "cross_ripleys_k_permutation_envelope"),
    "ecm_cross_ripleys_k_permutation_envelope": (
        "spatioev.tl.ecm",
        "cross_ripleys_k_permutation_envelope",
    ),
    "build_ecm_bipartite_graph_per_image": "spatioev.tl.ecm",
    "project_fiber_graph_per_image": "spatioev.tl.ecm",
    "detect_ecm_niches_per_image": "spatioev.tl.ecm",
    "assign_niches_to_fibers": "spatioev.tl.ecm",
    "compute_invasion_score": "spatioev.tl.ecm",
    "detect_tissue_regions_dbscan": "spatioev.tl.ecm",
    "filter_ecm_cell_inputs_by_tissue": "spatioev.tl.ecm",
    "build_ecm_cell_neighborhood_features": "spatioev.tl.ecm",
    "default_ecm_cell_neighborhood_feature_columns": "spatioev.tl.ecm",
    "cluster_ecm_cell_neighborhoods": "spatioev.tl.ecm",
    "summarize_ecm_cell_neighborhoods": "spatioev.tl.ecm",
    "score_col6_dark_neighborhoods": "spatioev.tl.ecm",
    "add_neighborhoods_to_obs": "spatioev.tl.ecm",
}

__all__ = sorted([*_SUBMODULE_MAP, *_EXPORTS])


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_MAP:
        module = import_module(_SUBMODULE_MAP[name])
        globals()[name] = module
        return module

    if name in _EXPORTS:
        target = _EXPORTS[name]
        if isinstance(target, tuple):
            module_name, attr_name = target
        else:
            module_name, attr_name = target, name
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
