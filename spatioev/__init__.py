"""SpatioEv — spatial feature extraction and evolution framework.

SpatioEv is a Python toolbox for spatial biology analyses over multiplexed
imaging and spatial transcriptomics data. It provides a Scimap-inspired
namespace organisation::

    import spatioev as sv

    sv.pp   # preprocessing and QC
    sv.tl   # analysis tools: statistics, niches, pseudotime, ECM
    sv.pl   # plotting
    sv.hl   # helper functions and building blocks
    sv.xe   # Xenium / spatial-transcriptomics helpers
    sv.io   # input / output

Heavy optional dependencies (Scanpy, scimap, Napari, ElPiGraph) are loaded
lazily, so ``import spatioev`` is lightweight in all environments.

Typical usage
-------------
>>> import anndata as ad, numpy as np, pandas as pd
>>> import spatioev as sv
>>> from spatioev.config import QCConfig
>>>
>>> adata = sv.pp.run_segmentation_qc(adata, QCConfig(pixel_size=0.325))
>>> tiled  = sv.tl.assign_tiles(adata, tile_size=128)
>>> mi     = sv.tl.morans_i(adata.obs[["X_centroid", "Y_centroid"]], adata.obs["area"])
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# Public subpackages (lazy-loaded on attribute access)
# --------------------------------------------------------------------------- #
_SUBMODULES = {
    "hl": "spatioev.hl",
    "io": "spatioev.io",
    "pl": "spatioev.pl",
    "pp": "spatioev.pp",
    "tl": "spatioev.tl",
    "xe": "spatioev.xe",
}

# --------------------------------------------------------------------------- #
# Flat convenience exports (delegated entirely to subpackages — no duplication)
# --------------------------------------------------------------------------- #
# Each entry maps  function_name -> subpackage_module
# The subpackage __init__.py is authoritative for which module provides it.
_EXPORTS: dict[str, str] = {
    # io
    "load_h5ad": "spatioev.io.load",
    # pp
    "zscore_normalize": "spatioev.pp",
    "add_obs_from_var": "spatioev.pp",
    "add_zscore_obs_features": "spatioev.pp",
    "run_segmentation_qc": "spatioev.pp",
    "generate_qc_summary": "spatioev.pp",
    "validate_spatial_coordinates": "spatioev.pp",
    "compute_tissue_areas": "spatioev.pp",
    "detect_edge_cells": "spatioev.pp",
    "extract_cell_pixel_features": "spatioev.pp",
    "extract_cell_pixel_features_for_fov": "spatioev.pp",
    "extract_xenium_dapi_features": "spatioev.pp",
    # tl — density
    "assign_tiles": "spatioev.tl",
    "compute_general_density": "spatioev.tl",
    "compute_phenotype_density": "spatioev.tl",
    "phenotype_density_correlation": "spatioev.tl",
    "compute_kde_density": "spatioev.tl",
    "compute_local_density_by_phenotype": "spatioev.tl",
    "compute_local_density_all_cells": "spatioev.tl",
    "compute_radius_density": "spatioev.tl",
    "phenotype_interaction_density": "spatioev.tl",
    # tl — spatial statistics
    "ripleys_k": "spatioev.tl",
    "ripleys_k_by_phenotype": "spatioev.tl",
    "cross_ripleys_k": "spatioev.tl",
    "cross_ripleys_k_by_phenotype": "spatioev.tl",
    "cross_ripleys_k_all_pairs": "spatioev.tl",
    "morans_i": "spatioev.tl",
    "morans_i_by_image": "spatioev.tl",
    "local_morans_i": "spatioev.tl",
    "add_local_morans_i": "spatioev.tl",
    "cross_morans_i": "spatioev.tl",
    "cross_morans_i_by_image": "spatioev.tl",
    "local_cross_morans_i": "spatioev.tl",
    "add_local_cross_morans_i": "spatioev.tl",
    # tl — niche
    "cluster_spatial_niches": "spatioev.tl",
    "build_niche_boundaries": "spatioev.tl",
    "build_cell_graph": "spatioev.tl",
    "build_niche_feature_table": "spatioev.tl",
    "build_niche_feature_table_batched": "spatioev.tl",
    # tl — pseudotime
    "prepare_pseudotime_feature_matrix": "spatioev.tl",
    "infer_branch_labels": "spatioev.tl",
    "compute_feature_trend_table": "spatioev.tl",
    # tl — ECM
    "build_cell_fiber_links": "spatioev.tl",
    "cell_to_fiber_distance": "spatioev.tl",
    "cross_morans_i_ecm_cells": "spatioev.tl",
    "build_ecm_bipartite_graph_per_image": "spatioev.tl",
    "compute_invasion_score": "spatioev.tl",
    # tl — phenotype / ML
    "cluster_cells": "spatioev.tl",
    "scimap_napari_gater": "spatioev.tl",
    "scimap_rescale": "spatioev.tl",
    "scimap_phenotype_cells": "spatioev.tl",
    "run_scimap_prior_knowledge_phenotyping": "spatioev.tl",
    "subset_cells": "spatioev.tl",
    "annotate_interactive": "spatioev.tl",
    "annotate_from_csv": "spatioev.tl",
    "merge_annotations": "spatioev.tl",
    "merge_refinements": "spatioev.tl",
    "refine_clusters": "spatioev.tl",
    "build_feature_matrix": "spatioev.tl",
    "build_marker_features": "spatioev.tl",
    "build_morphology_features": "spatioev.tl",
    "train_svm_classifier": "spatioev.tl",
    "predict_svm": "spatioev.tl",
    "run_svm_phenotyping": "spatioev.tl",
    "inspect_reassigned_cells": "spatioev.tl",
    "inspect_disagreements": "spatioev.tl",
    # tl — niche (extra)
    "add_niche_regions_to_obs": "spatioev.tl",
    "assign_cells_to_niche_regions": "spatioev.tl",
    "buffer_niche_boundaries": "spatioev.tl",
    "estimate_density_adaptive_dbscan_params": "spatioev.tl",
    "cluster_spatial_components": "spatioev.tl",
    "cluster_spatial_components_from_mask": "spatioev.tl",
    "cluster_spatial_components_hdbscan": "spatioev.tl",
    "summarize_niche_surrounding_context": "spatioev.tl",
    "summarize_niche_graph_features": "spatioev.tl",
    "summarize_niche_composition": "spatioev.tl",
    # tl — spatial stats (extra)
    "ripley_local_counts_by_phenotype": "spatioev.tl",
    "cross_ripley_local_counts": "spatioev.tl",
    "cross_ripley_envelope_by_phenotype": "spatioev.tl",
    "cross_ripley_permutation_envelope": "spatioev.tl",
    "cross_ripleys_curve_by_phenotype": "spatioev.tl",
    "morans_i_by_image_permutation_test": "spatioev.tl",
    "add_local_morans_i_quadrants": "spatioev.tl",
    "cross_morans_i_feature_matrix": "spatioev.tl",
    "add_local_cross_morans_i_between_phenotypes": "spatioev.tl",
    "add_local_cross_morans_i_quadrants": "spatioev.tl",
    # tl — pseudotime (extra)
    "add_branch_time_bins": "spatioev.tl",
    "block_balance_feature_matrix": "spatioev.tl",
    "branch_time_feature_matrix": "spatioev.tl",
    "find_branch_transition_features": "spatioev.tl",
    "summarize_branches": "spatioev.tl",
    "assign_pseudotime_bins": "spatioev.tl",
    "project_tree_nodes_to_embedding": "spatioev.tl",
    # tl — epithelial/niche dynamics
    "compute_epithelial_centered_interaction_dynamics": "spatioev.tl",
    "summarize_epithelial_interaction_dynamics": "spatioev.tl",
    "score_pdac_niche_pathology_modules": "spatioev.tl",
    # tl — ECM (extra)
    "build_ecm_cell_neighborhood_features": "spatioev.tl",
    "build_nearest_cell_fiber_map": "spatioev.tl",
    "assign_niches_to_fibers": "spatioev.tl",
    # pl — all plotting functions
    "plot_area_distribution": "spatioev.pl",
    "plot_nc_ratio_distribution": "spatioev.pl",
    "plot_cluster_heatmap": "spatioev.pl",
    "inspect_clusters": "spatioev.pl",
    "spatial_scatter_plot": "spatioev.pl",
    "plot_refinement_umaps": "spatioev.pl",
    "plot_density_heatmap": "spatioev.pl",
    "plot_phenotype_density_heatmap": "spatioev.pl",
    "plot_density_correlation": "spatioev.pl",
    "plot_kde_density": "spatioev.pl",
    "plot_local_density_map": "spatioev.pl",
    "plot_interaction_density": "spatioev.pl",
    "plot_interaction_overlay": "spatioev.pl",
    "plot_interaction_distribution": "spatioev.pl",
    "plot_niche_boundaries": "spatioev.pl",
    "plot_spatial_category": "spatioev.pl",
    "plot_spatial_feature": "spatioev.pl",
    "plot_correlation_heatmap": "spatioev.pl",
    # xe
    "compute_marker_set_scores": "spatioev.xe",
    "score_xenium_histology_modules": "spatioev.xe",
    "add_module_scores": "spatioev.xe",
    "summarize_cluster_marker_scores": "spatioev.xe",
    "assign_labels_from_marker_rules": "spatioev.xe",
}

__all__ = sorted([*_SUBMODULES, *_EXPORTS, "__version__"])


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module

    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
