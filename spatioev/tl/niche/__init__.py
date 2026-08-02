"""Spatial niche detection, boundaries, graphs and features.

Split by analysis stage:

    boundaries  spatial component clustering and boundary geometry
    graph       cell-level spatial graph and niche subgraphs
    features    per-niche feature tables and pathology module scoring

All names remain importable directly from this package, so
``from spatioev.tl.niche import build_cell_graph`` is unchanged.
"""

from __future__ import annotations

from .boundaries import (
    add_niche_regions_to_obs,
    assign_cells_to_niche_regions,
    buffer_niche_boundaries,
    build_niche_boundaries,
    cluster_spatial_components,
    cluster_spatial_components_from_mask,
    cluster_spatial_components_hdbscan,
    cluster_spatial_niches,
    estimate_density_adaptive_dbscan_params,
    estimate_spatial_component_params,
    summarize_niche_composition,
)
from .features import (
    build_niche_feature_table,
    build_niche_feature_table_batched,
    score_pdac_niche_pathology_modules,
    summarize_niche_graph_features,
    summarize_niche_surrounding_context,
)
from .graph import (
    build_cell_graph,
    extract_all_niche_subgraphs,
    extract_niche_subgraph,
)

__all__ = [
    "estimate_density_adaptive_dbscan_params",
    "estimate_spatial_component_params",
    "cluster_spatial_niches",
    "cluster_spatial_components",
    "cluster_spatial_components_hdbscan",
    "cluster_spatial_components_from_mask",
    "build_niche_boundaries",
    "buffer_niche_boundaries",
    "assign_cells_to_niche_regions",
    "summarize_niche_composition",
    "add_niche_regions_to_obs",
    "build_cell_graph",
    "extract_niche_subgraph",
    "extract_all_niche_subgraphs",
    "summarize_niche_graph_features",
    "build_niche_feature_table",
    "build_niche_feature_table_batched",
    "summarize_niche_surrounding_context",
    "score_pdac_niche_pathology_modules",
]
