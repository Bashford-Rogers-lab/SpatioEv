"""ECM (extracellular matrix) analysis.

Split by analysis stage:

    links          cell-to-fiber adjacency
    proximity      distance, fiber density, cross-type Ripley K
    moran          Moran's I over fibers and ECM-cell coupling
    regression     spatial regression, enrichment, fiber orientation
    graph          bipartite ECM graph, niche detection, invasion score
    neighborhoods  ECM-cell neighborhood features and clustering

All names remain importable directly from this package, so
``from spatioev.tl.ecm import build_cell_fiber_links`` is unchanged.
"""

from __future__ import annotations

# Canonical implementation lives in pp.spatial_prep; re-exported here because
# it has always been part of this module's public surface.
from ..preprocessing import compute_convex_hull_area
from .graph import (
    assign_niches_to_fibers,
    build_ecm_bipartite_graph_per_image,
    compute_invasion_score,
    detect_ecm_niches_per_image,
    project_fiber_graph_per_image,
)
from .links import (
    build_cell_fiber_links,
    build_nearest_cell_fiber_map,
)
from .moran import (
    cross_morans_i_ecm_cells,
    cross_morans_i_ecm_cells_permutation_test,
    local_cross_morans_i_ecm_cells,
    local_morans_i_fibers,
    map_cells_to_fibers,
    map_cells_to_fibers_kernel,
    morans_i_fibers,
)
from .neighborhoods import (
    add_neighborhoods_to_obs,
    build_ecm_cell_neighborhood_features,
    cluster_ecm_cell_neighborhoods,
    default_ecm_cell_neighborhood_feature_columns,
    detect_tissue_regions_dbscan,
    filter_ecm_cell_inputs_by_tissue,
    score_col6_dark_neighborhoods,
    summarize_ecm_cell_neighborhoods,
)
from .proximity import (
    cell_to_fiber_distance,
    cross_ripleys_k,
    cross_ripleys_k_permutation_envelope,
    fiber_density_near_cells,
)
from .regression import (
    cell_fiber_alignment,
    fiber_vectors,
    spatial_enrichment_score,
    spatial_linear_regression,
    spatial_mixed_model,
)

__all__ = [
    "build_cell_fiber_links",
    "build_nearest_cell_fiber_map",
    "cell_to_fiber_distance",
    "fiber_density_near_cells",
    "morans_i_fibers",
    "local_morans_i_fibers",
    "cross_morans_i_ecm_cells",
    "cross_morans_i_ecm_cells_permutation_test",
    "map_cells_to_fibers",
    "map_cells_to_fibers_kernel",
    "local_cross_morans_i_ecm_cells",
    "spatial_linear_regression",
    "spatial_mixed_model",
    "spatial_enrichment_score",
    "fiber_vectors",
    "cell_fiber_alignment",
    "cross_ripleys_k_permutation_envelope",
    "build_ecm_bipartite_graph_per_image",
    "project_fiber_graph_per_image",
    "detect_ecm_niches_per_image",
    "assign_niches_to_fibers",
    "compute_invasion_score",
    "detect_tissue_regions_dbscan",
    "filter_ecm_cell_inputs_by_tissue",
    "build_ecm_cell_neighborhood_features",
    "default_ecm_cell_neighborhood_feature_columns",
    "cluster_ecm_cell_neighborhoods",
    "summarize_ecm_cell_neighborhoods",
    "score_col6_dark_neighborhoods",
    "add_neighborhoods_to_obs",
    "compute_convex_hull_area",
    "cross_ripleys_k",
]
