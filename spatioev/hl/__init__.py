"""Helper-function API for reusable SpatioEv building blocks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "compute_convex_hull": "spatioev.archive.spatial.preprocessing",
    "compute_convex_hull_area": "spatioev.archive.spatial.preprocessing",
    "distance_to_convex_hull_boundary": "spatioev.archive.spatial.preprocessing",
    "calculate_polarity_score": "spatioev.archive.spatial.cell_pixel_features",
    "calculate_moment_of_inertia": "spatioev.archive.spatial.cell_pixel_features",
    "calculate_haralick_features": "spatioev.archive.spatial.cell_pixel_features",
    "calculate_haralick_features_rescaled": "spatioev.archive.spatial.cell_pixel_features",
    "calculate_entropy": "spatioev.archive.spatial.cell_pixel_features",
    "calculate_lacunarity": "spatioev.archive.spatial.cell_pixel_features",
    "calculate_channel_correlation": "spatioev.archive.spatial.cell_pixel_features",
    "FeatureMatrixResult": "spatioev.archive.spatial.pseudotime",
    "zscore_series": "spatioev.archive.spatial.pseudotime",
    "minmax_scale": "spatioev.archive.spatial.pseudotime",
    "score_signed_feature_module": "spatioev.archive.spatial.pseudotime",
    "assign_feature_blocks": "spatioev.archive.spatial.pseudotime",
    "tree_edges": "spatioev.archive.spatial.pseudotime",
    "node_graph": "spatioev.archive.spatial.pseudotime",
    "project_tree_nodes_to_embedding": "spatioev.archive.spatial.pseudotime",
    "benjamini_hochberg": "spatioev.archive.spatial.pseudotime_trends",
    "add_branch_time_bins": "spatioev.archive.spatial.pseudotime_trends",
    "branch_time_feature_matrix": "spatioev.archive.spatial.pseudotime_trends",
    "find_branch_transition_features": "spatioev.archive.spatial.pseudotime_trends",
    "available_feature_map": "spatioev.archive.xenium.niche_features",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

