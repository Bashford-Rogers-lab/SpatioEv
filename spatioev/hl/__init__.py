"""Helper-function API for reusable SpatioEv building blocks.

These are the lower-level building blocks that the ``pp``/``tl``/``pl``
namespaces are composed from. They are re-exported here so that user code can
reach them under a single stable name without depending on which module
currently implements them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # geometry helpers -> spatioev.pp.spatial_prep
    "compute_convex_hull": "spatioev.pp.spatial_prep",
    "compute_convex_hull_area": "spatioev.pp.spatial_prep",
    "distance_to_convex_hull_boundary": "spatioev.pp.spatial_prep",
    # per-cell pixel/morphology features -> spatioev.pp.pixel
    "calculate_polarity_score": "spatioev.pp.pixel",
    "calculate_moment_of_inertia": "spatioev.pp.pixel",
    "calculate_haralick_features": "spatioev.pp.pixel",
    "calculate_haralick_features_rescaled": "spatioev.pp.pixel",
    "calculate_entropy": "spatioev.pp.pixel",
    "calculate_lacunarity": "spatioev.pp.pixel",
    "calculate_channel_correlation": "spatioev.pp.pixel",
    # pseudotime building blocks -> spatioev.tl.pseudotime
    "FeatureMatrixResult": "spatioev.tl.pseudotime",
    "zscore_series": "spatioev.tl.pseudotime",
    "minmax_scale": "spatioev.tl.pseudotime",
    "score_signed_feature_module": "spatioev.tl.pseudotime",
    "assign_feature_blocks": "spatioev.tl.pseudotime",
    "tree_edges": "spatioev.tl.pseudotime",
    "node_graph": "spatioev.tl.pseudotime",
    "project_tree_nodes_to_embedding": "spatioev.tl.pseudotime",
    "benjamini_hochberg": "spatioev.tl.pseudotime",
    "add_branch_time_bins": "spatioev.tl.pseudotime",
    "branch_time_feature_matrix": "spatioev.tl.pseudotime",
    "find_branch_transition_features": "spatioev.tl.pseudotime",
    # Xenium niche features -> spatioev.xe.features
    "available_feature_map": "spatioev.xe.features",
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
