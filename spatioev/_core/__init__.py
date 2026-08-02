"""Shared internal building blocks.

Private to SpatioEv: nothing here is part of the public API. These helpers
exist so that the analysis modules stop re-deriving the same spatial
primitives (neighbour graphs, weight matrices, coordinate extraction) in
every function.
"""

from .coords import (
    clean_coords,
    ensure_list,
    get_coords,
    label_suffix,
    per_image,
    per_image_table,
    require_obs_columns,
)
from .neighbors import knn_weights, resolve_k
from .optional import require

__all__ = [
    "clean_coords",
    "ensure_list",
    "get_coords",
    "knn_weights",
    "label_suffix",
    "per_image",
    "per_image_table",
    "require",
    "require_obs_columns",
    "resolve_k",
]
