"""Thin re-export shim for geometry helpers used by tl.stats and tl.niche.

The canonical implementations live in ``spatioev.pp.spatial_prep``.
This module exists so that internal ``from .preprocessing import ...``
statements inside the ``tl`` sub-package resolve correctly without
circular imports.
"""

from spatioev.pp.spatial_prep import (
    compute_convex_hull,
    compute_convex_hull_area,
)

__all__ = [
    "compute_convex_hull",
    "compute_convex_hull_area",
]
