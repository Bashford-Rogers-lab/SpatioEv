"""Shared internals for the spatial statistics submodules.

Private to :mod:`spatioev.tl.stats`; nothing here is public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shared implementations live in spatioev._core. They are re-exported under
# the existing private names so call sites across this sub-package are
# unchanged; __all__ marks them as deliberate re-exports.
from ..._core.coords import (
    clean_coords as _clean_coords,
)
from ..._core.neighbors import resolve_k as _resolve_k

__all__ = [
    "_clean_coords",
    "_clean_paired_coords",
    "_permute_values_within_groups",
    "_random_points_in_hull",
    "_resolve_k",
    "_resolve_window_area",
]
from scipy.spatial import ConvexHull

from ..preprocessing import compute_convex_hull_area

# Permutation simulations are evaluated in blocks to bound peak memory:
# the (n, block) value matrix is the only large temporary.
_PERMUTATION_BLOCK = 128


def _random_points_in_hull(coords, n_points):
    """
    Generate random points inside the convex hull of coordinates.

    Used for Monte Carlo envelope testing.
    """
    try:
        from shapely.geometry import Point, Polygon
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Ripley envelope simulations require shapely. Install SpatioEv "
            "with its core dependencies or `pip install shapely`."
        ) from exc

    hull = ConvexHull(coords)

    polygon = Polygon(coords[hull.vertices])

    minx, miny, maxx, maxy = polygon.bounds

    points = []

    while len(points) < n_points:

        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)

        if polygon.contains(Point(x, y)):
            points.append((x, y))

    return np.array(points)


def _permute_values_within_groups(values, groups, rng):
    """
    Permute values independently within each group.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)

    permuted = values.copy()

    for group in pd.unique(groups):
        idx = np.where(groups == group)[0]
        permuted[idx] = rng.permutation(values[idx])

    return permuted




def _clean_paired_coords(source_coords, target_coords):
    """
    Drop non-finite rows from two coordinate arrays independently.
    """
    return _clean_coords(source_coords), _clean_coords(target_coords)




def _resolve_window_area(coords, window_coords=None):
    """
    Compute the observation-window area used for Ripley statistics.

    If ``window_coords`` is provided, its convex hull defines the tissue
    window; otherwise the hull of ``coords`` is used.
    """
    if window_coords is None:
        return compute_convex_hull_area(coords)

    window_coords = _clean_coords(window_coords)
    if len(window_coords) < 3:
        return np.nan

    return compute_convex_hull_area(window_coords)


__all__ = [
    "_PERMUTATION_BLOCK",
    "_random_points_in_hull",
    "_permute_values_within_groups",
    "_clean_coords",
    "_clean_paired_coords",
    "_resolve_k",
    "_resolve_window_area",
]
