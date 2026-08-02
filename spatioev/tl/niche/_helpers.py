"""Shared internals for the niche submodules (private)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Shared implementations live in spatioev._core. They are re-exported under
# the existing private names so call sites across this sub-package are
# unchanged; __all__ marks them as deliberate re-exports.
from ..._core.coords import (
    ensure_list as _ensure_list,
)
from ..._core.coords import (
    label_suffix as _label_suffix,
)
from ..._core.neighbors import resolve_k as _resolve_k

__all__ = [
    "_ensure_list",
    "_estimate_min_samples",
    "_label_suffix",
    "_make_component_geometry",
    "_make_density_mask_geometry",
    "_require_shapely",
    "_resolve_k",
]

if TYPE_CHECKING:  # pragma: no cover
    pass


from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
)

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:  # pragma: no cover - depends on sklearn version
    SklearnHDBSCAN = None



def _require_shapely():
    try:
        from shapely import concave_hull
        from shapely.geometry import LineString, MultiPoint, Point, box
        from shapely.ops import unary_union
        from shapely.validation import make_valid
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Niche boundary geometry functions require Shapely. "
            "Install it with `pip install shapely` or use the non-geometry "
            "niche graph/statistics helpers that do not require Shapely."
        ) from exc

    return concave_hull, LineString, MultiPoint, Point, box, unary_union, make_valid








def _estimate_min_samples(
    n_cells,
    k_eff,
    min_samples_base=1,
    min_samples_mode="capped_knn",
    min_samples_max=10,
):
    """
    Estimate a practical DBSCAN ``min_samples`` for spatial cell components.
    """
    if min_samples_mode == "sqrt":
        return int(max(min_samples_base, np.ceil(np.sqrt(n_cells))))

    if min_samples_mode == "singleton":
        return 1

    if k_eff is None:
        return int(min_samples_base)

    # This mirrors the original tumor-component workflow more closely:
    # small core neighborhoods, capped so large images do not force
    # unrealistically large min_samples values.
    return int(max(min_samples_base, min(min_samples_max, 2 * k_eff)))


def _make_component_geometry(
    coords,
    method="concave_hull",
    concavity=0.3,
    allow_holes=False,
    point_buffer=10.0,
    line_buffer=10.0,
    mask_resolution=5.0,
    mask_sigma=1.0,
    mask_threshold=0.15,
    mask_closing_size=3,
):
    """
    Build a robust geometry for one spatial component.
    """
    concave_hull, LineString, MultiPoint, Point, _box, _unary_union, make_valid = _require_shapely()

    coords = np.asarray(coords, dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]

    if len(coords) == 0:
        return None

    if len(coords) == 1:
        geom = Point(coords[0]).buffer(point_buffer)
    elif len(coords) == 2:
        geom = LineString(coords).buffer(line_buffer)
    elif method == "density_mask":
        geom = _make_density_mask_geometry(
            coords,
            resolution=mask_resolution,
            sigma=mask_sigma,
            threshold=mask_threshold,
            closing_size=mask_closing_size,
        )
    else:
        geom = concave_hull(
            MultiPoint(coords),
            ratio=concavity,
            allow_holes=allow_holes,
        )

    if geom is None or geom.is_empty:
        return None

    return make_valid(geom)


def _make_density_mask_geometry(
    coords,
    resolution=5.0,
    sigma=1.0,
    threshold=0.15,
    closing_size=3,
):
    """
    Build a component boundary from a rasterized density mask.

    This method tends to preserve fissures and abrupt shape changes better
    than a single concave hull.
    """
    _concave_hull, _LineString, _MultiPoint, _Point, box, unary_union, make_valid = _require_shapely()

    coords = np.asarray(coords, dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]

    if len(coords) < 3:
        return None

    minx, miny = coords.min(axis=0)
    maxx, maxy = coords.max(axis=0)

    pad = max(resolution * 2, 1.0)
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    nx = max(3, int(np.ceil((maxx - minx) / resolution)) + 1)
    ny = max(3, int(np.ceil((maxy - miny) / resolution)) + 1)

    grid = np.zeros((ny, nx), dtype=float)

    x_idx = np.clip(((coords[:, 0] - minx) / resolution).astype(int), 0, nx - 1)
    y_idx = np.clip(((coords[:, 1] - miny) / resolution).astype(int), 0, ny - 1)
    np.add.at(grid, (y_idx, x_idx), 1.0)

    smooth = gaussian_filter(grid, sigma=sigma)

    if smooth.max() <= 0:
        return None

    level = float(smooth.max() * threshold)
    mask = smooth >= level
    mask = binary_fill_holes(mask)

    if closing_size and closing_size > 1:
        structure = np.ones((closing_size, closing_size), dtype=bool)
        mask = binary_closing(mask, structure=structure)
        mask = binary_fill_holes(mask)

    pixels = np.argwhere(mask)

    if len(pixels) == 0:
        return None

    pixel_boxes = []
    for row, col in pixels:
        x0 = minx + col * resolution
        y0 = miny + row * resolution
        pixel_boxes.append(box(x0, y0, x0 + resolution, y0 + resolution))

    geom = unary_union(pixel_boxes)

    if geom.is_empty:
        return None

    # Light smoothing to reduce pixelation while preserving topology.
    geom = geom.buffer(resolution * 0.5).buffer(-resolution * 0.5)

    if geom.is_empty:
        return None

    return make_valid(geom)
