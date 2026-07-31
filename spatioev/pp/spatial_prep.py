from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    import anndata as ad


def validate_spatial_coordinates(
    adata: ad.AnnData,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    image_key: str = "imageid",
    allow_negative: bool = False,
    copy: bool = False,
) -> ad.AnnData:
    """
    Validate required spatial coordinate columns and basic integrity.

    Returns
    -------
    AnnData
        Original or copied AnnData with no modification unless copy=True.
    """
    if copy:
        adata = adata.copy()

    required = [x_key, y_key, image_key]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    x = adata.obs[x_key]
    y = adata.obs[y_key]

    if x.isna().any() or y.isna().any():
        raise ValueError("Spatial coordinates contain NaN values.")

    if not np.issubdtype(x.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("Spatial coordinates must be numeric.")

    if not allow_negative:
        if (x < 0).any() or (y < 0).any():
            raise ValueError("Negative spatial coordinates detected.")

    if adata.obs[image_key].isna().any():
        raise ValueError(f"{image_key} contains NaN values.")

    return adata


def compute_convex_hull(coords: np.ndarray) -> ConvexHull | None:
    """
    Compute convex hull from Nx2 coordinates.

    Returns
    -------
    hull : scipy.spatial.ConvexHull or None
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be an Nx2 array.")
    # Drop non-finite rows; scipy's Qhull raises on NaN/inf input.
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.shape[0] < 3:
        return None
    return ConvexHull(coords)


def compute_convex_hull_area(coords: np.ndarray) -> float:
    """
    Compute convex hull area from Nx2 coordinates.
    In 2D, scipy ConvexHull.volume is the enclosed area.
    """
    hull = compute_convex_hull(coords)
    if hull is None:
        return np.nan
    return float(hull.volume)


def compute_tissue_areas(
    adata: ad.AnnData,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    image_key: str = "imageid",
) -> pd.DataFrame:
    """
    Compute convex-hull tissue area for each image.

    Returns
    -------
    pd.DataFrame
        Columns: imageid, tissue_area
    """
    rows = []
    for img in adata.obs[image_key].unique():
        df = adata.obs.loc[adata.obs[image_key] == img, [x_key, y_key]]
        coords = df.to_numpy()
        area = compute_convex_hull_area(coords)
        rows.append({image_key: img, "tissue_area": area})

    out = pd.DataFrame(rows)

    if "spatial_stats" not in adata.uns:
        adata.uns["spatial_stats"] = {}
    adata.uns["spatial_stats"]["tissue_areas"] = out.copy()

    return out


def _point_to_segment_distance(points, a, b):
    """
    Distance from points (Nx2) to line segment ab.
    """
    ap = points - a
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 == 0:
        return np.linalg.norm(ap, axis=1)

    t = np.clip((ap @ ab) / ab2, 0, 1)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(points - proj, axis=1)


def distance_to_convex_hull_boundary(coords: np.ndarray) -> np.ndarray:
    """
    Compute minimum distance from each point to convex hull boundary.
    """
    coords = np.asarray(coords)
    hull = compute_convex_hull(coords)
    if hull is None:
        return np.full(coords.shape[0], np.nan)

    hull_pts = coords[hull.vertices]
    n = hull_pts.shape[0]

    dists = np.full(coords.shape[0], np.inf)
    for i in range(n):
        a = hull_pts[i]
        b = hull_pts[(i + 1) % n]
        seg_dist = _point_to_segment_distance(coords, a, b)
        dists = np.minimum(dists, seg_dist)

    return dists


def detect_edge_cells(
    adata: ad.AnnData,
    radius: float,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    image_key: str = "imageid",
    edge_key: str = "edge_cell",
    boundary_dist_key: str = "distance_to_boundary",
) -> ad.AnnData:
    """
    Detect cells whose radius-neighborhood intersects the convex hull boundary.

    Cells with distance_to_boundary < radius are flagged as edge cells.
    """
    adata.obs[edge_key] = False
    adata.obs[boundary_dist_key] = np.nan

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]
        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()

        dists = distance_to_convex_hull_boundary(coords)
        adata.obs.loc[idx, boundary_dist_key] = dists
        adata.obs.loc[idx, edge_key] = dists < radius

    return adata


__all__ = [
    "validate_spatial_coordinates",
    "compute_convex_hull",
    "compute_convex_hull_area",
    "distance_to_convex_hull_boundary",
    "compute_tissue_areas",
    "detect_edge_cells",
]
