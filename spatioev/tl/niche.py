# ============================================================
# Section 1: Niche boundaries  (from archive/spatial/spatial_niche_boundaries.py)
# ============================================================

"""
Spatial niche boundary module.

This module turns the notebook-style tumor component workflow into reusable
functions for:

1. clustering same-type cells within each image
2. adapting DBSCAN parameters to the density of the target population
3. building concave-hull niche boundaries
4. expanding or shrinking boundaries to define core and border regions
5. summarizing niche composition for downstream heterogeneity analyses

Typical use cases
-----------------
tumor-cell components
neighborhood-specific compartments
intra-tumor heterogeneity
intra-neighborhood heterogeneity
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad

import os
import re

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:  # pragma: no cover - depends on sklearn version
    SklearnHDBSCAN = None
import networkx as nx
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    generate_binary_structure,
)
from scipy.ndimage import (
    find_objects as ndi_find_objects,
)
from scipy.ndimage import (
    label as ndi_label,
)
from scipy.sparse import coo_matrix, csr_matrix, triu
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
from skimage.io import imread
from skimage.morphology import disk
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .preprocessing import compute_convex_hull_area


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


def _ensure_list(value):
    """
    Normalize a scalar or iterable label selection to a list.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    return list(value)


def _label_suffix(value, default):
    """
    Build a stable suffix for derived column names.
    """
    value = _ensure_list(value)

    if value is None or len(value) == 0:
        return default

    return "_".join(map(str, value))


def _resolve_k(n_obs, k):
    """
    Ensure nearest-neighbor queries use a valid number of neighbors.
    """
    if n_obs < 2:
        return None

    return min(k, n_obs - 1)


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


def estimate_density_adaptive_dbscan_params(
    adata: ad.AnnData,
    label_key: str,
    label_value: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    knn_k: int=5,
    eps_quantile: float=0.75,
    eps_scale: float=1.0,
    min_samples_base: int=1,
    min_samples_mode: str="capped_knn",
    min_samples_max: int=10,
) -> dict:
    """
    Estimate image-specific DBSCAN parameters from the density of the target cells.

    Parameters
    ----------
    label_key : str
        Column in ``adata.obs`` used to select the cells of interest.
        This can be a phenotype column or a neighborhood-identity column.
    label_value : str or list
        Target label or labels to cluster.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key, y_key : str
        Column names in ``adata.obs`` containing cell coordinates.
    knn_k : int
        Number of same-type neighbors used to estimate local spacing.
    eps_quantile : float
        Quantile of the kNN-distance distribution used as the base DBSCAN ``eps``.
    eps_scale : float
        Multiplier applied to the estimated ``eps``.
    min_samples_base : int
        Lower bound for DBSCAN ``min_samples``.
    min_samples_mode : {"capped_knn", "sqrt", "singleton"}
        Strategy used to estimate ``min_samples``.
        ``"capped_knn"`` is the most similar to the original tumor notebook.
    min_samples_max : int
        Upper cap used when ``min_samples_mode="capped_knn"``.

    Returns
    -------
    DataFrame
        One row per image with estimated ``eps`` and ``min_samples``.
    """
    label_value = _ensure_list(label_value)

    obs = adata.obs.copy()
    mask = obs[label_key].isin(label_value)
    target = obs.loc[mask, [image_key, x_key, y_key]].copy()

    rows = []

    for img in target[image_key].dropna().unique():
        sub = target[target[image_key] == img].copy()
        coords = sub[[x_key, y_key]].to_numpy(dtype=float)
        coords = coords[np.isfinite(coords).all(axis=1)]

        n_cells = len(coords)

        if n_cells < 2:
            rows.append({
                image_key: img,
                "n_cells": n_cells,
                "knn_k": np.nan,
                "knn_distance_median": np.nan,
                "knn_distance_quantile": np.nan,
                "eps": np.nan,
                "min_samples": np.nan,
            })
            continue

        k_eff = _resolve_k(n_cells, knn_k)

        nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
        nbrs.fit(coords)
        distances, _ = nbrs.kneighbors(coords)

        # drop self-distance in column 0
        kth_dist = distances[:, -1]
        eps = float(np.quantile(kth_dist, eps_quantile) * eps_scale)
        min_samples = _estimate_min_samples(
            n_cells=n_cells,
            k_eff=k_eff,
            min_samples_base=min_samples_base,
            min_samples_mode=min_samples_mode,
            min_samples_max=min_samples_max,
        )

        rows.append({
            image_key: img,
            "n_cells": n_cells,
            "knn_k": k_eff,
            "knn_distance_median": float(np.median(kth_dist)),
            "knn_distance_quantile": float(np.quantile(kth_dist, eps_quantile)),
            "eps": eps,
            "min_samples": min_samples,
        })

    return pd.DataFrame(rows)


def estimate_spatial_component_params(
    adata: ad.AnnData,
    label_key: str,
    label_value: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    knn_k: int=1,
    radius_quantile: float=0.9,
    radius_scale: float=1.0,
) -> dict:
    """
    Estimate image-specific connection radii for graph connected components.

    The radius is derived from within-class kNN distances and is meant to
    approximate the maximum gap still considered spatially contiguous.
    """
    label_value = _ensure_list(label_value)

    obs = adata.obs.copy()
    mask = obs[label_key].isin(label_value)
    target = obs.loc[mask, [image_key, x_key, y_key]].copy()

    rows = []

    for img in target[image_key].dropna().unique():
        sub = target[target[image_key] == img].copy()
        coords = sub[[x_key, y_key]].to_numpy(dtype=float)
        coords = coords[np.isfinite(coords).all(axis=1)]

        n_cells = len(coords)

        if n_cells < 2:
            rows.append({
                image_key: img,
                "n_cells": n_cells,
                "knn_k": np.nan,
                "knn_distance_median": np.nan,
                "knn_distance_quantile": np.nan,
                "radius": np.nan,
            })
            continue

        k_eff = _resolve_k(n_cells, knn_k)
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
        nbrs.fit(coords)
        distances, _ = nbrs.kneighbors(coords)

        kth_dist = distances[:, -1]
        radius = float(np.quantile(kth_dist, radius_quantile) * radius_scale)

        rows.append({
            image_key: img,
            "n_cells": n_cells,
            "knn_k": k_eff,
            "knn_distance_median": float(np.median(kth_dist)),
            "knn_distance_quantile": float(np.quantile(kth_dist, radius_quantile)),
            "radius": radius,
        })

    return pd.DataFrame(rows)


def cluster_spatial_niches(
    adata: ad.AnnData,
    label_key: str,
    label_value: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    component_key: str=None,
    target_key: str=None,
    eps: float=None,
    min_samples: int=None,
    knn_k: int=5,
    eps_quantile: float=0.75,
    eps_scale: float=1.0,
    min_samples_base: int=1,
    min_samples_mode: str="capped_knn",
    min_samples_max: int=10,
    assign_singletons: bool=True,
) -> ad.AnnData:
    """
    Cluster target cells within each image using DBSCAN.

    If ``eps`` or ``min_samples`` is not provided, they are estimated from
    the density of the selected cell population in each image.

    Parameters
    ----------
    label_key : str
        Column in ``adata.obs`` used to select the cells of interest.
        This can be a phenotype column or a neighborhood-identity column.
    label_value : str or list
        Target label or labels to cluster.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key, y_key : str
        Column names in ``adata.obs`` containing cell coordinates.
    component_key : str, optional
        Output column for component ids. If not provided, a name is derived automatically.
    target_key : str, optional
        Output boolean column marking whether each cell belongs to the selected target group.
    eps : float, optional
        DBSCAN ``eps``. If ``None``, estimated separately for each image.
    min_samples : int, optional
        DBSCAN ``min_samples``. If ``None``, estimated separately for each image.
    knn_k, eps_quantile, eps_scale, min_samples_base
        Parameters controlling the density-adaptive estimation.
    min_samples_mode : {"capped_knn", "sqrt", "singleton"}
        Strategy used when estimating ``min_samples``.
        ``"capped_knn"`` is more compatible with the original tumor component notebook.
    min_samples_max : int
        Upper cap used when ``min_samples_mode="capped_knn"``.
    assign_singletons : bool
        If ``True``, noise points receive unique singleton component ids instead of ``noise``.
        This is close to the original tumor workflow, which often used ``min_samples=1``.

    Returns
    -------
    AnnData
        Copy of ``adata`` with component annotations added to ``obs``.
    """
    label_value = _ensure_list(label_value)
    suffix = _label_suffix(label_value, "target")

    if component_key is None:
        component_key = f"{suffix}_component"

    if target_key is None:
        target_key = f"is_{suffix}"

    out = adata.copy()
    out.obs[target_key] = out.obs[label_key].isin(label_value)
    out.obs[component_key] = "unassigned"

    param_df = estimate_density_adaptive_dbscan_params(
        out,
        label_key=label_key,
        label_value=label_value,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        knn_k=knn_k,
        eps_quantile=eps_quantile,
        eps_scale=eps_scale,
        min_samples_base=min_samples_base,
        min_samples_mode=min_samples_mode,
        min_samples_max=min_samples_max,
    )

    params = param_df.set_index(image_key).to_dict(orient="index")

    for img in out.obs[image_key].dropna().unique():
        img_mask = out.obs[image_key] == img
        target_mask = img_mask & out.obs[target_key]
        sub = out.obs.loc[target_mask, [x_key, y_key]].copy()
        sub = sub[np.isfinite(sub[[x_key, y_key]]).all(axis=1)]

        if sub.empty:
            continue

        coords = sub[[x_key, y_key]].to_numpy(dtype=float)

        img_eps = eps
        if img_eps is None:
            img_eps = params.get(img, {}).get("eps", np.nan)

        img_min_samples = min_samples
        if img_min_samples is None:
            img_min_samples = params.get(img, {}).get("min_samples", np.nan)

        if not np.isfinite(img_eps) or not np.isfinite(img_min_samples):
            continue

        clustering = DBSCAN(
            eps=float(img_eps),
            min_samples=int(img_min_samples),
        ).fit(coords)

        labels = clustering.labels_.copy()
        component_labels = np.full(len(labels), "noise", dtype=object)

        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if label == -1:
                if assign_singletons:
                    for j, local_idx in enumerate(idx):
                        component_labels[local_idx] = f"{img}__singleton_{j}"
                continue

            component_labels[idx] = f"{img}__component_{label}"

        out.obs.loc[sub.index, component_key] = component_labels

    if "spatial_niches" not in out.uns:
        out.uns["spatial_niches"] = {}

    out.uns["spatial_niches"][f"{component_key}_params"] = param_df.copy()

    return out


def cluster_spatial_components(
    adata: ad.AnnData,
    label_key: str,
    label_value: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    component_key: str=None,
    target_key: str=None,
    radius: float=None,
    knn_k: int=1,
    radius_quantile: float=0.9,
    radius_scale: float=1.0,
    min_component_size: int=1,
    assign_singletons: bool=True,
) -> ad.AnnData:
    """
    Cluster target cells within each image by spatial graph connected components.

    Two target cells belong to the same component when they are connected by a
    chain of within-radius neighbors. This is often a better match to
    "separate non-contiguous region = separate gland" than density clustering.

    Returns
    -------
    AnnData
        Copy of ``adata`` with component annotations added to ``obs``.
    """
    label_value = _ensure_list(label_value)
    suffix = _label_suffix(label_value, "target")

    if component_key is None:
        component_key = f"{suffix}_component"

    if target_key is None:
        target_key = f"is_{suffix}"

    out = adata.copy()
    out.obs[target_key] = out.obs[label_key].isin(label_value)
    out.obs[component_key] = "unassigned"

    param_df = estimate_spatial_component_params(
        out,
        label_key=label_key,
        label_value=label_value,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        knn_k=knn_k,
        radius_quantile=radius_quantile,
        radius_scale=radius_scale,
    )

    params = param_df.set_index(image_key).to_dict(orient="index")

    for img in out.obs[image_key].dropna().unique():
        img_mask = out.obs[image_key] == img
        target_mask = img_mask & out.obs[target_key]
        sub = out.obs.loc[target_mask, [x_key, y_key]].copy()
        sub = sub[np.isfinite(sub[[x_key, y_key]]).all(axis=1)]

        if sub.empty:
            continue

        coords = sub[[x_key, y_key]].to_numpy(dtype=float)

        img_radius = radius
        if img_radius is None:
            img_radius = params.get(img, {}).get("radius", np.nan)

        if not np.isfinite(img_radius) or img_radius <= 0:
            continue

        if len(coords) == 1:
            if assign_singletons and min_component_size <= 1:
                out.obs.loc[sub.index, component_key] = f"{img}__singleton_0"
            continue

        nbrs = NearestNeighbors(radius=float(img_radius))
        nbrs.fit(coords)
        graph = nbrs.radius_neighbors_graph(coords, mode="connectivity")

        n_components, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

        component_labels = np.full(len(labels), "noise", dtype=object)

        singleton_counter = 0
        component_counter = 0
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]

            if len(idx) < int(min_component_size):
                if assign_singletons:
                    for local_idx in idx:
                        component_labels[local_idx] = f"{img}__singleton_{singleton_counter}"
                        singleton_counter += 1
                continue

            component_labels[idx] = f"{img}__component_{component_counter}"
            component_counter += 1

        out.obs.loc[sub.index, component_key] = component_labels

    if "spatial_niches" not in out.uns:
        out.uns["spatial_niches"] = {}

    out.uns["spatial_niches"][f"{component_key}_params"] = param_df.copy()

    return out


def cluster_spatial_components_hdbscan(
    adata: ad.AnnData,
    label_key: str,
    label_value: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    component_key: str=None,
    target_key: str=None,
    min_cluster_size: int=5,
    min_samples: int=None,
    cluster_selection_epsilon: float=0.0,
    cluster_selection_method: str="eom",
    allow_single_cluster: bool=False,
    assign_singletons: bool=False,
    probability_key: str=None,
) -> ad.AnnData:
    """
    Cluster target cells within each image using HDBSCAN.

    HDBSCAN can be a better fit than fixed-radius connected components when
    the target population spans multiple local densities within the same
    image. Instead of relying on a single connection radius, it finds stable
    dense clusters and can leave bridge-like or weakly supported cells as
    ``noise``.

    Parameters
    ----------
    label_key : str
        Column in ``adata.obs`` used to select the cells of interest.
    label_value : str or list
        Target label or labels to cluster.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key, y_key : str
        Column names in ``adata.obs`` containing cell coordinates.
    component_key : str, optional
        Output column for cluster ids. If not provided, a method-specific
        name is derived automatically.
    target_key : str, optional
        Output boolean column marking whether each cell belongs to the
        selected target group.
    min_cluster_size : int
        Minimum number of target cells needed to form a stable cluster.
    min_samples : int, optional
        HDBSCAN conservativeness parameter. Larger values tend to mark more
        cells as noise and suppress weak bridges.
    cluster_selection_epsilon : float
        Additional cluster-selection tolerance in coordinate units.
    cluster_selection_method : {"eom", "leaf"}
        HDBSCAN cluster selection strategy.
    allow_single_cluster : bool
        Whether HDBSCAN may return one cluster spanning the whole image.
    assign_singletons : bool
        If ``True``, cells labeled as HDBSCAN noise receive unique singleton
        component ids instead of ``noise``.
    probability_key : str, optional
        If provided, store HDBSCAN membership probabilities in this column.

    Returns
    -------
    AnnData
        Copy of ``adata`` with HDBSCAN component annotations added to ``obs``.
    """
    if SklearnHDBSCAN is None:
        raise ImportError(
            "sklearn.cluster.HDBSCAN is not available in this environment. "
            "Please install a compatible scikit-learn version."
        )

    label_value = _ensure_list(label_value)
    suffix = _label_suffix(label_value, "target")

    if component_key is None:
        component_key = f"{suffix}_hdbscan_component"

    if target_key is None:
        target_key = f"is_{suffix}"

    out = adata.copy()
    out.obs[target_key] = out.obs[label_key].isin(label_value)
    out.obs[component_key] = "unassigned"

    if probability_key is not None:
        out.obs[probability_key] = np.nan

    rows = []

    for img in out.obs[image_key].dropna().unique():
        img_mask = out.obs[image_key] == img
        target_mask = img_mask & out.obs[target_key]
        sub = out.obs.loc[target_mask, [x_key, y_key]].copy()
        sub = sub[np.isfinite(sub[[x_key, y_key]]).all(axis=1)]

        n_cells = len(sub)
        row = {
            image_key: img,
            "n_cells": int(n_cells),
            "min_cluster_size": int(min_cluster_size),
            "min_samples": np.nan if min_samples is None else int(min_samples),
            "cluster_selection_epsilon": float(cluster_selection_epsilon),
            "cluster_selection_method": cluster_selection_method,
            "allow_single_cluster": bool(allow_single_cluster),
            "n_components": 0,
            "n_noise": 0,
            "median_probability": np.nan,
        }

        if sub.empty:
            rows.append(row)
            continue

        if n_cells == 1:
            if assign_singletons:
                out.obs.loc[sub.index, component_key] = f"{img}__singleton_0"
                if probability_key is not None:
                    out.obs.loc[sub.index, probability_key] = 1.0
                row["n_noise"] = 0
            else:
                out.obs.loc[sub.index, component_key] = "noise"
                if probability_key is not None:
                    out.obs.loc[sub.index, probability_key] = 0.0
                row["n_noise"] = 1
            rows.append(row)
            continue

        coords = sub[[x_key, y_key]].to_numpy(dtype=float)
        clustering = SklearnHDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if min_samples is None else int(min_samples),
            cluster_selection_epsilon=float(cluster_selection_epsilon),
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
        ).fit(coords)

        labels = clustering.labels_.copy()
        probabilities = getattr(clustering, "probabilities_", None)
        component_labels = np.full(len(labels), "noise", dtype=object)

        singleton_counter = 0
        component_counter = 0
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if label == -1:
                if assign_singletons:
                    for local_idx in idx:
                        component_labels[local_idx] = f"{img}__singleton_{singleton_counter}"
                        singleton_counter += 1
                continue

            component_labels[idx] = f"{img}__component_{component_counter}"
            component_counter += 1

        out.obs.loc[sub.index, component_key] = component_labels

        if probability_key is not None and probabilities is not None:
            out.obs.loc[sub.index, probability_key] = probabilities

        row["n_components"] = int(np.sum(np.unique(labels) >= 0))
        row["n_noise"] = int(np.sum(labels == -1))
        if probabilities is not None and len(probabilities) > 0:
            row["median_probability"] = float(np.median(probabilities))
        rows.append(row)

    param_df = pd.DataFrame(rows)

    if "spatial_niches" not in out.uns:
        out.uns["spatial_niches"] = {}

    out.uns["spatial_niches"][f"{component_key}_params"] = param_df.copy()

    return out


def cluster_spatial_components_from_mask(
    adata: ad.AnnData,
    seg_dir: str,
    label_key: str,
    label_value: str,
    fov_key: str="fov",
    cell_label_key: str="label",
    component_key: str=None,
    target_key: str=None,
    seg_suffix: str="_whole_cell.tiff",
    dilation_radius: int=0,
    closing_radius: int=0,
    fill_holes: bool=False,
    connectivity: int=2,
    connection_mode: str="union_mask",
    gap_tolerance: int=0,
    stitch_across_fovs: bool=False,
    fov_grid_cols: int | None=None,
    stitch_gap_tolerance: int=None,
    min_component_size: int=1,
    assign_singletons: bool=True,
) -> ad.AnnData:
    """
    Cluster target cells within each FOV by connected components on a binary
    segmentation-derived mask.

    This approach uses the actual segmentation topology rather than centroid
    spacing. Cells are first selected by phenotype in ``adata.obs`` and mapped
    back to the whole-cell segmentation using ``fov_key`` + ``cell_label_key``.
    A binary mask is built from those selected labels, optional morphology is
    applied, and connected components are extracted from the resulting mask.

    Parameters
    ----------
    seg_dir : str
        Directory containing per-FOV whole-cell segmentation masks.
    label_key : str
        Column in ``adata.obs`` used to select the cells of interest.
    label_value : str or list
        Target label or labels to cluster.
    fov_key : str
        Column in ``adata.obs`` linking cells to segmentation files. Files are
        expected to follow ``f"{fov}{seg_suffix}"``.
    cell_label_key : str
        Column in ``adata.obs`` containing the integer segmentation label id.
    component_key : str, optional
        Output column for component ids.
    target_key : str, optional
        Output boolean column marking whether each cell belongs to the selected
        target group.
    seg_suffix : str
        Filename suffix appended to each FOV id.
    dilation_radius : int
        Optional binary dilation radius applied before component labeling.
    closing_radius : int
        Optional binary closing radius applied before component labeling.
    fill_holes : bool
        Whether to fill holes after morphology.
    connectivity : {1, 2}
        Pixel connectivity used for connected components. ``1`` gives 4-connectivity,
        ``2`` gives 8-connectivity.
    connection_mode : {"union_mask", "label_adjacency"}
        Strategy used to determine whether selected cells belong to the same
        component.
        ``"union_mask"`` uses connected components on the merged binary mask.
        ``"label_adjacency"`` is better suited to Mesmer-style instance masks
        and connects labels when they directly touch or are within
        ``gap_tolerance`` pixels.
    gap_tolerance : int
        Only used when ``connection_mode="label_adjacency"``.
        ``0`` means direct-touch adjacency only. ``1`` allows a one-pixel gap,
        ``2`` allows a two-pixel gap, and so on.
    stitch_across_fovs : bool
        If ``True``, run a second pass that stitches provisional components
        across neighboring FOV tile borders. This is useful when one gland or
        lesion crosses a tile boundary.
    fov_grid_cols : int, optional
        Number of tile columns in the FOV grid. Required when
        ``stitch_across_fovs=True``. FOV ids are assumed to follow the
        ``fov{index}`` pattern used during tiling.
    stitch_gap_tolerance : int, optional
        Allowed cross-tile border gap in pixels during stitching. If ``None``,
        defaults to ``gap_tolerance`` in ``label_adjacency`` mode and ``0``
        otherwise.
    min_component_size : int
        Minimum number of cells required before a component is kept as a named
        component.
    assign_singletons : bool
        If ``True``, cells in sub-threshold components receive unique singleton
        ids instead of ``noise``.

    Returns
    -------
    AnnData
        Copy of ``adata`` with mask-based component annotations added to ``obs``.
    """
    label_value = _ensure_list(label_value)
    suffix = _label_suffix(label_value, "target")

    if component_key is None:
        component_key = f"{suffix}_mask_component"

    if target_key is None:
        target_key = f"is_{suffix}"

    out = adata.copy()
    out.obs[target_key] = out.obs[label_key].isin(label_value)
    out.obs[component_key] = "unassigned"

    rows = []
    row_lookup = {}
    fov_states = {}
    structure = generate_binary_structure(2, int(connectivity))
    connection_mode = str(connection_mode)
    if connection_mode not in {"union_mask", "label_adjacency"}:
        raise ValueError(
            "connection_mode must be one of {'union_mask', 'label_adjacency'}."
        )
    if stitch_across_fovs and fov_grid_cols is None:
        raise ValueError("fov_grid_cols is required when stitch_across_fovs=True.")
    if stitch_gap_tolerance is None:
        stitch_gap_tolerance = gap_tolerance if connection_mode == "label_adjacency" else 0

    dilate_footprint = None
    if int(dilation_radius) > 0:
        dilate_footprint = disk(int(dilation_radius))

    close_footprint = None
    if int(closing_radius) > 0:
        close_footprint = disk(int(closing_radius))

    for fov in out.obs[fov_key].dropna().unique():
        fov_mask = out.obs[fov_key] == fov
        target_mask = fov_mask & out.obs[target_key]
        sub = out.obs.loc[target_mask, [cell_label_key]].copy()

        row = {
            fov_key: fov,
            "n_target_cells": int(len(sub)),
            "n_target_pixels": 0,
            "dilation_radius": int(dilation_radius),
            "closing_radius": int(closing_radius),
            "fill_holes": bool(fill_holes),
            "connectivity": int(connectivity),
            "connection_mode": connection_mode,
            "gap_tolerance": int(gap_tolerance),
            "n_mask_components": 0,
            "n_adjacency_edges": 0,
            "n_stitch_edges": 0,
            "n_components": 0,
            "n_singletons": 0,
            "n_noise": 0,
        }
        rows.append(row)
        row_lookup[fov] = row

        if sub.empty:
            continue

        seg_path = os.path.join(seg_dir, f"{fov}{seg_suffix}")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(
                f"Missing segmentation mask for FOV {fov}: {seg_path}"
            )

        seg_labels = imread(seg_path).astype(int)
        target_labels = (
            pd.to_numeric(sub[cell_label_key], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )

        if len(target_labels) == 0:
            continue

        original_mask = np.isin(seg_labels, target_labels)
        row["n_target_pixels"] = int(original_mask.sum())

        sub_labels = (
            pd.to_numeric(sub[cell_label_key], errors="coerce")
            .fillna(-1)
            .astype(int)
            .to_numpy()
        )

        if connection_mode == "union_mask":
            work_mask = original_mask
            if dilate_footprint is not None:
                work_mask = binary_dilation(work_mask, structure=dilate_footprint)
            if close_footprint is not None:
                work_mask = binary_closing(work_mask, structure=close_footprint)
            if fill_holes:
                work_mask = binary_fill_holes(work_mask)

            component_img, n_mask_components = ndi_label(work_mask, structure=structure)
            row["n_mask_components"] = int(n_mask_components)

            original_component_ids = component_img[original_mask]
            original_label_ids = seg_labels[original_mask].astype(int)

            valid_pixels = original_component_ids > 0
            original_component_ids = original_component_ids[valid_pixels]
            original_label_ids = original_label_ids[valid_pixels]

            component_by_label = {}
            if len(original_label_ids) > 0:
                order = np.argsort(original_label_ids, kind="mergesort")
                label_sorted = original_label_ids[order]
                component_sorted = original_component_ids[order]
                unique_labels, first_idx = np.unique(label_sorted, return_index=True)
                component_by_label = {
                    int(label): int(component_sorted[idx])
                    for label, idx in zip(unique_labels, first_idx)
                }

            cell_component_ids = np.array(
                [component_by_label.get(int(label), 0) for label in sub_labels],
                dtype=int,
            )
        else:
            label_index = {int(label): i for i, label in enumerate(target_labels)}
            slices = ndi_find_objects(seg_labels)
            search_iterations = int(max(1, int(gap_tolerance) + 1))
            edge_rows = []
            edge_cols = []

            for label in target_labels:
                label = int(label)
                if label <= 0 or label > len(slices):
                    continue
                slc = slices[label - 1]
                if slc is None:
                    continue

                y_slice, x_slice = slc
                y_min = max(y_slice.start - search_iterations, 0)
                y_max = min(y_slice.stop + search_iterations, seg_labels.shape[0])
                x_min = max(x_slice.start - search_iterations, 0)
                x_max = min(x_slice.stop + search_iterations, seg_labels.shape[1])

                cropped_labels = seg_labels[y_min:y_max, x_min:x_max]
                label_mask = cropped_labels == label
                if not np.any(label_mask):
                    continue

                search_mask = binary_dilation(
                    label_mask,
                    structure=structure,
                    iterations=search_iterations,
                )
                neighbor_labels = np.unique(cropped_labels[search_mask])
                neighbor_labels = neighbor_labels[
                    (neighbor_labels > 0)
                    & (neighbor_labels != label)
                    & np.isin(neighbor_labels, target_labels)
                ]

                src_idx = label_index[label]
                for neighbor in neighbor_labels:
                    dst_idx = label_index[int(neighbor)]
                    edge_rows.append(src_idx)
                    edge_cols.append(dst_idx)

            row["n_adjacency_edges"] = int(len(edge_rows))

            n_target = len(target_labels)
            if n_target == 0:
                cell_component_ids = np.zeros(len(sub_labels), dtype=int)
            else:
                graph = coo_matrix(
                    (
                        np.ones(len(edge_rows), dtype=int),
                        (edge_rows, edge_cols),
                    ),
                    shape=(n_target, n_target),
                ).tocsr()
                n_mask_components, comp_labels = connected_components(
                    graph,
                    directed=False,
                    return_labels=True,
                )
                row["n_mask_components"] = int(n_mask_components)
                component_by_label = {
                    int(label): int(comp_labels[idx] + 1)
                    for idx, label in enumerate(target_labels)
                }
                cell_component_ids = np.array(
                    [component_by_label.get(int(label), 0) for label in sub_labels],
                    dtype=int,
                )

        fov_states[fov] = {
            "seg_path": seg_path,
            "target_labels": np.asarray(target_labels, dtype=int),
            "component_by_label": {
                int(label): int(comp_id)
                for label, comp_id in component_by_label.items()
                if int(comp_id) > 0
            },
            "sub_index": sub.index.to_numpy(),
            "sub_labels": sub_labels,
            "cell_component_ids": cell_component_ids,
        }

    if stitch_across_fovs and fov_states:
        def _fov_idx(value):
            match = re.search(r"(\d+)$", str(value))
            if match is None:
                raise ValueError(
                    "stitch_across_fovs=True requires FOV ids ending in an integer, "
                    f"got {value!r}."
                )
            return int(match.group(1))

        node_lookup = {}
        node_counter = 0
        for fov, state in fov_states.items():
            for comp_id in np.unique(state["cell_component_ids"]):
                comp_id = int(comp_id)
                if comp_id <= 0:
                    continue
                node_lookup[(fov, comp_id)] = node_counter
                node_counter += 1

        edge_rows = []
        edge_cols = []

        def _add_edge(node_a, node_b):
            if node_a == node_b:
                return
            edge_rows.extend([node_a, node_b])
            edge_cols.extend([node_b, node_a])

        def _map_component_strip(strip_labels, component_by_label):
            if not component_by_label:
                return np.zeros(strip_labels.shape, dtype=int)
            mapper = np.frompyfunc(lambda x: component_by_label.get(int(x), 0), 1, 1)
            return mapper(strip_labels).astype(int)

        def _merge_neighbor_pair(fov_a, fov_b, axis):
            state_a = fov_states[fov_a]
            state_b = fov_states[fov_b]

            seg_a = imread(state_a["seg_path"]).astype(int)
            seg_b = imread(state_b["seg_path"]).astype(int)
            mask_a = np.isin(seg_a, state_a["target_labels"])
            mask_b = np.isin(seg_b, state_b["target_labels"])

            search = max(1, int(stitch_gap_tolerance) + 1)
            if axis == "horizontal":
                strip_mask_a = mask_a[:, -search:]
                strip_mask_b = mask_b[:, :search]
                strip_labels_a = seg_a[:, -search:]
                strip_labels_b = seg_b[:, :search]
                combined_mask = np.hstack([strip_mask_a, strip_mask_b])
                if int(stitch_gap_tolerance) > 0:
                    combined_mask = binary_dilation(
                        combined_mask,
                        structure=structure,
                        iterations=int(stitch_gap_tolerance),
                    )
                seam_img, _ = ndi_label(combined_mask, structure=structure)
                seam_left = seam_img[:, :search]
                seam_right = seam_img[:, search:]
                original_ids_a = seam_left[strip_mask_a]
                original_ids_b = seam_right[strip_mask_b]
            else:
                strip_mask_a = mask_a[-search:, :]
                strip_mask_b = mask_b[:search, :]
                strip_labels_a = seg_a[-search:, :]
                strip_labels_b = seg_b[:search, :]
                combined_mask = np.vstack([strip_mask_a, strip_mask_b])
                if int(stitch_gap_tolerance) > 0:
                    combined_mask = binary_dilation(
                        combined_mask,
                        structure=structure,
                        iterations=int(stitch_gap_tolerance),
                    )
                seam_img, _ = ndi_label(combined_mask, structure=structure)
                seam_left = seam_img[:search, :]
                seam_right = seam_img[search:, :]
                original_ids_a = seam_left[strip_mask_a]
                original_ids_b = seam_right[strip_mask_b]

            comp_strip_a = _map_component_strip(strip_labels_a, state_a["component_by_label"])
            comp_strip_b = _map_component_strip(strip_labels_b, state_b["component_by_label"])
            comp_ids_a = comp_strip_a[strip_mask_a]
            comp_ids_b = comp_strip_b[strip_mask_b]

            if len(original_ids_a) > 0 and len(original_ids_b) > 0:
                unique_seams = np.unique(np.concatenate([original_ids_a, original_ids_b]))
            else:
                unique_seams = np.array([], dtype=int)

            added = 0
            for seam_id in unique_seams:
                if seam_id <= 0:
                    continue
                comps_a = np.unique(comp_ids_a[original_ids_a == seam_id])
                comps_b = np.unique(comp_ids_b[original_ids_b == seam_id])
                comps_a = comps_a[comps_a > 0]
                comps_b = comps_b[comps_b > 0]
                for comp_a in comps_a:
                    for comp_b in comps_b:
                        node_a = node_lookup.get((fov_a, int(comp_a)))
                        node_b = node_lookup.get((fov_b, int(comp_b)))
                        if node_a is None or node_b is None:
                            continue
                        _add_edge(node_a, node_b)
                        added += 1

            row_lookup[fov_a]["n_stitch_edges"] += int(added)
            row_lookup[fov_b]["n_stitch_edges"] += int(added)

        fov_by_idx = {_fov_idx(fov): fov for fov in fov_states.keys()}
        n_cols = int(fov_grid_cols)
        for idx, fov in fov_by_idx.items():
            col_idx = idx % n_cols

            right_idx = idx + 1
            if (col_idx + 1) < n_cols and right_idx in fov_by_idx:
                _merge_neighbor_pair(fov, fov_by_idx[right_idx], axis="horizontal")

            down_idx = idx + n_cols
            if down_idx in fov_by_idx:
                _merge_neighbor_pair(fov, fov_by_idx[down_idx], axis="vertical")

        if node_counter > 0:
            graph = coo_matrix(
                (
                    np.ones(len(edge_rows), dtype=int),
                    (edge_rows, edge_cols),
                ),
                shape=(node_counter, node_counter),
            ).tocsr()
            _, global_labels = connected_components(
                graph,
                directed=False,
                return_labels=True,
            )
        else:
            global_labels = np.array([], dtype=int)

        for fov, state in fov_states.items():
            remapped = np.zeros_like(state["cell_component_ids"], dtype=int)
            for local_comp_id in np.unique(state["cell_component_ids"]):
                local_comp_id = int(local_comp_id)
                if local_comp_id <= 0:
                    continue
                node_id = node_lookup[(fov, local_comp_id)]
                remapped[state["cell_component_ids"] == local_comp_id] = int(global_labels[node_id] + 1)
            state["cell_component_ids"] = remapped

    if fov_states:
        all_component_ids = np.concatenate(
            [state["cell_component_ids"] for state in fov_states.values()]
        )
        valid_global = all_component_ids[all_component_ids > 0]
        if len(valid_global) > 0:
            unique_ids, counts = np.unique(valid_global, return_counts=True)
            global_sizes = {int(comp_id): int(count) for comp_id, count in zip(unique_ids, counts)}
        else:
            global_sizes = {}
    else:
        global_sizes = {}

    if stitch_across_fovs:
        component_name_map = {}
        component_counter = 0
        for comp_id in sorted(global_sizes):
            if global_sizes[comp_id] < int(min_component_size):
                continue
            component_name_map[comp_id] = f"global__component_{component_counter}"
            component_counter += 1
        global_singleton_counter = 0

    for fov, state in fov_states.items():
        cell_component_ids = state["cell_component_ids"]
        component_labels = np.full(len(cell_component_ids), "noise", dtype=object)

        if stitch_across_fovs:
            for idx, comp_id in enumerate(cell_component_ids):
                comp_id = int(comp_id)
                if comp_id <= 0:
                    continue
                if comp_id in component_name_map:
                    component_labels[idx] = component_name_map[comp_id]
                elif assign_singletons:
                    component_labels[idx] = f"global__singleton_{global_singleton_counter}"
                    global_singleton_counter += 1
        else:
            singleton_counter = 0
            component_counter = 0
            for comp_id in np.unique(cell_component_ids):
                comp_id = int(comp_id)
                if comp_id <= 0:
                    continue

                idx = np.where(cell_component_ids == comp_id)[0]
                if len(idx) < int(min_component_size):
                    if assign_singletons:
                        for local_idx in idx:
                            component_labels[local_idx] = f"{fov}__singleton_{singleton_counter}"
                            singleton_counter += 1
                    continue

                component_labels[idx] = f"{fov}__component_{component_counter}"
                component_counter += 1

        row = row_lookup[fov]
        label_series = pd.Series(component_labels)
        row["n_components"] = int(
            label_series[label_series.str.contains("component", na=False)].nunique()
        )
        row["n_singletons"] = int(
            label_series[label_series.str.contains("singleton", na=False)].nunique()
        )
        row["n_noise"] = int(np.sum(component_labels == "noise"))
        out.obs.loc[state["sub_index"], component_key] = component_labels

    param_df = pd.DataFrame(rows)

    if "spatial_niches" not in out.uns:
        out.uns["spatial_niches"] = {}

    out.uns["spatial_niches"][f"{component_key}_params"] = param_df.copy()

    return out


def build_niche_boundaries(
    adata: ad.AnnData,
    component_key: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    min_cluster_size: int=20,
    method: str="concave_hull",
    concavity: float=0.3,
    allow_holes: bool=False,
    point_buffer: float=10.0,
    line_buffer: float=10.0,
    mask_resolution: float=5.0,
    mask_sigma: float=1.0,
    mask_threshold: float=0.15,
    mask_closing_size: int=3,
) -> pd.DataFrame:
    """
    Build one geometry per clustered niche component.

    This function converts each clustered component into a spatial region geometry.
    For most use cases, the main tuning decision is whether to use a point-based
    concave hull or a rasterized density mask:

    - ``method="concave_hull"``:
      fast and effective for smooth, compact nests
    - ``method="density_mask"``:
      better for irregular nests, fissures, sharp indentations, and fragmented
      tumour regions that need a more image-like boundary

    Practical tuning guide
    ----------------------
    If the boundary is too fragmented:
        increase ``mask_sigma`` or ``mask_closing_size``
        decrease ``mask_threshold`` slightly

    If the boundary is too smooth and loses fissures:
        decrease ``mask_sigma``
        decrease ``mask_closing_size``
        decrease ``mask_resolution`` for finer detail

    If inward shrinking fails later:
        the geometry is still too thin or fragmented locally
        try larger ``mask_sigma`` and ``mask_closing_size`` first

    Good starting ranges for ``method="density_mask"``:
        ``mask_resolution=5 to 8``
        ``mask_sigma=1.5 to 2.5``
        ``mask_threshold=0.05 to 0.12``
        ``mask_closing_size=5 to 9``

    Parameters
    ----------
    component_key : str
        Column in ``adata.obs`` containing component labels.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key, y_key : str
        Column names in ``adata.obs`` containing cell coordinates.
    min_cluster_size : int
        Minimum number of cells required before a component gets a boundary.
    method : {"concave_hull", "density_mask"}
        Boundary-building backend.
        ``"concave_hull"`` is fast and works well for smooth nests.
        ``"density_mask"`` is often better for fissured or abruptly changing nests.
    concavity : float
        Shapely concave-hull ratio in ``[0, 1]``.
        Lower values produce tighter shapes.
    allow_holes : bool
        If ``True``, allow the concave hull to preserve enclosed holes.
    point_buffer, line_buffer : float
        Fallback buffer sizes for very small components with 1-2 points.
    mask_resolution : float
        Pixel size of the density grid used when ``method="density_mask"``.
        Smaller values give more detailed boundaries but can increase fragmentation.
        Larger values produce smoother, coarser boundaries.
    mask_sigma : float
        Gaussian smoothing applied to the density grid for ``density_mask`` boundaries.
        Lower values preserve sharper fissures and local detail.
        Higher values merge nearby gaps and create a more solid nest.
    mask_threshold : float
        Fraction of the maximum smoothed density used to extract the contour.
        Lower values produce larger, more inclusive boundaries.
        Higher values produce tighter, more conservative boundaries.
    mask_closing_size : int
        Binary closing kernel size used to seal small gaps in the density mask.
        Larger values create smoother, more solid nests and connect nearby fragments.

    Returns
    -------
    DataFrame
        One row per component with its geometry and summary statistics.
    """
    rows = []
    obs = adata.obs.copy()

    valid = obs[component_key].notna() & ~obs[component_key].isin(["unassigned", "noise"])
    grouped = obs.loc[valid].groupby(component_key)

    for component, sub in grouped:
        if len(sub) < min_cluster_size:
            continue

        coords = sub[[x_key, y_key]].to_numpy(dtype=float)
        geom = _make_component_geometry(
            coords,
            method=method,
            concavity=concavity,
            allow_holes=allow_holes,
            point_buffer=point_buffer,
            line_buffer=line_buffer,
            mask_resolution=mask_resolution,
            mask_sigma=mask_sigma,
            mask_threshold=mask_threshold,
            mask_closing_size=mask_closing_size,
        )

        if geom is None or geom.is_empty:
            continue

        image_ids = sub[image_key].dropna().unique()
        img = image_ids[0] if len(image_ids) > 0 else np.nan

        rows.append({
            component_key: component,
            image_key: img,
            "n_cells": len(sub),
            "method": method,
            "geometry": geom,
            "area": geom.area,
            "bounds": geom.bounds,
        })

    return pd.DataFrame(rows)


def buffer_niche_boundaries(
    boundary_df: pd.DataFrame,
    component_key: str,
    expand_by: float=0.0,
    shrink_by: float=0.0,
) -> object:
    """
    Expand or shrink niche boundaries to define outer and inner regions.

    Parameters
    ----------
    boundary_df : DataFrame
        Output from ``build_niche_boundaries``.
    component_key : str
        Column name containing component ids.
    expand_by : float
        Distance used to expand the component boundary outward.
    shrink_by : float
        Distance used to shrink the component boundary inward.

    Returns
    -------
    DataFrame
        Boundary table with added buffered geometries.
    """
    _concave_hull, _LineString, _MultiPoint, _Point, _box, _unary_union, make_valid = _require_shapely()

    out = boundary_df.copy()
    out["expanded_geometry"] = None
    out["shrunk_geometry"] = None

    for idx, row in out.iterrows():
        geom = row["geometry"]

        expanded = make_valid(geom.buffer(expand_by)) if expand_by > 0 else geom
        shrunk = make_valid(geom.buffer(-shrink_by)) if shrink_by > 0 else None

        if shrunk is not None and shrunk.is_empty:
            shrunk = None

        out.at[idx, "expanded_geometry"] = expanded
        out.at[idx, "shrunk_geometry"] = shrunk

    return out


def assign_cells_to_niche_regions(
    adata: ad.AnnData,
    boundary_df: pd.DataFrame,
    component_key: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    region_key: str="region",
    mode: str="distance_to_edge",
    boundary_width: float=None,
) -> ad.AnnData:
    """
    Assign cells to niche core and border regions.

    Region labels are:
    - ``core``: interior of the niche away from the edge
    - ``inner_border``: cells near the niche edge
    - ``outer_border``: cells outside the niche but inside the expanded boundary

    Two assignment modes are supported:
    - ``mode="distance_to_edge"``:
      classify cells inside the niche by their distance to the niche exterior.
      This is recommended for irregular or fissured tumour nests.
    - ``mode="buffer"``:
      use the original / shrunk / expanded geometries directly.

    Returns
    -------
    DataFrame
        Long-format table with one row per cell-component membership.
    """
    if mode not in {"distance_to_edge", "buffer"}:
        raise ValueError("mode must be either 'distance_to_edge' or 'buffer'")

    _concave_hull, _LineString, _MultiPoint, Point, _box, _unary_union, _make_valid = _require_shapely()

    rows = []
    obs = adata.obs.copy()

    valid_cells = obs[[image_key, x_key, y_key]].dropna().index
    obs = obs.loc[valid_cells]

    for _, row in boundary_df.iterrows():
        img = row[image_key]
        component = row[component_key]
        geom = row["geometry"]
        expanded = row.get("expanded_geometry", geom)
        shrunk = row.get("shrunk_geometry", None)

        width = boundary_width
        if width is None and mode == "distance_to_edge":
            if expanded is not None and expanded is not geom:
                width = max(0.0, expanded.distance(geom))
            else:
                width = 0.0

        sub = obs.loc[obs[image_key] == img, [x_key, y_key]].copy()
        if sub.empty:
            continue

        for cell_id, cell in sub.iterrows():
            pt = Point(float(cell[x_key]), float(cell[y_key]))

            region = None

            if mode == "distance_to_edge":
                if geom.covers(pt):
                    edge_distance = geom.boundary.distance(pt)
                    if width > 0 and edge_distance <= width:
                        region = "inner_border"
                    else:
                        region = "core"
                elif expanded is not None and expanded.covers(pt):
                    region = "outer_border"
            else:
                if shrunk is not None and shrunk.covers(pt):
                    region = "core"
                elif geom.covers(pt):
                    region = "inner_border" if shrunk is not None else "component"
                elif expanded is not None and expanded.covers(pt):
                    region = "outer_border"

            if region is None:
                continue

            rows.append({
                "cell_id": cell_id,
                image_key: img,
                component_key: component,
                region_key: region,
            })

    return pd.DataFrame(rows)


def summarize_niche_composition(
    adata: ad.AnnData,
    assignments_df: pd.DataFrame,
    component_key: str,
    phenotype_key: str,
    image_key: str="imageid",
    region_key: str="region",
    normalize: bool=True,
) -> pd.DataFrame:
    """
    Summarize phenotype composition within niche regions.

    Parameters
    ----------
    assignments_df : DataFrame
        Output from ``assign_cells_to_niche_regions``.
    component_key : str
        Column name containing component ids.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype or neighborhood labels.
    image_key : str
        Column identifying the image.
    region_key : str
        Column containing region labels such as ``core`` or ``outer_border``.
    normalize : bool
        If ``True``, convert counts to proportions within each niche region.

    Returns
    -------
    DataFrame
        Long-format composition table.
    """
    if assignments_df.empty:
        return pd.DataFrame()

    merged = assignments_df.merge(
        adata.obs[[phenotype_key]],
        left_on="cell_id",
        right_index=True,
        how="left",
    )

    counts = (
        merged.groupby([image_key, component_key, region_key, phenotype_key])
        .size()
        .rename("count")
        .reset_index()
    )

    if not normalize:
        return counts

    totals = (
        counts.groupby([image_key, component_key, region_key])["count"]
        .transform("sum")
    )
    counts["proportion"] = counts["count"] / totals

    return counts


def add_niche_regions_to_obs(
    adata: ad.AnnData,
    assignments_df: pd.DataFrame,
    region_key: str="region",
    component_key: str="component",
    cell_id_col: str="cell_id",
    out_region_key: str=None,
    out_component_key: str=None,
    outside_label: str="outside",
    region_priority: list[str] | None=None,
) -> ad.AnnData:
    """
    Merge niche-region assignments back into ``adata.obs``.

    When a cell appears in multiple rows of ``assignments_df``, this helper keeps
    one primary assignment using a configurable priority order.

    Default priority:
    ``core`` > ``inner_border`` > ``outer_border`` > ``component``

    Parameters
    ----------
    assignments_df : DataFrame
        Output from ``assign_cells_to_niche_regions``.
    region_key : str
        Column in ``assignments_df`` containing region labels.
    component_key : str
        Column in ``assignments_df`` containing component ids.
    cell_id_col : str
        Column in ``assignments_df`` containing cell ids matching ``adata.obs.index``.
    out_region_key : str, optional
        Output column name added to ``adata.obs`` for the primary region label.
        Defaults to ``region_key``.
    out_component_key : str, optional
        Output column name added to ``adata.obs`` for the component label.
        Defaults to ``f"{region_key}_component"``.
    outside_label : str
        Label assigned to cells not present in ``assignments_df``.
    region_priority : dict, optional
        Mapping from region label to numeric priority.
        Lower values win when duplicate assignments exist.

    Returns
    -------
    AnnData
        Copy of ``adata`` with region/component columns added to ``obs``.
    """
    if out_region_key is None:
        out_region_key = region_key

    if out_component_key is None:
        out_component_key = f"{region_key}_component"

    if region_priority is None:
        region_priority = {
            "core": 0,
            "inner_border": 1,
            "outer_border": 2,
            "component": 3,
        }

    out = adata.copy()

    out.obs[out_region_key] = outside_label
    out.obs[out_component_key] = pd.NA

    if assignments_df.empty:
        return out

    assignment_primary = assignments_df.copy()
    assignment_primary["__priority"] = (
        assignment_primary[region_key].map(region_priority).fillna(99)
    )

    sort_cols = [cell_id_col, "__priority"]
    if component_key in assignment_primary.columns:
        sort_cols.append(component_key)

    assignment_primary = (
        assignment_primary
        .sort_values(sort_cols)
        .drop_duplicates(subset=cell_id_col, keep="first")
    )

    out.obs.loc[
        assignment_primary[cell_id_col],
        out_region_key,
    ] = assignment_primary[region_key].values

    out.obs.loc[
        assignment_primary[cell_id_col],
        out_component_key,
    ] = assignment_primary[component_key].values

    return out



# ============================================================
# Section 2: Cell graph  (from archive/spatial/spatial_cell_graph.py)
# ============================================================

"""
Spatial cell graph construction utilities.

This module builds a Layer 1 cell graph where:
- nodes = cells
- edges = spatial proximity within each image
- node features = selected morphology/intensity features plus optional phenotype encoding

The graph is stored in AnnData-friendly form:
- ``adata.obsm["cell_features"]`` for processed node features
- ``adata.obsp["cell_graph_connectivities"]`` for the weighted or binary adjacency
- ``adata.obsp["cell_graph_distances"]`` for edge distances
- ``adata.uns["cell_graph"]`` for graph metadata

These outputs are designed to support niche-level induced subgraphs for
downstream trajectory or representation analyses.
"""

def _auto_log_transform(df, feature_cols, skew_threshold=1.0):
    """
    Apply ``log1p`` to non-negative, skewed continuous features.
    """
    df_out = df.copy()
    transformed = []

    for col in feature_cols:
        values = pd.to_numeric(df_out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)

        if not finite.any():
            continue

        finite_values = values[finite]
        if np.all(finite_values >= 0):
            skewness = pd.Series(finite_values).skew()
            if pd.notna(skewness) and skewness > skew_threshold:
                df_out.loc[finite, col] = np.log1p(finite_values)
                transformed.append(col)

    return df_out, transformed


def _prepare_node_features(
    adata,
    feature_cols,
    phenotype_key=None,
    auto_log=True,
    skew_threshold=1.0,
    scale_features=True,
    scale_binary_features=False,
    phenotype_weight=1.0,
):
    """
    Prepare a processed node-feature matrix aligned to ``adata.obs``.

    Continuous features are optionally log-transformed and scaled.
    Phenotype dummies are optionally appended and are not scaled by default.
    Rows with non-finite continuous features are retained in the output but
    marked invalid via the returned mask.
    """
    obs = adata.obs.copy()

    missing = [col for col in feature_cols if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df_cont = obs[feature_cols].apply(pd.to_numeric, errors="coerce")

    transformed_cols = []
    if auto_log:
        df_cont, transformed_cols = _auto_log_transform(
            df_cont,
            feature_cols=feature_cols,
            skew_threshold=skew_threshold,
        )

    valid_cont_mask = np.isfinite(df_cont.to_numpy(dtype=float)).all(axis=1)
    n_obs = adata.n_obs

    cont_array = df_cont.to_numpy(dtype=float)
    if scale_features and len(feature_cols) > 0 and valid_cont_mask.any():
        scaler = StandardScaler()
        cont_scaled = np.full_like(cont_array, np.nan, dtype=float)
        cont_scaled[valid_cont_mask] = scaler.fit_transform(cont_array[valid_cont_mask])
    else:
        cont_scaled = cont_array.astype(float, copy=True)

    matrices = []
    feature_names = []

    if len(feature_cols) > 0:
        matrices.append(cont_scaled)
        feature_names.extend(feature_cols)

    phenotype_cols = []
    if phenotype_key is not None:
        if phenotype_key not in obs.columns:
            raise ValueError(f"{phenotype_key} not found in adata.obs")

        pheno_onehot = pd.get_dummies(obs[phenotype_key], prefix="pheno", dtype=float)
        pheno_array = pheno_onehot.to_numpy(dtype=float)

        if scale_binary_features and pheno_array.shape[1] > 0:
            scaler = StandardScaler()
            pheno_array = scaler.fit_transform(pheno_array)

        pheno_array = pheno_array * float(phenotype_weight)
        matrices.append(pheno_array)
        phenotype_cols = pheno_onehot.columns.tolist()
        feature_names.extend(phenotype_cols)

    if matrices:
        X = np.concatenate(matrices, axis=1)
    else:
        X = np.empty((n_obs, 0), dtype=float)

    # Missing phenotype values are allowed; invalidity is driven by continuous features.
    return X, feature_names, transformed_cols, valid_cont_mask


def _build_radius_graph(coords, radius):
    """
    Build an undirected radius graph from coordinates.

    Returns local edge endpoints and distances with each pair included once.
    """
    if len(coords) < 2:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0, dtype=float)

    nbrs = NearestNeighbors(radius=radius)
    nbrs.fit(coords)
    graph = nbrs.radius_neighbors_graph(coords, mode="distance")
    graph = triu(graph, k=1).tocoo()

    return graph.row.astype(int), graph.col.astype(int), graph.data.astype(float)


def _resolve_sigma(values, sigma, default=1.0):
    """
    Resolve kernel width from data when not provided explicitly.
    """
    if sigma is not None:
        return float(sigma)

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]

    if len(positive) == 0:
        return float(default)

    return float(np.median(positive))


def _compute_edge_weights(row, col, dists, features, sigma_space=None, sigma_feat=None):
    """
    Compute edge weights using spatial decay and feature similarity.
    """
    sigma_space = _resolve_sigma(dists, sigma_space, default=1.0)
    spatial_weights = np.exp(-(dists ** 2) / (2.0 * sigma_space ** 2))

    if features.shape[1] == 0:
        return spatial_weights, sigma_space, None

    diffs = features[row] - features[col]
    feat_dists = np.linalg.norm(diffs, axis=1)
    sigma_feat = _resolve_sigma(feat_dists, sigma_feat, default=1.0)
    feat_weights = np.exp(-(feat_dists ** 2) / (2.0 * sigma_feat ** 2))

    return spatial_weights * feat_weights, sigma_space, sigma_feat


def build_cell_graph(
    adata: ad.AnnData,
    feature_cols: list[str],
    phenotype_key: str=None,
    radius: float=40.0,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    auto_log: bool=True,
    skew_threshold: float=1.0,
    scale_features: bool=True,
    scale_binary_features: bool=False,
    phenotype_weight: float=1.0,
    compute_weights: bool=True,
    sigma_space: float=None,
    sigma_feat: float=None,
    feature_obsm_key: str="cell_features",
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    graph_obs_key: str="cell_graph_valid",
) -> ad.AnnData:
    """
    Build a Layer 1 spatial cell graph across all images.

    The resulting adjacency is block-diagonal across images because edges are
    only built within each image.
    """
    obs = adata.obs
    required_cols = [x_key, y_key, image_key]
    missing = [col for col in required_cols if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    adata_out = adata.copy()

    X, feature_names, transformed_cols, valid_feature_mask = _prepare_node_features(
        adata_out,
        feature_cols=feature_cols,
        phenotype_key=phenotype_key,
        auto_log=auto_log,
        skew_threshold=skew_threshold,
        scale_features=scale_features,
        scale_binary_features=scale_binary_features,
        phenotype_weight=phenotype_weight,
    )

    coords_df = adata_out.obs[[x_key, y_key]].apply(pd.to_numeric, errors="coerce")
    coords = coords_df.to_numpy(dtype=float)
    valid_coord_mask = np.isfinite(coords).all(axis=1)

    if compute_weights:
        graph_valid_mask = valid_coord_mask & valid_feature_mask
    else:
        graph_valid_mask = valid_coord_mask

    adata_out.obsm[feature_obsm_key] = X
    adata_out.obs[graph_obs_key] = graph_valid_mask

    rows = []
    cols = []
    weights = []
    distances = []
    image_graphs = {}

    for img in adata_out.obs[image_key].dropna().unique():
        img_mask = adata_out.obs[image_key] == img
        local_mask = img_mask.to_numpy() & graph_valid_mask
        local_idx = np.flatnonzero(local_mask)

        if len(local_idx) < 2:
            continue

        local_coords = coords[local_idx]
        row_local, col_local, dist_local = _build_radius_graph(local_coords, radius=radius)

        if len(row_local) == 0:
            image_graphs[img] = {
                "cell_ids": adata_out.obs_names[local_idx].tolist(),
                "obs_positions": local_idx.tolist(),
                "n_nodes": int(len(local_idx)),
                "n_edges": 0,
            }
            continue

        row_global = local_idx[row_local]
        col_global = local_idx[col_local]

        if compute_weights:
            weight_local, sigma_space_img, sigma_feat_img = _compute_edge_weights(
                row=row_local,
                col=col_local,
                dists=dist_local,
                features=X[local_idx],
                sigma_space=sigma_space,
                sigma_feat=sigma_feat,
            )
        else:
            weight_local = np.ones(len(dist_local), dtype=float)
            sigma_space_img = None
            sigma_feat_img = None

        rows.extend(row_global.tolist())
        cols.extend(col_global.tolist())
        weights.extend(weight_local.tolist())
        distances.extend(dist_local.tolist())

        image_graphs[img] = {
            "cell_ids": adata_out.obs_names[local_idx].tolist(),
            "obs_positions": local_idx.tolist(),
            "n_nodes": int(len(local_idx)),
            "n_edges": int(len(row_local)),
            "sigma_space": sigma_space_img,
            "sigma_feat": sigma_feat_img,
        }

    n_obs = adata_out.n_obs
    if len(rows) > 0:
        row_array = np.asarray(rows, dtype=int)
        col_array = np.asarray(cols, dtype=int)
        weight_array = np.asarray(weights, dtype=float)
        dist_array = np.asarray(distances, dtype=float)

        adjacency = coo_matrix(
            (
                np.concatenate([weight_array, weight_array]),
                (
                    np.concatenate([row_array, col_array]),
                    np.concatenate([col_array, row_array]),
                ),
            ),
            shape=(n_obs, n_obs),
        ).tocsr()

        distance_matrix = coo_matrix(
            (
                np.concatenate([dist_array, dist_array]),
                (
                    np.concatenate([row_array, col_array]),
                    np.concatenate([col_array, row_array]),
                ),
            ),
            shape=(n_obs, n_obs),
        ).tocsr()
    else:
        adjacency = csr_matrix((n_obs, n_obs), dtype=float)
        distance_matrix = csr_matrix((n_obs, n_obs), dtype=float)

    adata_out.obsp[adjacency_key] = adjacency
    adata_out.obsp[distance_key] = distance_matrix
    adata_out.uns["cell_graph"] = {
        "feature_obsm_key": feature_obsm_key,
        "adjacency_key": adjacency_key,
        "distance_key": distance_key,
        "graph_obs_key": graph_obs_key,
        "feature_names": feature_names,
        "log_transformed": transformed_cols,
        "radius": float(radius),
        "x_key": x_key,
        "y_key": y_key,
        "image_key": image_key,
        "phenotype_key": phenotype_key,
        "compute_weights": bool(compute_weights),
        "scale_features": bool(scale_features),
        "scale_binary_features": bool(scale_binary_features),
        "phenotype_weight": float(phenotype_weight),
        "images": image_graphs,
        "node_ids": adata_out.obs_names.tolist(),
    }

    return adata_out


def extract_niche_subgraph(
    adata: ad.AnnData,
    niche_key: str,
    niche_value: str,
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    feature_obsm_key: str="cell_features",
) -> object:
    """
    Extract the induced cell subgraph for one niche label.

    Returns a dictionary containing the niche membership, adjacency, distances,
    and processed node features aligned to the returned cell IDs.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    mask = (adata.obs[niche_key] == niche_value).to_numpy()
    idx = np.flatnonzero(mask)

    adjacency = adata.obsp[adjacency_key][idx][:, idx].copy()

    result = {
        "niche_key": niche_key,
        "niche_value": niche_value,
        "cell_ids": adata.obs_names[idx].tolist(),
        "obs_positions": idx.tolist(),
        "adjacency": adjacency,
    }

    if distance_key in adata.obsp:
        result["distances"] = adata.obsp[distance_key][idx][:, idx].copy()

    if feature_obsm_key in adata.obsm:
        result["node_features"] = np.asarray(adata.obsm[feature_obsm_key][idx])

    return result


def extract_all_niche_subgraphs(
    adata: ad.AnnData,
    niche_key: str,
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    feature_obsm_key: str="cell_features",
    include_values: list[str] | None=None,
    include_prefix: str | None=None,
    exclude_values: list[str] | None=("unassigned", "noise"),
    exclude_prefixes: list[str] | None=None,
    min_cells: int=1,
) -> dict:
    """
    Extract induced cell subgraphs for multiple niche labels.

    This is a convenience wrapper around ``extract_niche_subgraph`` that reads
    niche labels from ``adata.obs[niche_key]``, filters them, and returns a
    dictionary keyed by niche label.

    Parameters
    ----------
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    include_values : iterable, optional
        Explicit niche labels to include. If provided, only these labels are
        considered.
    include_prefix : str, optional
        If provided, keep only labels whose string form starts with this prefix.
    exclude_values : iterable, optional
        Labels to exclude exactly. By default excludes ``"unassigned"`` and
        ``"noise"``.
    exclude_prefixes : iterable, optional
        Exclude labels whose string form starts with any of these prefixes.
        Useful for filtering singleton labels.
    min_cells : int
        Minimum number of cells required for a niche to be returned.

    Returns
    -------
    dict
        ``{niche_value: subgraph_dict}``
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    labels = pd.Series(adata.obs[niche_key]).dropna()
    if labels.empty:
        return {}

    unique_values = labels.unique().tolist()

    if include_values is not None:
        include_set = set(include_values)
        unique_values = [value for value in unique_values if value in include_set]

    if include_prefix is not None:
        unique_values = [
            value for value in unique_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        unique_values = [value for value in unique_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        unique_values = [
            value for value in unique_values
            if not str(value).startswith(exclude_prefixes)
        ]

    subgraphs = {}

    for niche_value in unique_values:
        subgraph = extract_niche_subgraph(
            adata,
            niche_key=niche_key,
            niche_value=niche_value,
            adjacency_key=adjacency_key,
            distance_key=distance_key,
            feature_obsm_key=feature_obsm_key,
        )

        if len(subgraph["cell_ids"]) < int(min_cells):
            continue

        subgraphs[niche_value] = subgraph

    return subgraphs



# ============================================================
# Section 3: Niche graph features  (from archive/spatial/spatial_niche_graph_features.py)
# ============================================================

"""
Niche-level graph descriptor utilities.

This module summarizes each niche as the induced subgraph of the Layer 1
cell graph plus a set of handcrafted descriptors that capture:

- graph topology
- geometry-aware graph structure
- node-feature organization on the graph
- optional boundary/core contrasts

The output is a per-niche feature table suitable for downstream clustering,
trajectory inference, or pseudotime modeling.
"""

def _safe_scalar(value):
    """
    Convert scalar-like results to a float, returning NaN when invalid.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if np.isfinite(value):
        return value

    return np.nan


def _safe_cv(values):
    """
    Coefficient of variation with NaN-safe handling.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    mean = values.mean()
    if np.isclose(mean, 0.0):
        return np.nan

    return float(values.std(ddof=0) / mean)


def _safe_ratio(num, denom):
    """
    Safe scalar ratio with NaN on zero or non-finite denominator.
    """
    num = _safe_scalar(num)
    denom = _safe_scalar(denom)

    if not np.isfinite(num) or not np.isfinite(denom) or np.isclose(denom, 0.0):
        return np.nan

    return float(num / denom)


def _nanmean_if_any(values):
    """
    NaN-safe mean that returns NaN when no finite values are present.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def _normalized_entropy(values, n_bins):
    """
    Compute entropy normalized to [0, 1].
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    counts, _ = np.histogram(values, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return np.nan

    probs = counts[counts > 0] / total
    if len(probs) <= 1:
        return 0.0

    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(n_bins))


def _sanitize_label(value):
    """
    Convert arbitrary labels into stable column-name fragments.
    """
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _graph_morans_i_from_adjacency(adjacency, values):
    """
    Moran-like autocorrelation on a graph adjacency matrix.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if adjacency.shape[0] != len(values):
        return np.nan

    if valid.sum() < 3:
        return np.nan

    idx = np.flatnonzero(valid)
    sub_adj = adjacency[idx][:, idx].astype(float).tocsr()
    x = values[idx]

    W = float(sub_adj.sum())
    n = len(x)

    if n < 3 or np.isclose(W, 0.0):
        return np.nan

    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2))
    if np.isclose(denom, 0.0):
        return np.nan

    numerator = float(x_centered @ (sub_adj @ x_centered))
    return float((n / W) * (numerator / denom))


def _build_knn_skeleton(coords, k=6):
    """
    Build a sparse Euclidean skeleton graph from a symmetric kNN graph and
    extract its minimum spanning tree.

    This provides a less trivial topology summary than the original niche graph
    when niches were themselves defined as connected components of a radius
    graph.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)

    if n < 2:
        return csr_matrix((n, n), dtype=float)

    k_eff = min(int(k), n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
    nbrs.fit(coords)
    knn_graph = nbrs.kneighbors_graph(coords, mode="distance")
    knn_graph = knn_graph.maximum(knn_graph.T).tocsr()

    mst = minimum_spanning_tree(knn_graph)
    mst = mst.maximum(mst.T).tocsr()

    return mst


def _edge_feature_stats(values, edge_rows, edge_cols):
    """
    Edge-based coherence summaries for one node feature.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    keep = valid[edge_rows] & valid[edge_cols]
    if keep.sum() < 2:
        return {
            "neighbor_corr": np.nan,
            "edge_abs_diff_mean": np.nan,
        }

    x = values[edge_rows[keep]]
    y = values[edge_cols[keep]]

    if np.std(x) == 0 or np.std(y) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(x, y)[0, 1]

    return {
        "neighbor_corr": _safe_scalar(corr),
        "edge_abs_diff_mean": _safe_scalar(np.mean(np.abs(x - y))),
    }


def _quantile_bin(values, n_bins=3):
    """
    Bin a continuous feature by quantiles for assortativity calculations.
    """
    series = pd.Series(values, dtype=float)
    valid = series.notna()

    if valid.sum() < n_bins:
        return None

    try:
        binned = pd.qcut(series[valid], q=n_bins, duplicates="drop")
    except ValueError:
        return None

    if binned.nunique(dropna=True) < 2:
        return None

    out = pd.Series(index=series.index, dtype=object)
    out.loc[valid] = binned.astype(str).values
    return out.to_numpy(dtype=object)


def _spearman_feature(values, reference):
    """
    Spearman correlation with NaN-safe handling.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(values) & np.isfinite(reference)

    if valid.sum() < 3:
        return np.nan

    x = values[valid]
    y = reference[valid]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan

    stat = spearmanr(x, y).statistic
    return _safe_scalar(stat)


def _summarize_topology(G, include_path_metrics=True):
    """
    Topological summaries for one induced niche graph.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)

    out = {
        "topology__n_nodes": float(n_nodes),
        "topology__n_edges": float(n_edges),
        "topology__density": _safe_scalar(nx.density(G)) if n_nodes > 1 else 0.0,
        "topology__avg_degree": _safe_scalar(degrees.mean()) if len(degrees) > 0 else np.nan,
        "topology__degree_var": _safe_scalar(degrees.var(ddof=0)) if len(degrees) > 0 else np.nan,
        "topology__degree_cv": _safe_cv(degrees),
        "topology__leaf_fraction": _safe_scalar(np.mean(degrees == 1)) if len(degrees) > 0 else np.nan,
        "topology__isolates_fraction": _safe_scalar(np.mean(degrees == 0)) if len(degrees) > 0 else np.nan,
        "topology__avg_clustering": _safe_scalar(nx.average_clustering(G)) if n_nodes > 1 else np.nan,
        "topology__transitivity": _safe_scalar(nx.transitivity(G)) if n_nodes > 2 else np.nan,
        "topology__n_connected_components": float(nx.number_connected_components(G)) if n_nodes > 0 else 0.0,
    }

    if n_nodes > 0:
        component_sizes = np.array([len(c) for c in nx.connected_components(G)], dtype=float)
        out["topology__largest_component_size"] = _safe_scalar(component_sizes.max())
        out["topology__largest_component_fraction"] = _safe_scalar(component_sizes.max() / n_nodes)
        out["topology__component_size_cv"] = _safe_cv(component_sizes)
    else:
        out["topology__largest_component_size"] = np.nan
        out["topology__largest_component_fraction"] = np.nan
        out["topology__component_size_cv"] = np.nan

    if n_nodes < 3 or n_edges == 0 or len(np.unique(degrees)) < 2:
        out["topology__degree_assortativity"] = np.nan
    else:
        try:
            out["topology__degree_assortativity"] = _safe_scalar(
                nx.degree_assortativity_coefficient(G)
            )
        except Exception:
            out["topology__degree_assortativity"] = np.nan

    if n_nodes > 0 and n_edges > 0:
        core_numbers = np.array(list(nx.core_number(G).values()), dtype=float)
        out["topology__mean_core_number"] = _safe_scalar(core_numbers.mean())
        out["topology__max_core_number"] = _safe_scalar(core_numbers.max())
        if include_path_metrics:
            out["topology__bridge_fraction"] = _safe_scalar(
                len(list(nx.bridges(G))) / n_edges
            )
        else:
            out["topology__bridge_fraction"] = np.nan
    else:
        out["topology__mean_core_number"] = np.nan
        out["topology__max_core_number"] = np.nan
        out["topology__bridge_fraction"] = np.nan

    if n_nodes == 0 or not include_path_metrics:
        out["topology__diameter_lcc"] = np.nan
        out["topology__avg_shortest_path_lcc"] = np.nan
        return out

    largest_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_nodes).copy()

    if G_lcc.number_of_nodes() <= 1:
        out["topology__diameter_lcc"] = 0.0
        out["topology__avg_shortest_path_lcc"] = 0.0
    else:
        out["topology__diameter_lcc"] = _safe_scalar(nx.diameter(G_lcc))
        out["topology__avg_shortest_path_lcc"] = _safe_scalar(
            nx.average_shortest_path_length(G_lcc)
        )

    return out


def _summarize_skeleton_topology(coords):
    """
    Pathology-oriented topology summaries from a sparse Euclidean skeleton.

    In PDAC this tends to separate compact gland-forming niches from elongated,
    budding, or infiltrative epithelial structures better than dense radius
    graphs alone.
    """
    coords = np.asarray(coords, dtype=float)
    n_nodes = len(coords)

    out = {
        "topology__skeleton_n_edges": np.nan,
        "topology__skeleton_leaf_fraction": np.nan,
        "topology__skeleton_branchpoint_fraction": np.nan,
        "topology__skeleton_avg_degree": np.nan,
        "topology__skeleton_degree_cv": np.nan,
        "topology__skeleton_total_length": np.nan,
        "topology__skeleton_mean_edge_length": np.nan,
        "topology__skeleton_edge_length_cv": np.nan,
        "topology__skeleton_diameter": np.nan,
        "topology__skeleton_avg_shortest_path": np.nan,
        "topology__skeleton_tortuosity": np.nan,
    }

    if n_nodes < 2:
        return out

    skeleton = _build_knn_skeleton(coords, k=6)
    if skeleton.nnz == 0:
        return out

    G_skel = nx.from_scipy_sparse_array(skeleton)
    degrees = np.array([deg for _, deg in G_skel.degree()], dtype=float)
    edge_lengths = skeleton.data.astype(float)

    out["topology__skeleton_n_edges"] = float(G_skel.number_of_edges())
    out["topology__skeleton_leaf_fraction"] = _safe_scalar(np.mean(degrees == 1))
    out["topology__skeleton_branchpoint_fraction"] = _safe_scalar(np.mean(degrees >= 3))
    out["topology__skeleton_avg_degree"] = _safe_scalar(degrees.mean())
    out["topology__skeleton_degree_cv"] = _safe_cv(degrees)
    out["topology__skeleton_total_length"] = _safe_scalar(edge_lengths.sum())
    out["topology__skeleton_mean_edge_length"] = _safe_scalar(edge_lengths.mean())
    out["topology__skeleton_edge_length_cv"] = _safe_cv(edge_lengths)

    if G_skel.number_of_nodes() <= 1:
        out["topology__skeleton_diameter"] = 0.0
        out["topology__skeleton_avg_shortest_path"] = 0.0
        out["topology__skeleton_tortuosity"] = np.nan
        return out

    largest_nodes = max(nx.connected_components(G_skel), key=len)
    G_lcc = G_skel.subgraph(largest_nodes).copy()

    out["topology__skeleton_diameter"] = _safe_scalar(nx.diameter(G_lcc))
    out["topology__skeleton_avg_shortest_path"] = _safe_scalar(
        nx.average_shortest_path_length(G_lcc)
    )

    if len(coords) >= 2:
        centered = coords - coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvecs = np.linalg.eigh(cov)[1]
        major_axis = eigvecs[:, -1]
        major_span = np.ptp(centered @ major_axis)
        out["topology__skeleton_tortuosity"] = _safe_ratio(edge_lengths.sum(), major_span)

    return out


def _summarize_geometry(coords, edge_rows, edge_cols, edge_lengths):
    """
    Geometry-aware summaries for one niche subgraph.
    """
    coords = np.asarray(coords, dtype=float)
    out = {}

    out["geometry__mean_edge_length"] = _safe_scalar(np.mean(edge_lengths)) if len(edge_lengths) > 0 else np.nan
    out["geometry__edge_length_std"] = _safe_scalar(np.std(edge_lengths, ddof=0)) if len(edge_lengths) > 0 else np.nan
    out["geometry__edge_length_cv"] = _safe_cv(edge_lengths)
    out["geometry__median_edge_length"] = _safe_scalar(np.median(edge_lengths)) if len(edge_lengths) > 0 else np.nan

    if len(edge_lengths) > 0:
        vecs = coords[edge_cols] - coords[edge_rows]
        angles = np.mod(np.arctan2(vecs[:, 1], vecs[:, 0]), np.pi)
        out["geometry__orientation_entropy"] = _normalized_entropy(angles, n_bins=8)

        doubled = 2.0 * angles
        cos_mean = np.mean(np.cos(doubled))
        sin_mean = np.mean(np.sin(doubled))
        out["geometry__orientation_coherence"] = _safe_scalar(
            np.sqrt(cos_mean ** 2 + sin_mean ** 2)
        )
    else:
        out["geometry__orientation_entropy"] = np.nan
        out["geometry__orientation_coherence"] = np.nan

    if len(coords) >= 2:
        centroid = coords.mean(axis=0)
        radial = np.linalg.norm(coords - centroid, axis=1)
        out["geometry__mean_radial_distance"] = _safe_scalar(radial.mean())
        out["geometry__radial_distance_cv"] = _safe_cv(radial)

        cov = np.cov(coords.T)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
        major = eigvals[0]
        minor = eigvals[1] if len(eigvals) > 1 else 0.0

        denom = major + minor
        out["geometry__node_cloud_anisotropy"] = (
            _safe_scalar((major - minor) / denom) if not np.isclose(denom, 0.0) else np.nan
        )
        out["geometry__node_cloud_elongation"] = (
            _safe_scalar(major / minor) if minor > 0 else np.nan
        )
        proj = coords - centroid
        eigvecs = np.linalg.eigh(cov)[1]
        major_axis = eigvecs[:, -1]
        minor_axis = eigvecs[:, 0]
        major_proj = proj @ major_axis
        minor_proj = proj @ minor_axis
        major_span = np.ptp(major_proj)
        minor_span = np.ptp(minor_proj)
        out["geometry__major_axis_span"] = _safe_scalar(major_span)
        out["geometry__minor_axis_span"] = _safe_scalar(minor_span)
        out["geometry__span_ratio"] = _safe_ratio(major_span, minor_span)

        width = max(2, int(np.ceil(np.sqrt(len(coords)))))
        x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), width + 1)
        y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), width + 1)
        hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_bins, y_bins])
        probs = hist.ravel()
        probs = probs[probs > 0] / probs.sum()
        if len(probs) <= 1:
            out["geometry__spatial_entropy"] = 0.0
        else:
            out["geometry__spatial_entropy"] = _safe_scalar(
                -np.sum(probs * np.log(probs)) / np.log(len(hist.ravel()))
            )

        if len(coords) >= 3:
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(coords)
            dists, _ = nn.kneighbors(coords)
            nn_dists = dists[:, 1]
            out["geometry__mean_nearest_neighbor_distance"] = _safe_scalar(nn_dists.mean())
            out["geometry__nearest_neighbor_distance_cv"] = _safe_cv(nn_dists)
        else:
            out["geometry__mean_nearest_neighbor_distance"] = np.nan
            out["geometry__nearest_neighbor_distance_cv"] = np.nan

        hull_area = compute_convex_hull_area(coords)
        out["geometry__convex_hull_area"] = _safe_scalar(hull_area)
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                hull_perimeter = float(hull.area)
            except Exception:
                hull_perimeter = np.nan
        else:
            hull_perimeter = np.nan
        out["geometry__convex_hull_perimeter"] = _safe_scalar(hull_perimeter)
        if np.isfinite(hull_area) and hull_area > 0 and np.isfinite(hull_perimeter) and hull_perimeter > 0:
            out["geometry__hull_circularity"] = _safe_scalar(
                4.0 * np.pi * hull_area / (hull_perimeter ** 2)
            )
        else:
            out["geometry__hull_circularity"] = np.nan
        out["geometry__cell_density_hull"] = (
            _safe_scalar(len(coords) / hull_area) if np.isfinite(hull_area) and hull_area > 0 else np.nan
        )
        out["geometry__edge_density_hull"] = (
            _safe_scalar(len(edge_lengths) / hull_area) if np.isfinite(hull_area) and hull_area > 0 else np.nan
        )
        out["geometry__cells_per_major_axis_span"] = _safe_ratio(len(coords), major_span)
    else:
        out["geometry__mean_radial_distance"] = np.nan
        out["geometry__radial_distance_cv"] = np.nan
        out["geometry__node_cloud_anisotropy"] = np.nan
        out["geometry__node_cloud_elongation"] = np.nan
        out["geometry__major_axis_span"] = np.nan
        out["geometry__minor_axis_span"] = np.nan
        out["geometry__span_ratio"] = np.nan
        out["geometry__spatial_entropy"] = np.nan
        out["geometry__mean_nearest_neighbor_distance"] = np.nan
        out["geometry__nearest_neighbor_distance_cv"] = np.nan
        out["geometry__convex_hull_area"] = np.nan
        out["geometry__convex_hull_perimeter"] = np.nan
        out["geometry__hull_circularity"] = np.nan
        out["geometry__cell_density_hull"] = np.nan
        out["geometry__edge_density_hull"] = np.nan
        out["geometry__cells_per_major_axis_span"] = np.nan

    return out


def _summarize_feature_organization(
    obs_sub,
    adjacency_binary,
    edge_rows,
    edge_cols,
    feature_cols=None,
    phenotype_key=None,
    morphology_bin_count=3,
):
    """
    Node-feature organization descriptors within one niche graph.
    """
    out = {}
    G = nx.from_scipy_sparse_array(adjacency_binary)

    if phenotype_key is not None and phenotype_key in obs_sub.columns:
        phenotype_values = obs_sub[phenotype_key].astype("object").to_numpy()
        valid = pd.notna(phenotype_values)

        if valid.sum() >= 2 and pd.Series(phenotype_values[valid]).nunique() >= 2:
            G_pheno = G.subgraph(np.flatnonzero(valid)).copy()
            mapping = {old: i for i, old in enumerate(G_pheno.nodes())}
            G_pheno = nx.relabel_nodes(G_pheno, mapping)
            pheno_valid = phenotype_values[valid]
            for i, value in enumerate(pheno_valid):
                G_pheno.nodes[i]["phenotype"] = value
            if G_pheno.number_of_edges() == 0:
                out["features__phenotype_assortativity"] = np.nan
            else:
                try:
                    out["features__phenotype_assortativity"] = _safe_scalar(
                        nx.attribute_assortativity_coefficient(G_pheno, "phenotype")
                    )
                except Exception:
                    out["features__phenotype_assortativity"] = np.nan
        else:
            out["features__phenotype_assortativity"] = np.nan

    if feature_cols is None:
        return out

    degree = np.asarray(adjacency_binary.sum(axis=1)).ravel().astype(float)

    for feature in feature_cols:
        if feature not in obs_sub.columns:
            continue

        values = pd.to_numeric(obs_sub[feature], errors="coerce").to_numpy(dtype=float)

        out[f"features__{feature}__graph_morans_i"] = _graph_morans_i_from_adjacency(
            adjacency_binary,
            values,
        )

        edge_stats = _edge_feature_stats(values, edge_rows, edge_cols)
        out[f"features__{feature}__neighbor_corr"] = edge_stats["neighbor_corr"]
        out[f"features__{feature}__edge_abs_diff_mean"] = edge_stats["edge_abs_diff_mean"]
        out[f"features__{feature}__degree_spearman"] = _spearman_feature(values, degree)

        binned = _quantile_bin(values, n_bins=morphology_bin_count)
        if binned is not None:
            valid = pd.notna(binned)
            G_bin = G.subgraph(np.flatnonzero(valid)).copy()
            mapping = {old: i for i, old in enumerate(G_bin.nodes())}
            G_bin = nx.relabel_nodes(G_bin, mapping)
            binned_valid = binned[valid]
            for i, value in enumerate(binned_valid):
                G_bin.nodes[i]["feature_bin"] = value
            if G_bin.number_of_edges() == 0 or pd.Series(binned_valid).nunique() < 2:
                out[f"features__{feature}__bin_assortativity"] = np.nan
            else:
                try:
                    out[f"features__{feature}__bin_assortativity"] = _safe_scalar(
                        nx.attribute_assortativity_coefficient(G_bin, "feature_bin")
                    )
                except Exception:
                    out[f"features__{feature}__bin_assortativity"] = np.nan
        else:
            out[f"features__{feature}__bin_assortativity"] = np.nan

    return out


def _summarize_boundary_core(
    obs_sub,
    adjacency_binary,
    feature_cols=None,
    phenotype_key=None,
    region_key=None,
    boundary_labels=("inner_border",),
    core_labels=("core",),
):
    """
    Boundary/core summaries for one niche graph.
    """
    out = {}

    if region_key is None or region_key not in obs_sub.columns:
        return out

    region_values = obs_sub[region_key].astype("object")
    boundary_mask = region_values.isin(boundary_labels).to_numpy()
    core_mask = region_values.isin(core_labels).to_numpy()

    out["boundary__boundary_fraction"] = _safe_scalar(boundary_mask.mean())

    degree = np.asarray(adjacency_binary.sum(axis=1)).ravel().astype(float)
    if boundary_mask.any():
        out["boundary__mean_degree_boundary"] = _safe_scalar(np.nanmean(degree[boundary_mask]))
    else:
        out["boundary__mean_degree_boundary"] = np.nan

    if core_mask.any():
        out["boundary__mean_degree_core"] = _safe_scalar(np.nanmean(degree[core_mask]))
    else:
        out["boundary__mean_degree_core"] = np.nan

    if boundary_mask.any() and core_mask.any():
        out["boundary__degree_boundary_minus_core"] = _safe_scalar(
            np.nanmean(degree[boundary_mask]) - np.nanmean(degree[core_mask])
        )
    else:
        out["boundary__degree_boundary_minus_core"] = np.nan

    if feature_cols is not None:
        for feature in feature_cols:
            if feature not in obs_sub.columns:
                continue

            values = pd.to_numeric(obs_sub[feature], errors="coerce").to_numpy(dtype=float)
            if boundary_mask.any() and core_mask.any():
                out[f"boundary__{feature}__boundary_minus_core"] = _safe_scalar(
                    np.nanmean(values[boundary_mask]) - np.nanmean(values[core_mask])
                )
            else:
                out[f"boundary__{feature}__boundary_minus_core"] = np.nan

    if phenotype_key is not None and phenotype_key in obs_sub.columns:
        phenotypes = obs_sub[phenotype_key].astype("object")
        if boundary_mask.any():
            out["boundary__phenotype_entropy_boundary"] = _safe_scalar(
                phenotypes[boundary_mask].value_counts(normalize=True, dropna=True).pipe(
                    lambda probs: -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
                )
            )
        else:
            out["boundary__phenotype_entropy_boundary"] = np.nan

        if core_mask.any():
            out["boundary__phenotype_entropy_core"] = _safe_scalar(
                phenotypes[core_mask].value_counts(normalize=True, dropna=True).pipe(
                    lambda probs: -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
                )
            )
        else:
            out["boundary__phenotype_entropy_core"] = np.nan

    return out


def _get_n_hop_external_layers(adjacency_full, niche_idx, max_hops=1):
    """
    Collect external nodes around a niche in successive graph hops.

    Hop 1 contains external neighbors directly adjacent to niche cells.
    Hop 2 contains previously unseen external neighbors adjacent to hop-1 nodes,
    and so on. Niche nodes themselves are never returned.
    """
    niche_idx = np.asarray(niche_idx, dtype=int)

    if len(niche_idx) == 0 or max_hops < 1:
        return {}

    niche_set = set(niche_idx.tolist())
    visited = set(niche_idx.tolist())
    frontier = set(niche_idx.tolist())
    layers = {}

    for hop in range(1, int(max_hops) + 1):
        next_frontier = set()
        for node in frontier:
            neighbors = adjacency_full.getrow(node).indices
            for neighbor in neighbors:
                if neighbor in niche_set or neighbor in visited:
                    continue
                next_frontier.add(int(neighbor))

        if not next_frontier:
            break

        layers[hop] = np.array(sorted(next_frontier), dtype=int)
        visited.update(next_frontier)
        frontier = next_frontier

    return layers


def _summarize_graph_surroundings(
    adata,
    niche_idx,
    adjacency_full,
    niche_key,
    niche_value,
    feature_cols=None,
    phenotype_key=None,
    surround_hops=1,
    numeric_cache=None,
    phenotype_array=None,
):
    """
    Fast graph-defined niche boundary/core/surround summaries.

    Boundary cells are niche cells with at least one edge to a non-niche cell.
    Core cells are niche cells with no direct external neighbors.
    Surrounding cells are non-niche cells reached within ``surround_hops``.
    """
    niche_idx = np.asarray(niche_idx, dtype=int)
    out = {}

    if len(niche_idx) == 0:
        return out

    niche_set = set(niche_idx.tolist())

    boundary_local = []
    external_degree = np.zeros(len(niche_idx), dtype=float)
    cross_edges = []

    for local_i, global_i in enumerate(niche_idx):
        neighbors = adjacency_full.getrow(global_i).indices
        external_neighbors = [j for j in neighbors if j not in niche_set]
        external_degree[local_i] = len(external_neighbors)

        if external_neighbors:
            boundary_local.append(local_i)
            for j in external_neighbors:
                cross_edges.append((int(global_i), int(j)))

    boundary_local = np.array(boundary_local, dtype=int)
    core_local = np.array(
        sorted(set(range(len(niche_idx))) - set(boundary_local.tolist())),
        dtype=int,
    )

    boundary_idx = niche_idx[boundary_local] if len(boundary_local) > 0 else np.empty(0, dtype=int)
    core_idx = niche_idx[core_local] if len(core_local) > 0 else np.empty(0, dtype=int)
    surround_layers = _get_n_hop_external_layers(
        adjacency_full=adjacency_full,
        niche_idx=niche_idx,
        max_hops=surround_hops,
    )

    surround_idx = (
        np.concatenate(list(surround_layers.values()))
        if len(surround_layers) > 0
        else np.empty(0, dtype=int)
    )

    out["graph_boundary__n_boundary_cells"] = float(len(boundary_idx))
    out["graph_boundary__boundary_fraction"] = _safe_scalar(len(boundary_idx) / len(niche_idx))
    out["graph_boundary__n_core_cells"] = float(len(core_idx))
    out["graph_boundary__core_fraction"] = _safe_scalar(len(core_idx) / len(niche_idx))
    out["graph_boundary__mean_external_degree"] = _safe_scalar(external_degree.mean())
    out["graph_boundary__max_external_degree"] = _safe_scalar(external_degree.max()) if len(external_degree) > 0 else np.nan
    out["graph_boundary__boundary_external_degree_mean"] = (
        _safe_scalar(external_degree[boundary_local].mean()) if len(boundary_local) > 0 else np.nan
    )
    out["graph_surround__n_total"] = float(len(surround_idx))
    out["graph_surround__surround_to_niche_ratio"] = _safe_scalar(len(surround_idx) / len(niche_idx))
    out["graph_surround__n_cross_edges"] = float(len(cross_edges))
    out["graph_surround__cross_edges_per_niche_cell"] = _safe_scalar(len(cross_edges) / len(niche_idx))

    for hop in range(1, int(surround_hops) + 1):
        hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
        out[f"graph_surround__hop_{hop}__n_cells"] = float(len(hop_idx))
        out[f"graph_surround__hop_{hop}__fraction_of_niche"] = _safe_scalar(
            len(hop_idx) / len(niche_idx)
        )

    if phenotype_key is not None and phenotype_array is not None:
        phenotype_series = pd.Series(phenotype_array)

        if len(cross_edges) > 0:
            same = []
            for src, dst in cross_edges:
                p_src = phenotype_series.iloc[src]
                p_dst = phenotype_series.iloc[dst]
                same.append(pd.notna(p_src) and pd.notna(p_dst) and p_src == p_dst)
            out["graph_surround__cross_edge_same_phenotype_fraction"] = _safe_scalar(np.mean(same))
        else:
            out["graph_surround__cross_edge_same_phenotype_fraction"] = np.nan

        if len(surround_idx) > 0:
            surround_pheno = phenotype_series.iloc[surround_idx]
            probs = surround_pheno.value_counts(normalize=True, dropna=True)
            out["graph_surround__phenotype_entropy"] = _safe_scalar(
                -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
            )
        else:
            out["graph_surround__phenotype_entropy"] = np.nan

        for hop in range(1, int(surround_hops) + 1):
            hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
            if len(hop_idx) > 0:
                hop_probs = phenotype_series.iloc[hop_idx].value_counts(normalize=True, dropna=True)
                out[f"graph_surround__hop_{hop}__phenotype_entropy"] = _safe_scalar(
                    -np.sum(hop_probs * np.log(hop_probs)) if len(hop_probs) > 1 else 0.0
                )
            else:
                out[f"graph_surround__hop_{hop}__phenotype_entropy"] = np.nan

    if feature_cols is not None:
        for feature in feature_cols:
            if numeric_cache is None or feature not in numeric_cache:
                continue

            values = numeric_cache[feature]
            niche_values = values[niche_idx]

            if len(boundary_idx) > 0 and len(core_idx) > 0:
                out[f"graph_boundary__{feature}__boundary_minus_core"] = _safe_scalar(
                    _nanmean_if_any(values[boundary_idx]) - _nanmean_if_any(values[core_idx])
                )
            else:
                out[f"graph_boundary__{feature}__boundary_minus_core"] = np.nan

            if len(surround_idx) > 0:
                out[f"graph_surround__{feature}__surround_minus_niche"] = _safe_scalar(
                    _nanmean_if_any(values[surround_idx]) - _nanmean_if_any(niche_values)
                )
            else:
                out[f"graph_surround__{feature}__surround_minus_niche"] = np.nan

            for hop in range(1, int(surround_hops) + 1):
                hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
                if len(hop_idx) > 0:
                    out[f"graph_surround__hop_{hop}__{feature}__minus_niche"] = _safe_scalar(
                        _nanmean_if_any(values[hop_idx]) - _nanmean_if_any(niche_values)
                    )
                else:
                    out[f"graph_surround__hop_{hop}__{feature}__minus_niche"] = np.nan

    return out


def _summarize_niche_state(
    obs_sub,
    state_feature_cols=None,
    phenotype_key=None,
    state_summary_stats=("mean", "median", "std", "iqr", "p10", "p90"),
):
    """
    Summarize original cell-level features at the niche level.

    This keeps more of the single-cell information than a pure mean-only
    aggregation by using richer distribution summaries.
    """
    out = {}

    if state_feature_cols is not None:
        for feature in state_feature_cols:
            if feature not in obs_sub.columns:
                continue

            values = pd.to_numeric(obs_sub[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                for stat in state_summary_stats:
                    out[f"state__{feature}__{stat}"] = np.nan
                continue

            if "mean" in state_summary_stats:
                out[f"state__{feature}__mean"] = _safe_scalar(np.mean(values))
            if "median" in state_summary_stats:
                out[f"state__{feature}__median"] = _safe_scalar(np.median(values))
            if "std" in state_summary_stats:
                out[f"state__{feature}__std"] = _safe_scalar(np.std(values, ddof=0))
            if "iqr" in state_summary_stats:
                q75, q25 = np.percentile(values, [75, 25])
                out[f"state__{feature}__iqr"] = _safe_scalar(q75 - q25)
            if "p10" in state_summary_stats:
                out[f"state__{feature}__p10"] = _safe_scalar(np.percentile(values, 10))
            if "p90" in state_summary_stats:
                out[f"state__{feature}__p90"] = _safe_scalar(np.percentile(values, 90))

    if phenotype_key is not None and phenotype_key in obs_sub.columns:
        phenotypes = obs_sub[phenotype_key].astype("object")
        probs = phenotypes.value_counts(normalize=True, dropna=True)

        out["state__phenotype_entropy"] = _safe_scalar(
            -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
        )
        out["state__phenotype_dominant_fraction"] = _safe_scalar(
            probs.max() if len(probs) > 0 else np.nan
        )

        for label, proportion in probs.items():
            safe_label = _sanitize_label(label)
            out[f"state__phenotype__{safe_label}__proportion"] = _safe_scalar(proportion)

    return out


def summarize_niche_graph_features(
    adata: ad.AnnData,
    niche_key: str,
    feature_cols: list[str]=None,
    state_feature_cols: list[str]=None,
    include_values: list[str] | None=None,
    phenotype_key: str=None,
    region_key: str=None,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    boundary_labels: list[str] | None=("inner_border",),
    core_labels: list[str] | None=("core",),
    exclude_labels: list[str] | None=("unassigned", "noise"),
    morphology_bin_count: int=3,
    min_cells: int=3,
    include_graph_surroundings: bool=True,
    surround_hops: int=1,
    include_state_summaries: bool=True,
    state_summary_stats: list[str]=("mean", "median", "std", "iqr", "p10", "p90"),
    lightweight: bool=False,
    show_progress: bool=False,
    progress_desc: str="Niche features",
) -> pd.DataFrame:
    """
    Summarize each niche as a graph-derived descriptor vector.

    Parameters
    ----------
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    feature_cols : list, optional
        Continuous node features for feature-organization descriptors.
    state_feature_cols : list, optional
        Continuous cell features to summarize as niche-level state descriptors.
        If ``None``, defaults to ``feature_cols``.
    include_values : iterable, optional
        If provided, only summarize these niche labels.
    phenotype_key : str, optional
        Categorical phenotype label for assortativity and entropy summaries.
    region_key : str, optional
        Optional region label column such as ``core`` / ``inner_border``.
    adjacency_key : str
        Key in ``adata.obsp`` containing the Layer 1 cell graph adjacency.
    distance_key : str
        Key in ``adata.obsp`` containing edge distances.
    boundary_labels, core_labels : tuple
        Region labels used for boundary/core contrasts.
    exclude_labels : tuple
        Niche labels to skip.
    min_cells : int
        Minimum number of cells required to summarize one niche.
    include_graph_surroundings : bool
        If ``True``, add fast graph-defined boundary/core/surround features that
        do not require precomputed geometric region labels.
    surround_hops : int
        Number of external graph hops used to define niche surroundings.
    include_state_summaries : bool
        If ``True``, append niche-level summaries of the original cell features.
    state_summary_stats : tuple
        Summary statistics used for the niche state block.
    lightweight : bool
        If ``True``, skip the most expensive topology metrics such as bridge
        fraction, diameter, and average shortest path length.
    show_progress : bool
        If ``True``, display a progress bar over niche groups.
    progress_desc : str
        Description shown in the progress bar.

    Returns
    -------
    DataFrame
        One row per niche with graph-derived descriptors.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
        raise ValueError(f"{x_key} and/or {y_key} not found in adata.obs")

    feature_cols = list(feature_cols) if feature_cols is not None else None
    if state_feature_cols is None:
        state_feature_cols = feature_cols
    else:
        state_feature_cols = list(state_feature_cols)
    adjacency = adata.obsp[adjacency_key].tocsr()
    distance_matrix = adata.obsp[distance_key] if distance_key in adata.obsp else None
    include_set = set(include_values) if include_values is not None else None
    numeric_cache = {}
    if feature_cols is not None:
        for feature in feature_cols:
            if feature in adata.obs.columns:
                numeric_cache[feature] = pd.to_numeric(
                    adata.obs[feature],
                    errors="coerce",
                ).to_numpy(dtype=float)
    phenotype_array = None
    if phenotype_key is not None and phenotype_key in adata.obs.columns:
        phenotype_array = adata.obs[phenotype_key].astype("object").to_numpy()

    rows = []
    group_cols = [image_key, niche_key] if image_key in adata.obs.columns else [niche_key]
    grouped = adata.obs.groupby(group_cols, dropna=True, observed=False)
    grouped_iter = grouped
    if show_progress:
        grouped_iter = tqdm(grouped, total=grouped.ngroups, desc=progress_desc)

    for group_key, obs_sub in grouped_iter:
        if len(group_cols) == 2:
            image_value, niche_value = group_key
        else:
            image_value = np.nan
            niche_value = group_key

        if include_set is not None and niche_value not in include_set:
            continue

        if pd.isna(niche_value) or niche_value in exclude_labels:
            continue

        idx = adata.obs_names.get_indexer(obs_sub.index)
        idx = idx[idx >= 0]

        if len(idx) < min_cells:
            continue

        obs_sub = adata.obs.iloc[idx].copy()
        coords = obs_sub[[x_key, y_key]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid_coords = np.isfinite(coords).all(axis=1)
        if valid_coords.sum() < min_cells:
            continue

        idx = idx[valid_coords]
        obs_sub = obs_sub.iloc[np.flatnonzero(valid_coords)].copy()
        coords = coords[valid_coords]

        adjacency_sub = adjacency[idx][:, idx].astype(float).tocsr()
        adjacency_binary = adjacency_sub.copy()
        adjacency_binary.data = np.ones_like(adjacency_binary.data, dtype=float)

        if adjacency_binary.nnz == 0:
            edge_rows = np.empty(0, dtype=int)
            edge_cols = np.empty(0, dtype=int)
            edge_lengths = np.empty(0, dtype=float)
        else:
            upper = triu(adjacency_binary, k=1).tocoo()
            edge_rows = upper.row.astype(int)
            edge_cols = upper.col.astype(int)

            if distance_matrix is not None:
                dist_sub = distance_matrix[idx][:, idx].tocsr()
                edge_lengths = np.asarray(dist_sub[edge_rows, edge_cols]).ravel().astype(float)
            else:
                edge_lengths = np.linalg.norm(coords[edge_cols] - coords[edge_rows], axis=1)

        G = nx.from_scipy_sparse_array(adjacency_binary)

        row = {
            niche_key: niche_value,
            "image_id": image_value,
            "n_cells": float(len(obs_sub)),
        }

        row.update(_summarize_topology(G, include_path_metrics=not lightweight))
        row.update(_summarize_skeleton_topology(coords))
        row.update(_summarize_geometry(coords, edge_rows, edge_cols, edge_lengths))
        row.update(
            _summarize_feature_organization(
                obs_sub=obs_sub,
                adjacency_binary=adjacency_binary,
                edge_rows=edge_rows,
                edge_cols=edge_cols,
                feature_cols=feature_cols,
                phenotype_key=phenotype_key,
                morphology_bin_count=morphology_bin_count,
            )
        )
        row.update(
            _summarize_boundary_core(
                obs_sub=obs_sub,
                adjacency_binary=adjacency_binary,
                feature_cols=feature_cols,
                phenotype_key=phenotype_key,
                region_key=region_key,
                boundary_labels=boundary_labels,
                core_labels=core_labels,
            )
        )
        if include_graph_surroundings:
            row.update(
                _summarize_graph_surroundings(
                    adata=adata,
                    niche_idx=idx,
                    adjacency_full=adjacency,
                    niche_key=niche_key,
                    niche_value=niche_value,
                    feature_cols=feature_cols,
                    phenotype_key=phenotype_key,
                    surround_hops=surround_hops,
                    numeric_cache=numeric_cache,
                    phenotype_array=phenotype_array,
                )
            )
        if include_state_summaries:
            row.update(
                _summarize_niche_state(
                    obs_sub=obs_sub,
                    state_feature_cols=state_feature_cols,
                    phenotype_key=phenotype_key,
                    state_summary_stats=state_summary_stats,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_niche_feature_table(
    adata: ad.AnnData,
    niche_key: str,
    feature_cols: list[str]=None,
    state_feature_cols: list[str]=None,
    phenotype_key: str=None,
    region_key: str=None,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    include_values: list[str] | None=None,
    include_prefix: str | None=None,
    exclude_values: list[str] | None=("unassigned", "noise"),
    exclude_prefixes: list[str] | None=None,
    boundary_labels: list[str] | None=("inner_border",),
    core_labels: list[str] | None=("core",),
    morphology_bin_count: int=3,
    min_cells: int=3,
    include_graph_surroundings: bool=True,
    surround_hops: int=1,
    include_state_summaries: bool=True,
    state_summary_stats: list[str]=("mean", "median", "std", "iqr", "p10", "p90"),
    lightweight: bool=False,
    show_progress: bool=False,
    progress_desc: str="Niche features",
) -> pd.DataFrame:
    """
    Build a filtered per-niche graph-descriptor table in one call.

    This is a notebook-friendly wrapper around
    ``summarize_niche_graph_features`` that first filters niche labels from
    ``adata.obs[niche_key]`` and then computes graph-derived descriptors only
    for the retained niches.

    Parameters
    ----------
    include_values : iterable, optional
        Explicit niche labels to include.
    include_prefix : str, optional
        Keep only niche labels whose string form starts with this prefix.
    exclude_values : iterable, optional
        Exact niche labels to exclude.
    exclude_prefixes : iterable, optional
        Exclude niche labels whose string form starts with any of these prefixes.

    Returns
    -------
    DataFrame
        One row per retained niche with graph-derived descriptors.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    niche_series = pd.Series(adata.obs[niche_key]).dropna()
    if niche_series.empty:
        return pd.DataFrame()

    niche_values = niche_series.unique().tolist()

    if include_values is not None:
        include_set = set(include_values)
        niche_values = [value for value in niche_values if value in include_set]

    if include_prefix is not None:
        niche_values = [
            value for value in niche_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        niche_values = [value for value in niche_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        niche_values = [
            value for value in niche_values
            if not str(value).startswith(exclude_prefixes)
        ]

    if len(niche_values) == 0:
        return pd.DataFrame()

    summary_df = summarize_niche_graph_features(
        adata=adata,
        niche_key=niche_key,
        feature_cols=feature_cols,
        state_feature_cols=state_feature_cols,
        include_values=niche_values,
        phenotype_key=phenotype_key,
        region_key=region_key,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        adjacency_key=adjacency_key,
        distance_key=distance_key,
        boundary_labels=boundary_labels,
        core_labels=core_labels,
        exclude_labels=(),
        morphology_bin_count=morphology_bin_count,
        min_cells=min_cells,
        include_graph_surroundings=include_graph_surroundings,
        surround_hops=surround_hops,
        include_state_summaries=include_state_summaries,
        state_summary_stats=state_summary_stats,
        lightweight=lightweight,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    if summary_df.empty:
        return summary_df

    return summary_df[summary_df[niche_key].isin(niche_values)].reset_index(drop=True)


def build_niche_feature_table_batched(
    adata: ad.AnnData,
    niche_key: str,
    batch_size: int=100,
    include_values: list[str] | None=None,
    show_progress: bool=False,
    progress_desc: str="Niche batches",
    **kwargs,
) -> pd.DataFrame:
    """
    Build the niche feature table in batches of niche labels.

    This is a safer option for large whole-slide datasets than trying to process
    every niche in one notebook call.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if include_values is None:
        niche_values = pd.Series(adata.obs[niche_key]).dropna().unique().tolist()
    else:
        niche_values = list(include_values)

    include_prefix = kwargs.get("include_prefix", None)
    exclude_values = kwargs.get("exclude_values", ("unassigned", "noise"))
    exclude_prefixes = kwargs.get("exclude_prefixes", None)

    if include_prefix is not None:
        niche_values = [
            value for value in niche_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        niche_values = [value for value in niche_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        niche_values = [
            value for value in niche_values
            if not str(value).startswith(exclude_prefixes)
        ]

    if len(niche_values) == 0:
        return pd.DataFrame()

    inner_show_progress = kwargs.pop("show_progress_inner", False)
    inner_progress_desc = kwargs.pop("progress_desc_inner", "Niches in batch")
    batch_frames = []
    batch_starts = range(0, len(niche_values), int(batch_size))
    if show_progress:
        total_batches = int(np.ceil(len(niche_values) / int(batch_size)))
        batch_starts = tqdm(batch_starts, total=total_batches, desc=progress_desc)

    for start in batch_starts:
        batch_values = niche_values[start:start + int(batch_size)]
        batch_df = build_niche_feature_table(
            adata=adata,
            niche_key=niche_key,
            include_values=batch_values,
            show_progress=inner_show_progress,
            progress_desc=inner_progress_desc,
            **kwargs,
        )
        if not batch_df.empty:
            batch_frames.append(batch_df)

    if len(batch_frames) == 0:
        return pd.DataFrame()

    return pd.concat(batch_frames, ignore_index=True)


def summarize_niche_surrounding_context(
    adata: ad.AnnData,
    niche_key: str,
    phenotype_key: str,
    feature_cols: list[str]=None,
    phenotype_feature_map: dict[str, list[str]]=None,
    include_values: list[str] | None=None,
    image_key: str="imageid",
    adjacency_key: str="cell_graph_connectivities",
    surround_hops: int=1,
    phenotype_labels: list[str] | None=None,
    min_cells: int=3,
    summary_stats: list[str]=("mean", "median"),
    show_progress: bool=False,
    progress_desc: str="Niche surroundings",
) -> pd.DataFrame:
    """
    Summarize the graph-defined surroundings of each niche.

    This is a notebook-friendly wrapper around the same n-hop surround logic
    used by ``summarize_niche_graph_features``, but exposes phenotype
    composition and phenotype-restricted feature summaries in a reusable table.

    Parameters
    ----------
    adata : AnnData
        Annotated table containing cell graph and metadata.
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels for surrounding-cell
        composition summaries.
    feature_cols : list, optional
        Numeric features in ``adata.obs`` to summarize across all surrounding
        cells.
    phenotype_feature_map : dict, optional
        Mapping ``{phenotype_label: [feature_cols...]}`` describing phenotype-
        restricted feature summaries to compute in the surround.
    include_values : iterable, optional
        If provided, restrict output to these niche labels.
    image_key : str
        Optional image/FOV identifier used to keep niches image-local.
    adjacency_key : str
        Key in ``adata.obsp`` holding the cell graph adjacency.
    surround_hops : int
        Number of graph hops used to define the surrounding context.
    phenotype_labels : list, optional
        Explicit phenotype labels to report. If ``None``, all observed labels
        are used.
    min_cells : int
        Minimum niche size required for summarization.
    summary_stats : tuple
        Subset of ``("mean", "median", "std")`` to compute.

    Returns
    -------
    DataFrame
        One row per niche with surrounding-context summaries.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")
    if phenotype_key not in adata.obs.columns:
        raise ValueError(f"{phenotype_key} not found in adata.obs")
    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    feature_cols = list(feature_cols) if feature_cols is not None else []
    phenotype_feature_map = phenotype_feature_map or {}
    include_set = set(include_values) if include_values is not None else None

    missing_features = [col for col in feature_cols if col not in adata.obs.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in adata.obs: {missing_features}")

    missing_map_features = []
    for cols in phenotype_feature_map.values():
        for col in cols:
            if col not in adata.obs.columns:
                missing_map_features.append(col)
    if missing_map_features:
        missing_map_features = sorted(set(missing_map_features))
        raise ValueError(f"Missing phenotype_feature_map columns in adata.obs: {missing_map_features}")

    if phenotype_labels is None:
        phenotype_labels = pd.Series(adata.obs[phenotype_key]).dropna().unique().tolist()
    else:
        phenotype_labels = list(phenotype_labels)

    numeric_cache = {}
    for feature in sorted(set(feature_cols).union(*[set(v) for v in phenotype_feature_map.values()])):
        numeric_cache[feature] = pd.to_numeric(
            adata.obs[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

    adjacency = adata.obsp[adjacency_key].tocsr()
    group_cols = [image_key, niche_key] if image_key in adata.obs.columns else [niche_key]
    grouped = adata.obs.groupby(group_cols, dropna=True, observed=False)
    grouped_iter = grouped
    if show_progress:
        grouped_iter = tqdm(grouped, total=grouped.ngroups, desc=progress_desc)

    rows = []
    for group_key, obs_sub in grouped_iter:
        if len(group_cols) == 2:
            image_value, niche_value = group_key
        else:
            image_value = np.nan
            niche_value = group_key

        if include_set is not None and niche_value not in include_set:
            continue
        if pd.isna(niche_value) or niche_value in ("unassigned", "noise"):
            continue

        idx = adata.obs_names.get_indexer(obs_sub.index)
        idx = idx[idx >= 0]
        if len(idx) < int(min_cells):
            continue

        surround_layers = _get_n_hop_external_layers(
            adjacency_full=adjacency,
            niche_idx=idx,
            max_hops=surround_hops,
        )
        surround_idx = (
            np.concatenate(list(surround_layers.values()))
            if len(surround_layers) > 0
            else np.empty(0, dtype=int)
        )

        row = {
            niche_key: niche_value,
            "image_id": image_value,
            "n_cells": float(len(idx)),
            "n_surround": float(len(surround_idx)),
        }
        for hop in range(1, int(surround_hops) + 1):
            hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
            row[f"surround__hop_{hop}__n_cells"] = float(len(hop_idx))

        if len(surround_idx) == 0:
            for label in phenotype_labels:
                safe_label = _sanitize_label(label)
                row[f"surround_prop__{safe_label}"] = 0.0
                row[f"surround__{safe_label}__n_cells"] = 0.0
            for feature in feature_cols:
                for stat in summary_stats:
                    row[f"surround__{feature}__{stat}"] = np.nan
            for label, cols in phenotype_feature_map.items():
                safe_label = _sanitize_label(label)
                for feature in cols:
                    row[f"surround__{safe_label}__{feature}__n_cells"] = 0.0
                    for stat in summary_stats:
                        row[f"surround__{safe_label}__{feature}__{stat}"] = np.nan
            rows.append(row)
            continue

        surround_obs = adata.obs.iloc[surround_idx]
        surround_pheno = surround_obs[phenotype_key].astype("object")
        pheno_probs = surround_pheno.value_counts(normalize=True, dropna=True)

        for label in phenotype_labels:
            safe_label = _sanitize_label(label)
            label_mask = surround_pheno == label
            row[f"surround_prop__{safe_label}"] = _safe_scalar(pheno_probs.get(label, 0.0))
            row[f"surround__{safe_label}__n_cells"] = float(label_mask.sum())

        for feature in feature_cols:
            values = numeric_cache[feature][surround_idx]
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                for stat in summary_stats:
                    row[f"surround__{feature}__{stat}"] = np.nan
                continue
            if "mean" in summary_stats:
                row[f"surround__{feature}__mean"] = _safe_scalar(np.mean(finite))
            if "median" in summary_stats:
                row[f"surround__{feature}__median"] = _safe_scalar(np.median(finite))
            if "std" in summary_stats:
                row[f"surround__{feature}__std"] = _safe_scalar(np.std(finite, ddof=0))

        for label, cols in phenotype_feature_map.items():
            safe_label = _sanitize_label(label)
            label_idx = surround_idx[surround_pheno.to_numpy(dtype=object) == label]
            for feature in cols:
                row[f"surround__{safe_label}__{feature}__n_cells"] = float(len(label_idx))
                values = numeric_cache[feature][label_idx] if len(label_idx) > 0 else np.array([], dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    for stat in summary_stats:
                        row[f"surround__{safe_label}__{feature}__{stat}"] = np.nan
                    continue
                if "mean" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__mean"] = _safe_scalar(np.mean(finite))
                if "median" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__median"] = _safe_scalar(np.median(finite))
                if "std" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__std"] = _safe_scalar(np.std(finite, ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


def score_pdac_niche_pathology_modules(
    feature_df: pd.DataFrame,
    niche_key: str=None,
    image_key: str="image_id",
    polarity_high_is_organized: bool=True,
    min_features_per_module: int=2,
) -> pd.DataFrame:
    """
    Score PDAC pathology-inspired niche modules from a niche feature table.

    The function is intentionally tolerant of partially available inputs: each
    module uses whatever relevant columns are present in ``feature_df`` and
    averages signed per-column z-scores across the resolved features.

    Returns a compact per-niche table containing:
    - base pathology scores, including a separate proliferation module
    - condensed trajectory-oriented scores/axes
    - per-module feature counts for transparency
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")

    df = feature_df.copy()
    out = pd.DataFrame(index=df.index)

    if niche_key is not None and niche_key in df.columns:
        out[niche_key] = df[niche_key]
    if image_key is not None and image_key in df.columns:
        out[image_key] = df[image_key]

    def _first_existing(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _zscore(col):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        out_arr = np.full(len(values), np.nan, dtype=float)
        if finite.sum() < 2:
            return out_arr
        mean = values[finite].mean()
        std = values[finite].std(ddof=0)
        if np.isclose(std, 0.0):
            return out_arr
        out_arr[finite] = (values[finite] - mean) / std
        return out_arr

    polarity_sign_pos = 1.0 if polarity_high_is_organized else -1.0
    polarity_sign_neg = -1.0 * polarity_sign_pos

    module_specs = {
        "pdac_duct_organization_score": [
            (1.0, ["geometry__hull_circularity"]),
            (1.0, ["topology__largest_component_fraction"]),
            (1.0, ["geometry__mean_nearest_neighbor_distance", "geometry__median_edge_length", "topology__skeleton_mean_edge_length"]),
            (1.0, ["state__CK19_expr_z__mean", "state__CK19_expr__mean", "state__CK19_expr__median"]),
            (1.0, ["state__NaKATPase_expr_z__mean", "state__NaKATPase_expr__mean", "state__NaKATPase_expr__median"]),
            (polarity_sign_pos, ["state__polarity_score__median", "state__polarity_score__mean"]),
            (-1.0, ["graph_boundary__boundary_fraction"]),
            (-1.0, ["graph_boundary__mean_external_degree", "graph_boundary__boundary_external_degree_mean"]),
            (-1.0, ["topology__bridge_fraction"]),
            (-1.0, ["topology__skeleton_branchpoint_fraction"]),
            (-1.0, ["geometry__spatial_entropy"]),
        ],
        "pdac_dysplasia_score": [
            (1.0, ["state__nc_ratio_z__mean", "state__nc_ratio__mean", "state__nc_ratio__median", "state__nc_ratio__p90"]),
            (1.0, ["state__boundary_irregularity_z__mean", "state__boundary_irregularity__mean"]),
            (1.0, ["state__major_minor_axis_ratio_z__mean", "state__major_minor_axis_ratio__mean"]),
            (1.0, ["state__centroid_dif_z__mean", "state__centroid_dif__mean"]),
            (1.0, ["state__num_concavities_z__mean", "state__num_concavities__mean"]),
            (polarity_sign_neg, ["state__polarity_score__median", "state__polarity_score__mean"]),
            (1.0, ["features__nc_ratio__graph_morans_i"]),
        ],
        "pdac_architectural_complexity_score": [
            (1.0, ["topology__degree_var"]),
            (1.0, ["topology__avg_clustering"]),
            (1.0, ["topology__skeleton_branchpoint_fraction"]),
            (1.0, ["topology__skeleton_leaf_fraction"]),
            (1.0, ["geometry__orientation_entropy"]),
            (1.0, ["geometry__edge_length_cv"]),
            (1.0, ["state__lacunarity_z__mean", "state__lacunarity__mean"]),
            (1.0, ["state__boundary_irregularity__mean"]),
            (-1.0, ["geometry__hull_circularity"]),
        ],
        "pdac_invasion_desmoplasia_score": [
            (1.0, ["graph_boundary__boundary_fraction"]),
            (1.0, ["graph_boundary__mean_external_degree", "graph_boundary__boundary_external_degree_mean"]),
            (1.0, ["graph_surround__cross_edges_per_niche_cell", "graph_surround__n_cross_edges"]),
            (1.0, ["graph_surround__phenotype_entropy", "graph_surround__hop_1__phenotype_entropy"]),
            (1.0, ["graph_surround__hop_1__fraction_of_niche", "graph_surround__surround_to_niche_ratio"]),
            (1.0, ["surround_prop__Fibroblasts", "surround__Fibroblasts__fraction"]),
            (1.0, ["surround__Fibroblasts__FAP_expr_z__mean", "surround__Fibroblasts__FAP_expr__mean", "fibro_surround__FAP_expr_z__mean", "fibro_surround__FAP_expr__mean"]),
            (1.0, ["surround__Fibroblasts__aSMA_expr_z__mean", "surround__Fibroblasts__aSMA_expr__mean", "fibro_surround__aSMA_expr_z__mean", "fibro_surround__aSMA_expr__mean"]),
            (-1.0, ["geometry__hull_circularity"]),
        ],
        "pdac_proliferation_score": [
            (1.0, ["state__Ki67_expr_z__mean", "state__Ki67_expr__mean", "state__Ki67_expr__median"]),
            (1.0, ["state__Ki67_expr_z__p90", "state__Ki67_expr__p90"]),
        ],
        "pdac_dedifferentiation_score": [
            (1.0, ["state__entropy_z__mean", "state__entropy__mean"]),
            (1.0, ["state__inertia_z__mean", "state__inertia__mean"]),
            (1.0, ["state__haralick_contrast__mean"]),
            (1.0, ["geometry__cell_density_hull", "geometry__edge_density_hull"]),
            (1.0, ["topology__skeleton_branchpoint_fraction"]),
            (-1.0, ["geometry__mean_nearest_neighbor_distance", "topology__skeleton_mean_edge_length"]),
            (-1.0, ["geometry__hull_circularity"]),
            (polarity_sign_neg, ["state__polarity_score__median", "state__polarity_score__mean"]),
        ],
    }

    resolved_map = {}
    z_cache = {}

    for module_name, spec in module_specs.items():
        signed_parts = []
        resolved_features = []
        for sign, candidates in spec:
            resolved = _first_existing(candidates)
            if resolved is None:
                continue
            if resolved not in z_cache:
                z_cache[resolved] = _zscore(resolved)
            signed_parts.append(sign * z_cache[resolved])
            resolved_features.append((resolved, float(sign)))

        resolved_map[module_name] = resolved_features
        out[f"{module_name}__n_features"] = float(len(resolved_features))

        if len(signed_parts) < int(min_features_per_module):
            out[module_name] = np.nan
            continue

        stacked = np.vstack(signed_parts)
        out[module_name] = np.nanmean(stacked, axis=0)

    def _module_z(col):
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        result = np.full(len(values), np.nan, dtype=float)
        if finite.sum() < 2:
            return result
        mean = values[finite].mean()
        std = values[finite].std(ddof=0)
        if np.isclose(std, 0.0):
            return result
        result[finite] = (values[finite] - mean) / std
        return result

    base_cols = [
        "pdac_duct_organization_score",
        "pdac_dysplasia_score",
        "pdac_architectural_complexity_score",
        "pdac_invasion_desmoplasia_score",
        "pdac_proliferation_score",
        "pdac_dedifferentiation_score",
    ]
    base_z = {col: _module_z(col) for col in base_cols}

    condensed_specs = {
        "pdac_early_duct_anchor_score": [
            base_z["pdac_duct_organization_score"],
            -base_z["pdac_dysplasia_score"],
            -base_z["pdac_invasion_desmoplasia_score"],
            -base_z["pdac_proliferation_score"],
            -base_z["pdac_dedifferentiation_score"],
        ],
        "pdac_panin_like_dysplasia_score": [
            base_z["pdac_dysplasia_score"],
            base_z["pdac_architectural_complexity_score"],
            -base_z["pdac_duct_organization_score"],
        ],
        "pdac_invasive_gland_forming_score": [
            base_z["pdac_duct_organization_score"],
            base_z["pdac_invasion_desmoplasia_score"],
            base_z["pdac_proliferation_score"],
            -base_z["pdac_dedifferentiation_score"],
        ],
        "pdac_invasion_desmoplasia_axis": [
            base_z["pdac_invasion_desmoplasia_score"],
        ],
        "pdac_proliferation_axis": [
            base_z["pdac_proliferation_score"],
        ],
        "pdac_dedifferentiation_axis": [
            base_z["pdac_dedifferentiation_score"],
            -base_z["pdac_duct_organization_score"],
        ],
    }

    for module_name, arrays in condensed_specs.items():
        stacked = np.vstack(arrays)
        finite_counts = np.isfinite(stacked).sum(axis=0)
        out[f"{module_name}__n_features"] = finite_counts.astype(float)
        out[module_name] = np.nanmean(stacked, axis=0)
        out.loc[finite_counts == 0, module_name] = np.nan

    out.attrs["resolved_module_features"] = resolved_map
    return out



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
