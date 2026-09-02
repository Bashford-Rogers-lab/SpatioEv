"""Spatial component clustering and niche boundary geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad

import os
import re

from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
)
from scipy.ndimage import (
    find_objects as ndi_find_objects,
)
from scipy.ndimage import (
    label as ndi_label,
)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.io import imread
from skimage.morphology import disk
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:  # pragma: no cover - depends on sklearn version
    SklearnHDBSCAN = None


from ._helpers import (
    _ensure_list,
    _estimate_min_samples,
    _label_suffix,
    _make_component_geometry,
    _require_shapely,
    _resolve_k,
)


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

    _require_shapely()
    # Shapely 2.x exposes vectorised predicates that operate on whole point
    # arrays. Testing cells one at a time made this O(boundaries x cells) in
    # Python; these run the same predicates in one native call per boundary.
    import shapely

    frames = []
    obs = adata.obs.copy()

    valid_cells = obs[[image_key, x_key, y_key]].dropna().index
    obs = obs.loc[valid_cells]

    # Group once rather than filtering the full table per boundary row.
    by_image = {img: sub for img, sub in obs[[image_key, x_key, y_key]].groupby(image_key)}

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

        sub = by_image.get(img)
        if sub is None or sub.empty:
            continue

        points = shapely.points(
            sub[x_key].to_numpy(dtype=float),
            sub[y_key].to_numpy(dtype=float),
        )

        region = np.full(len(sub), None, dtype=object)

        if mode == "distance_to_edge":
            inside = shapely.covers(geom, points)
            if inside.any():
                edge_distance = shapely.distance(geom.boundary, points[inside])
                inner = (width > 0) & (edge_distance <= width)
                region[np.flatnonzero(inside)] = np.where(inner, "inner_border", "core")
            if expanded is not None:
                outer = ~inside & shapely.covers(expanded, points)
                region[outer] = "outer_border"
        else:
            remaining = np.ones(len(sub), dtype=bool)
            if shrunk is not None:
                core = shapely.covers(shrunk, points)
                region[core] = "core"
                remaining &= ~core
            inner = remaining & shapely.covers(geom, points)
            region[inner] = "inner_border" if shrunk is not None else "component"
            remaining &= ~inner
            if expanded is not None:
                outer = remaining & shapely.covers(expanded, points)
                region[outer] = "outer_border"

        assigned = region != None  # noqa: E711 - element-wise, not an identity test
        if not assigned.any():
            continue

        frames.append(
            pd.DataFrame({
                "cell_id": sub.index[assigned],
                image_key: img,
                component_key: component,
                region_key: region[assigned],
            })
        )

    if not frames:
        return pd.DataFrame(columns=["cell_id", image_key, component_key, region_key])

    return pd.concat(frames, ignore_index=True)


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
