"""Shared internals for the ECM submodules (private)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..._core.neighbors import cross_morans_i_from_weights, knn_weights
from ..preprocessing import compute_convex_hull_area

if TYPE_CHECKING:  # pragma: no cover
    pass


# ============================================================
# Helper functions
# ============================================================
def _filter_fibers(
    fiber_df,
    fiber_type_key,
    fiber_type,
):
    fiber_type = _ensure_list(fiber_type)

    if fiber_type is None:
        return fiber_df

    return fiber_df[
        fiber_df[fiber_type_key].isin(fiber_type)
    ]

def _filter_cells(
    adata,
    phenotype_key,
    phenotype,
):
    phenotype = _ensure_list(phenotype)

    cells = adata.obs

    if phenotype is None:
        return cells

    if phenotype_key is None:
        raise ValueError(
            "phenotype_key must be provided when phenotype filtering is used."
        )

    return cells[cells[phenotype_key].isin(phenotype)]


def _subset_links(
    links_df, 
    cells, 
    fibers,
    cell_image_key="imageid",
    fiber_image_key="imageid",
    link_image_key="imageid",
):
    """
    CRITICAL FIX:
    Ensures links are restricted to same image + valid ids
    """
    valid_images = np.intersect1d(
        cells[cell_image_key].unique(),
        fibers[fiber_image_key].unique()
    )

    links = links_df.copy()

    links = links[links[link_image_key].isin(valid_images)]
    links = links[links["cell_id"].isin(cells.index)]
    links = links[links["fiber_id"].isin(fibers.index)]

    return links


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


def _clean_coords(coords):
    """
    Drop rows with non-finite x/y coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be an Nx2 array.")

    return coords[np.isfinite(coords).all(axis=1)]


def _resolve_k(n_obs, k):
    """
    Ensure kNN graph construction uses a valid number of neighbors.
    """
    if n_obs < 2:
        return None

    return min(k, n_obs - 1)


def _cross_ripleys_curve_from_links(
    cells,
    fibers,
    links,
    radii,
    cell_x="X_centroid",
    cell_y="Y_centroid",
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    check_radius=True,
):
    """
    Compute cross Ripley's K/L curve from pre-filtered cells, fibers, and links.
    """
    if cells.empty or fibers.empty:
        return pd.DataFrame()

    cell_valid = cells[[cell_x, cell_y]].notna().all(axis=1)
    fiber_valid = fibers[[fiber_x, fiber_y]].notna().all(axis=1)

    cells = cells.loc[cell_valid]
    fibers = fibers.loc[fiber_valid]
    links = _subset_links(links, cells, fibers)

    if cells.empty or fibers.empty:
        return pd.DataFrame()

    cell_coords = _clean_coords(cells[[cell_x, cell_y]].to_numpy())
    fiber_coords = _clean_coords(fibers[[fiber_x, fiber_y]].to_numpy())

    area = compute_convex_hull_area(
        np.vstack([cell_coords, fiber_coords])
    )

    if np.isnan(area):
        return pd.DataFrame()

    if links.empty:
        if check_radius:
            raise ValueError(
                "No cell-fiber links found. Increase link radius to cover the requested radii."
            )

        counts = np.zeros(len(radii), dtype=float)
    else:
        if check_radius and max(radii) > links["distance"].max():
            raise ValueError("Increase link radius to cover all radii")

        counts = np.array(
            [(links["distance"] <= r).sum() for r in radii],
            dtype=float,
        )

    n_cells = len(cells)
    n_fibers = len(fibers)

    K_vals = (area / (n_cells * n_fibers)) * counts
    L_vals = np.sqrt(K_vals / np.pi)

    return pd.DataFrame({
        "radius": radii,
        "K": K_vals,
        "L": L_vals,
        "L_minus_r": L_vals - radii,
    })


def _permute_mask_within_images(mask, image_ids, rng):
    """
    Permute a boolean selection mask independently within each image.
    """
    mask = np.asarray(mask, dtype=bool)
    image_ids = np.asarray(image_ids)

    permuted = np.zeros_like(mask, dtype=bool)

    for img in pd.unique(image_ids):
        idx = np.where(image_ids == img)[0]
        permuted[idx] = rng.permutation(mask[idx])

    return permuted


def _permute_values_within_images(values, image_ids, rng):
    """
    Permute feature values independently within each image.
    """
    values = np.asarray(values)
    image_ids = np.asarray(image_ids)

    permuted = values.copy()

    for img in pd.unique(image_ids):
        idx = np.where(image_ids == img)[0]
        permuted[idx] = rng.permutation(values[idx])

    return permuted


def _map_cell_feature_to_fibers(
    adata,
    fiber_df,
    links_df,
    cell_feature,
    cell_values=None,
    image_key="imageid",
    fiber_image_key="imageid",
    weight_scale=50,
):
    """
    Aggregate a cell feature onto fibers using distance-weighted links.
    """
    valid_images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[fiber_image_key].unique(),
    )

    cells = adata.obs[adata.obs[image_key].isin(valid_images)]
    fibers = fiber_df[fiber_df[fiber_image_key].isin(valid_images)]
    links = _subset_links(
        links_df,
        cells,
        fibers,
        cell_image_key=image_key,
        fiber_image_key=fiber_image_key,
        link_image_key="imageid",
    )

    if cell_values is None:
        cell_values = adata.obs[cell_feature]

    cell_values = pd.Series(cell_values, index=adata.obs.index, name=cell_feature)

    merged = links.merge(
        cell_values.to_frame(),
        left_on="cell_id",
        right_index=True,
    )

    merged = merged[np.isfinite(merged[cell_feature])]

    if merged.empty:
        return pd.Series(dtype=float)

    merged["weight"] = np.exp(-merged["distance"] / weight_scale)

    return merged.groupby("fiber_id")[[cell_feature, "weight"]].apply(
        lambda x: np.average(x[cell_feature], weights=x["weight"])
    )


def _fiber_cross_moran_inputs(
    fiber_df,
    feature_fiber,
    feature_cell,
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
):
    """Extract the valid (coords, x, y) triple shared by the cross-Moran paths."""
    valid = fiber_df[[fiber_x, fiber_y, feature_fiber, feature_cell]].dropna()
    coords = fiber_df.loc[valid.index, [fiber_x, fiber_y]].to_numpy()
    x = valid[feature_fiber].to_numpy(dtype=float)
    y = valid[feature_cell].to_numpy(dtype=float)
    return valid.index, coords, x, y


def _compute_cross_morans_i_from_fiber_table(
    fiber_df,
    feature_fiber,
    feature_cell,
    k=8,
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    W=None,
):
    """
    Compute global cross Moran's I from fiber-level features.

    Parameters
    ----------
    W : scipy.sparse matrix, optional
        Precomputed row-normalised spatial weight matrix over the *valid*
        fibers. Supplying it avoids rebuilding the kNN graph, which is the
        dominant cost when this is called repeatedly under permutation with
        fixed fiber coordinates.
    """
    _, coords, x, y = _fiber_cross_moran_inputs(
        fiber_df, feature_fiber, feature_cell, fiber_x, fiber_y
    )

    n = len(x)

    if n < 3:
        return np.nan

    if W is None:
        k_eff = _resolve_k(n, k)

        if k_eff is None:
            return np.nan

        W = knn_weights(coords, k_eff, normalize=True)

    return cross_morans_i_from_weights(W, x, y)
