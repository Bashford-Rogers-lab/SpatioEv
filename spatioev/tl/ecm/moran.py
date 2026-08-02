"""Moran's I over ECM fibers and ECM-cell coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad
from sklearn.neighbors import kneighbors_graph

from ._helpers import (
    _compute_cross_morans_i_from_fiber_table,
    _ensure_list,
    _filter_cells,
    _label_suffix,
    _map_cell_feature_to_fibers,
    _permute_values_within_images,
    _resolve_k,
    _subset_links,
)

# ============================================================
# 4. Moran’s I for ECM structure
# ============================================================

def morans_i_fibers(
    fiber_df: pd.DataFrame,
    feature: str,
    k: int=8,
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
) -> pd.DataFrame:
    """
    Spatial autocorrelation of ECM features.

    Biological application
    ----------------------
    Detects spatial clustering of ECM morphology or architecture.

    Example questions
    -----------------
    Do long collagen fibers cluster spatially?
    Do aligned fibers form stromal invasion tracks?
    Do dense ECM bundles form fibrotic niches?

    Parameters
    ----------
    fiber_df : DataFrame
        ECM fiber dataset

    feature : str
        Fiber feature to analyze
        Example:
            major_axis_length
            eccentricity
            alignment_score

    k : int
        Number of nearest neighbors used to construct spatial graph

    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.

    fiber_type : str or list, optional
        Restrict analysis to specific fiber types
        Example:
            "collagen"
            ["collagen", "fibronectin"]

    Returns
    -------
    float
        Moran's I spatial autocorrelation statistic

    Interpretation
    --------------
    Moran's I > 0
        clustering of similar ECM features
        stromal remodeling zones

    Moran's I ≈ 0
        random ECM organization

    Moran's I < 0
        spatial dispersion of ECM features
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    fiber_type = _ensure_list(fiber_type)

    if fiber_type is not None:

        fiber_subset = fiber_subset[
            fiber_subset[fiber_type_key].isin(fiber_type)
        ]

    if fiber_subset.empty:
        raise ValueError("No fibers found for specified fiber_type")

    # --------------------------------------------------
    # Extract coordinates and feature
    # --------------------------------------------------

    valid = (
        fiber_subset[[fiber_x, fiber_y, feature]]
        .notna()
        .all(axis=1)
    )

    fiber_subset = fiber_subset[valid]

    coords = fiber_subset[[fiber_x, fiber_y]].to_numpy()
    values = fiber_subset[feature].to_numpy()

    if len(values) < 3:
        return np.nan

    k_eff = _resolve_k(len(values), k)

    if k_eff is None:
        return np.nan

    # --------------------------------------------------
    # Spatial weights (kNN graph)
    # --------------------------------------------------

    W = kneighbors_graph(coords, k_eff, mode="connectivity").toarray()

    # NORMALIZATION FIX
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # --------------------------------------------------
    # Moran's I calculation
    # --------------------------------------------------

    x = values - values.mean()

    denom = np.sum(x**2)

    if denom == 0:
        return np.nan

    return (len(x) / W.sum()) * (
        np.sum(W * np.outer(x, x)) / denom
    )


# ============================================================
# 5. Local Moran’s I for ECM features
# ============================================================
def local_morans_i_fibers(
    fiber_df: pd.DataFrame,
    feature: str,
    k: int=8,
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
) -> np.ndarray:
    """
    Local Moran's I spatial autocorrelation for ECM features.

    Computes a spatial clustering statistic for each fiber.

    Biological application
    ----------------------
    Detects spatial hotspots of ECM remodeling.

    Examples
    --------
    aligned collagen tracks
    fibrotic ECM bundles
    stromal remodeling zones near tumors

    Parameters
    ----------
    fiber_df : DataFrame
        ECM fiber dataset

    feature : str
        Fiber feature to analyze
        Example:
            major_axis_length
            eccentricity
            alignment_score

    k : int
        Number of nearest neighbors

    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.

    fiber_type : str or list, optional
        Restrict analysis to specific fiber types

    Returns
    -------
    DataFrame
        Original dataframe with added column:
            local_moran_<feature>

    Interpretation
    --------------
    high positive values
        clusters of similar ECM features

    negative values
        spatial contrast / boundary regions
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    fiber_type = _ensure_list(fiber_type)

    if fiber_type is not None:

        fiber_subset = fiber_subset[
            fiber_subset[fiber_type_key].isin(fiber_type)
        ]

    if fiber_subset.empty:
        raise ValueError("No fibers found for specified fiber_type")

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    valid = (
        fiber_subset[[fiber_x, fiber_y, feature]]
        .notna()
        .all(axis=1)
    )

    fiber_subset = fiber_subset[valid]

    coords = fiber_subset[[fiber_x, fiber_y]].to_numpy()
    values = fiber_subset[feature].to_numpy()

    n = len(values)

    if n < 3:
        result = fiber_df.copy()
        result[f"local_moran_{feature}"] = np.nan
        return result

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        result = fiber_df.copy()
        result[f"local_moran_{feature}"] = np.nan
        return result

    # --------------------------------------------------
    # Build spatial weights
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k_eff,
        mode="connectivity",
        include_self=False
    )

    W = W.toarray()

    # --------------------------------------------------
    # Center feature values
    # --------------------------------------------------

    x = values - values.mean()

    m2 = np.sum(x**2) / n

    if m2 == 0:
        result = fiber_df.copy()
        result[f"local_moran_{feature}"] = np.nan
        return result

    # --------------------------------------------------
    # Compute Local Moran's I
    # --------------------------------------------------

    local_I = x * (W @ x) / m2

    # --------------------------------------------------
    # Store result
    # --------------------------------------------------

    result = fiber_df.copy()

    result.loc[fiber_subset.index, f"local_moran_{feature}"] = local_I

    return result


# ============================================================
# 6. Cross Moran’s I (ECM–cell coupling)
# ============================================================
def cross_morans_i_ecm_cells(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    feature_fiber: str,
    cell_feature: str,
    k: int=8,
) -> pd.DataFrame:
    
    """
    Cross Moran's I between ECM features and nearby cell features.

    Automatically maps cell features from adata.obs onto fiber locations
    using spatial proximity.

    Biological application
    ----------------------
    Detects spatial coupling between ECM architecture and nearby
    cellular niches.

    Example questions
    -----------------
    Do aligned collagen fibers occur near tumor cells?
    Do thick ECM bundles occur near macrophage clusters?

    Parameters
    ----------
    feature_fiber : str
        Column in ``fiber_df`` containing the ECM feature to analyze.
    cell_feature : str
        Column in ``adata.obs`` containing the cell feature to aggregate onto fibers.
    k : int
        Number of nearest neighbors used to define the fiber-level spatial graph.

    Returns
    -------
    float
        cross Moran's I statistic
    """
    fiber_cell = _map_cell_feature_to_fibers(
        adata,
        fiber_df,
        links_df,
        cell_feature,
    )

    fiber_local = fiber_df.copy()
    fiber_local["cell_local"] = fiber_local.index.map(fiber_cell)

    return _compute_cross_morans_i_from_fiber_table(
        fiber_local,
        feature_fiber,
        "cell_local",
        k=k,
    )


def cross_morans_i_ecm_cells_permutation_test(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    feature_fiber: str,
    cell_feature: str,
    k: int=8,
    n_sim: int=999,
    image_key: str="imageid",
    random_state: int=None,
) -> pd.DataFrame:
    """
    Permutation test for global cross Moran's I between ECM and cell features.

    Preserves cell and fiber coordinates while shuffling the cell feature
    within each image.

    Parameters
    ----------
    feature_fiber : str
        Column in ``fiber_df`` containing the ECM feature to analyze.
    cell_feature : str
        Column in ``adata.obs`` containing the cell feature to aggregate onto fibers.
    k : int
        Number of nearest neighbors used to define the fiber-level spatial graph.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.

    Returns
    -------
    dict
        observed
        p_value
        null_mean
        null_std
        z_score
        n_sim
    """
    observed = cross_morans_i_ecm_cells(
        adata,
        fiber_df,
        links_df,
        feature_fiber,
        cell_feature,
        k=k,
    )

    if not np.isfinite(observed):
        return {
            "observed": np.nan,
            "p_value": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "z_score": np.nan,
            "n_sim": 0,
        }

    rng = np.random.default_rng(random_state)
    cell_values = adata.obs[cell_feature].to_numpy()
    cell_images = adata.obs[image_key].to_numpy()
    sims = []

    for i in range(n_sim):
        permuted = _permute_values_within_images(
            cell_values,
            cell_images,
            rng,
        )

        fiber_cell = _map_cell_feature_to_fibers(
            adata,
            fiber_df,
            links_df,
            cell_feature,
            cell_values=permuted,
            image_key=image_key,
        )

        fiber_local = fiber_df.copy()
        fiber_local["cell_local"] = fiber_local.index.map(fiber_cell)

        sim_stat = _compute_cross_morans_i_from_fiber_table(
            fiber_local,
            feature_fiber,
            "cell_local",
            k=k,
        )

        if np.isfinite(sim_stat):
            sims.append(sim_stat)

    if len(sims) == 0:
        raise ValueError(
            "All permutation simulations returned invalid statistics."
        )

    sims = np.asarray(sims, dtype=float)
    null_mean = sims.mean()
    null_std = sims.std(ddof=1) if len(sims) > 1 else 0.0

    # Two-sided Monte Carlo p-value with +1 correction.
    p_value = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (len(sims) + 1)

    if null_std == 0:
        z_score = np.nan
    else:
        z_score = (observed - null_mean) / null_std

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_sim": len(sims),
    }



# helper function for local cross Moran's I
def map_cells_to_fibers(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    phenotype_key: str,
    phenotype: str,
) -> pd.DataFrame:
    """
    Map nearby cell features onto ECM fiber locations.

    Computes spatial relationships between cells and fibers.

    Generated features
    ------------------
    <phenotype>_density
        number of cells within radius of each fiber

    <phenotype>_presence
        whether any cells exist near fiber

    <phenotype>_mean_distance
        mean distance to nearby cells

    Biological applications
    -----------------------
    tumor_density
        tumor invasion tracks

    CD8_density
        immune infiltration

    macrophage_density
        stromal immune niches

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list
        Cell phenotype label or list of labels to map onto fibers.
    """

    phenotype = _ensure_list(phenotype)
    cells = _filter_cells(adata, phenotype_key, phenotype)
    links = _subset_links(links_df, cells, fiber_df)

    density = links.groupby("fiber_id").size()
    presence = density.gt(0).astype(float)
    mean_dist = links.groupby("fiber_id")["distance"].mean()

    name = _label_suffix(phenotype, "cells")

    fiber_df = fiber_df.copy()
    fiber_df[f"{name}_density"] = np.nan
    fiber_df[f"{name}_presence"] = np.nan
    fiber_df[f"{name}_mean_distance"] = np.nan

    fiber_df[f"{name}_density"] = fiber_df.index.map(density)
    fiber_df[f"{name}_presence"] = fiber_df.index.map(presence)
    fiber_df[f"{name}_mean_distance"] = fiber_df.index.map(mean_dist)

    return fiber_df



def map_cells_to_fibers_kernel(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    phenotype_key: str,
    phenotype: str,
    bandwidth: float=50,
) -> pd.DataFrame:
    """
    Map cell influence onto ECM fibers using Gaussian kernel density.

    Produces smooth spatial signals of cell presence around fibers.

    Generated features
    ------------------
    <phenotype>_kernel_density
        distance-weighted cell density

    <phenotype>_nearest_distance
        distance to nearest cell

    Biological interpretation
    -------------------------
    high kernel density
        strong cellular niche around fiber

    low kernel density
        fiber located in cell-poor region

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list
        Cell phenotype label or list of labels to map onto fibers.
    bandwidth : float
        Gaussian kernel bandwidth in the same spatial units as coordinates.
    """

    phenotype = _ensure_list(phenotype)
    cells = _filter_cells(adata, phenotype_key, phenotype)

    links = _subset_links(links_df, cells, fiber_df).copy()

    d = links["distance"].to_numpy()
    links["weight"] = np.exp(-(d**2)/(2*bandwidth**2))

    kernel = links.groupby("fiber_id")["weight"].sum()
    nearest = links.groupby("fiber_id")["distance"].min()

    name = _label_suffix(phenotype, "cells")

    fiber_df = fiber_df.copy()
    fiber_df[f"{name}_kernel_density"] = np.nan
    fiber_df[f"{name}_nearest_distance"] = np.nan

    fiber_df[f"{name}_kernel_density"] = fiber_df.index.map(kernel)
    fiber_df[f"{name}_nearest_distance"] = fiber_df.index.map(nearest)

    return fiber_df


# ============================================================
# 7. Local cross Moran’s I
# ============================================================

def local_cross_morans_i_ecm_cells(
    fiber_df: pd.DataFrame,
    feature_fiber: str,
    feature_cell: str,
    k: int=8,
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
) -> np.ndarray:
    """
    Local cross Moran's I.

    Produces spatial map of ECM–cell coupling.

    Biological application
    ----------------------
    Detects spatial hotspots where ECM morphology
    interacts with cellular niches.

    Example interpretations
    -----------------------
    high positive
        aligned ECM near tumor invasion fronts

    high negative
        aligned ECM associated with immune exclusion

    Parameters
    ----------
    feature_fiber : str
        Column in ``fiber_df`` containing the ECM feature to analyze.
    feature_cell : str
        Column in ``fiber_df`` containing a cell-derived feature already mapped to fibers.
    k : int
        Number of nearest neighbors used to define the fiber-level spatial graph.
    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    fiber_type = _ensure_list(fiber_type)

    if fiber_type is not None:

        fiber_subset = fiber_subset[
            fiber_subset[fiber_type_key].isin(fiber_type)
        ]

    if fiber_subset.empty:
        raise ValueError("No fibers found for specified fiber_type")

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    valid = (
        fiber_subset[[fiber_x, fiber_y, feature_fiber, feature_cell]]
        .notna()
        .all(axis=1)
    )

    fiber_subset = fiber_subset[valid]

    coords = fiber_subset[[fiber_x, fiber_y]].to_numpy()

    x = fiber_subset[feature_fiber].to_numpy(dtype=float)
    y = fiber_subset[feature_cell].to_numpy(dtype=float)

    n = len(x)

    if n < 3:
        result = fiber_df.copy()
        result[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return result

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        result = fiber_df.copy()
        result[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return result

    # --------------------------------------------------
    # Spatial weights
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k_eff,
        mode="connectivity",
        include_self=False
    ).toarray()

    # row normalization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # --------------------------------------------------
    # Center variables
    # --------------------------------------------------

    x_c = x - x.mean()
    y_c = y - y.mean()

    m2 = np.sum(y_c**2) / n

    if m2 == 0:
        result = fiber_df.copy()
        result[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return result

    local_I = x_c * (W @ y_c) / m2

    result = fiber_df.copy()

    result.loc[
        fiber_subset.index,
        f"local_cross_moran_{feature_fiber}_{feature_cell}"
    ] = local_I

    return result



# ============================================================
