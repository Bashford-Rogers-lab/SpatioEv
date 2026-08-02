"""Moran's I: global, local, and cross-variable spatial autocorrelation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, kneighbors_graph

from ..._core.neighbors import (
    knn_weights,
    morans_i_batch,
    morans_i_from_weights,
)
from ._helpers import _PERMUTATION_BLOCK, _resolve_k

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad


# ============================================================
# 6. MORAN'S I (GLOBAL SPATIAL AUTOCORRELATION)
# ============================================================

def morans_i(
    coords: np.ndarray,
    values: np.ndarray,
    k: int = 8,
    *,
    W=None,
) -> float:
    """Compute the global Moran's I spatial autocorrelation statistic.

    Moran's I measures whether similar feature values cluster spatially
    (positive autocorrelation) or repel each other (negative autocorrelation).

    Interpretation:
        I > 0  → spatial clustering (similar values are neighbours)
        I ≈ 0  → spatial randomness (CSR)
        I < 0  → spatial dispersion (dissimilar values are neighbours)

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial (x, y) coordinates for each cell.
    values : array-like of shape (n,)
        Continuous feature measured at each coordinate (e.g. marker
        expression, area, local density).  Non-finite values are dropped.
    k : int
        Number of nearest neighbours used to build the spatial weight
        matrix.  Default is 8.

    Returns
    -------
    float
        Moran's I statistic.  Returns ``np.nan`` if fewer than 3 valid
        observations are present or if all values are identical.

    Examples
    --------
    >>> import numpy as np
    >>> import spatioev as sv
    >>> coords = np.random.uniform(0, 500, (300, 2))
    >>> values = np.random.normal(size=300)
    >>> I = sv.tl.morans_i(coords, values)
    >>> -1.0 < I < 1.0
    True
    """

    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)

    coords = coords[valid]
    values = values[valid]

    n = len(values)

    if n < 3:
        return np.nan

    if W is None:
        k_eff = _resolve_k(n, k)

        if k_eff is None:
            return np.nan

        # build spatial neighbor graph
        W = knn_weights(coords, k_eff)

    return morans_i_from_weights(W, values)


def morans_i_permutation_test(coords: np.ndarray, values: np.ndarray, k: int=8, n_sim: int=999, random_state: int=None) -> dict:
    """
    Permutation test for global Moran's I.
    """
    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)
    coords = coords[valid]
    values = values[valid]

    observed = morans_i(coords, values, k=k)

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
    n = len(values)

    # The coordinates are identical across permutations, so the spatial weight
    # matrix is built once and reused; only the values are shuffled. The
    # statistics are then evaluated in blocks with a single sparse product per
    # block instead of one graph build plus one product per simulation.
    k_eff = _resolve_k(n, k)
    if k_eff is None:
        return {
            "observed": observed,
            "p_value": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "z_score": np.nan,
            "n_sim": 0,
        }

    W = knn_weights(coords, k_eff)

    sims = []
    remaining = n_sim
    while remaining > 0:
        block = min(_PERMUTATION_BLOCK, remaining)
        # (n, block) matrix of independently permuted value vectors
        V = np.empty((n, block), dtype=float)
        for j in range(block):
            V[:, j] = rng.permutation(values)
        stats_block = morans_i_batch(W, V)
        sims.append(stats_block[np.isfinite(stats_block)])
        remaining -= block

    sims = np.concatenate(sims) if sims else np.empty(0, dtype=float)

    if len(sims) == 0:
        raise ValueError("All permutation simulations returned invalid statistics.")

    null_mean = sims.mean()
    null_std = sims.std(ddof=1) if len(sims) > 1 else 0.0
    p_value = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (len(sims) + 1)
    z_score = np.nan if null_std == 0 else (observed - null_mean) / null_std

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_sim": len(sims),
    }


def morans_i_by_image(
    adata: ad.AnnData,
    value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8, # number of neighbors for spatial weights
) -> pd.DataFrame:
    """
    Compute Moran's I per image for a spatial feature.

    Parameters
    ----------
    value_key : str
        Column in ``adata.obs`` containing the numeric feature to analyze.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    rows = []

    for img in adata.obs[image_key].unique():

        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()

        values = adata.obs.loc[idx, value_key].to_numpy()

        rows.append({
            image_key: img,
            "morans_i": morans_i(coords, values, k=k)
        })

    return pd.DataFrame(rows)


def morans_i_by_image_permutation_test(
    adata: ad.AnnData,
    value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
    n_sim: int=999,
    random_state: int=None,
) -> pd.DataFrame:
    """
    Permutation test for Moran's I per image.
    """
    rows = []
    rng = np.random.default_rng(random_state)

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]
        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        values = adata.obs.loc[idx, value_key].to_numpy()

        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        stats = morans_i_permutation_test(
            coords,
            values,
            k=k,
            n_sim=n_sim,
            random_state=seed,
        )

        rows.append({
            image_key: img,
            **stats,
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. LOCAL MORAN'S I (SPATIAL HOTSPOTS)
# ============================================================

def local_morans_i(
        coords: np.ndarray, 
        values: np.ndarray, 
        k: int=8 # number of neighbors for spatial weights
    ) -> np.ndarray:
    """
    Local Moran's I.

    Detects spatial hotspots of similar values.

    High positive values indicate clusters of high values.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates for each observation.
    values : array-like of shape (n,)
        Continuous feature measured at each coordinate.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)

    coords_v = coords[valid]
    values_v = values[valid]

    out = np.full(len(values), np.nan)

    n = len(values_v)

    if n < 3:
        return out

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return out

    W = kneighbors_graph(
        coords_v,
        k_eff,
        mode="connectivity",
        include_self=False,
    )

    x = values_v - values_v.mean()

    m2 = np.sum(x ** 2) / n

    if m2 == 0:
        return out

    local_I = x * (W @ x) / m2

    out[np.where(valid)[0]] = local_I

    return out


def add_local_morans_i(
    adata: ad.AnnData,
    value_key: str,
    out_key: str=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> ad.AnnData:
    """
    Compute Local Moran's I and add to adata.obs.

    Each cell receives a spatial autocorrelation value.

    Parameters
    ----------
    value_key : str
        Column in ``adata.obs`` containing the numeric feature to analyze.
    out_key : str, optional
        Name of the output column added to ``adata.obs``.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    if out_key is None:
        out_key = f"local_morans_i__{value_key}"

    adata.obs[out_key] = np.nan

    for img in adata.obs[image_key].unique():

        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()

        values = adata.obs.loc[idx, value_key].to_numpy()

        local_I = local_morans_i(coords, values, k=k)

        adata.obs.loc[idx, out_key] = local_I

    return adata


def classify_local_morans_i(
    values: np.ndarray,
    local_i: np.ndarray,
    value_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> np.ndarray:
    """
    Classify local Moran's I into hotspot/coldspot/outlier quadrants.

    Parameters
    ----------
    values : array-like
        Feature values, typically centered or z-scored so that 0 separates
        high from low.
    local_i : array-like
        Local Moran's I values for the same observations.
    value_threshold : float, default 0.0
        Threshold separating high vs low feature values.
    local_i_threshold : float, default 0.0
        Threshold separating positive vs negative local Moran's I.

    Returns
    -------
    np.ndarray of dtype object
        One of: ``high-high``, ``low-low``, ``high-low``, ``low-high``,
        or ``unclassified`` for missing values.
    """
    values = np.asarray(values, dtype=float)
    local_i = np.asarray(local_i, dtype=float)

    if values.shape[0] != local_i.shape[0]:
        raise ValueError("values and local_i must have the same length.")

    labels = np.full(values.shape[0], "unclassified", dtype=object)
    valid = np.isfinite(values) & np.isfinite(local_i)

    high_value = values > value_threshold
    low_value = values < value_threshold
    positive_i = local_i > local_i_threshold
    negative_i = local_i < local_i_threshold

    labels[valid & high_value & positive_i] = "high-high"
    labels[valid & low_value & positive_i] = "low-low"
    labels[valid & high_value & negative_i] = "high-low"
    labels[valid & low_value & negative_i] = "low-high"

    return labels


def add_local_morans_i_quadrants(
    adata: ad.AnnData,
    value_key: str,
    local_i_key: str=None,
    out_key: str=None,
    value_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> ad.AnnData:
    """
    Add quadrant-style local Moran classification to ``adata.obs``.

    This labels each cell as:
    - ``high-high``: high value surrounded by high values
    - ``low-low``: low value surrounded by low values
    - ``high-low``: high value surrounded by low values
    - ``low-high``: low value surrounded by high values
    """
    if local_i_key is None:
        local_i_key = f"local_morans_i__{value_key}"

    if local_i_key not in adata.obs.columns:
        raise ValueError(
            f"{local_i_key!r} not found in adata.obs. "
            "Run add_local_morans_i(...) first or provide local_i_key."
        )

    if value_key not in adata.obs.columns:
        raise ValueError(f"{value_key!r} not found in adata.obs.")

    if out_key is None:
        out_key = f"local_morans_quadrant__{value_key}"

    adata.obs[out_key] = classify_local_morans_i(
        values=adata.obs[value_key].to_numpy(),
        local_i=adata.obs[local_i_key].to_numpy(),
        value_threshold=value_threshold,
        local_i_threshold=local_i_threshold,
    )

    return adata

# ============================================================
# 8. CROSS MORAN'S I
# ============================================================

def cross_morans_i(coords: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, k: int=8) -> float:
    """
    Cross Moran's I.

    Measures spatial association between two features.

    Example
    -------
    tumor_density vs fibroblast_density

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates shared by both features.
    x_values, y_values : array-like of shape (n,)
        Continuous features measured at the same coordinates.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)

    x = np.asarray(x_values)
    y = np.asarray(y_values)

    valid = np.isfinite(x) & np.isfinite(y)

    coords = coords[valid]
    x = x[valid]
    y = y[valid]

    n = len(x)

    if n < 3:
        return np.nan

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return np.nan

    W = kneighbors_graph(coords, k_eff, mode="connectivity", include_self=False)

    x = x - x.mean()
    y = y - y.mean()

    Wy = W @ y
    numerator = np.dot(x, Wy)

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))

    if denom == 0:
        return np.nan

    I = (n / W.sum()) * (numerator / denom)

    return I


def cross_morans_i_by_image(
    adata: ad.AnnData,
    x_value_key: str,
    y_value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> pd.DataFrame:
    """
    Compute cross Moran's I per image for two spatial features.

    Parameters
    ----------
    x_value_key, y_value_key : str
        Columns in ``adata.obs`` containing the numeric features to analyze.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    rows = []

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        x_values = adata.obs.loc[idx, x_value_key].to_numpy()
        y_values = adata.obs.loc[idx, y_value_key].to_numpy()

        rows.append({
            image_key: img,
            "x_feature": x_value_key,
            "y_feature": y_value_key,
            "cross_morans_i": cross_morans_i(
                coords,
                x_values,
                y_values,
                k=k,
            ),
        })

    return pd.DataFrame(rows)


def cross_morans_i_permutation_test(
    coords: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    k: int=8,
    n_sim: int=999,
    permute: bool="y",
    random_state: int=None,
) -> dict:
    """
    Permutation test for global cross Moran's I.

    Parameters
    ----------
    permute : {"x", "y"}
        Which feature to shuffle for the null model.
    """
    if permute not in {"x", "y"}:
        raise ValueError("permute must be either 'x' or 'y'")

    coords = np.asarray(coords)
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x_values) & np.isfinite(y_values)
    coords = coords[valid]
    x_values = x_values[valid]
    y_values = y_values[valid]

    observed = cross_morans_i(coords, x_values, y_values, k=k)

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
    sims = []

    for i in range(n_sim):
        if permute == "x":
            sim_x = rng.permutation(x_values)
            sim_y = y_values
        else:
            sim_x = x_values
            sim_y = rng.permutation(y_values)

        sim_stat = cross_morans_i(coords, sim_x, sim_y, k=k)

        if np.isfinite(sim_stat):
            sims.append(sim_stat)

    sims = np.asarray(sims, dtype=float)

    if len(sims) == 0:
        raise ValueError("All permutation simulations returned invalid statistics.")

    null_mean = sims.mean()
    null_std = sims.std(ddof=1) if len(sims) > 1 else 0.0
    p_value = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (len(sims) + 1)
    z_score = np.nan if null_std == 0 else (observed - null_mean) / null_std

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_sim": len(sims),
    }


def cross_morans_i_by_image_permutation_test(
    adata: ad.AnnData,
    x_value_key: str,
    y_value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
    n_sim: int=999,
    permute: bool="y",
    random_state: int=None,
) -> pd.DataFrame:
    """
    Permutation test for cross Moran's I per image.
    """

    rows = []
    rng = np.random.default_rng(random_state)

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        x_values = adata.obs.loc[idx, x_value_key].to_numpy()
        y_values = adata.obs.loc[idx, y_value_key].to_numpy()

        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        stats = cross_morans_i_permutation_test(
            coords,
            x_values,
            y_values,
            k=k,
            n_sim=n_sim,
            permute=permute,
            random_state=seed,
        )

        rows.append({
            image_key: img,
            "x_feature": x_value_key,
            "y_feature": y_value_key,
            **stats,
        })

    return pd.DataFrame(rows)


def local_cross_morans_i(coords: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, k: int=8) -> np.ndarray:
    """
    Local cross Moran's I.

    Detects spatial regions where two features interact.

    Example
    -------
    tumor density correlated with collagen alignment.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates shared by both features.
    x_values, y_values : array-like of shape (n,)
        Continuous features measured at the same coordinates.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)

    x = np.asarray(x_values)
    y = np.asarray(y_values)

    valid = np.isfinite(x) & np.isfinite(y)

    coords_v = coords[valid]
    x_v = x[valid]
    y_v = y[valid]

    out = np.full(len(x), np.nan)

    n = len(x_v)

    if n < 3:
        return out

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return out

    W = kneighbors_graph(
        coords_v,
        k_eff,
        mode="connectivity",
        include_self=False,
    )

    x_c = x_v - x_v.mean()
    y_c = y_v - y_v.mean()

    m2 = np.sum(y_c**2) / n

    if m2 == 0:
        return out

    local_I = x_c * (W @ y_c) / m2

    out[np.where(valid)[0]] = local_I

    return out


def add_local_cross_morans_i(
    adata: ad.AnnData,
    x_value_key: str,
    y_value_key: str,
    out_key: str=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> ad.AnnData:
    """
    Compute local cross Moran's I and add it to ``adata.obs``.

    Each cell receives a local spatial association value describing
    how strongly ``x_value_key`` aligns with neighboring ``y_value_key``.
    """

    if out_key is None:
        out_key = f"local_cross_morans_i__{x_value_key}__{y_value_key}"

    adata.obs[out_key] = np.nan

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        x_values = adata.obs.loc[idx, x_value_key].to_numpy()
        y_values = adata.obs.loc[idx, y_value_key].to_numpy()

        local_I = local_cross_morans_i(
            coords,
            x_values,
            y_values,
            k=k,
        )

        adata.obs.loc[idx, out_key] = local_I

    return adata


def classify_local_cross_morans_i(
    source_values: np.ndarray,
    target_neighbor_values: np.ndarray,
    local_cross_i: np.ndarray,
    source_threshold: float=0.0,
    target_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> np.ndarray:
    """
    Classify local cross Moran's I into quadrant-style labels.

    Parameters
    ----------
    source_values : array-like
        Source-cell feature values.
    target_neighbor_values : array-like
        Source-centered neighborhood summaries of the target feature.
    local_cross_i : array-like
        Local cross Moran's I values.
    source_threshold, target_threshold : float, default 0.0
        Thresholds separating high vs low values. These work best when the
        features are centered or z-scored.
    local_i_threshold : float, default 0.0
        Threshold separating positive vs negative local cross Moran's I.

    Returns
    -------
    np.ndarray of dtype object
        One of: ``high-high``, ``low-low``, ``high-low``, ``low-high``,
        or ``unclassified``.
    """
    source_values = np.asarray(source_values, dtype=float)
    target_neighbor_values = np.asarray(target_neighbor_values, dtype=float)
    local_cross_i = np.asarray(local_cross_i, dtype=float)

    n = len(source_values)
    if len(target_neighbor_values) != n or len(local_cross_i) != n:
        raise ValueError("All input arrays must have the same length.")

    labels = np.full(n, "unclassified", dtype=object)
    valid = (
        np.isfinite(source_values)
        & np.isfinite(target_neighbor_values)
        & np.isfinite(local_cross_i)
    )

    source_high = source_values > source_threshold
    source_low = source_values < source_threshold
    target_high = target_neighbor_values > target_threshold
    target_low = target_neighbor_values < target_threshold
    local_pos = local_cross_i > local_i_threshold
    local_neg = local_cross_i < local_i_threshold

    labels[valid & source_high & target_high & local_pos] = "high-high"
    labels[valid & source_low & target_low & local_pos] = "low-low"
    labels[valid & source_high & target_low & local_neg] = "high-low"
    labels[valid & source_low & target_high & local_neg] = "low-high"

    return labels


def add_local_cross_morans_i_quadrants(
    adata: ad.AnnData,
    source_value_key: str,
    target_neighbor_value_key: str,
    local_i_key: str=None,
    out_key: str=None,
    source_threshold: float=0.0,
    target_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> ad.AnnData:
    """
    Add quadrant-style local cross Moran classification to ``adata.obs``.

    This labels each source cell as:
    - ``high-high``: high source value in a high target-neighborhood context
    - ``low-low``: low source value in a low target-neighborhood context
    - ``high-low``: high source value in a low target-neighborhood context
    - ``low-high``: low source value in a high target-neighborhood context

    The sign of local cross Moran's I determines whether the association is
    locally concordant (positive) or discordant (negative).
    """
    if local_i_key is None:
        local_i_key = f"local_cross_morans_i__{source_value_key}__{target_neighbor_value_key}"

    missing = [
        key for key in [source_value_key, target_neighbor_value_key, local_i_key]
        if key not in adata.obs.columns
    ]
    if missing:
        raise ValueError(f"Required keys not found in adata.obs: {missing}")

    if out_key is None:
        out_key = f"local_cross_morans_quadrant__{source_value_key}__{target_neighbor_value_key}"

    adata.obs[out_key] = classify_local_cross_morans_i(
        source_values=adata.obs[source_value_key].to_numpy(),
        target_neighbor_values=adata.obs[target_neighbor_value_key].to_numpy(),
        local_cross_i=adata.obs[local_i_key].to_numpy(),
        source_threshold=source_threshold,
        target_threshold=target_threshold,
        local_i_threshold=local_i_threshold,
    )

    return adata


def summarize_target_features_around_source_cells(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    target_feature_keys: list[str],
    radius: float=None,
    k_neighbors: int=None,
    agg: str="mean",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    source_only: bool=True,
) -> pd.DataFrame:
    """
    Summarize target-cell features in neighborhoods around source cells.

    Parameters
    ----------
    target_feature_keys : list of str
        Numeric target-cell features to aggregate around each source cell.
    radius : float, optional
        Radius-based neighborhood. Exactly one of ``radius`` or ``k_neighbors``
        must be provided.
    k_neighbors : int, optional
        kNN-based neighborhood in the target phenotype.
    agg : {"mean", "median"}
        Aggregation applied to target-feature values in each source-centered
        neighborhood.
    source_only : bool, default True
        If True, returns only source cells with aggregated target summaries.
        If False, merges the summary columns back onto a copy of ``adata.obs``.

    Returns
    -------
    pd.DataFrame
        Source-cell table with neighborhood target-feature summaries.
    """
    if (radius is None) == (k_neighbors is None):
        raise ValueError("Specify exactly one of radius or k_neighbors.")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be either 'mean' or 'median'.")

    missing = [c for c in target_feature_keys if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Target features not found in adata.obs: {missing}")

    rows = []

    for img in adata.obs[image_key].dropna().unique():
        df = adata.obs[adata.obs[image_key] == img]
        source = df[df[phenotype_key] == source_phenotype]
        target = df[df[phenotype_key] == target_phenotype]

        if source.empty or target.empty:
            continue

        source_coords = source[[x_key, y_key]].to_numpy()
        target_coords = target[[x_key, y_key]].to_numpy()

        tree = BallTree(target_coords)
        if radius is not None:
            neighbor_idx = tree.query_radius(source_coords, r=radius)
        else:
            k_eff = _resolve_k(len(target_coords), k_neighbors + 1 if source_phenotype == target_phenotype else k_neighbors)
            if k_eff is None:
                continue
            _, neighbor_idx = tree.query(source_coords, k=k_eff, return_distance=True)
            if k_eff == 1:
                neighbor_idx = neighbor_idx.reshape(-1, 1)

        target_values = target[target_feature_keys]

        for src_pos, cell_id in enumerate(source.index):
            idx = np.asarray(neighbor_idx[src_pos], dtype=int)

            if source_phenotype == target_phenotype and len(idx) > 0:
                src_label = source.index[src_pos]
                target_ids = target.index.to_numpy()
                idx = idx[target_ids[idx] != src_label]

            row = {
                "cell_id": cell_id,
                image_key: img,
                "source_phenotype": source_phenotype,
                "target_phenotype": target_phenotype,
                "n_target_neighbors": len(idx),
            }

            if len(idx) == 0:
                for feature in target_feature_keys:
                    row[f"neighbor_{agg}__{feature}"] = np.nan
            else:
                subset = target_values.iloc[idx]
                if agg == "mean":
                    vals = subset.mean(axis=0, numeric_only=True)
                else:
                    vals = subset.median(axis=0, numeric_only=True)
                for feature in target_feature_keys:
                    row[f"neighbor_{agg}__{feature}"] = vals.get(feature, np.nan)

            rows.append(row)

    out = pd.DataFrame(rows)

    if source_only:
        return out

    merged = adata.obs.copy()
    if not out.empty:
        merged = merged.merge(out, how="left", left_index=True, right_on="cell_id")
    return merged


def cross_morans_i_feature_matrix(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    source_feature_keys: list[str],
    target_feature_keys: list[str],
    radius: float=None,
    k_neighbors: int=None,
    agg: str="mean",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> pd.DataFrame:
    """
    Compute a feature-by-feature cross Moran's I matrix between two phenotypes.

    Workflow:
    - summarize target-feature neighborhoods around each source cell
    - compute cross Moran's I between each source feature and each summarized
      target feature at the source-cell coordinates
    """
    missing_source = [c for c in source_feature_keys if c not in adata.obs.columns]
    missing_target = [c for c in target_feature_keys if c not in adata.obs.columns]
    if missing_source:
        raise ValueError(f"Source features not found in adata.obs: {missing_source}")
    if missing_target:
        raise ValueError(f"Target features not found in adata.obs: {missing_target}")

    source_df = adata.obs[adata.obs[phenotype_key] == source_phenotype].copy()
    neighbor_df = summarize_target_features_around_source_cells(
        adata=adata,
        phenotype_key=phenotype_key,
        source_phenotype=source_phenotype,
        target_phenotype=target_phenotype,
        target_feature_keys=target_feature_keys,
        radius=radius,
        k_neighbors=k_neighbors,
        agg=agg,
        x_key=x_key,
        y_key=y_key,
        image_key=image_key,
        source_only=True,
    )

    if neighbor_df.empty or source_df.empty:
        return pd.DataFrame()

    source_df = source_df.merge(
        neighbor_df,
        how="left",
        left_index=True,
        right_on="cell_id",
        suffixes=("", "__neighbor"),
    )

    if f"{image_key}__neighbor" in source_df.columns:
        source_df = source_df.drop(columns=[f"{image_key}__neighbor"])

    rows = []
    neighbor_cols = [f"neighbor_{agg}__{f}" for f in target_feature_keys]

    for src_feat in source_feature_keys:
        for tgt_feat, tgt_col in zip(target_feature_keys, neighbor_cols):
            for img in source_df[image_key].dropna().unique():
                df_img = source_df[source_df[image_key] == img]
                coords = df_img[[x_key, y_key]].to_numpy()
                x_values = df_img[src_feat].to_numpy()
                y_values = df_img[tgt_col].to_numpy()

                rows.append({
                    image_key: img,
                    "source_phenotype": source_phenotype,
                    "target_phenotype": target_phenotype,
                    "source_feature": src_feat,
                    "target_feature": tgt_feat,
                    "target_summary_feature": tgt_col,
                    "cross_morans_i": cross_morans_i(
                        coords,
                        x_values,
                        y_values,
                        k=k,
                    ),
                    "n_source_cells": len(df_img),
                    "n_nonmissing_pairs": int(np.sum(np.isfinite(x_values) & np.isfinite(y_values))),
                })

    return pd.DataFrame(rows)


def add_local_cross_morans_i_between_phenotypes(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    source_feature_key: str,
    target_feature_key: str,
    radius: float=None,
    k_neighbors: int=None,
    agg: str="mean",
    out_key: str=None,
    neighbor_feature_key: str=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> ad.AnnData:
    """
    Compute local cross Moran's I between a source-cell feature and a
    source-centered neighborhood summary of a target-cell feature.

    This is a one-step wrapper around:
    1. ``summarize_target_features_around_source_cells(...)``
    2. merging the neighborhood summary onto source cells
    3. ``add_local_cross_morans_i(...)``

    Returns
    -------
    anndata.AnnData
        AnnData containing only source-phenotype cells, with both the
        summarized target feature and local cross Moran's I added to ``obs``.
    """
    if source_feature_key not in adata.obs.columns:
        raise ValueError(f"Source feature {source_feature_key!r} not found in adata.obs.")
    if target_feature_key not in adata.obs.columns:
        raise ValueError(f"Target feature {target_feature_key!r} not found in adata.obs.")

    neighbor_df = summarize_target_features_around_source_cells(
        adata=adata,
        phenotype_key=phenotype_key,
        source_phenotype=source_phenotype,
        target_phenotype=target_phenotype,
        target_feature_keys=[target_feature_key],
        radius=radius,
        k_neighbors=k_neighbors,
        agg=agg,
        x_key=x_key,
        y_key=y_key,
        image_key=image_key,
        source_only=True,
    )

    source_adata = adata[adata.obs[phenotype_key] == source_phenotype].copy()
    if source_adata.n_obs == 0:
        return source_adata

    default_neighbor_key = f"neighbor_{agg}__{target_feature_key}"
    if neighbor_feature_key is None:
        neighbor_feature_key = default_neighbor_key

    if not neighbor_df.empty:
        merge_df = neighbor_df[["cell_id", default_neighbor_key]].rename(
            columns={default_neighbor_key: neighbor_feature_key}
        )
        merge_df = merge_df.set_index("cell_id")
        source_adata.obs = source_adata.obs.join(merge_df, how="left")
    else:
        source_adata.obs[neighbor_feature_key] = np.nan

    if out_key is None:
        out_key = (
            f"local_cross_morans_i__{source_feature_key}"
            f"__{source_phenotype}__vs__{target_phenotype}"
            f"__{neighbor_feature_key}"
        )

    source_adata = add_local_cross_morans_i(
        source_adata,
        x_value_key=source_feature_key,
        y_value_key=neighbor_feature_key,
        out_key=out_key,
        x_key=x_key,
        y_key=y_key,
        image_key=image_key,
        k=k,
    )

    return source_adata
