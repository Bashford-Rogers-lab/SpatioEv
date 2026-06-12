"""Marker expression normalization and obs feature construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad


def zscore_normalize(adata: ad.AnnData) -> ad.AnnData:
    """Z-score normalise marker expression across all cells.

    Stores the original expression in ``adata.layers["raw"]`` (if not
    already present) and the z-scored matrix in ``adata.layers["scaled"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with a marker expression matrix in ``adata.X``.

    Returns
    -------
    anndata.AnnData
        Same object with ``adata.X`` replaced by z-scored values and the
        two new layers populated.

    Examples
    --------
    >>> adata = sv.pp.zscore_normalize(adata)
    >>> "scaled" in adata.layers
    True
    """

    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    mean = np.asarray(np.mean(X, axis=0)).ravel()
    std = np.asarray(np.std(X, axis=0)).ravel()

    std[std == 0] = 1

    adata.X = (X - mean) / std

    adata.layers["scaled"] = adata.X.copy()

    return adata


def add_obs_from_var(
    adata: ad.AnnData,
    markers: list[str],
    zscore: bool = True,
    obs_suffix: str = "_expr",
    zscore_suffix: str = "_z",
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Copy one or more features from ``adata.X``/``adata.var_names`` into ``adata.obs``.

    Parameters
    ----------
    markers : list-like of str
        Marker names to pull from ``adata.var_names``.
    zscore : bool, default True
        Whether to add z-scored versions of the copied values.
    obs_suffix : str, default "_expr"
        Suffix for the raw copied columns in ``adata.obs``.
    zscore_suffix : str, default "_z"
        Suffix appended to the raw copied column name for z-scored columns.
    overwrite : bool, default False
        Whether to overwrite existing columns in ``adata.obs``.
    """
    markers = list(markers)
    missing = [m for m in markers if m not in adata.var_names]
    if missing:
        raise ValueError(f"Markers not found in adata.var_names: {missing}")

    expr = adata[:, markers].X
    if sparse.issparse(expr):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=float)

    expr_df = pd.DataFrame(
        expr,
        index=adata.obs_names,
        columns=[f"{marker}{obs_suffix}" for marker in markers],
    )

    for col in expr_df.columns:
        if (col in adata.obs.columns) and not overwrite:
            raise ValueError(
                f"Column {col!r} already exists in adata.obs. "
                "Set overwrite=True to replace it."
            )
        adata.obs[col] = expr_df[col]

        if zscore:
            z_col = f"{col}{zscore_suffix}"
            if (z_col in adata.obs.columns) and not overwrite:
                raise ValueError(
                    f"Column {z_col!r} already exists in adata.obs. "
                    "Set overwrite=True to replace it."
                )

            vals = expr_df[col].to_numpy(dtype=float)
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std == 0 or np.isnan(std):
                adata.obs[z_col] = np.nan
            else:
                adata.obs[z_col] = (vals - mean) / std

    return adata


def add_zscore_obs_features(
    adata: ad.AnnData,
    feature_keys: list[str],
    suffix: str = "_z",
    overwrite: bool = False,
) -> ad.AnnData:
    """
    Add z-scored versions of numeric ``adata.obs`` columns.

    Parameters
    ----------
    feature_keys : list-like of str
        Existing columns in ``adata.obs`` to z-score.
    suffix : str, default "_z"
        Suffix appended to each feature name for the z-scored output column.
    overwrite : bool, default False
        Whether to overwrite existing z-scored columns.
    """
    feature_keys = list(feature_keys)
    missing = [f for f in feature_keys if f not in adata.obs.columns]
    if missing:
        raise ValueError(f"Features not found in adata.obs: {missing}")

    for feature in feature_keys:
        z_col = f"{feature}{suffix}"
        if (z_col in adata.obs.columns) and not overwrite:
            raise ValueError(
                f"Column {z_col!r} already exists in adata.obs. "
                "Set overwrite=True to replace it."
            )

        vals = pd.to_numeric(adata.obs[feature], errors="coerce").to_numpy(dtype=float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std == 0 or np.isnan(std):
            adata.obs[z_col] = np.nan
        else:
            adata.obs[z_col] = (vals - mean) / std

    return adata


__all__ = [
    "zscore_normalize",
    "add_obs_from_var",
    "add_zscore_obs_features",
]
