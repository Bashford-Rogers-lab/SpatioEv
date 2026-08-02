"""Tile-based and KDE spatial density.

Cells are binned onto a regular grid and summarised per tile, or smoothed
with a Gaussian kernel density estimate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad
from scipy.stats import gaussian_kde


def assign_tiles(
    adata: ad.AnnData,
    tile_size: int = 128,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
) -> pd.DataFrame:
    """
    Assign each cell to a spatial tile based on its x/y coordinates.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    tile_size : int
        Side length of each square tile in the same units as the coordinates.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.

    Returns
    -------
    DataFrame
        Copy of ``adata.obs`` with added ``tile_x`` and ``tile_y`` integer columns.

    Examples
    --------
    >>> import spatioev as sv
    >>> tile_df = sv.tl.assign_tiles(adata, tile_size=128)
    """

    df = adata.obs.copy()

    df["tile_x"] = (df[x_key] // tile_size).astype(int)
    df["tile_y"] = (df[y_key] // tile_size).astype(int)

    return df


def compute_general_density(
    df: pd.DataFrame,
    tile_size: int = 128,
) -> pd.DataFrame:
    """
    Compute object and pixel density per spatial tile for all cells.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`assign_tiles` containing ``imageid``, ``tile_x``,
        ``tile_y``, ``label``, and ``area`` columns.
    tile_size : int
        Side length of each square tile in coordinate units.

    Returns
    -------
    DataFrame
        One row per (imageid, tile_x, tile_y) with columns:
        ``object_count``, ``pixel_sum``, ``object_density``, ``pixel_density``.

    Examples
    --------
    >>> import spatioev as sv
    >>> tile_df = sv.tl.assign_tiles(adata, tile_size=128)
    >>> density_df = sv.tl.compute_general_density(tile_df, tile_size=128)
    """

    tile_area = tile_size ** 2

    stats = (
        df.groupby(["imageid", "tile_x", "tile_y"])
        .agg(
            object_count=("label", "nunique"),
            pixel_sum=("area", "sum")
        )
        .reset_index()
    )

    stats["object_density"] = stats["object_count"] / tile_area * 100
    stats["pixel_density"] = stats["pixel_sum"] / tile_area * 100

    return stats


def compute_phenotype_density(
    df: pd.DataFrame,
    phenotype_key: str = "phenotype",
    tile_size: int = 128,
) -> pd.DataFrame:
    """
    Compute object and pixel density per spatial tile for each phenotype.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`assign_tiles` containing ``imageid``, ``tile_x``,
        ``tile_y``, ``label``, ``area``, and a phenotype column.
    phenotype_key : str
        Column in ``df`` containing phenotype labels.
    tile_size : int
        Side length of each square tile in coordinate units.

    Returns
    -------
    DataFrame
        One row per (imageid, tile_x, tile_y, phenotype) with columns:
        ``object_count``, ``pixel_sum``, ``object_density``, ``pixel_density``.

    Examples
    --------
    >>> import spatioev as sv
    >>> tile_df = sv.tl.assign_tiles(adata, tile_size=128)
    >>> pheno_density = sv.tl.compute_phenotype_density(tile_df, phenotype_key="phenotype")
    """

    tile_area = tile_size ** 2

    stats = (
        df.groupby(["imageid", "tile_x", "tile_y", phenotype_key])
        .agg(
            object_count=("label", "nunique"),
            pixel_sum=("area", "sum")
        )
        .reset_index()
    )

    stats["object_density"] = stats["object_count"] / tile_area * 100
    stats["pixel_density"] = stats["pixel_sum"] / tile_area * 100

    return stats


def phenotype_density_correlation(
    density_df: pd.DataFrame,
    phenotype_key: str="phenotype",
    value: str="object_density"
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations of phenotype densities across tiles.

    Parameters
    ----------
    density_df : DataFrame
        Output of :func:`compute_phenotype_density` containing tile coordinates
        and a phenotype column.
    phenotype_key : str
        Column in ``density_df`` containing phenotype labels.
    value : str
        Density column used as values in the pivot table.

    Returns
    -------
    DataFrame
        Square correlation matrix with phenotypes as rows and columns.

    Examples
    --------
    >>> import spatioev as sv
    >>> corr = sv.tl.phenotype_density_correlation(pheno_density)
    >>> corr.style.background_gradient(cmap="coolwarm")
    """

    pivot = density_df.pivot_table(
        index=["tile_y", "tile_x"],
        columns=phenotype_key,
        values=value
    )

    corr = pivot.corr()

    return corr


def compute_kde_density(
    adata: ad.AnnData,
    phenotype: str=None,
    phenotype_key: str="phenotype",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    grid_size: int=200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a kernel density estimate (KDE) on a regular grid.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    phenotype : str, optional
        If provided, restrict KDE to cells of this phenotype.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    grid_size : int
        Number of grid points along each axis.

    Returns
    -------
    Xgrid : ndarray of shape (grid_size, grid_size)
        X-coordinates of the evaluation grid.
    Ygrid : ndarray of shape (grid_size, grid_size)
        Y-coordinates of the evaluation grid.
    Z : ndarray of shape (grid_size, grid_size)
        Estimated density values on the grid.

    Examples
    --------
    >>> import spatioev as sv
    >>> Xg, Yg, Z = sv.tl.compute_kde_density(adata, phenotype="Tumor")
    >>> sv.pl.plot_kde_density(Xg, Yg, Z)
    """

    df = adata.obs.copy()

    if phenotype is not None:
        df = df[df[phenotype_key] == phenotype]

    x = df[x_key].values
    y = df[y_key].values

    # shift coordinates so origin is 0
    x = x - x.min()
    y = y - y.min()

    kde = gaussian_kde(np.vstack([x, y]))

    xgrid = np.linspace(0, x.max(), grid_size)
    ygrid = np.linspace(0, y.max(), grid_size)

    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

    Z = kde(positions).reshape(Xgrid.shape)

    return Xgrid, Ygrid, Z


__all__ = [
    "assign_tiles",
    "compute_general_density",
    "compute_phenotype_density",
    "phenotype_density_correlation",
    "compute_kde_density",
]
