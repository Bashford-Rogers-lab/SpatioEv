"""k-nearest-neighbour local density."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad
from sklearn.neighbors import BallTree


def compute_local_density_by_phenotype(
    adata: ad.AnnData,
    phenotype_key: str="phenotype",
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    k_neighbors: int=5
) -> pd.DataFrame:
    """
    Compute local density within each phenotype using kNN distances.

    Density is defined as 1 / mean distance to the k nearest neighbors
    of the same phenotype within each image.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid, Y_centroid, imageid, and phenotype columns in adata.obs.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    k_neighbors : int
        Number of nearest same-phenotype neighbors to use.

    Returns
    -------
    AnnData
        Modified in-place with added columns in ``adata.obs``:
        ``mean_dist_pheno`` (mean kNN distance) and
        ``density_pheno`` (reciprocal density estimate).

    Examples
    --------
    >>> import spatioev as sv
    >>> adata = sv.tl.compute_local_density_by_phenotype(adata, phenotype_key="phenotype")
    """

    adata.obs["mean_dist_pheno"] = np.nan
    adata.obs["density_pheno"] = np.nan

    for image in adata.obs[image_key].unique():

        img_mask = adata.obs[image_key] == image
        img_data = adata[img_mask]

        for phenotype in img_data.obs[phenotype_key].unique():

            ph_mask = img_data.obs[phenotype_key] == phenotype
            ph_cells = img_data[ph_mask]

            coords = ph_cells.obs[[x_key, y_key]].values

            n_cells = coords.shape[0]

            if n_cells <= 1:
                continue

            k = min(k_neighbors + 1, n_cells)

            tree = BallTree(coords)

            distances, _ = tree.query(coords, k=k)

            mean_dist = distances[:, 1:].mean(axis=1)

            density = 1 / mean_dist

            adata.obs.loc[ph_cells.obs.index, "mean_dist_pheno"] = mean_dist
            adata.obs.loc[ph_cells.obs.index, "density_pheno"] = density

    return adata


def compute_local_density_all_cells(
    adata: ad.AnnData,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    k_neighbors: int=5
) -> pd.DataFrame:
    """
    Compute local density for all cells using kNN distances.

    Density is defined as 1 / mean distance to the k nearest neighbors
    within each image, regardless of phenotype.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    k_neighbors : int
        Number of nearest neighbors to use.

    Returns
    -------
    AnnData
        Modified in-place with added columns in ``adata.obs``:
        ``mean_dist_all`` (mean kNN distance) and
        ``density_all`` (reciprocal density estimate).

    Examples
    --------
    >>> import spatioev as sv
    >>> adata = sv.tl.compute_local_density_all_cells(adata)
    """

    adata.obs["mean_dist_all"] = np.nan
    adata.obs["density_all"] = np.nan

    for image in adata.obs[image_key].unique():

        img_mask = adata.obs[image_key] == image
        img_data = adata[img_mask]

        coords = img_data.obs[[x_key, y_key]].values

        n_cells = coords.shape[0]

        if n_cells <= 1:
            continue

        k = min(k_neighbors + 1, n_cells)

        tree = BallTree(coords)

        distances, _ = tree.query(coords, k=k)

        mean_dist = distances[:, 1:].mean(axis=1)

        density = 1 / mean_dist

        adata.obs.loc[img_data.obs.index, "mean_dist_all"] = mean_dist
        adata.obs.loc[img_data.obs.index, "density_all"] = density

    return adata


__all__ = [
    "compute_local_density_by_phenotype",
    "compute_local_density_all_cells",
]
