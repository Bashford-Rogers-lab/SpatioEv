"""Fixed-radius local density."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
from sklearn.neighbors import BallTree


def compute_radius_density(
    adata: ad.AnnData,
    radius: float=50,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    density_key: str="radius_density",
    edge_key: str=None,
    exclude_edge: bool=False,
) -> ad.AnnData:
    """
    Compute local density within a fixed radius for all cells.

    Density is defined as (number of neighbors within radius) / (pi * radius^2),
    giving a per-unit-area cell density. Self is excluded from the count.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    radius : float
        Search radius in the same coordinate units as ``x_key``/``y_key``.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    density_key : str
        Output column name added to ``adata.obs``.
    edge_key : str, optional
        Boolean column marking edge cells. Only used when ``exclude_edge=True``.
    exclude_edge : bool
        If ``True``, set density to NaN for cells flagged by ``edge_key``.

    Returns
    -------
    AnnData
        Modified in-place with ``density_key`` added to ``adata.obs``.

    Examples
    --------
    >>> import spatioev as sv
    >>> adata = sv.tl.compute_radius_density(adata, radius=50)
    """
    adata.obs[density_key] = np.nan
    area = np.pi * (radius ** 2)

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]
        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()

        if coords.shape[0] < 2:
            continue

        tree = BallTree(coords)
        neigh = tree.query_radius(coords, r=radius)
        counts = np.array([len(n) - 1 for n in neigh], dtype=float)
        dens = counts / area

        adata.obs.loc[idx, density_key] = dens

        if exclude_edge and edge_key is not None and edge_key in adata.obs.columns:
            edge_idx = idx[adata.obs.loc[idx, edge_key].to_numpy()]
            adata.obs.loc[edge_idx, density_key] = np.nan

    return adata


__all__ = ["compute_radius_density"]
