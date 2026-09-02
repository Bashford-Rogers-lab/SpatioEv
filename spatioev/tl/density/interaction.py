"""Pairwise phenotype interaction density."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
from sklearn.neighbors import BallTree


def phenotype_interaction_density(
    adata: ad.AnnData,
    phenotype_key: str,
    source_pheno: str,
    target_pheno: str,
    radius: float=50,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    out_key: str=None,
    edge_key: str=None,
    exclude_edge: bool=False,
) -> ad.AnnData:
    """
    For each source cell, count target phenotype cells within radius,
    and normalize by local circle area.
    """
    if out_key is None:
        out_key = f"interaction_density__{source_pheno}__to__{target_pheno}"

    adata.obs[out_key] = np.nan
    area = np.pi * (radius ** 2)

    for img in adata.obs[image_key].unique():
        img_mask = adata.obs[image_key] == img
        img_data = adata[img_mask]

        src = img_data[img_data.obs[phenotype_key] == source_pheno]
        tgt = img_data[img_data.obs[phenotype_key] == target_pheno]

        if src.n_obs == 0 or tgt.n_obs == 0:
            continue

        src_coords = src.obs[[x_key, y_key]].to_numpy()
        tgt_coords = tgt.obs[[x_key, y_key]].to_numpy()

        tree = BallTree(tgt_coords)
        neigh = tree.query_radius(src_coords, r=radius)
        counts = np.array([len(n) for n in neigh], dtype=float)
        dens = counts / area

        adata.obs.loc[src.obs_names, out_key] = dens

        if exclude_edge and edge_key is not None and edge_key in adata.obs.columns:
            edge_src = src.obs_names[adata.obs.loc[src.obs_names, edge_key].to_numpy()]
            adata.obs.loc[edge_src, out_key] = np.nan

    return adata


__all__ = ["phenotype_interaction_density"]
