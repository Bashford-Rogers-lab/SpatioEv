import scanpy as sc
import numpy as np


def cluster_cells(adata, config):

    markers = config.markers

    adata_sub = adata[:, markers].copy()

    sc.tl.pca(adata_sub)
    sc.pp.neighbors(
        adata_sub,
        n_neighbors=config.n_neighbors,
        n_pcs=config.n_pcs
    )

    sc.tl.umap(adata_sub)

    sc.tl.leiden(
        adata_sub,
        resolution=config.resolution
    )

    return adata_sub