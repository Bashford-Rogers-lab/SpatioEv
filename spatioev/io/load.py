import anndata as ad


def load_h5ad(path):

    adata = ad.read_h5ad(path)

    return adata