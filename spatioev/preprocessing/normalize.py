import numpy as np


def zscore_normalize(adata):

    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()

    mean = np.mean(adata.X, axis=0)
    std = np.std(adata.X, axis=0)

    std[std == 0] = 1

    adata.X = (adata.X - mean) / std

    adata.layers["scaled"] = adata.X.copy()

    return adata