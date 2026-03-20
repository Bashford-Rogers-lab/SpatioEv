import numpy as np
from sklearn.neighbors import BallTree


def compute_radius_density(
    adata,
    radius=50,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    density_key="radius_density",
    edge_key=None,
    exclude_edge=False,
):
    """
    Compute local density within a fixed radius.

    Density = number of neighbors within radius / (pi * r^2)
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