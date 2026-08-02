"""Cell-to-fiber adjacency: radius links and nearest-fiber maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad
from sklearn.neighbors import BallTree

# ------------------------------------------------
# Radius-based adjacency map
# ------------------------------------------------

def build_cell_fiber_links(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    radius: float=50,
    cell_x: str="X_centroid",
    cell_y: str="Y_centroid",
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    image_key: str="imageid",
    fiber_image_key: str="imageid",
) -> pd.DataFrame:
    """
    Build bidirectional adjacency table linking cells and fibers.

    Returns
    -------
    DataFrame

        imageid
        cell_id
        fiber_id
        distance
    """

    rows = []

    images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[fiber_image_key].unique(),
    )

    for img in images:

        cells = adata.obs[adata.obs[image_key] == img]
        fibers = fiber_df[fiber_df[fiber_image_key] == img]

        if cells.empty or fibers.empty:
            continue

        cell_coords = cells[[cell_x, cell_y]].to_numpy()
        fiber_coords = fibers[[fiber_x, fiber_y]].to_numpy()

        tree = BallTree(cell_coords)

        neighbors = tree.query_radius(fiber_coords, r=radius)

        for i, idxs in enumerate(neighbors):

            if len(idxs) == 0:
                continue

            fiber_id = fibers.index[i]
            fiber_point = fiber_coords[i]

            dists = np.linalg.norm(cell_coords[idxs] - fiber_point, axis=1)

            for j, dist in zip(idxs, dists):

                rows.append(
                    {
                        "imageid": img,
                        "cell_id": cells.index[j],
                        "fiber_id": fiber_id,
                        "distance": dist,
                    }
                )

    return pd.DataFrame(rows)



# ------------------------------------------------
# Nearest cell-fiber mapping
# ------------------------------------------------

def build_nearest_cell_fiber_map(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    cell_x: str="X_centroid",
    cell_y: str="Y_centroid",
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    image_key: str="imageid",
    fiber_image_key: str="imageid",
) -> pd.DataFrame:
    """
    Compute nearest fiber for every cell.

    Returns
    -------
    DataFrame

        imageid
        cell_id
        fiber_id
        distance
    """

    rows = []

    images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[fiber_image_key].unique(),
    )

    for img in images:

        cells = adata.obs[adata.obs[image_key] == img]
        fibers = fiber_df[fiber_df[fiber_image_key] == img]

        if cells.empty or fibers.empty:
            continue

        cell_coords = cells[[cell_x, cell_y]].to_numpy()
        fiber_coords = fibers[[fiber_x, fiber_y]].to_numpy()

        tree = BallTree(fiber_coords)

        dist, idx = tree.query(cell_coords, k=1)

        for i in range(len(cells)):

            rows.append(
                {
                    "imageid": img,
                    "cell_id": cells.index[i],
                    "fiber_id": fibers.index[idx[i, 0]],
                    "distance": dist[i, 0],
                }
            )

    return pd.DataFrame(rows)
