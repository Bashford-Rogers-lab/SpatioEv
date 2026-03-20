"""
Spatial ECM Link Construction
=============================

Builds reusable spatial relationship tables linking cells and ECM fibers.

Relationship table structure
----------------------------

imageid
cell_id
fiber_id
distance

One cell may link to many fibers.
One fiber may link to many cells.

This table can be reused across all ECM spatial statistics.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ------------------------------------------------
# Radius-based adjacency map
# ------------------------------------------------

def build_cell_fiber_links(
    adata,
    fiber_df,
    radius=50,
    cell_x="X_centroid",
    cell_y="Y_centroid",
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    image_key="imageid",
    fiber_image_key="imageid",
):
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
    adata,
    fiber_df,
    cell_x="X_centroid",
    cell_y="Y_centroid",
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    image_key="imageid",
    fiber_image_key="imageid",
):
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