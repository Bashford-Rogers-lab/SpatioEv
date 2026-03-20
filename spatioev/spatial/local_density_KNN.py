import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Local density by phenotype
# ------------------------------------------------------------

def compute_local_density_by_phenotype(
    adata,
    phenotype_key="phenotype",
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    k_neighbors=5
):
    """
    Compute local density within each phenotype.

    Density = 1 / mean distance to k nearest phenotype neighbors.
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


# ------------------------------------------------------------
# Local density of all cells
# ------------------------------------------------------------

def compute_local_density_all_cells(
    adata,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    k_neighbors=5
):
    """
    Compute local density for all cells.

    Density = 1 / mean distance to k nearest neighbors.
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


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def plot_local_density_map(
    adata,
    image_id,
    density_key="density_all",
    phenotype_key=None,
    phenotype=None,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    cmap="viridis",
    point_size=3
):
    """
    Plot spatial density map.
    """

    data = adata[adata.obs[image_key] == image_id]

    if phenotype_key and phenotype:

        data = data[data.obs[phenotype_key] == phenotype]

    x = data.obs[x_key]
    y = data.obs[y_key]
    density = data.obs[density_key]

    fig, ax = plt.subplots(figsize=(8,8))

    sc = ax.scatter(
        x,
        y,
        c=density,
        cmap=cmap,
        s=point_size,
        edgecolor="none"
    )

    plt.colorbar(sc, ax=ax, label="Local density")

    sns.kdeplot(
        x=x,
        y=y,
        levels=10,
        color="black",
        linewidths=1,
        ax=ax
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid")
    ax.set_ylabel("Y centroid")

    title = f"Density map: {image_id}"

    if phenotype:
        title += f" | {phenotype}"

    ax.set_title(title)

    plt.tight_layout()

    return fig, ax