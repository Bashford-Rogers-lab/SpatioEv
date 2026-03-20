import numpy as np
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt


def phenotype_interaction_density(
    adata,
    phenotype_key,
    source_pheno,
    target_pheno,
    radius=50,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    out_key=None,
    edge_key=None,
    exclude_edge=False,
):
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

def plot_interaction_density(
    adata,
    image_id,
    source_pheno,
    target_pheno,
    phenotype_key="annotated_clusters_update3",
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    size=1
):
    
    feature = f"interaction_density__{source_pheno}__to__{target_pheno}"

    data = adata[
        (adata.obs[image_key] == image_id) &
        (adata.obs[phenotype_key] == source_pheno)
    ]

    x = data.obs[x_key]
    y = data.obs[y_key]
    values = data.obs[feature]

    plt.figure(figsize=(8,8))

    sc = plt.scatter(
        x,
        y,
        c=values,
        cmap="viridis",
        s=size
    )

    plt.colorbar(sc,label=feature)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")

    plt.title(f"{source_pheno} interaction with {target_pheno}")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.show()

def plot_interaction_overlay(
    adata,
    image_id,
    source_pheno,
    target_pheno,
    phenotype_key="annotated_clusters_update3",
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    background_size=1,
    source_size=2
):

    feature = f"interaction_density__{source_pheno}__to__{target_pheno}"

    img = adata[adata.obs[image_key] == image_id]

    src = img[img.obs[phenotype_key] == source_pheno]

    plt.figure(figsize=(8,8))

    # background cells
    plt.scatter(
        img.obs[x_key],
        img.obs[y_key],
        color="lightgrey",
        s=background_size,
        alpha=0.3
    )

    # source cells colored by interaction
    sc = plt.scatter(
        src.obs[x_key],
        src.obs[y_key],
        c=src.obs[feature],
        cmap="viridis",
        s=source_size
    )

    plt.colorbar(sc,label="interaction density")

    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal")

    plt.title(f"{source_pheno} proximity to {target_pheno}")

    plt.show()

import seaborn as sns

def plot_interaction_distribution(
    adata,
    source_pheno,
    target_pheno,
    phenotype_key="annotated_clusters_update3"
):

    feature = f"interaction_density__{source_pheno}__to__{target_pheno}"

    data = adata.obs[adata.obs[phenotype_key] == source_pheno]

    sns.histplot(
        data[feature],
        bins=50
    )

    plt.xlabel("interaction density")
    plt.title(f"{source_pheno} proximity to {target_pheno}")

    plt.show()