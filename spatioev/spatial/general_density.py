import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde


# ----------------------------------------------------
# 1. Assign cells to tiles
# ----------------------------------------------------

def assign_tiles(
    adata,
    tile_size=128,
    x_key="X_centroid",
    y_key="Y_centroid"
):
    """
    Assign each cell to a spatial tile.
    """

    df = adata.obs.copy()

    df["tile_x"] = (df[x_key] // tile_size).astype(int)
    df["tile_y"] = (df[y_key] // tile_size).astype(int)

    return df


# ----------------------------------------------------
# 2. General density (all cells)
# ----------------------------------------------------

def compute_general_density(
    df,
    tile_size=128
):
    """
    Compute density per tile for all cells.
    """

    tile_area = tile_size ** 2

    stats = (
        df.groupby(["imageid", "tile_x", "tile_y"])
        .agg(
            object_count=("label", "nunique"),
            pixel_sum=("area", "sum")
        )
        .reset_index()
    )

    stats["object_density"] = stats["object_count"] / tile_area * 100
    stats["pixel_density"] = stats["pixel_sum"] / tile_area * 100

    return stats


# ----------------------------------------------------
# 3. Phenotype density
# ----------------------------------------------------

def compute_phenotype_density(
    df,
    phenotype_key="phenotype",
    tile_size=128
):
    """
    Compute density per tile for each phenotype.
    """

    tile_area = tile_size ** 2

    stats = (
        df.groupby(["imageid", "tile_x", "tile_y", phenotype_key])
        .agg(
            object_count=("label", "nunique"),
            pixel_sum=("area", "sum")
        )
        .reset_index()
    )

    stats["object_density"] = stats["object_count"] / tile_area * 100
    stats["pixel_density"] = stats["pixel_sum"] / tile_area * 100

    return stats


# ----------------------------------------------------
# 4. Density heatmap
# ----------------------------------------------------

def plot_density_heatmap(
    density_df,
    imageid,
    value="object_density"
):
    """
    Plot tile density heatmap.
    """

    df = density_df[density_df["imageid"] == imageid]

    heatmap = df.pivot(
        index="tile_y",
        columns="tile_x",
        values=value
    )

    plt.figure(figsize=(8, 8))

    sns.heatmap(
        heatmap,
        cmap="viridis",
        square=True
    )

    plt.title(f"{imageid} {value}")
    plt.tight_layout()

def plot_phenotype_density_heatmap(
    density_df,
    phenotype,
    imageid=None,
    phenotype_key="phenotype",
    value="object_density"
):
    """
    Plot tile density heatmap for a specific phenotype.
    """

    df = density_df.copy()

    df = df[df[phenotype_key] == phenotype]

    if imageid is not None:
        df = df[df["imageid"] == imageid]

    heatmap = df.pivot(
        index="tile_y",
        columns="tile_x",
        values=value
    )

    plt.figure(figsize=(8,8))

    sns.heatmap(
        heatmap,
        cmap="viridis",
        square=True,
        cbar_kws={"label": value}
    )

    plt.title(f"{phenotype} density")
    plt.xlabel("Tile X")
    plt.ylabel("Tile Y")

    plt.tight_layout()

# ----------------------------------------------------
# 5. Phenotype density correlation
# ----------------------------------------------------

def phenotype_density_correlation(
    density_df,
    phenotype_key="phenotype",
    value="object_density"
):
    """
    Compute phenotype density correlations across tiles.
    """

    pivot = density_df.pivot_table(
        index=["tile_y", "tile_x"],
        columns=phenotype_key,
        values=value
    )

    corr = pivot.corr()

    return corr

def plot_density_correlation(
    corr_matrix,
    figsize=(6,6),
    annot=True,
    cmap="coolwarm"
):
    """
    Plot phenotype density correlation heatmap.

    Parameters
    ----------
    corr_matrix : DataFrame
        Correlation matrix.
    """

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Pearson correlation",
            "shrink": 0.8
        }
    )

    plt.title("Phenotype density correlation", pad=15)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    return ax

# ----------------------------------------------------
# 6. KDE density (smoothed density)
# ----------------------------------------------------

def compute_kde_density(
    adata,
    phenotype=None,
    phenotype_key="phenotype",
    x_key="X_centroid",
    y_key="Y_centroid",
    grid_size=200
):

    df = adata.obs.copy()

    if phenotype is not None:
        df = df[df[phenotype_key] == phenotype]

    x = df[x_key].values
    y = df[y_key].values

    # shift coordinates so origin is 0
    x = x - x.min()
    y = y - y.min()

    kde = gaussian_kde(np.vstack([x, y]))

    xgrid = np.linspace(0, x.max(), grid_size)
    ygrid = np.linspace(0, y.max(), grid_size)

    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])

    Z = kde(positions).reshape(Xgrid.shape)

    return Xgrid, Ygrid, Z


# ----------------------------------------------------
# 7. Plot KDE density
# ----------------------------------------------------

def plot_kde_density(
    Xgrid,
    Ygrid,
    Z
):

    plt.figure(figsize=(8,8))

    plt.imshow(
        Z,
        cmap="viridis",
        origin="lower",
        extent=[
            Xgrid.min(),
            Xgrid.max(),
            Ygrid.min(),
            Ygrid.max()
        ]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.colorbar(label="Density")

    plt.tight_layout()