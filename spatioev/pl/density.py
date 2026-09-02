"""Density and interaction plots.

These render the outputs of :mod:`spatioev.tl.density`. They live in ``pl``
because they are presentation, not computation; ``spatioev.pl`` re-exports
them and remains the supported import path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    import anndata as ad


def plot_density_heatmap(
    density_df: pd.DataFrame,
    imageid: str,
    value: str = "object_density",
) -> None:
    """
    Plot a tile-level density heatmap for a single image.

    Parameters
    ----------
    density_df : DataFrame
        Output of :func:`compute_general_density`.
    imageid : str
        Image identifier to visualize.
    value : str
        Column to use as heatmap values, e.g. ``"object_density"`` or
        ``"pixel_density"``.

    Returns
    -------
    None
        Displays the heatmap via matplotlib.

    Examples
    --------
    >>> import spatioev as sv
    >>> sv.pl.plot_density_heatmap(density_df, imageid="sample1")
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
    density_df: pd.DataFrame,
    phenotype: str,
    imageid: str | None = None,
    phenotype_key: str = "phenotype",
    value: str = "object_density",
) -> None:
    """
    Plot a tile-level density heatmap for a specific phenotype.

    Parameters
    ----------
    density_df : DataFrame
        Output of :func:`compute_phenotype_density`.
    phenotype : str
        Phenotype label to plot.
    imageid : str, optional
        If provided, restrict the heatmap to this image.
    phenotype_key : str
        Column in ``density_df`` containing phenotype labels.
    value : str
        Column to use as heatmap values.

    Returns
    -------
    None
        Displays the heatmap via matplotlib.

    Examples
    --------
    >>> import spatioev as sv
    >>> sv.pl.plot_phenotype_density_heatmap(pheno_density, phenotype="Tumor")
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


def plot_density_correlation(
    corr_matrix: pd.DataFrame,
    figsize: tuple[int, int]=(6,6),
    annot: bool=True,
    cmap: str="coolwarm"
) -> None:
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


def plot_kde_density(
    Xgrid: np.ndarray,
    Ygrid: np.ndarray,
    Z: np.ndarray
) -> None:
    """
    Display a KDE density surface as a spatial heatmap.

    Parameters
    ----------
    Xgrid : ndarray of shape (grid_size, grid_size)
        X-coordinates returned by :func:`compute_kde_density`.
    Ygrid : ndarray of shape (grid_size, grid_size)
        Y-coordinates returned by :func:`compute_kde_density`.
    Z : ndarray of shape (grid_size, grid_size)
        Density values returned by :func:`compute_kde_density`.

    Returns
    -------
    None
        Displays the heatmap via matplotlib.

    Examples
    --------
    >>> import spatioev as sv
    >>> Xg, Yg, Z = sv.tl.compute_kde_density(adata)
    >>> sv.pl.plot_kde_density(Xg, Yg, Z)
    """

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


def plot_local_density_map(
    adata: ad.AnnData,
    image_id: str,
    density_key: str="density_all",
    phenotype_key: str=None,
    phenotype: str=None,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    cmap: str="viridis",
    point_size: int=3
) -> None:
    """
    Plot a spatial density scatter map with optional KDE contours.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid, Y_centroid, and a density column in adata.obs.
    image_id : str
        Image identifier to visualize.
    density_key : str
        Column in ``adata.obs`` containing per-cell density values.
    phenotype_key : str, optional
        Column in ``adata.obs`` containing phenotype labels.
    phenotype : str, optional
        If provided along with ``phenotype_key``, restrict the plot to this
        phenotype.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    cmap : str
        Matplotlib colormap name.
    point_size : float
        Marker size passed to ``scatter``.

    Returns
    -------
    tuple
        ``(fig, ax)`` matplotlib figure and axes objects.

    Examples
    --------
    >>> import spatioev as sv
    >>> fig, ax = sv.pl.plot_local_density_map(adata, image_id="sample1")
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


def plot_interaction_density(
    adata: ad.AnnData,
    image_id: str,
    source_pheno: str,
    target_pheno: str,
    phenotype_key: str="annotated_clusters_update3",
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    size: int=1
) -> None:
    """Scatter plot of source cells coloured by interaction density with a target phenotype.

    Requires :func:`phenotype_interaction_density` to have been run first so
    that the ``interaction_density__{source}__to__{target}`` column exists in
    ``adata.obs``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with interaction-density columns in ``adata.obs``.
    image_id : str
        Image/FOV identifier to subset.
    source_pheno, target_pheno : str
        Phenotype labels defining the interaction pair.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    image_key : str
        Column identifying image/FOV membership.
    x_key, y_key : str
        Column names for x and y centroid coordinates.
    size : int
        Scatter point size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    feature = f"interaction_density__{source_pheno}__to__{target_pheno}"

    data = adata[
        (adata.obs[image_key] == image_id) &
        (adata.obs[phenotype_key] == source_pheno)
    ]

    x = data.obs[x_key]
    y = data.obs[y_key]
    values = data.obs[feature]

    fig, ax = plt.subplots(figsize=(8, 8))

    sc = ax.scatter(x, y, c=values, cmap="viridis", s=size)
    fig.colorbar(sc, ax=ax, label=feature)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{source_pheno} interaction with {target_pheno}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    return fig


def plot_interaction_overlay(
    adata: ad.AnnData,
    image_id: str,
    source_pheno: str,
    target_pheno: str,
    phenotype_key: str="annotated_clusters_update3",
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    background_size: int=1,
    source_size: int=2
) -> None:
    """Spatial scatter plot overlaying source cells on a grey-background tissue map.

    All cells are plotted in light grey; source phenotype cells are coloured by
    their interaction-density score with the target phenotype.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with interaction-density columns in ``adata.obs``.
    image_id : str
        Image/FOV identifier to subset.
    source_pheno, target_pheno : str
        Phenotype labels defining the interaction pair.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    image_key : str
        Column identifying image/FOV membership.
    x_key, y_key : str
        Column names for x and y centroid coordinates.
    background_size : int
        Point size for background (all) cells.
    source_size : int
        Point size for source phenotype cells.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    feature = f"interaction_density__{source_pheno}__to__{target_pheno}"

    img = adata[adata.obs[image_key] == image_id]
    src = img[img.obs[phenotype_key] == source_pheno]

    fig, ax = plt.subplots(figsize=(8, 8))

    # background cells
    ax.scatter(img.obs[x_key], img.obs[y_key],
               color="lightgrey", s=background_size, alpha=0.3)

    # source cells colored by interaction density
    sc = ax.scatter(src.obs[x_key], src.obs[y_key],
                    c=src.obs[feature], cmap="viridis", s=source_size)
    fig.colorbar(sc, ax=ax, label="interaction density")

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"{source_pheno} proximity to {target_pheno}")
    fig.tight_layout()
    return fig


def plot_interaction_distribution(
    adata: ad.AnnData,
    source_pheno: str,
    target_pheno: str,
    phenotype_key: str="annotated_clusters_update3"
) -> None:
    """Plot a histogram of interaction-density scores across all images.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with interaction-density columns in ``adata.obs``.
    source_pheno, target_pheno : str
        Phenotype labels defining the interaction pair.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    feature = f"interaction_density__{source_pheno}__to__{target_pheno}"

    data = adata.obs[adata.obs[phenotype_key] == source_pheno]

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data[feature], bins=50, ax=ax)
    ax.set_xlabel("interaction density")
    ax.set_title(f"{source_pheno} proximity to {target_pheno}")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_density_heatmap",
    "plot_phenotype_density_heatmap",
    "plot_density_correlation",
    "plot_kde_density",
    "plot_local_density_map",
    "plot_interaction_density",
    "plot_interaction_overlay",
    "plot_interaction_distribution",
]
