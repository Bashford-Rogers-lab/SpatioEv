# ============================================================
# Section 1: General / tile / KDE density  (from archive/spatial/general_density.py)
# ============================================================

"""
General spatial density utilities.

Provides tile-based and KDE-based density computation and visualization
for all cells and per-phenotype cell populations in multiplexed imaging data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    import anndata as ad


# ----------------------------------------------------
# 1. Assign cells to tiles
# ----------------------------------------------------

def assign_tiles(
    adata: ad.AnnData,
    tile_size: int = 128,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
) -> pd.DataFrame:
    """
    Assign each cell to a spatial tile based on its x/y coordinates.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    tile_size : int
        Side length of each square tile in the same units as the coordinates.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.

    Returns
    -------
    DataFrame
        Copy of ``adata.obs`` with added ``tile_x`` and ``tile_y`` integer columns.

    Examples
    --------
    >>> import spatioev as sv
    >>> tile_df = sv.tl.assign_tiles(adata, tile_size=128)
    """

    df = adata.obs.copy()

    df["tile_x"] = (df[x_key] // tile_size).astype(int)
    df["tile_y"] = (df[y_key] // tile_size).astype(int)

    return df


# ----------------------------------------------------
# 2. General density (all cells)
# ----------------------------------------------------

def compute_general_density(
    df: pd.DataFrame,
    tile_size: int = 128,
) -> pd.DataFrame:
    """
    Compute object and pixel density per spatial tile for all cells.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`assign_tiles` containing ``imageid``, ``tile_x``,
        ``tile_y``, ``label``, and ``area`` columns.
    tile_size : int
        Side length of each square tile in coordinate units.

    Returns
    -------
    DataFrame
        One row per (imageid, tile_x, tile_y) with columns:
        ``object_count``, ``pixel_sum``, ``object_density``, ``pixel_density``.

    Examples
    --------
    >>> import spatioev as sv
    >>> tile_df = sv.tl.assign_tiles(adata, tile_size=128)
    >>> density_df = sv.tl.compute_general_density(tile_df, tile_size=128)
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
    df: pd.DataFrame,
    phenotype_key: str = "phenotype",
    tile_size: int = 128,
) -> pd.DataFrame:
    """
    Compute object and pixel density per spatial tile for each phenotype.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`assign_tiles` containing ``imageid``, ``tile_x``,
        ``tile_y``, ``label``, ``area``, and a phenotype column.
    phenotype_key : str
        Column in ``df`` containing phenotype labels.
    tile_size : int
        Side length of each square tile in coordinate units.

    Returns
    -------
    DataFrame
        One row per (imageid, tile_x, tile_y, phenotype) with columns:
        ``object_count``, ``pixel_sum``, ``object_density``, ``pixel_density``.

    Examples
    --------
    >>> import spatioev as sv
    >>> tile_df = sv.tl.assign_tiles(adata, tile_size=128)
    >>> pheno_density = sv.tl.compute_phenotype_density(tile_df, phenotype_key="phenotype")
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

# ----------------------------------------------------
# 5. Phenotype density correlation
# ----------------------------------------------------

def phenotype_density_correlation(
    density_df: pd.DataFrame,
    phenotype_key: str="phenotype",
    value: str="object_density"
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations of phenotype densities across tiles.

    Parameters
    ----------
    density_df : DataFrame
        Output of :func:`compute_phenotype_density` containing tile coordinates
        and a phenotype column.
    phenotype_key : str
        Column in ``density_df`` containing phenotype labels.
    value : str
        Density column used as values in the pivot table.

    Returns
    -------
    DataFrame
        Square correlation matrix with phenotypes as rows and columns.

    Examples
    --------
    >>> import spatioev as sv
    >>> corr = sv.tl.phenotype_density_correlation(pheno_density)
    >>> corr.style.background_gradient(cmap="coolwarm")
    """

    pivot = density_df.pivot_table(
        index=["tile_y", "tile_x"],
        columns=phenotype_key,
        values=value
    )

    corr = pivot.corr()

    return corr

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

# ----------------------------------------------------
# 6. KDE density (smoothed density)
# ----------------------------------------------------

def compute_kde_density(
    adata: ad.AnnData,
    phenotype: str=None,
    phenotype_key: str="phenotype",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    grid_size: int=200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a kernel density estimate (KDE) on a regular grid.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    phenotype : str, optional
        If provided, restrict KDE to cells of this phenotype.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    grid_size : int
        Number of grid points along each axis.

    Returns
    -------
    Xgrid : ndarray of shape (grid_size, grid_size)
        X-coordinates of the evaluation grid.
    Ygrid : ndarray of shape (grid_size, grid_size)
        Y-coordinates of the evaluation grid.
    Z : ndarray of shape (grid_size, grid_size)
        Estimated density values on the grid.

    Examples
    --------
    >>> import spatioev as sv
    >>> Xg, Yg, Z = sv.tl.compute_kde_density(adata, phenotype="Tumor")
    >>> sv.pl.plot_kde_density(Xg, Yg, Z)
    """

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



# ============================================================
# Section 2: kNN local density  (from archive/spatial/local_density_KNN.py)
# ============================================================

"""
KNN-based local cell density utilities.

Computes per-cell local density using k-nearest-neighbor distances,
supporting both within-phenotype and global-population analyses.
"""

from sklearn.neighbors import BallTree

# ------------------------------------------------------------
# Local density by phenotype
# ------------------------------------------------------------

def compute_local_density_by_phenotype(
    adata: ad.AnnData,
    phenotype_key: str="phenotype",
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    k_neighbors: int=5
) -> pd.DataFrame:
    """
    Compute local density within each phenotype using kNN distances.

    Density is defined as 1 / mean distance to the k nearest neighbors
    of the same phenotype within each image.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid, Y_centroid, imageid, and phenotype columns in adata.obs.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    k_neighbors : int
        Number of nearest same-phenotype neighbors to use.

    Returns
    -------
    AnnData
        Modified in-place with added columns in ``adata.obs``:
        ``mean_dist_pheno`` (mean kNN distance) and
        ``density_pheno`` (reciprocal density estimate).

    Examples
    --------
    >>> import spatioev as sv
    >>> adata = sv.tl.compute_local_density_by_phenotype(adata, phenotype_key="phenotype")
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
    adata: ad.AnnData,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    k_neighbors: int=5
) -> pd.DataFrame:
    """
    Compute local density for all cells using kNN distances.

    Density is defined as 1 / mean distance to the k nearest neighbors
    within each image, regardless of phenotype.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    k_neighbors : int
        Number of nearest neighbors to use.

    Returns
    -------
    AnnData
        Modified in-place with added columns in ``adata.obs``:
        ``mean_dist_all`` (mean kNN distance) and
        ``density_all`` (reciprocal density estimate).

    Examples
    --------
    >>> import spatioev as sv
    >>> adata = sv.tl.compute_local_density_all_cells(adata)
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



# ============================================================
# Section 3: Radius local density  (from archive/spatial/local_density_radius.py)
# ============================================================

"""
Radius-based local cell density utilities.

Computes per-cell local density as the number of neighbors within a fixed
radius divided by the area of the corresponding circle.
"""



def compute_radius_density(
    adata: ad.AnnData,
    radius: float=50,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    density_key: str="radius_density",
    edge_key: str=None,
    exclude_edge: bool=False,
) -> ad.AnnData:
    """
    Compute local density within a fixed radius for all cells.

    Density is defined as (number of neighbors within radius) / (pi * radius^2),
    giving a per-unit-area cell density. Self is excluded from the count.

    Parameters
    ----------
    adata : AnnData
        AnnData with X_centroid and Y_centroid columns in adata.obs.
    radius : float
        Search radius in the same coordinate units as ``x_key``/``y_key``.
    x_key : str
        Column in ``adata.obs`` containing x-coordinates.
    y_key : str
        Column in ``adata.obs`` containing y-coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    density_key : str
        Output column name added to ``adata.obs``.
    edge_key : str, optional
        Boolean column marking edge cells. Only used when ``exclude_edge=True``.
    exclude_edge : bool
        If ``True``, set density to NaN for cells flagged by ``edge_key``.

    Returns
    -------
    AnnData
        Modified in-place with ``density_key`` added to ``adata.obs``.

    Examples
    --------
    >>> import spatioev as sv
    >>> adata = sv.tl.compute_radius_density(adata, radius=50)
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



# ============================================================
# Section 4: Phenotype interaction density  (from archive/spatial/interaction.py)
# ============================================================

"""
Cell-cell interaction density utilities.

Provides radius-based phenotype interaction density computation and
visualization functions for source-to-target cell proximity analysis.
"""



def phenotype_interaction_density(
    adata: ad.AnnData,
    phenotype_key: str,
    source_pheno: str,
    target_pheno: str,
    radius: float=50,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    out_key: str=None,
    edge_key: str=None,
    exclude_edge: bool=False,
) -> ad.AnnData:
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
    "assign_tiles",
    "compute_general_density",
    "compute_phenotype_density",
    "phenotype_density_correlation",
    "compute_kde_density",
    "plot_density_heatmap",
    "plot_phenotype_density_heatmap",
    "plot_density_correlation",
    "plot_kde_density",
    "compute_local_density_by_phenotype",
    "compute_local_density_all_cells",
    "plot_local_density_map",
    "compute_radius_density",
    "phenotype_interaction_density",
    "plot_interaction_density",
    "plot_interaction_overlay",
    "plot_interaction_distribution",
]
