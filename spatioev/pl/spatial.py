from __future__ import annotations

# ============================================================
# Section 1: Cluster heatmap  (from archive/visualization/heatmap.py)
# ============================================================

def _require_scanpy():
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "plot_cluster_heatmap requires the optional Scanpy dependency. "
            "Install SpatioEv with `pip install -e '.[scanpy]'` or "
            "`pip install scanpy`."
        ) from exc

    return sc


def plot_cluster_heatmap(
    adata: ad.AnnData,
    markers: list[str],
    cluster_key: str="leiden"
) -> None:
    """Plot a marker-expression matrix grouped by cluster.

    Wraps ``scanpy.pl.matrixplot`` with variance-scaled expression.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with clustering labels in ``adata.obs[cluster_key]``.
    markers : list of str
        Marker names to include in the heatmap (must be in ``adata.var_names``).
    cluster_key : str
        Column in ``adata.obs`` containing cluster labels.

    Returns
    -------
    None
        Displays the matrix plot via Scanpy's interactive backend.
    """
    sc = _require_scanpy()

    sc.pl.matrixplot(
        adata,
        markers,
        groupby=cluster_key,
        use_raw=False, 
        cmap="vlag", 
        standard_scale='var',
    )



# ============================================================
# Section 2: UMAP overview  (from archive/visualization/overview.py)
# ============================================================

def plot_refinement_umaps(results: dict) -> None:
    """Plot UMAP embeddings for a collection of sub-clustered AnnData objects.

    Iterates over the *results* dict (output of :func:`~spatioev.tl.phenotype.refine_clusters`)
    and calls ``scanpy.pl.umap`` on each entry, colouring by the ``leiden``
    cluster label.

    Parameters
    ----------
    results : dict of str → AnnData
        Mapping from cluster name to sub-clustered AnnData with UMAP
        coordinates in ``obsm["X_umap"]`` and ``"leiden"`` in ``obs``.

    Returns
    -------
    None
        Displays one UMAP panel per entry via Scanpy's interactive backend.
    """
    sc = _require_scanpy()

    for name, adata in results.items():

        sc.pl.umap(
            adata,
            color="leiden",
            title=name
        )



# ============================================================
# Section 3: Spatial scatter  (from archive/visualization/spatial_scatter.py)
# ============================================================

def spatial_scatter_plot(
    adata: ad.AnnData,
    colorBy: str,
    topLayer: str | None=None,
    x_coordinate: str='X_centroid',
    y_coordinate: str='Y_centroid',
    imageid: str='imageid',
    layer: str | None=None,
    subset: str | None=None,
    s: int=None,
    ncols: int=None,
    alpha: float=1,
    dpi: int=200,
    fontsize: int=None,
    plotLegend: bool=True,
    cmap: str='RdBu_r',
    catCmap: str='tab20',
    vmin: float | None=None,
    vmax: float | None=None,
    customColors: dict | None=None,
    figsize: tuple[int, int]=(5, 5),
    invert_yaxis: bool=True,
    saveDir: str | None=None,
    fileName: str | None='ScatterPlot.png',
    **kwargs,
) -> object:
    """Spatial scatter plot coloured by one or more marker/metadata columns.

    Renders each cell as a point at its (x, y) tissue coordinates, coloured
    either by continuous expression (colormap) or by categorical annotation
    (palette).  Supports multi-panel layouts when multiple columns are supplied.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with cell coordinates in ``adata.obs``.
    colorBy : str or list of str
        Marker name(s) (from ``adata.var_names``) or metadata column(s) (from
        ``adata.obs``) to colour by.  Multiple names produce a multi-panel figure.
    topLayer : str or list of str or None
        Categorical value(s) to render on top of all other cells (e.g. to
        highlight a rare phenotype).
    x_coordinate, y_coordinate : str
        Column names for x and y centroid coordinates in ``adata.obs``.
    imageid : str
        Column identifying image/FOV membership.
    layer : str or None
        AnnData layer to use for expression values.  ``None`` uses ``adata.X``;
        ``"raw"`` uses ``adata.raw.X``.
    subset : str or list of str or None
        Image IDs to display.  ``None`` shows all images.
    s : int or None
        Scatter point size.  Auto-scaled from dataset size if ``None``.
    ncols : int or None
        Number of columns in the multi-panel grid.
    alpha : float
        Point transparency (0–1).
    dpi : int
        Figure resolution.
    fontsize : int or None
        Font size for tick labels and legend.
    plotLegend : bool
        Whether to draw a colorbar (continuous) or legend (categorical).
    cmap : str
        Colormap for continuous columns.
    catCmap : str
        Colormap used to assign colours to categorical values.
    vmin, vmax : float or None
        Colormap limits for continuous columns.
    customColors : dict or None
        Optional mapping from category label to colour for categorical columns.
    figsize : tuple of int
        ``(width, height)`` in inches for the whole figure.
    invert_yaxis : bool
        If ``True``, invert the y-axis so tissue coordinates match image space.
    saveDir : str or None
        Directory to save the figure.  If ``None``, the figure is returned and
        displayed inline in Jupyter (no ``plt.show()`` call inside the function).
    fileName : str or None
        File name for saved figure (requires *saveDir*).

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object when *saveDir* is ``None``; ``None`` when the figure
        has been saved and closed.
    """
    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import anndata as ad

    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    else:
        adata = adata.copy()

    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        if layer == 'raw':
            bdata = adata.copy()
            bdata.X = adata.raw.X
            bdata = bdata[bdata.obs[imageid].isin(subset)]
        else:
            bdata = adata.copy()
            bdata = bdata[bdata.obs[imageid].isin(subset)]
    else:
        bdata = adata.copy()

    if layer is None:
        data = pd.DataFrame(bdata.X, index=bdata.obs.index, columns=bdata.var.index)
    elif layer == 'raw':
        data = pd.DataFrame(bdata.raw.X, index=bdata.obs.index, columns=bdata.var.index)
    else:
        data = pd.DataFrame(bdata.layers[layer], index=bdata.obs.index, columns=bdata.var.index)

    meta = bdata.obs

    if isinstance(topLayer, str):
        topLayer = [topLayer]

    if isinstance(colorBy, str):
        colorBy = [colorBy]

    data_cols = [col for col in data.columns if col in colorBy]
    meta_cols = [col for col in meta.columns if col in colorBy]
    colorColumns = pd.concat([data[data_cols], meta[meta_cols]], axis=1)

    x = meta[x_coordinate]
    y = meta[y_coordinate]

    def calculate_grid_dimensions(num_items, num_columns=None):
        if num_columns is None:
            num_rows_columns = int(math.ceil(math.sqrt(num_items)))
            return num_rows_columns, num_rows_columns
        else:
            num_rows = int(math.ceil(num_items / num_columns))
            return num_rows, num_columns

    nrows, ncols = calculate_grid_dimensions(len(colorColumns.columns), num_columns=ncols)

    if s is None:
        s = (10000 / bdata.shape[0]) / len(colorColumns.columns)

    cmap_cat = plt.get_cmap(catCmap)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axs = [axs] if nrows == 1 and ncols == 1 else axs.flatten()

    for i, col in enumerate(colorColumns):
        ax = axs[i]
        ax.set_aspect('equal')  # <- Ensures x/y axis scale is equal
        if invert_yaxis:
            ax.invert_yaxis()

        if colorColumns[col].dtype.kind in 'iufc':
            scatter = ax.scatter(x=x, y=y, c=colorColumns[col], cmap=cmap, s=s,
                                 vmin=vmin, vmax=vmax, linewidths=0, alpha=alpha, **kwargs)
            if plotLegend:
                cbar = plt.colorbar(scatter, ax=ax, pad=0)
                cbar.ax.tick_params(labelsize=fontsize)
        else:
            categories = colorColumns[col].unique()
            colors = {cat: customColors[cat] for cat in categories if customColors and cat in customColors}
            if not customColors:
                colors = {cat: cmap_cat(i) for i, cat in enumerate(categories)}

            categories_to_plot_last = [cat for cat in topLayer if cat in categories] if topLayer else []
            categories_to_plot_first = [cat for cat in categories if cat not in categories_to_plot_last]

            for cat in categories_to_plot_first + categories_to_plot_last:
                cat_mask = colorColumns[col] == cat
                ax.scatter(x=x[cat_mask], y=y[cat_mask],
                           c=[colors.get(cat, cmap_cat(np.where(categories == cat)[0][0]))],
                           s=s, linewidths=0, alpha=alpha, **kwargs)

            if plotLegend:
                handles = [mpatches.Patch(color=colors.get(cat, cmap_cat(np.where(categories == cat)[0][0])), label=cat)
                           for cat in categories]
                ax.legend(handles=handles, bbox_to_anchor=(1.0, 1.0), loc='upper left',
                          bbox_transform=ax.transAxes, fontsize=fontsize)

        ax.set_title(col)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for i in range(len(colorColumns.columns), nrows * ncols):
        fig.delaxes(axs[i])

    plt.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()

    if saveDir:
        os.makedirs(saveDir, exist_ok=True)
        full_path = os.path.join(saveDir, fileName)
        fig.savefig(full_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved plot to {full_path}")
        return None
    else:
        return fig



# ============================================================
# Section 4: Cluster inspection  (from archive/visualization/spatial.py)
# ============================================================

def _require_image_viewer_dependencies():
    try:
        import napari
        import scimap as sm
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime
        if exc.name == "pkg_resources":
            raise ImportError(
                "SCIMAP's image-viewer dependency mpl-scatter-density requires "
                "the legacy pkg_resources module. Install a compatible runtime "
                "with `python -m pip install 'setuptools<82'`."
            ) from exc
        raise ImportError(
            "inspect_clusters requires optional image viewer dependencies. "
            "Install SpatioEv with `pip install -e '.[viewer]'` or install "
            "`scimap[napari]`."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "inspect_clusters requires optional image viewer dependencies. "
            "Install SpatioEv with `pip install -e '.[viewer]'` or install "
            "`scimap[napari]`."
        ) from exc

    return sm, napari


def inspect_clusters(
    adata: ad.AnnData,
    image_path: str | None,
    label: str="leiden",
    block: bool=True
) -> None:
    """Open a Napari image viewer to inspect cluster labels overlaid on the tissue.

    Requires the optional ``scimap[napari]`` installation.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with cluster labels in ``adata.obs[label]`` and cell
        coordinate columns in ``adata.obs``.
    image_path : str or None
        Path to the tissue image to display in the background.
    label : str
        Column in ``adata.obs`` to use as the overlay annotation.
    block : bool
        If ``True``, blocks Python execution until the Napari viewer is closed.

    Returns
    -------
    None
        Opens an interactive Napari viewer.
    """
    sm, napari = _require_image_viewer_dependencies()

    # open viewer via scimap
    sm.pl.image_viewer(
        image_path=image_path,
        adata=adata,
        overlay=label,
        point_size=10,
        point_color="white"
    )

    if not block:
        return

    print("\nInspect clusters in napari. Close the viewer to continue.\n")

    viewer = napari.current_viewer()

    if viewer is None:
        print("Warning: napari viewer not detected.")
        return

    # Wait until viewer closes
    from qtpy.QtCore import QEventLoop

    loop = QEventLoop()

    viewer.window._qt_window.destroyed.connect(loop.quit)

    loop.exec_()



# ============================================================
# Section 5: Spatial feature plots  (from archive/spatial/visualization.py)
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_spatial_feature(
    adata: ad.AnnData,
    feature: str,
    image_id: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    cmap: str="viridis",
    point_size: int=4,
    invert_y: bool=True,
    vmin: float | None=None,
    vmax: float | None=None,
    figsize: tuple[int, int]=(8, 8),
) -> None:
    """
    Scatter plot of a spatial feature.
    """
    data = adata[adata.obs[image_key] == image_id]

    x = data.obs[x_key].to_numpy()
    y = data.obs[y_key].to_numpy()
    val = data.obs[feature].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        x,
        y,
        c=val,
        cmap=cmap,
        s=point_size,
        edgecolor="none",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(sc, ax=ax, label=feature)

    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{feature} | {image_id}")
    plt.tight_layout()

    return fig, ax


def plot_spatial_category(
    adata: ad.AnnData,
    feature: str,
    image_id: str=None,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    point_size: int=4,
    alpha: float=0.85,
    palette: dict | None="tab20",
    show_legend: bool=False,
    invert_y: bool=True,
    figsize: tuple[int, int]=(8, 8),
    ax: object | None=None,
) -> None:
    """
    Scatter plot of a categorical spatial feature.

    This is useful for visualizing fields such as:
    - tumour components
    - niche regions
    - phenotypes
    - neighborhood identities

    Parameters
    ----------
    feature : str
        Column in ``adata.obs`` containing the categorical labels to plot.
    image_id : str, optional
        If provided, only cells from this image are plotted.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    point_size : float
        Marker size for the scatter plot.
    alpha : float
        Marker transparency.
    palette : str or list
        Seaborn/matplotlib palette used for categories.
    show_legend : bool
        If ``False``, suppress the legend. This is recommended for
        high-cardinality labels such as tumour components.
    invert_y : bool
        Whether to invert the y-axis to match image coordinates.
    figsize : tuple
        Figure size used when ``ax`` is not supplied.
    ax : matplotlib Axes, optional
        Existing axes to draw on.
    """
    if image_id is not None:
        data = adata[adata.obs[image_key] == image_id].obs.copy()
    else:
        data = adata.obs.copy()

    data = data[[x_key, y_key, feature]].dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.scatterplot(
        data=data,
        x=x_key,
        y=y_key,
        hue=feature,
        s=point_size,
        linewidth=0,
        alpha=alpha,
        palette=palette,
        legend=show_legend,
        ax=ax,
    )

    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    title = feature if image_id is None else f"{feature} | {image_id}"
    ax.set_title(title)
    ax.grid(False)

    if not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    plt.tight_layout()
    return fig, ax


def plot_niche_boundaries(
    adata: ad.AnnData,
    boundary_df: pd.DataFrame,
    image_id: str,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    point_size: int=4,
    point_color: str="lightgray",
    point_alpha: float=0.6,
    boundary_color: str="black",
    boundary_linewidth: float=1.5,
    show_expanded: bool=True,
    expanded_color: str="red",
    expanded_linewidth: float=1.0,
    expanded_linestyle: str="--",
    show_shrunk: bool=True,
    shrunk_color: str="blue",
    shrunk_linewidth: float=1.0,
    shrunk_linestyle: str=":",
    invert_y: bool=True,
    figsize: tuple[int, int]=(10, 10),
    ax: object | None=None,
) -> None:
    """
    Overlay niche boundary geometries on top of spatial cell coordinates.

    This helper is intended for QC and visualization checks of outputs from
    ``build_niche_boundaries()`` and ``buffer_niche_boundaries()``.

    Parameters
    ----------
    boundary_df : DataFrame
        Boundary table containing at least ``geometry`` and optionally
        ``expanded_geometry``.
    image_id : str
        Image identifier to plot.
    image_key : str
        Column in both ``adata.obs`` and ``boundary_df`` identifying the image.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    point_size : float
        Marker size for cells.
    point_color : str
        Color used for background cells.
    point_alpha : float
        Marker transparency for cells.
    boundary_color : str
        Color used for the primary niche boundary.
    boundary_linewidth : float
        Line width for the primary niche boundary.
    show_expanded : bool
        Whether to overlay ``expanded_geometry`` when present.
    expanded_color : str
        Color used for the expanded niche boundary, if present.
    expanded_linewidth : float
        Line width for the expanded boundary.
    expanded_linestyle : str
        Line style for the expanded boundary.
    show_shrunk : bool
        Whether to overlay ``shrunk_geometry`` when present.
    shrunk_color : str
        Color used for the shrunk niche boundary, if present.
    shrunk_linewidth : float
        Line width for the shrunk boundary.
    shrunk_linestyle : str
        Line style for the shrunk boundary.
    invert_y : bool
        Whether to invert the y-axis to match image coordinates.
    figsize : tuple
        Figure size used when ``ax`` is not supplied.
    ax : matplotlib Axes, optional
        Existing axes to draw on.
    """
    plot_df = adata.obs.loc[adata.obs[image_key] == image_id].copy()
    plot_bounds = boundary_df.loc[boundary_df[image_key] == image_id].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.scatterplot(
        data=plot_df,
        x=x_key,
        y=y_key,
        s=point_size,
        color=point_color,
        linewidth=0,
        alpha=point_alpha,
        legend=False,
        ax=ax,
    )

    def _plot_geometry(geom, color, linewidth, linestyle="-"):
        from shapely.geometry import MultiPolygon, Polygon

        if geom is None:
            return

        if isinstance(geom, Polygon):
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle)

    for _, row in plot_bounds.iterrows():
        _plot_geometry(
            row.get("geometry"),
            color=boundary_color,
            linewidth=boundary_linewidth,
        )
        if show_expanded:
            _plot_geometry(
                row.get("expanded_geometry"),
                color=expanded_color,
                linewidth=expanded_linewidth,
                linestyle=expanded_linestyle,
            )
        if show_shrunk:
            _plot_geometry(
                row.get("shrunk_geometry"),
                color=shrunk_color,
                linewidth=shrunk_linewidth,
                linestyle=shrunk_linestyle,
            )

    ax.set_title(f"Niche boundaries: {image_id}")
    if invert_y:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    plt.tight_layout()
    return fig, ax


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    figsize: tuple[int, int]=(6, 6),
    annot: bool=True,
    cmap: str="coolwarm",
    title: str | None="Correlation",
) -> None:
    """
    Pretty lower-triangle correlation heatmap.
    """
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
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
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig, ax



__all__ = [
    "plot_cluster_heatmap",
    "plot_refinement_umaps",
    "spatial_scatter_plot",
    "inspect_clusters",
    "plot_spatial_feature",
    "plot_spatial_category",
    "plot_niche_boundaries",
    "plot_correlation_heatmap",
]
