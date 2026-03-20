import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def plot_spatial_feature(
    adata,
    feature,
    image_id,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    cmap="viridis",
    point_size=4,
    invert_y=True,
    vmin=None,
    vmax=None,
    figsize=(8, 8),
):
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
    adata,
    feature,
    image_id=None,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    point_size=4,
    alpha=0.85,
    palette="tab20",
    show_legend=False,
    invert_y=True,
    figsize=(8, 8),
    ax=None,
):
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
    adata,
    boundary_df,
    image_id,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    point_size=4,
    point_color="lightgray",
    point_alpha=0.6,
    boundary_color="black",
    boundary_linewidth=1.5,
    show_expanded=True,
    expanded_color="red",
    expanded_linewidth=1.0,
    expanded_linestyle="--",
    show_shrunk=True,
    shrunk_color="blue",
    shrunk_linewidth=1.0,
    shrunk_linestyle=":",
    invert_y=True,
    figsize=(10, 10),
    ax=None,
):
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
    corr_matrix,
    figsize=(6, 6),
    annot=True,
    cmap="coolwarm",
    title="Correlation",
):
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
