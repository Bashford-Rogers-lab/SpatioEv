def spatial_scatter_plot(
    adata,
    colorBy,
    topLayer=None,
    x_coordinate='X_centroid',
    y_coordinate='Y_centroid',
    imageid='imageid',
    layer=None,
    subset=None,
    s=None,
    ncols=None,
    alpha=1,
    dpi=200,
    fontsize=None,
    plotLegend=True,
    cmap='RdBu_r',
    catCmap='tab20',
    vmin=None,
    vmax=None,
    customColors=None,
    figsize=(5, 5),
    invert_yaxis=True,
    saveDir=None,
    fileName='ScatterPlot.png',
    **kwargs,
):
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
        plt.savefig(full_path, dpi=dpi)
        plt.close()
        print(f"Saved plot to {full_path}")
    else:
        plt.show()