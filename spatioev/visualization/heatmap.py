import scanpy as sc


def plot_cluster_heatmap(
    adata,
    markers,
    cluster_key="leiden"
):

    sc.pl.matrixplot(
        adata,
        markers,
        groupby=cluster_key,
        use_raw=False, 
        cmap="vlag", 
        standard_scale='var',
    )