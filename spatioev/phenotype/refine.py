from .clustering import cluster_cells

def refine_clusters(
    adata,
    clusters,
    configs,
    annotation_key="annotation"
):

    results = {}

    for cluster in clusters:

        if cluster not in configs:
            raise ValueError(f"No config provided for cluster '{cluster}'")

        subset = adata[
            adata.obs[annotation_key] == cluster
        ].copy()

        if subset.n_obs == 0:
            print(f"Skipping {cluster} (no cells)")
            continue

        config = configs[cluster]

        clustered = cluster_cells(subset, config)

        results[cluster] = clustered

    return results