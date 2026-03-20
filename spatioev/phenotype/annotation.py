import pandas as pd


def annotate_interactive(
    adata,
    cluster_key="leiden",
    new_key="annotation"
):

    clusters = sorted(adata.obs[cluster_key].unique())

    mapping = {}

    print("\nEnter annotation for each cluster\n")

    for c in clusters:

        label = input(f"Cluster {c}: ")

        mapping[str(c)] = label

    adata.obs[new_key] = adata.obs[cluster_key].map(mapping)

    return adata, mapping


def annotate_from_csv(
    adata,
    csv_file,
    cluster_key="leiden",
    new_key="annotation"
):

    mapping_df = pd.read_csv(csv_file)

    mapping = dict(
        zip(
            mapping_df.cluster.astype(str),
            mapping_df.annotation
        )
    )

    adata.obs[new_key] = adata.obs[cluster_key].map(mapping)

    return adata