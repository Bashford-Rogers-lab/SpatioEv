def subset_cells(
    adata,
    annotation_key,
    labels
):

    if isinstance(labels, str):
        labels = [labels]

    subset = adata[
        adata.obs[annotation_key].isin(labels)
    ].copy()

    return subset