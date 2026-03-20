def merge_annotations(
    adata,
    subset,
    new_key="annotation_level2"
):

    if new_key not in adata.obs:
        adata.obs[new_key] = None

    adata.obs.loc[
        subset.obs_names,
        new_key
    ] = subset.obs["annotation"]

    return adata

def merge_refinements(
    adata,
    refined_results,
    new_key="annotation_level2"
):

    if new_key not in adata.obs:
        adata.obs[new_key] = adata.obs.get("annotation", None)

    # convert categorical to string to allow new labels
    adata.obs[new_key] = adata.obs[new_key].astype("object")

    for subset in refined_results.values():

        if "annotation" not in subset.obs:
            continue

        adata.obs.loc[
            subset.obs_names,
            new_key
        ] = subset.obs["annotation"].astype("object")

    return adata