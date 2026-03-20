import scimap as sm

def inspect_reassigned_cells(
    adata,
    image_path,
    original_label="annotated_clusters_update3",
    predicted_label="svm_prediction",
    original_value="Unknown",
    point_size=6
):
    """
    Visualize cells reassigned by SVM.

    Parameters
    ----------
    adata : AnnData
    image_path : str
    original_label : str
        Column containing manual annotation
    predicted_label : str
        Column containing SVM prediction
    original_value : str
        Original class to inspect (e.g. "Unknown")
    """

    mask = adata.obs[original_label] == original_value

    print(f"Cells originally '{original_value}': {mask.sum()}")

    subset = adata[mask].copy()

    sm.pl.image_viewer(
        image_path=image_path,
        adata=subset,
        overlay=predicted_label,
        point_size=point_size,
        point_color="white"
    )

def inspect_disagreements(
    adata,
    image_path,
    original_label="annotated_clusters_update3",
    predicted_label="svm_prediction",
    point_size=6
):
    """
    Visualize cells where manual annotation and SVM prediction disagree.
    """

    mask = adata.obs[original_label] != adata.obs[predicted_label]

    print(f"Disagreement cells: {mask.sum()}")

    subset = adata[mask].copy()

    sm.pl.image_viewer(
        image_path=image_path,
        adata=subset,
        overlay=predicted_label,
        point_size=point_size,
        point_color="white"
    )