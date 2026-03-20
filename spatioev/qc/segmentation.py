# spatioev/qc/segmentation.py

import numpy as np
import pandas as pd
from spatioev.config import QCConfig


def compute_area_um2(adata, pixel_size):
    adata.obs["area_um2"] = adata.obs["area"] * (pixel_size ** 2)
    return adata


def categorize_area(adata, min_area, max_area):
    conditions = [
        adata.obs["area_um2"] < min_area,
        adata.obs["area_um2"] > max_area,
    ]
    choices = ["debris_fragment", "merged_cell"]

    adata.obs["area_category"] = np.select(
        conditions,
        choices,
        default="normal_area"
    )
    return adata


def categorize_nc_ratio(adata, max_ratio):
    adata.obs["nc_ratio_category"] = np.where(
        adata.obs["nc_ratio"] > max_ratio,
        "abnormal_nc_ratio",
        "normal_nc_ratio"
    )
    return adata


def filter_segmentation_errors(adata):
    mask = (
        (adata.obs["area_category"] == "normal_area") &
        (adata.obs["nc_ratio_category"] == "normal_nc_ratio")
    )
    return adata[mask].copy()


def run_segmentation_qc(adata, config: QCConfig):
    adata = compute_area_um2(adata, config.pixel_size)
    adata = categorize_area(adata, config.min_area_um2, config.max_area_um2)
    adata = categorize_nc_ratio(adata, config.max_nc_ratio)
    return adata


def generate_qc_summary(
    adata,
    groupby: str | None = None,
    verbose: bool = False
):
    """
    Generate a QC summary table from QC-annotated AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object after running run_segmentation_qc().
        Must contain:
            - area_category
            - nc_ratio_category

    groupby : str, optional
        Column in adata.obs to group by (e.g., "fov", "sample").
        If None, summary is computed for the entire dataset.

    verbose : bool
        If True, prints a formatted summary.

    Returns
    -------
    summary_df : pd.DataFrame
        QC summary table.
    """

    required_cols = ["area_category", "nc_ratio_category"]
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                "Run run_segmentation_qc() first."
            )

    if groupby is None:

        total = adata.n_obs

        debris = (adata.obs["area_category"] == "debris_fragment").sum()
        merged = (adata.obs["area_category"] == "merged_cell").sum()
        abnormal_nc = (adata.obs["nc_ratio_category"] == "abnormal_nc_ratio").sum()

        removed = debris + merged + abnormal_nc
        normal = total - removed

        summary_dict = {
            "total_cells": total,
            "normal_cells": normal,
            "debris_fragment": debris,
            "merged_cell": merged,
            "abnormal_nc_ratio": abnormal_nc,
            "removed_total": removed,
            "percent_removed": round(removed / total * 100, 2)
        }

        summary_df = pd.DataFrame([summary_dict])

    else:

        summaries = []

        for group_name, group_df in adata.obs.groupby(groupby):

            total = group_df.shape[0]

            debris = (group_df["area_category"] == "debris_fragment").sum()
            merged = (group_df["area_category"] == "merged_cell").sum()
            abnormal_nc = (group_df["nc_ratio_category"] == "abnormal_nc_ratio").sum()

            removed = debris + merged + abnormal_nc
            normal = total - removed

            summaries.append({
                groupby: group_name,
                "total_cells": total,
                "normal_cells": normal,
                "debris_fragment": debris,
                "merged_cell": merged,
                "abnormal_nc_ratio": abnormal_nc,
                "removed_total": removed,
                "percent_removed": round(removed / total * 100, 2)
            })

        summary_df = pd.DataFrame(summaries)

    if verbose:
        print("\nQC Summary")
        print(summary_df)

    return summary_df