"""Segmentation quality-control functions.

Filters cells by morphology metrics (area, nucleus/cytoplasm ratio) and
produces per-image QC summary tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

import anndata as ad

from spatioev.config import QCConfig


def compute_area_um2(adata: ad.AnnData, pixel_size: float) -> ad.AnnData:
    """Convert pixel-area to physical area in µm².

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with an ``area`` column in ``adata.obs`` holding
        cell area in pixels².
    pixel_size : float
        Physical size of one pixel in micrometres.

    Returns
    -------
    anndata.AnnData
        Same object with an added ``area_um2`` column in ``adata.obs``.
    """
    adata.obs["area_um2"] = adata.obs["area"] * (pixel_size ** 2)
    return adata


def categorize_area(adata: ad.AnnData, min_area: float, max_area: float) -> ad.AnnData:
    """Categorise cells by their physical area.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with ``area_um2`` in ``adata.obs`` (add via
        :func:`compute_area_um2`).
    min_area : float
        Minimum valid area in µm².  Cells below this are labelled
        ``"debris_fragment"``.
    max_area : float
        Maximum valid area in µm².  Cells above this are labelled
        ``"merged_cell"``.

    Returns
    -------
    anndata.AnnData
        Same object with an added ``area_category`` column
        (``"normal_area"`` | ``"debris_fragment"`` | ``"merged_cell"``).
    """
    conditions = [
        adata.obs["area_um2"] < min_area,
        adata.obs["area_um2"] > max_area,
    ]
    choices = ["debris_fragment", "merged_cell"]
    adata.obs["area_category"] = np.select(
        conditions, choices, default="normal_area"
    )
    return adata


def categorize_nc_ratio(adata: ad.AnnData, max_ratio: float) -> ad.AnnData:
    """Categorise cells by nucleus-to-cytoplasm ratio.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with an ``nc_ratio`` column in ``adata.obs``.
    max_ratio : float
        Upper bound for a valid NC ratio.  Cells above this are labelled
        ``"abnormal_nc_ratio"``.

    Returns
    -------
    anndata.AnnData
        Same object with an added ``nc_ratio_category`` column
        (``"normal_nc_ratio"`` | ``"abnormal_nc_ratio"``).
    """
    adata.obs["nc_ratio_category"] = np.where(
        adata.obs["nc_ratio"] > max_ratio,
        "abnormal_nc_ratio",
        "normal_nc_ratio",
    )
    return adata


def filter_segmentation_errors(adata: ad.AnnData) -> ad.AnnData:
    """Remove cells flagged as debris, merged, or abnormal NC ratio.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData after :func:`run_segmentation_qc` has been applied.

    Returns
    -------
    anndata.AnnData
        Filtered copy containing only cells with ``area_category ==
        "normal_area"`` and ``nc_ratio_category == "normal_nc_ratio"``.
    """
    mask = (
        (adata.obs["area_category"] == "normal_area")
        & (adata.obs["nc_ratio_category"] == "normal_nc_ratio")
    )
    return adata[mask].copy()


def run_segmentation_qc(adata: ad.AnnData, config: QCConfig) -> ad.AnnData:
    """Run the full segmentation QC pipeline.

    Applies area conversion, area categorisation, and NC-ratio
    categorisation in sequence.  Does **not** filter cells — call
    :func:`filter_segmentation_errors` afterwards if needed.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with ``area`` and ``nc_ratio`` columns in ``adata.obs``.
    config : QCConfig
        Configuration object specifying ``pixel_size``, ``min_area_um2``,
        ``max_area_um2``, and ``max_nc_ratio``.

    Returns
    -------
    anndata.AnnData
        Same object (modified in place and returned) with added columns
        ``area_um2``, ``area_category``, ``nc_ratio_category``.

    Examples
    --------
    >>> import anndata as ad, numpy as np, pandas as pd
    >>> import spatioev as sv
    >>> from spatioev.config import QCConfig
    >>> adata = ad.AnnData(
    ...     obs=pd.DataFrame({"area": [80, 5, 2000], "nc_ratio": [0.4, 0.3, 1.2]})
    ... )
    >>> adata = sv.pp.run_segmentation_qc(adata, QCConfig(pixel_size=0.325))
    >>> adata.obs["area_category"].tolist()
    ['normal_area', 'debris_fragment', 'merged_cell']
    """
    adata = compute_area_um2(adata, config.pixel_size)
    adata = categorize_area(adata, config.min_area_um2, config.max_area_um2)
    adata = categorize_nc_ratio(adata, config.max_nc_ratio)
    return adata


def generate_qc_summary(
    adata: ad.AnnData,
    groupby: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate a QC summary table from a QC-annotated AnnData.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData after :func:`run_segmentation_qc`.  Must contain
        ``area_category`` and ``nc_ratio_category`` in ``adata.obs``.
    groupby : str or None
        Column in ``adata.obs`` to group by (e.g., ``"imageid"``).
        If ``None``, the summary covers the entire dataset.
    verbose : bool
        If ``True``, prints the summary table to stdout.

    Returns
    -------
    pandas.DataFrame
        Columns: ``total_cells``, ``normal_cells``, ``debris_fragment``,
        ``merged_cell``, ``abnormal_nc_ratio``, ``removed_total``,
        ``percent_removed``.  One row per group (or one row overall when
        *groupby* is ``None``).

    Examples
    --------
    >>> summary = sv.pp.generate_qc_summary(adata, groupby="imageid")
    >>> summary[["imageid", "total_cells", "percent_removed"]]
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
            "percent_removed": round(removed / total * 100, 2),
        }
        summary_df = pd.DataFrame([summary_dict])

    else:
        summaries = []
        for group_name, group_df in adata.obs.groupby(groupby):
            total = group_df.shape[0]
            debris = (group_df["area_category"] == "debris_fragment").sum()
            merged = (group_df["area_category"] == "merged_cell").sum()
            abnormal_nc = (
                group_df["nc_ratio_category"] == "abnormal_nc_ratio"
            ).sum()
            removed = debris + merged + abnormal_nc
            normal = total - removed
            summaries.append(
                {
                    groupby: group_name,
                    "total_cells": total,
                    "normal_cells": normal,
                    "debris_fragment": debris,
                    "merged_cell": merged,
                    "abnormal_nc_ratio": abnormal_nc,
                    "removed_total": removed,
                    "percent_removed": round(removed / total * 100, 2),
                }
            )
        summary_df = pd.DataFrame(summaries)

    if verbose:
        print("\nQC Summary")
        print(summary_df)

    return summary_df


__all__ = [
    "compute_area_um2",
    "categorize_area",
    "categorize_nc_ratio",
    "filter_segmentation_errors",
    "run_segmentation_qc",
    "generate_qc_summary",
]
