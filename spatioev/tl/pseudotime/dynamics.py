"""Spatial interaction dynamics along pseudotime.

Runs after a pseudotime workflow has assigned a niche- or cell-level
pseudotime back onto ``adata.obs``. Built on the source-centred interaction
functions in :mod:`spatioev.tl.stats` rather than a second framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..stats import cross_ripley_local_counts

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad


def _resolve_radius(radius=None, radius_um=None, pixel_size_um=None):
    """
    Resolve a neighborhood radius in the coordinate units stored in ``adata.obs``.

    Exactly one of ``radius`` or ``radius_um`` must be provided. When
    ``radius_um`` is used, ``pixel_size_um`` must also be supplied so the value
    can be converted into pixel units.
    """
    if (radius is None) == (radius_um is None):
        raise ValueError("Specify exactly one of radius or radius_um.")

    if radius is not None:
        return float(radius)

    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required when radius_um is provided.")
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive.")

    return float(radius_um) / float(pixel_size_um)


def assign_pseudotime_bins(
    values: np.ndarray,
    n_bins: int=8,
    method: str="quantile",
) -> ad.AnnData:
    """
    Assign integer pseudotime bins and return per-bin pseudotime summaries.

    Parameters
    ----------
    values : array-like or Series
        Continuous pseudotime values.
    n_bins : int, default 8
        Target number of bins. The realized number may be smaller when there
        are too few unique pseudotime values.
    method : {"quantile", "equal_width"}, default "quantile"
        Binning strategy.

    Returns
    -------
    tuple
        ``(bin_codes, bin_summary_df)``, where ``bin_codes`` is a nullable
        integer Series aligned to ``values`` and ``bin_summary_df`` contains
        one row per realized bin with min / max / median pseudotime.
    """
    series = pd.Series(values).copy()
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()

    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    if valid.empty:
        return out, pd.DataFrame(
            columns=[
                "pseudotime_bin",
                "n_cells",
                "pseudotime_min",
                "pseudotime_max",
                "pseudotime_median",
            ]
        )

    n_unique = int(valid.nunique())
    n_bins_eff = max(1, min(int(n_bins), n_unique))

    if method == "quantile":
        bins = pd.qcut(valid, q=n_bins_eff, labels=False, duplicates="drop")
    elif method == "equal_width":
        bins = pd.cut(valid, bins=n_bins_eff, labels=False, include_lowest=True)
    else:
        raise ValueError("method must be 'quantile' or 'equal_width'.")

    if bins is None:
        return out, pd.DataFrame(
            columns=[
                "pseudotime_bin",
                "n_cells",
                "pseudotime_min",
                "pseudotime_max",
                "pseudotime_median",
            ]
        )

    bins = bins.astype("Int64")
    out.loc[valid.index] = bins

    summary = (
        pd.DataFrame(
            {
                "pseudotime": valid,
                "pseudotime_bin": bins.to_numpy(),
            }
        )
        .groupby("pseudotime_bin", dropna=True)
        .agg(
            n_cells=("pseudotime", "size"),
            pseudotime_min=("pseudotime", "min"),
            pseudotime_max=("pseudotime", "max"),
            pseudotime_median=("pseudotime", "median"),
        )
        .reset_index()
        .sort_values("pseudotime_bin")
        .reset_index(drop=True)
    )

    return out, summary


def compute_epithelial_centered_interaction_dynamics(
    adata: ad.AnnData,
    pseudotime_key: str,
    phenotype_key: str,
    target_phenotypes: list[str] | None,
    source_phenotype: str="pancreatic ductal epithelium",
    radius: float=None,
    radius_um: float=None,
    pixel_size_um: float=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    pseudotime_bin_count: int=8,
    pseudotime_bin_method: str="quantile",
    min_source_cells: int=1,
    min_target_cells: int=1,
) -> pd.DataFrame:
    """
    Compute epithelial-centered local interaction metrics along pseudotime.

    Workflow
    --------
    1. Restrict to source cells with a valid epithelial pseudotime value.
    2. For each requested target phenotype, call
       ``cross_ripley_local_counts(...)`` to quantify the local target-cell
       neighborhood of each source cell.
    3. Merge source-cell pseudotime values and pseudotime bins onto the
       interaction table.

    Parameters
    ----------
    adata : AnnData
        Annotated dataset containing phenotype labels and pseudotime values in
        ``adata.obs``.
    pseudotime_key : str
        Column in ``adata.obs`` containing epithelial-centered pseudotime.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels such as ``Tier_A``
        or ``Tier_B``.
    target_phenotypes : iterable of str
        Target phenotypes to quantify around the epithelial source cells.
    source_phenotype : str, default "pancreatic ductal epithelium"
        Source phenotype used as the pseudotime anchor.
    radius, radius_um : float, optional
        Neighborhood radius in coordinate units or microns. Exactly one must be
        provided.
    pixel_size_um : float, optional
        Pixel size in microns. Required when ``radius_um`` is used.

    Returns
    -------
    DataFrame
        One row per source cell per target phenotype with:
        - local cross-Ripley neighborhood metrics
        - epithelial pseudotime
        - pseudotime bin
    """
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"{pseudotime_key!r} not found in adata.obs.")
    if phenotype_key not in adata.obs.columns:
        raise ValueError(f"{phenotype_key!r} not found in adata.obs.")

    radius_value = _resolve_radius(
        radius=radius,
        radius_um=radius_um,
        pixel_size_um=pixel_size_um,
    )

    source_mask = adata.obs[phenotype_key] == source_phenotype
    source_df = adata.obs.loc[source_mask, [pseudotime_key]].copy()
    source_df[pseudotime_key] = pd.to_numeric(source_df[pseudotime_key], errors="coerce")
    source_df = source_df[source_df[pseudotime_key].notna()].copy()

    if source_df.empty:
        return pd.DataFrame(
            columns=[
                "cell_id",
                image_key,
                "source",
                "target",
                "radius",
                "target_neighbor_count",
                "expected_target_neighbor_count",
                "target_neighbor_excess",
                "target_neighbor_ratio",
                "is_cross_ripley_hotspot",
                pseudotime_key,
                "pseudotime_bin",
                "has_target_neighbor",
            ]
        )

    source_df = source_df.reset_index().rename(columns={"index": "cell_id"})
    source_df["pseudotime_bin"], bin_summary = assign_pseudotime_bins(
        source_df[pseudotime_key],
        n_bins=pseudotime_bin_count,
        method=pseudotime_bin_method,
    )

    all_rows = []
    for target in target_phenotypes:
        local_df = cross_ripley_local_counts(
            adata=adata,
            phenotype_key=phenotype_key,
            source_phenotype=source_phenotype,
            target_phenotype=target,
            radius=radius_value,
            x_key=x_key,
            y_key=y_key,
            image_key=image_key,
            min_source_cells=min_source_cells,
            min_target_cells=min_target_cells,
            add_to_obs=False,
        )

        if local_df.empty:
            continue

        merged = local_df.merge(
            source_df[["cell_id", pseudotime_key, "pseudotime_bin"]],
            on="cell_id",
            how="inner",
        )
        if merged.empty:
            continue

        merged["has_target_neighbor"] = merged["target_neighbor_count"] > 0
        merged["target_phenotype"] = target
        all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "cell_id",
                image_key,
                "source",
                "target",
                "radius",
                "target_neighbor_count",
                "expected_target_neighbor_count",
                "target_neighbor_excess",
                "target_neighbor_ratio",
                "is_cross_ripley_hotspot",
                pseudotime_key,
                "pseudotime_bin",
                "has_target_neighbor",
                "target_phenotype",
            ]
        )

    out = pd.concat(all_rows, ignore_index=True)
    out.attrs["pseudotime_bin_summary"] = bin_summary
    out.attrs["radius_value"] = radius_value
    out.attrs["source_phenotype"] = source_phenotype
    out.attrs["phenotype_key"] = phenotype_key
    out.attrs["pseudotime_key"] = pseudotime_key
    return out


def summarize_epithelial_interaction_dynamics(
    interaction_df: pd.DataFrame,
    pseudotime_key: str="elpigraph_pseudotime_pathology",
    target_col: str="target_phenotype",
    pseudotime_bin_key: str="pseudotime_bin",
) -> pd.DataFrame:
    """
    Aggregate epithelial-centered local interaction metrics by pseudotime bin.

    Parameters
    ----------
    interaction_df : DataFrame
        Output from ``compute_epithelial_centered_interaction_dynamics(...)``.

    Returns
    -------
    DataFrame
        Tidy bin-level summary table suitable for line plots or heatmaps.
    """
    if interaction_df.empty:
        return pd.DataFrame(
            columns=[
                target_col,
                pseudotime_bin_key,
                "n_source_cells",
                "pseudotime_min",
                "pseudotime_max",
                "pseudotime_median",
                "mean_target_neighbor_count",
                "mean_target_neighbor_excess",
                "mean_target_neighbor_ratio",
                "fraction_source_with_target_neighbor",
                "hotspot_fraction",
            ]
        )

    required = [
        target_col,
        pseudotime_bin_key,
        pseudotime_key,
        "target_neighbor_count",
        "target_neighbor_excess",
        "target_neighbor_ratio",
        "has_target_neighbor",
        "is_cross_ripley_hotspot",
    ]
    missing = [col for col in required if col not in interaction_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in interaction_df: {missing}")

    summary = (
        interaction_df.dropna(subset=[pseudotime_bin_key, pseudotime_key])
        .groupby([target_col, pseudotime_bin_key], dropna=True)
        .agg(
            n_source_cells=("cell_id", "size"),
            pseudotime_min=(pseudotime_key, "min"),
            pseudotime_max=(pseudotime_key, "max"),
            pseudotime_median=(pseudotime_key, "median"),
            mean_target_neighbor_count=("target_neighbor_count", "mean"),
            mean_target_neighbor_excess=("target_neighbor_excess", "mean"),
            mean_target_neighbor_ratio=("target_neighbor_ratio", "mean"),
            fraction_source_with_target_neighbor=("has_target_neighbor", "mean"),
            hotspot_fraction=("is_cross_ripley_hotspot", "mean"),
        )
        .reset_index()
        .sort_values([target_col, pseudotime_bin_key])
        .reset_index(drop=True)
    )
    return summary


__all__ = [
    "assign_pseudotime_bins",
    "compute_epithelial_centered_interaction_dynamics",
    "summarize_epithelial_interaction_dynamics",
]
