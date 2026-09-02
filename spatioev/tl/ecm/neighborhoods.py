"""ECM-cell neighborhood features and clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._helpers import _ensure_list

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad
import re

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

# Section 4: ECM cell neighborhoods  (from archive/spatial/spatial_ecm_neighborhoods.py)
# ============================================================

"""
ECM-cell spatial neighborhood analysis.

This module implements a scimap-style radius-neighborhood workflow for
cell-ECM data. Around each cell, it summarizes nearby cell phenotypes and
nearby ECM fibers, then clusters those local profiles into spatial
neighborhoods.
"""




def _safe_label(value):
    """
    Convert category labels into stable column-name suffixes.
    """
    value = str(value)
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    return value.strip("_")


def _circle_area_mm2(radius, pixel_size_um):
    """
    Area of the circular neighborhood in mm2.
    """
    radius_um = radius * pixel_size_um
    return np.pi * (radius_um ** 2) / 1e6


def detect_tissue_regions_dbscan(
    adata: ad.AnnData,
    eps: float=200,
    min_samples: int=50,
    min_cells: int=1600,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    cluster_key: str="tissue_region",
    keep_key: list[str] | None="keep_tissue",
) -> pd.DataFrame:
    """
    Identify large tissue pieces from cell coordinates using DBSCAN.

    This is useful before radius-neighborhood analysis because isolated cells,
    dust-like segmentation artifacts, and tiny disconnected fragments can
    otherwise form misleading low-ECM neighborhoods.

    Parameters
    ----------
    adata : AnnData
        Cell-level object. ``adata.obs`` must contain image ids and spatial
        coordinates.
    eps : float
        DBSCAN neighborhood radius in the same coordinate units as
        ``x_key``/``y_key``. In pixel coordinates, this should be pixels.
    min_samples : int
        Minimum local cell count for DBSCAN core points.
    min_cells : int
        Minimum total number of cells required to retain a tissue component.
        Components smaller than this are marked as not kept.
    image_key : str
        Column in ``adata.obs`` identifying each tissue image/sample.
    x_key, y_key : str
        Coordinate columns in ``adata.obs``.
    cluster_key : str
        Output column for tissue-component labels.
    keep_key : str
        Output boolean column indicating retained tissue cells.

    Returns
    -------
    tissue_df : DataFrame
        One row per cell with DBSCAN labels, component size, and keep flag.
    summary_df : DataFrame
        One row per DBSCAN component with size and retained/removed status.
    """
    obs = adata.obs
    required = [image_key, x_key, y_key]
    missing = [col for col in required if col not in obs.columns]
    if missing:
        raise KeyError(f"adata.obs is missing required columns: {missing}")

    tissue_rows = []
    summary_rows = []

    for img, sub in obs.groupby(image_key, observed=True):
        sub = sub.dropna(subset=[x_key, y_key]).copy()
        if sub.empty:
            continue

        coords = sub[[x_key, y_key]].to_numpy(dtype=float)
        valid = np.isfinite(coords).all(axis=1)
        sub = sub.loc[valid].copy()
        coords = coords[valid]

        if len(sub) < min_samples:
            labels = np.full(len(sub), -1, dtype=int)
        else:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

        raw_col = f"{cluster_key}_raw"
        sub[raw_col] = labels
        sizes = sub[raw_col].value_counts()
        keep_labels = [
            int(label)
            for label, size in sizes.items()
            if label >= 0 and size >= min_cells
        ]
        keep_labels = set(keep_labels)

        sub[keep_key] = sub[raw_col].isin(keep_labels)
        sub[f"{cluster_key}_size"] = sub[raw_col].map(sizes).astype(int)
        sub[cluster_key] = pd.Series(pd.NA, index=sub.index, dtype="object")
        for label in keep_labels:
            mask = sub[raw_col].eq(label)
            sub.loc[mask, cluster_key] = f"{img}_tissue_{label}"

        for label, size in sizes.items():
            summary_rows.append({
                image_key: img,
                "dbscan_label": int(label),
                "n_cells": int(size),
                "kept": bool(label in keep_labels),
            })

        tissue_rows.append(
            sub[[image_key, x_key, y_key, raw_col, f"{cluster_key}_size", cluster_key, keep_key]]
            .assign(cell_id=sub.index)
        )

    if not tissue_rows:
        return pd.DataFrame(), pd.DataFrame(summary_rows)

    tissue_df = pd.concat(tissue_rows, axis=0)
    tissue_df = tissue_df.set_index("cell_id", drop=False)
    summary_df = pd.DataFrame(summary_rows)

    return tissue_df, summary_df


def filter_ecm_cell_inputs_by_tissue(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    tissue_df: pd.DataFrame,
    keep_key: list[str] | None="keep_tissue",
    cell_id_col: str="cell_id",
    fiber_id_col: str="fiber_id",
) -> ad.AnnData:
    """
    Filter cells, fibers, and links to retained tissue components.

    Fibers are retained when they are linked to at least one retained tissue
    cell. This keeps ECM features aligned with the tissue-cleaned cell anchors
    used for radius-neighborhood analysis.
    """
    if tissue_df.empty:
        raise ValueError("tissue_df is empty.")
    if keep_key not in tissue_df.columns:
        raise KeyError(f"{keep_key!r} is not present in tissue_df.")

    keep_cells = tissue_df.loc[tissue_df[keep_key], cell_id_col].astype(str)
    keep_cells = set(keep_cells)

    adata_clean = adata[adata.obs.index.astype(str).isin(keep_cells)].copy()
    links_clean = links_df[links_df[cell_id_col].astype(str).isin(keep_cells)].copy()

    keep_fibers = set(links_clean[fiber_id_col].astype(str))
    fiber_clean = fiber_df[fiber_df.index.astype(str).isin(keep_fibers)].copy()
    links_clean = links_clean[links_clean[fiber_id_col].astype(str).isin(fiber_clean.index.astype(str))].copy()

    return adata_clean, fiber_clean, links_clean


def build_ecm_cell_neighborhood_features(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    radius: float,
    phenotype_key: str="phenotype",
    image_key: str="imageid",
    group_key: str="pathology",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_types: list[str] | None=("COL6A1", "CHP"),
    cell_phenotypes: list[str] | None=None,
    include_self: bool=False,
    pixel_size_um: float=0.325,
    distance_weight_scale: float=None,
) -> pd.DataFrame:
    """
    Build radius-based ECM-cell neighborhood features around each cell.

    This mirrors the idea of scimap radius neighborhoods: every cell becomes
    an anchor point, and the local neighborhood is represented by nearby cell
    phenotype composition plus nearby ECM fiber abundance.

    Parameters
    ----------
    adata : AnnData
        Cell-level object. ``adata.obs`` must contain coordinates, image ids,
        and phenotype annotations.
    fiber_df : DataFrame
        Fiber table containing coordinates, image ids, and fiber types.
    links_df : DataFrame
        Precomputed cell-fiber links with ``cell_id``, ``fiber_id``,
        ``imageid``, and ``distance`` columns.
    radius : float
        Spatial radius in the same units as ``X_centroid``/``Y_centroid``.
        If coordinates are pixels, pass a pixel radius.
    phenotype_key : str
        Column in ``adata.obs`` containing cell phenotype labels.
    image_key : str
        Column shared by cells, fibers, and links identifying each image.
    group_key : str
        Optional biological group column, for example RA/OA pathology.
    x_key, y_key : str
        Coordinate columns used for cell neighborhoods.
    fiber_type_key : str
        Column in ``fiber_df`` containing fiber type labels.
    fiber_types : sequence of str
        Fiber types to include as ECM features, for example ``COL6A1`` and
        ``CHP`` for collagen VI dark-zone discovery.
    cell_phenotypes : sequence of str, optional
        Phenotypes to include. If ``None``, all observed phenotypes are used.
    include_self : bool
        If ``False``, subtract each anchor cell from its own phenotype count.
        This makes the features describe the surrounding neighborhood rather
        than the anchor label itself.
    pixel_size_um : float
        Pixel size used to convert radius-neighborhood counts into densities.
    distance_weight_scale : float, optional
        Distance decay scale for fiber weights. Defaults to ``radius / 2``.

    Returns
    -------
    DataFrame
        One row per cell with local cell-composition and ECM-fiber features.
    """
    fiber_types = _ensure_list(fiber_types)
    if fiber_types is None:
        fiber_types = []

    obs = adata.obs.copy()
    required_cell_cols = [image_key, x_key, y_key, phenotype_key]
    missing = [col for col in required_cell_cols if col not in obs.columns]
    if missing:
        raise KeyError(f"adata.obs is missing required columns: {missing}")

    required_fiber_cols = [image_key, fiber_type_key]
    missing = [col for col in required_fiber_cols if col not in fiber_df.columns]
    if missing:
        raise KeyError(f"fiber_df is missing required columns: {missing}")

    required_link_cols = [image_key, "cell_id", "fiber_id", "distance"]
    missing = [col for col in required_link_cols if col not in links_df.columns]
    if missing:
        raise KeyError(f"links_df is missing required columns: {missing}")

    if cell_phenotypes is None:
        cell_phenotypes = sorted(obs[phenotype_key].dropna().unique())
    else:
        cell_phenotypes = list(cell_phenotypes)

    distance_weight_scale = radius / 2 if distance_weight_scale is None else distance_weight_scale
    neighborhood_area = _circle_area_mm2(radius, pixel_size_um)

    rows = []
    images = np.intersect1d(obs[image_key].dropna().unique(), fiber_df[image_key].dropna().unique())

    fiber_meta = fiber_df[[fiber_type_key]].copy()

    for img in images:
        cells = obs.loc[obs[image_key].eq(img)].dropna(subset=[x_key, y_key]).copy()
        if cells.empty:
            continue

        coords = cells[[x_key, y_key]].to_numpy(dtype=float)
        valid = np.isfinite(coords).all(axis=1)
        cells = cells.loc[valid].copy()
        coords = coords[valid]
        if cells.empty:
            continue

        image_features = pd.DataFrame(index=cells.index)
        image_features["cell_id"] = cells.index
        image_features[image_key] = img
        if group_key in cells.columns:
            image_features[group_key] = cells[group_key].to_numpy()
        image_features[phenotype_key] = cells[phenotype_key].to_numpy()
        image_features[x_key] = cells[x_key].to_numpy(dtype=float)
        image_features[y_key] = cells[y_key].to_numpy(dtype=float)

        for phenotype in cell_phenotypes:
            suffix = _safe_label(phenotype)
            phenotype_mask = cells[phenotype_key].eq(phenotype).to_numpy()
            phenotype_coords = coords[phenotype_mask]

            if len(phenotype_coords) == 0:
                counts = np.zeros(len(cells), dtype=float)
            else:
                tree = KDTree(phenotype_coords)
                counts = tree.query_radius(coords, r=radius, count_only=True).astype(float)
                if not include_self:
                    counts = counts - phenotype_mask.astype(float)
                    counts[counts < 0] = 0

            image_features[f"cell_count_{suffix}"] = counts
            image_features[f"cell_density_{suffix}"] = counts / neighborhood_area

        cell_count_cols = [f"cell_count_{_safe_label(p)}" for p in cell_phenotypes]
        total_cells = image_features[cell_count_cols].sum(axis=1).to_numpy(dtype=float)
        image_features["neighbor_cell_count"] = total_cells
        image_features["neighbor_cell_density"] = total_cells / neighborhood_area

        denom = np.where(total_cells > 0, total_cells, 1.0)
        for phenotype in cell_phenotypes:
            suffix = _safe_label(phenotype)
            image_features[f"cell_fraction_{suffix}"] = (
                image_features[f"cell_count_{suffix}"].to_numpy(dtype=float) / denom
            )

        image_links = links_df.loc[
            links_df[image_key].eq(img) & links_df["distance"].le(radius),
            [image_key, "cell_id", "fiber_id", "distance"],
        ].copy()

        if not image_links.empty and fiber_types:
            image_links = image_links.merge(
                fiber_meta,
                left_on="fiber_id",
                right_index=True,
                how="left",
            )
            image_links = image_links[image_links[fiber_type_key].isin(fiber_types)].copy()

        fiber_count_total = np.zeros(len(image_features), dtype=float)
        if image_links.empty or not fiber_types:
            for fiber_type in fiber_types:
                suffix = _safe_label(fiber_type)
                image_features[f"fiber_count_{suffix}"] = 0.0
                image_features[f"fiber_density_{suffix}"] = 0.0
                image_features[f"fiber_weight_{suffix}"] = 0.0
                image_features[f"nearest_distance_{suffix}"] = np.nan
        else:
            image_links["weight"] = np.exp(-image_links["distance"] / distance_weight_scale)
            count_table = (
                image_links
                .groupby(["cell_id", fiber_type_key], observed=True)
                .size()
                .unstack(fill_value=0)
            )
            weight_table = (
                image_links
                .groupby(["cell_id", fiber_type_key], observed=True)["weight"]
                .sum()
                .unstack(fill_value=0)
            )
            nearest_table = (
                image_links
                .groupby(["cell_id", fiber_type_key], observed=True)["distance"]
                .min()
                .unstack()
            )

            for fiber_type in fiber_types:
                suffix = _safe_label(fiber_type)
                counts = count_table[fiber_type] if fiber_type in count_table else pd.Series(dtype=float)
                weights = weight_table[fiber_type] if fiber_type in weight_table else pd.Series(dtype=float)
                nearest = nearest_table[fiber_type] if fiber_type in nearest_table else pd.Series(dtype=float)

                image_features[f"fiber_count_{suffix}"] = image_features.index.map(counts).fillna(0).to_numpy(dtype=float)
                image_features[f"fiber_density_{suffix}"] = image_features[f"fiber_count_{suffix}"] / neighborhood_area
                image_features[f"fiber_weight_{suffix}"] = image_features.index.map(weights).fillna(0).to_numpy(dtype=float)
                image_features[f"nearest_distance_{suffix}"] = image_features.index.map(nearest).to_numpy(dtype=float)
                fiber_count_total += image_features[f"fiber_count_{suffix}"].to_numpy(dtype=float)

        image_features["neighbor_fiber_count"] = fiber_count_total
        fiber_denom = np.where(fiber_count_total > 0, fiber_count_total, 1.0)
        for fiber_type in fiber_types:
            suffix = _safe_label(fiber_type)
            image_features[f"fiber_fraction_{suffix}"] = image_features[f"fiber_count_{suffix}"] / fiber_denom

        if {"COL6A1", "CHP"}.issubset(set(map(str, fiber_types))):
            col6 = image_features["fiber_count_COL6A1"].to_numpy(dtype=float)
            chp = image_features["fiber_count_CHP"].to_numpy(dtype=float)
            image_features["COL6A1_CHP_count_ratio"] = (col6 + 1.0) / (chp + 1.0)

        rows.append(image_features)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, axis=0)


def default_ecm_cell_neighborhood_feature_columns(
    neighborhood_df: pd.DataFrame,
    include_cell_fractions: bool=True,
    include_fiber_features: bool=True,
    include_cell_density: bool=False,
) -> object:
    """
    Select sensible default features for K-means neighborhood clustering.
    """
    cols = []
    if include_cell_fractions:
        cols.extend([c for c in neighborhood_df.columns if c.startswith("cell_fraction_")])
    if include_fiber_features:
        cols.extend([c for c in neighborhood_df.columns if c.startswith("fiber_count_")])
        cols.extend([c for c in neighborhood_df.columns if c.startswith("fiber_density_")])
        cols.extend([c for c in neighborhood_df.columns if c.startswith("fiber_weight_")])
        cols.extend([c for c in neighborhood_df.columns if c.startswith("nearest_distance_")])
    if include_cell_density and "neighbor_cell_density" in neighborhood_df.columns:
        cols.append("neighbor_cell_density")
    return cols


def cluster_ecm_cell_neighborhoods(
    neighborhood_df: pd.DataFrame,
    feature_columns: list[str]=None,
    feature_weights: dict[str, float] | None=None,
    n_clusters: int=10,
    label_key: str="ecm_cell_neighborhood",
    random_state: int=0,
    scale: bool=True,
) -> ad.AnnData:
    """
    Cluster ECM-cell neighborhood features with K-means.

    Parameters
    ----------
    neighborhood_df : DataFrame
        Output of :func:`build_ecm_cell_neighborhood_features`.
    feature_columns : sequence of str, optional
        Columns used for clustering. If ``None``, a default set of cell
        fractions, fiber densities, fiber fractions, and total cell density is
        used.
    feature_weights : dict, optional
        Optional per-feature weights applied after scaling. This is useful when
        one feature block has many more columns than another, for example all
        cell-type fractions versus a smaller COL6A1/CHP ECM block.
    n_clusters : int
        Number of K-means neighborhoods.
    label_key : str
        Name of the output label column.
    random_state : int
        Random seed for reproducibility.
    scale : bool
        If ``True``, standardize features before K-means.

    Returns
    -------
    clustered_df : DataFrame
        Input table with added K-means labels.
    model : KMeans
        Fitted K-means model.
    scaler : StandardScaler or None
        Fitted scaler when ``scale=True``.
    feature_columns : list
        Columns used for clustering.
    """
    if neighborhood_df.empty:
        raise ValueError("neighborhood_df is empty.")

    if feature_columns is None:
        feature_columns = default_ecm_cell_neighborhood_feature_columns(neighborhood_df)
    feature_columns = list(feature_columns)

    missing = [col for col in feature_columns if col not in neighborhood_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    X = neighborhood_df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X.to_numpy(dtype=float)

    if feature_weights is not None:
        weights = np.array(
            [feature_weights.get(col, 1.0) for col in feature_columns],
            dtype=float,
        )
        X_fit = X_fit * weights

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )
    labels = model.fit_predict(X_fit)

    out = neighborhood_df.copy()
    out[label_key] = labels
    out[f"{label_key}_label"] = pd.Categorical([f"N{label}" for label in labels])
    return out, model, scaler, feature_columns


def summarize_ecm_cell_neighborhoods(
    neighborhood_df: pd.DataFrame,
    label_key: str="ecm_cell_neighborhood",
    phenotype_key: str="phenotype",
    group_key: str="pathology",
) -> pd.DataFrame:
    """
    Summarize K-means neighborhoods by size, phenotype composition, and features.
    """
    if label_key not in neighborhood_df.columns:
        raise KeyError(f"{label_key!r} is not present in neighborhood_df.")

    feature_cols = [
        c for c in neighborhood_df.columns
        if c.startswith((
            "cell_fraction_",
            "fiber_count_",
            "fiber_density_",
            "fiber_weight_",
            "fiber_fraction_",
            "nearest_distance_",
        ))
        or c in {"neighbor_cell_density", "neighbor_cell_count", "neighbor_fiber_count", "COL6A1_CHP_count_ratio"}
    ]

    summary = (
        neighborhood_df
        .groupby(label_key, observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            **{f"median_{col}": (col, "median") for col in feature_cols}
        )
        .reset_index()
    )

    if group_key in neighborhood_df.columns:
        group_counts = (
            neighborhood_df
            .groupby([group_key, label_key], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        totals = group_counts.groupby(group_key, observed=True)["n_cells"].transform("sum")
        group_counts["fraction"] = group_counts["n_cells"] / totals
    else:
        group_counts = pd.DataFrame()

    if phenotype_key in neighborhood_df.columns:
        phenotype_counts = (
            neighborhood_df
            .groupby([label_key, phenotype_key], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        totals = phenotype_counts.groupby(label_key, observed=True)["n_cells"].transform("sum")
        phenotype_counts["fraction"] = phenotype_counts["n_cells"] / totals
    else:
        phenotype_counts = pd.DataFrame()

    return summary, group_counts, phenotype_counts


def score_col6_dark_neighborhoods(
    neighborhood_df: pd.DataFrame,
    label_key: str="ecm_cell_neighborhood",
    col6_density_col: str="fiber_density_COL6A1",
    chp_density_col: str="fiber_density_CHP",
    cell_density_col: str="neighbor_cell_density",
    low_quantile: float=0.35,
    min_cell_density_quantile: float=0.10,
    output_key: str="COL6_dark_neighborhood",
) -> pd.DataFrame:
    """
    Mark K-means neighborhoods that look COL6A1/CHP-poor inside tissue.

    The rule is intentionally transparent:
    - low median local COL6A1 fiber density;
    - low median local CHP fiber density;
    - enough local cells to avoid selecting sparse/tissue-edge artifacts.

    Returns a copy of ``neighborhood_df`` plus a cluster-level score table.
    """
    required = [label_key, col6_density_col, chp_density_col, cell_density_col]
    missing = [col for col in required if col not in neighborhood_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    cluster_scores = (
        neighborhood_df
        .groupby(label_key, observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            median_col6_density=(col6_density_col, "median"),
            median_chp_density=(chp_density_col, "median"),
            median_cell_density=(cell_density_col, "median"),
        )
        .reset_index()
    )

    col6_cutoff = cluster_scores["median_col6_density"].quantile(low_quantile)
    chp_cutoff = cluster_scores["median_chp_density"].quantile(low_quantile)
    cell_density_cutoff = cluster_scores["median_cell_density"].quantile(min_cell_density_quantile)

    cluster_scores["low_col6"] = cluster_scores["median_col6_density"] <= col6_cutoff
    cluster_scores["low_chp"] = cluster_scores["median_chp_density"] <= chp_cutoff
    cluster_scores["inside_tissue_like"] = cluster_scores["median_cell_density"] >= cell_density_cutoff
    cluster_scores["col6_dark_candidate"] = (
        cluster_scores["low_col6"]
        & cluster_scores["low_chp"]
        & cluster_scores["inside_tissue_like"]
    )

    out = neighborhood_df.copy()
    dark_clusters = set(
        cluster_scores.loc[
            cluster_scores["col6_dark_candidate"],
            label_key,
        ]
    )
    out[output_key] = np.where(out[label_key].isin(dark_clusters), "COL6_dark_candidate", "other")
    out[output_key] = pd.Categorical(out[output_key], categories=["COL6_dark_candidate", "other"])

    return out, cluster_scores


def add_neighborhoods_to_obs(
    adata: ad.AnnData,
    neighborhood_df: pd.DataFrame,
    columns: list[str],
) -> ad.AnnData:
    """
    Copy neighborhood labels/features back into ``adata.obs`` by cell id.

    Parameters
    ----------
    adata : AnnData
        Cell-level object to annotate.
    neighborhood_df : DataFrame
        Table indexed by cell id or containing a ``cell_id`` column.
    columns : sequence of str
        Columns to copy into ``adata.obs``.

    Returns
    -------
    AnnData
        A copy of ``adata`` with added observation columns.
    """
    out = adata.copy()
    columns = list(columns)
    missing = [col for col in columns if col not in neighborhood_df.columns]
    if missing:
        raise KeyError(f"neighborhood_df is missing columns: {missing}")

    if "cell_id" in neighborhood_df.columns:
        mapped = neighborhood_df.set_index("cell_id")[columns]
    else:
        mapped = neighborhood_df[columns]

    for col in columns:
        out.obs[col] = out.obs.index.map(mapped[col])

    return out
