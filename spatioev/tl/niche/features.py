"""Niche feature tables and pathology module scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad


import networkx as nx
from scipy.sparse import triu
from tqdm.auto import tqdm

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:  # pragma: no cover - depends on sklearn version
    SklearnHDBSCAN = None


from ._metrics import (
    _get_n_hop_external_layers,
    _safe_scalar,
    _sanitize_label,
    _summarize_boundary_core,
    _summarize_feature_organization,
    _summarize_geometry,
    _summarize_graph_surroundings,
    _summarize_niche_state,
    _summarize_skeleton_topology,
    _summarize_topology,
)


def summarize_niche_graph_features(
    adata: ad.AnnData,
    niche_key: str,
    feature_cols: list[str]=None,
    state_feature_cols: list[str]=None,
    include_values: list[str] | None=None,
    phenotype_key: str=None,
    region_key: str=None,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    boundary_labels: list[str] | None=("inner_border",),
    core_labels: list[str] | None=("core",),
    exclude_labels: list[str] | None=("unassigned", "noise"),
    morphology_bin_count: int=3,
    min_cells: int=3,
    include_graph_surroundings: bool=True,
    surround_hops: int=1,
    include_state_summaries: bool=True,
    state_summary_stats: list[str]=("mean", "median", "std", "iqr", "p10", "p90"),
    lightweight: bool=False,
    show_progress: bool=False,
    progress_desc: str="Niche features",
) -> pd.DataFrame:
    """
    Summarize each niche as a graph-derived descriptor vector.

    Parameters
    ----------
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    feature_cols : list, optional
        Continuous node features for feature-organization descriptors.
    state_feature_cols : list, optional
        Continuous cell features to summarize as niche-level state descriptors.
        If ``None``, defaults to ``feature_cols``.
    include_values : iterable, optional
        If provided, only summarize these niche labels.
    phenotype_key : str, optional
        Categorical phenotype label for assortativity and entropy summaries.
    region_key : str, optional
        Optional region label column such as ``core`` / ``inner_border``.
    adjacency_key : str
        Key in ``adata.obsp`` containing the Layer 1 cell graph adjacency.
    distance_key : str
        Key in ``adata.obsp`` containing edge distances.
    boundary_labels, core_labels : tuple
        Region labels used for boundary/core contrasts.
    exclude_labels : tuple
        Niche labels to skip.
    min_cells : int
        Minimum number of cells required to summarize one niche.
    include_graph_surroundings : bool
        If ``True``, add fast graph-defined boundary/core/surround features that
        do not require precomputed geometric region labels.
    surround_hops : int
        Number of external graph hops used to define niche surroundings.
    include_state_summaries : bool
        If ``True``, append niche-level summaries of the original cell features.
    state_summary_stats : tuple
        Summary statistics used for the niche state block.
    lightweight : bool
        If ``True``, skip the most expensive topology metrics such as bridge
        fraction, diameter, and average shortest path length.
    show_progress : bool
        If ``True``, display a progress bar over niche groups.
    progress_desc : str
        Description shown in the progress bar.

    Returns
    -------
    DataFrame
        One row per niche with graph-derived descriptors.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
        raise ValueError(f"{x_key} and/or {y_key} not found in adata.obs")

    feature_cols = list(feature_cols) if feature_cols is not None else None
    if state_feature_cols is None:
        state_feature_cols = feature_cols
    else:
        state_feature_cols = list(state_feature_cols)
    adjacency = adata.obsp[adjacency_key].tocsr()
    distance_matrix = adata.obsp[distance_key] if distance_key in adata.obsp else None
    include_set = set(include_values) if include_values is not None else None
    numeric_cache = {}
    if feature_cols is not None:
        for feature in feature_cols:
            if feature in adata.obs.columns:
                numeric_cache[feature] = pd.to_numeric(
                    adata.obs[feature],
                    errors="coerce",
                ).to_numpy(dtype=float)
    phenotype_array = None
    if phenotype_key is not None and phenotype_key in adata.obs.columns:
        phenotype_array = adata.obs[phenotype_key].astype("object").to_numpy()

    rows = []
    group_cols = [image_key, niche_key] if image_key in adata.obs.columns else [niche_key]
    grouped = adata.obs.groupby(group_cols, dropna=True, observed=False)
    grouped_iter = grouped
    if show_progress:
        grouped_iter = tqdm(grouped, total=grouped.ngroups, desc=progress_desc)

    for group_key, obs_sub in grouped_iter:
        if len(group_cols) == 2:
            image_value, niche_value = group_key
        else:
            image_value = np.nan
            niche_value = group_key

        if include_set is not None and niche_value not in include_set:
            continue

        if pd.isna(niche_value) or niche_value in exclude_labels:
            continue

        idx = adata.obs_names.get_indexer(obs_sub.index)
        idx = idx[idx >= 0]

        if len(idx) < min_cells:
            continue

        obs_sub = adata.obs.iloc[idx].copy()
        coords = obs_sub[[x_key, y_key]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid_coords = np.isfinite(coords).all(axis=1)
        if valid_coords.sum() < min_cells:
            continue

        idx = idx[valid_coords]
        obs_sub = obs_sub.iloc[np.flatnonzero(valid_coords)].copy()
        coords = coords[valid_coords]

        adjacency_sub = adjacency[idx][:, idx].astype(float).tocsr()
        adjacency_binary = adjacency_sub.copy()
        adjacency_binary.data = np.ones_like(adjacency_binary.data, dtype=float)

        if adjacency_binary.nnz == 0:
            edge_rows = np.empty(0, dtype=int)
            edge_cols = np.empty(0, dtype=int)
            edge_lengths = np.empty(0, dtype=float)
        else:
            upper = triu(adjacency_binary, k=1).tocoo()
            edge_rows = upper.row.astype(int)
            edge_cols = upper.col.astype(int)

            if distance_matrix is not None:
                dist_sub = distance_matrix[idx][:, idx].tocsr()
                edge_lengths = np.asarray(dist_sub[edge_rows, edge_cols]).ravel().astype(float)
            else:
                edge_lengths = np.linalg.norm(coords[edge_cols] - coords[edge_rows], axis=1)

        G = nx.from_scipy_sparse_array(adjacency_binary)

        row = {
            niche_key: niche_value,
            "image_id": image_value,
            "n_cells": float(len(obs_sub)),
        }

        row.update(_summarize_topology(G, include_path_metrics=not lightweight))
        row.update(_summarize_skeleton_topology(coords))
        row.update(_summarize_geometry(coords, edge_rows, edge_cols, edge_lengths))
        row.update(
            _summarize_feature_organization(
                obs_sub=obs_sub,
                adjacency_binary=adjacency_binary,
                edge_rows=edge_rows,
                edge_cols=edge_cols,
                feature_cols=feature_cols,
                phenotype_key=phenotype_key,
                morphology_bin_count=morphology_bin_count,
            )
        )
        row.update(
            _summarize_boundary_core(
                obs_sub=obs_sub,
                adjacency_binary=adjacency_binary,
                feature_cols=feature_cols,
                phenotype_key=phenotype_key,
                region_key=region_key,
                boundary_labels=boundary_labels,
                core_labels=core_labels,
            )
        )
        if include_graph_surroundings:
            row.update(
                _summarize_graph_surroundings(
                    adata=adata,
                    niche_idx=idx,
                    adjacency_full=adjacency,
                    niche_key=niche_key,
                    niche_value=niche_value,
                    feature_cols=feature_cols,
                    phenotype_key=phenotype_key,
                    surround_hops=surround_hops,
                    numeric_cache=numeric_cache,
                    phenotype_array=phenotype_array,
                )
            )
        if include_state_summaries:
            row.update(
                _summarize_niche_state(
                    obs_sub=obs_sub,
                    state_feature_cols=state_feature_cols,
                    phenotype_key=phenotype_key,
                    state_summary_stats=state_summary_stats,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_niche_feature_table(
    adata: ad.AnnData,
    niche_key: str,
    feature_cols: list[str]=None,
    state_feature_cols: list[str]=None,
    phenotype_key: str=None,
    region_key: str=None,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    adjacency_key: str="cell_graph_connectivities",
    distance_key: str="cell_graph_distances",
    include_values: list[str] | None=None,
    include_prefix: str | None=None,
    exclude_values: list[str] | None=("unassigned", "noise"),
    exclude_prefixes: list[str] | None=None,
    boundary_labels: list[str] | None=("inner_border",),
    core_labels: list[str] | None=("core",),
    morphology_bin_count: int=3,
    min_cells: int=3,
    include_graph_surroundings: bool=True,
    surround_hops: int=1,
    include_state_summaries: bool=True,
    state_summary_stats: list[str]=("mean", "median", "std", "iqr", "p10", "p90"),
    lightweight: bool=False,
    show_progress: bool=False,
    progress_desc: str="Niche features",
) -> pd.DataFrame:
    """
    Build a filtered per-niche graph-descriptor table in one call.

    This is a notebook-friendly wrapper around
    ``summarize_niche_graph_features`` that first filters niche labels from
    ``adata.obs[niche_key]`` and then computes graph-derived descriptors only
    for the retained niches.

    Parameters
    ----------
    include_values : iterable, optional
        Explicit niche labels to include.
    include_prefix : str, optional
        Keep only niche labels whose string form starts with this prefix.
    exclude_values : iterable, optional
        Exact niche labels to exclude.
    exclude_prefixes : iterable, optional
        Exclude niche labels whose string form starts with any of these prefixes.

    Returns
    -------
    DataFrame
        One row per retained niche with graph-derived descriptors.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    niche_series = pd.Series(adata.obs[niche_key]).dropna()
    if niche_series.empty:
        return pd.DataFrame()

    niche_values = niche_series.unique().tolist()

    if include_values is not None:
        include_set = set(include_values)
        niche_values = [value for value in niche_values if value in include_set]

    if include_prefix is not None:
        niche_values = [
            value for value in niche_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        niche_values = [value for value in niche_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        niche_values = [
            value for value in niche_values
            if not str(value).startswith(exclude_prefixes)
        ]

    if len(niche_values) == 0:
        return pd.DataFrame()

    summary_df = summarize_niche_graph_features(
        adata=adata,
        niche_key=niche_key,
        feature_cols=feature_cols,
        state_feature_cols=state_feature_cols,
        include_values=niche_values,
        phenotype_key=phenotype_key,
        region_key=region_key,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        adjacency_key=adjacency_key,
        distance_key=distance_key,
        boundary_labels=boundary_labels,
        core_labels=core_labels,
        exclude_labels=(),
        morphology_bin_count=morphology_bin_count,
        min_cells=min_cells,
        include_graph_surroundings=include_graph_surroundings,
        surround_hops=surround_hops,
        include_state_summaries=include_state_summaries,
        state_summary_stats=state_summary_stats,
        lightweight=lightweight,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    if summary_df.empty:
        return summary_df

    return summary_df[summary_df[niche_key].isin(niche_values)].reset_index(drop=True)


def build_niche_feature_table_batched(
    adata: ad.AnnData,
    niche_key: str,
    batch_size: int=100,
    include_values: list[str] | None=None,
    show_progress: bool=False,
    progress_desc: str="Niche batches",
    **kwargs,
) -> pd.DataFrame:
    """
    Build the niche feature table in batches of niche labels.

    This is a safer option for large whole-slide datasets than trying to process
    every niche in one notebook call.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if include_values is None:
        niche_values = pd.Series(adata.obs[niche_key]).dropna().unique().tolist()
    else:
        niche_values = list(include_values)

    include_prefix = kwargs.get("include_prefix", None)
    exclude_values = kwargs.get("exclude_values", ("unassigned", "noise"))
    exclude_prefixes = kwargs.get("exclude_prefixes", None)

    if include_prefix is not None:
        niche_values = [
            value for value in niche_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        niche_values = [value for value in niche_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        niche_values = [
            value for value in niche_values
            if not str(value).startswith(exclude_prefixes)
        ]

    if len(niche_values) == 0:
        return pd.DataFrame()

    inner_show_progress = kwargs.pop("show_progress_inner", False)
    inner_progress_desc = kwargs.pop("progress_desc_inner", "Niches in batch")
    batch_frames = []
    batch_starts = range(0, len(niche_values), int(batch_size))
    if show_progress:
        total_batches = int(np.ceil(len(niche_values) / int(batch_size)))
        batch_starts = tqdm(batch_starts, total=total_batches, desc=progress_desc)

    for start in batch_starts:
        batch_values = niche_values[start:start + int(batch_size)]
        batch_df = build_niche_feature_table(
            adata=adata,
            niche_key=niche_key,
            include_values=batch_values,
            show_progress=inner_show_progress,
            progress_desc=inner_progress_desc,
            **kwargs,
        )
        if not batch_df.empty:
            batch_frames.append(batch_df)

    if len(batch_frames) == 0:
        return pd.DataFrame()

    return pd.concat(batch_frames, ignore_index=True)


def summarize_niche_surrounding_context(
    adata: ad.AnnData,
    niche_key: str,
    phenotype_key: str,
    feature_cols: list[str]=None,
    phenotype_feature_map: dict[str, list[str]]=None,
    include_values: list[str] | None=None,
    image_key: str="imageid",
    adjacency_key: str="cell_graph_connectivities",
    surround_hops: int=1,
    phenotype_labels: list[str] | None=None,
    min_cells: int=3,
    summary_stats: list[str]=("mean", "median"),
    show_progress: bool=False,
    progress_desc: str="Niche surroundings",
) -> pd.DataFrame:
    """
    Summarize the graph-defined surroundings of each niche.

    This is a notebook-friendly wrapper around the same n-hop surround logic
    used by ``summarize_niche_graph_features``, but exposes phenotype
    composition and phenotype-restricted feature summaries in a reusable table.

    Parameters
    ----------
    adata : AnnData
        Annotated table containing cell graph and metadata.
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels for surrounding-cell
        composition summaries.
    feature_cols : list, optional
        Numeric features in ``adata.obs`` to summarize across all surrounding
        cells.
    phenotype_feature_map : dict, optional
        Mapping ``{phenotype_label: [feature_cols...]}`` describing phenotype-
        restricted feature summaries to compute in the surround.
    include_values : iterable, optional
        If provided, restrict output to these niche labels.
    image_key : str
        Optional image/FOV identifier used to keep niches image-local.
    adjacency_key : str
        Key in ``adata.obsp`` holding the cell graph adjacency.
    surround_hops : int
        Number of graph hops used to define the surrounding context.
    phenotype_labels : list, optional
        Explicit phenotype labels to report. If ``None``, all observed labels
        are used.
    min_cells : int
        Minimum niche size required for summarization.
    summary_stats : tuple
        Subset of ``("mean", "median", "std")`` to compute.

    Returns
    -------
    DataFrame
        One row per niche with surrounding-context summaries.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")
    if phenotype_key not in adata.obs.columns:
        raise ValueError(f"{phenotype_key} not found in adata.obs")
    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    feature_cols = list(feature_cols) if feature_cols is not None else []
    phenotype_feature_map = phenotype_feature_map or {}
    include_set = set(include_values) if include_values is not None else None

    missing_features = [col for col in feature_cols if col not in adata.obs.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in adata.obs: {missing_features}")

    missing_map_features = []
    for cols in phenotype_feature_map.values():
        for col in cols:
            if col not in adata.obs.columns:
                missing_map_features.append(col)
    if missing_map_features:
        missing_map_features = sorted(set(missing_map_features))
        raise ValueError(f"Missing phenotype_feature_map columns in adata.obs: {missing_map_features}")

    if phenotype_labels is None:
        phenotype_labels = pd.Series(adata.obs[phenotype_key]).dropna().unique().tolist()
    else:
        phenotype_labels = list(phenotype_labels)

    numeric_cache = {}
    for feature in sorted(set(feature_cols).union(*[set(v) for v in phenotype_feature_map.values()])):
        numeric_cache[feature] = pd.to_numeric(
            adata.obs[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

    adjacency = adata.obsp[adjacency_key].tocsr()
    group_cols = [image_key, niche_key] if image_key in adata.obs.columns else [niche_key]
    grouped = adata.obs.groupby(group_cols, dropna=True, observed=False)
    grouped_iter = grouped
    if show_progress:
        grouped_iter = tqdm(grouped, total=grouped.ngroups, desc=progress_desc)

    rows = []
    for group_key, obs_sub in grouped_iter:
        if len(group_cols) == 2:
            image_value, niche_value = group_key
        else:
            image_value = np.nan
            niche_value = group_key

        if include_set is not None and niche_value not in include_set:
            continue
        if pd.isna(niche_value) or niche_value in ("unassigned", "noise"):
            continue

        idx = adata.obs_names.get_indexer(obs_sub.index)
        idx = idx[idx >= 0]
        if len(idx) < int(min_cells):
            continue

        surround_layers = _get_n_hop_external_layers(
            adjacency_full=adjacency,
            niche_idx=idx,
            max_hops=surround_hops,
        )
        surround_idx = (
            np.concatenate(list(surround_layers.values()))
            if len(surround_layers) > 0
            else np.empty(0, dtype=int)
        )

        row = {
            niche_key: niche_value,
            "image_id": image_value,
            "n_cells": float(len(idx)),
            "n_surround": float(len(surround_idx)),
        }
        for hop in range(1, int(surround_hops) + 1):
            hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
            row[f"surround__hop_{hop}__n_cells"] = float(len(hop_idx))

        if len(surround_idx) == 0:
            for label in phenotype_labels:
                safe_label = _sanitize_label(label)
                row[f"surround_prop__{safe_label}"] = 0.0
                row[f"surround__{safe_label}__n_cells"] = 0.0
            for feature in feature_cols:
                for stat in summary_stats:
                    row[f"surround__{feature}__{stat}"] = np.nan
            for label, cols in phenotype_feature_map.items():
                safe_label = _sanitize_label(label)
                for feature in cols:
                    row[f"surround__{safe_label}__{feature}__n_cells"] = 0.0
                    for stat in summary_stats:
                        row[f"surround__{safe_label}__{feature}__{stat}"] = np.nan
            rows.append(row)
            continue

        surround_obs = adata.obs.iloc[surround_idx]
        surround_pheno = surround_obs[phenotype_key].astype("object")
        pheno_probs = surround_pheno.value_counts(normalize=True, dropna=True)

        for label in phenotype_labels:
            safe_label = _sanitize_label(label)
            label_mask = surround_pheno == label
            row[f"surround_prop__{safe_label}"] = _safe_scalar(pheno_probs.get(label, 0.0))
            row[f"surround__{safe_label}__n_cells"] = float(label_mask.sum())

        for feature in feature_cols:
            values = numeric_cache[feature][surround_idx]
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                for stat in summary_stats:
                    row[f"surround__{feature}__{stat}"] = np.nan
                continue
            if "mean" in summary_stats:
                row[f"surround__{feature}__mean"] = _safe_scalar(np.mean(finite))
            if "median" in summary_stats:
                row[f"surround__{feature}__median"] = _safe_scalar(np.median(finite))
            if "std" in summary_stats:
                row[f"surround__{feature}__std"] = _safe_scalar(np.std(finite, ddof=0))

        for label, cols in phenotype_feature_map.items():
            safe_label = _sanitize_label(label)
            label_idx = surround_idx[surround_pheno.to_numpy(dtype=object) == label]
            for feature in cols:
                row[f"surround__{safe_label}__{feature}__n_cells"] = float(len(label_idx))
                values = numeric_cache[feature][label_idx] if len(label_idx) > 0 else np.array([], dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    for stat in summary_stats:
                        row[f"surround__{safe_label}__{feature}__{stat}"] = np.nan
                    continue
                if "mean" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__mean"] = _safe_scalar(np.mean(finite))
                if "median" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__median"] = _safe_scalar(np.median(finite))
                if "std" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__std"] = _safe_scalar(np.std(finite, ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


def score_pdac_niche_pathology_modules(
    feature_df: pd.DataFrame,
    niche_key: str=None,
    image_key: str="image_id",
    polarity_high_is_organized: bool=True,
    min_features_per_module: int=2,
) -> pd.DataFrame:
    """
    Score PDAC pathology-inspired niche modules from a niche feature table.

    The function is intentionally tolerant of partially available inputs: each
    module uses whatever relevant columns are present in ``feature_df`` and
    averages signed per-column z-scores across the resolved features.

    Returns a compact per-niche table containing:
    - base pathology scores, including a separate proliferation module
    - condensed trajectory-oriented scores/axes
    - per-module feature counts for transparency
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")

    df = feature_df.copy()
    out = pd.DataFrame(index=df.index)

    if niche_key is not None and niche_key in df.columns:
        out[niche_key] = df[niche_key]
    if image_key is not None and image_key in df.columns:
        out[image_key] = df[image_key]

    def _first_existing(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _zscore(col):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        out_arr = np.full(len(values), np.nan, dtype=float)
        if finite.sum() < 2:
            return out_arr
        mean = values[finite].mean()
        std = values[finite].std(ddof=0)
        if np.isclose(std, 0.0):
            return out_arr
        out_arr[finite] = (values[finite] - mean) / std
        return out_arr

    polarity_sign_pos = 1.0 if polarity_high_is_organized else -1.0
    polarity_sign_neg = -1.0 * polarity_sign_pos

    module_specs = {
        "pdac_duct_organization_score": [
            (1.0, ["geometry__hull_circularity"]),
            (1.0, ["topology__largest_component_fraction"]),
            (1.0, ["geometry__mean_nearest_neighbor_distance", "geometry__median_edge_length", "topology__skeleton_mean_edge_length"]),
            (1.0, ["state__CK19_expr_z__mean", "state__CK19_expr__mean", "state__CK19_expr__median"]),
            (1.0, ["state__NaKATPase_expr_z__mean", "state__NaKATPase_expr__mean", "state__NaKATPase_expr__median"]),
            (polarity_sign_pos, ["state__polarity_score__median", "state__polarity_score__mean"]),
            (-1.0, ["graph_boundary__boundary_fraction"]),
            (-1.0, ["graph_boundary__mean_external_degree", "graph_boundary__boundary_external_degree_mean"]),
            (-1.0, ["topology__bridge_fraction"]),
            (-1.0, ["topology__skeleton_branchpoint_fraction"]),
            (-1.0, ["geometry__spatial_entropy"]),
        ],
        "pdac_dysplasia_score": [
            (1.0, ["state__nc_ratio_z__mean", "state__nc_ratio__mean", "state__nc_ratio__median", "state__nc_ratio__p90"]),
            (1.0, ["state__boundary_irregularity_z__mean", "state__boundary_irregularity__mean"]),
            (1.0, ["state__major_minor_axis_ratio_z__mean", "state__major_minor_axis_ratio__mean"]),
            (1.0, ["state__centroid_dif_z__mean", "state__centroid_dif__mean"]),
            (1.0, ["state__num_concavities_z__mean", "state__num_concavities__mean"]),
            (polarity_sign_neg, ["state__polarity_score__median", "state__polarity_score__mean"]),
            (1.0, ["features__nc_ratio__graph_morans_i"]),
        ],
        "pdac_architectural_complexity_score": [
            (1.0, ["topology__degree_var"]),
            (1.0, ["topology__avg_clustering"]),
            (1.0, ["topology__skeleton_branchpoint_fraction"]),
            (1.0, ["topology__skeleton_leaf_fraction"]),
            (1.0, ["geometry__orientation_entropy"]),
            (1.0, ["geometry__edge_length_cv"]),
            (1.0, ["state__lacunarity_z__mean", "state__lacunarity__mean"]),
            (1.0, ["state__boundary_irregularity__mean"]),
            (-1.0, ["geometry__hull_circularity"]),
        ],
        "pdac_invasion_desmoplasia_score": [
            (1.0, ["graph_boundary__boundary_fraction"]),
            (1.0, ["graph_boundary__mean_external_degree", "graph_boundary__boundary_external_degree_mean"]),
            (1.0, ["graph_surround__cross_edges_per_niche_cell", "graph_surround__n_cross_edges"]),
            (1.0, ["graph_surround__phenotype_entropy", "graph_surround__hop_1__phenotype_entropy"]),
            (1.0, ["graph_surround__hop_1__fraction_of_niche", "graph_surround__surround_to_niche_ratio"]),
            (1.0, ["surround_prop__Fibroblasts", "surround__Fibroblasts__fraction"]),
            (1.0, ["surround__Fibroblasts__FAP_expr_z__mean", "surround__Fibroblasts__FAP_expr__mean", "fibro_surround__FAP_expr_z__mean", "fibro_surround__FAP_expr__mean"]),
            (1.0, ["surround__Fibroblasts__aSMA_expr_z__mean", "surround__Fibroblasts__aSMA_expr__mean", "fibro_surround__aSMA_expr_z__mean", "fibro_surround__aSMA_expr__mean"]),
            (-1.0, ["geometry__hull_circularity"]),
        ],
        "pdac_proliferation_score": [
            (1.0, ["state__Ki67_expr_z__mean", "state__Ki67_expr__mean", "state__Ki67_expr__median"]),
            (1.0, ["state__Ki67_expr_z__p90", "state__Ki67_expr__p90"]),
        ],
        "pdac_dedifferentiation_score": [
            (1.0, ["state__entropy_z__mean", "state__entropy__mean"]),
            (1.0, ["state__inertia_z__mean", "state__inertia__mean"]),
            (1.0, ["state__haralick_contrast__mean"]),
            (1.0, ["geometry__cell_density_hull", "geometry__edge_density_hull"]),
            (1.0, ["topology__skeleton_branchpoint_fraction"]),
            (-1.0, ["geometry__mean_nearest_neighbor_distance", "topology__skeleton_mean_edge_length"]),
            (-1.0, ["geometry__hull_circularity"]),
            (polarity_sign_neg, ["state__polarity_score__median", "state__polarity_score__mean"]),
        ],
    }

    resolved_map = {}
    z_cache = {}

    for module_name, spec in module_specs.items():
        signed_parts = []
        resolved_features = []
        for sign, candidates in spec:
            resolved = _first_existing(candidates)
            if resolved is None:
                continue
            if resolved not in z_cache:
                z_cache[resolved] = _zscore(resolved)
            signed_parts.append(sign * z_cache[resolved])
            resolved_features.append((resolved, float(sign)))

        resolved_map[module_name] = resolved_features
        out[f"{module_name}__n_features"] = float(len(resolved_features))

        if len(signed_parts) < int(min_features_per_module):
            out[module_name] = np.nan
            continue

        stacked = np.vstack(signed_parts)
        out[module_name] = np.nanmean(stacked, axis=0)

    def _module_z(col):
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        result = np.full(len(values), np.nan, dtype=float)
        if finite.sum() < 2:
            return result
        mean = values[finite].mean()
        std = values[finite].std(ddof=0)
        if np.isclose(std, 0.0):
            return result
        result[finite] = (values[finite] - mean) / std
        return result

    base_cols = [
        "pdac_duct_organization_score",
        "pdac_dysplasia_score",
        "pdac_architectural_complexity_score",
        "pdac_invasion_desmoplasia_score",
        "pdac_proliferation_score",
        "pdac_dedifferentiation_score",
    ]
    base_z = {col: _module_z(col) for col in base_cols}

    condensed_specs = {
        "pdac_early_duct_anchor_score": [
            base_z["pdac_duct_organization_score"],
            -base_z["pdac_dysplasia_score"],
            -base_z["pdac_invasion_desmoplasia_score"],
            -base_z["pdac_proliferation_score"],
            -base_z["pdac_dedifferentiation_score"],
        ],
        "pdac_panin_like_dysplasia_score": [
            base_z["pdac_dysplasia_score"],
            base_z["pdac_architectural_complexity_score"],
            -base_z["pdac_duct_organization_score"],
        ],
        "pdac_invasive_gland_forming_score": [
            base_z["pdac_duct_organization_score"],
            base_z["pdac_invasion_desmoplasia_score"],
            base_z["pdac_proliferation_score"],
            -base_z["pdac_dedifferentiation_score"],
        ],
        "pdac_invasion_desmoplasia_axis": [
            base_z["pdac_invasion_desmoplasia_score"],
        ],
        "pdac_proliferation_axis": [
            base_z["pdac_proliferation_score"],
        ],
        "pdac_dedifferentiation_axis": [
            base_z["pdac_dedifferentiation_score"],
            -base_z["pdac_duct_organization_score"],
        ],
    }

    for module_name, arrays in condensed_specs.items():
        stacked = np.vstack(arrays)
        finite_counts = np.isfinite(stacked).sum(axis=0)
        out[f"{module_name}__n_features"] = finite_counts.astype(float)
        out[module_name] = np.nanmean(stacked, axis=0)
        out.loc[finite_counts == 0, module_name] = np.nan

    out.attrs["resolved_module_features"] = resolved_map
    return out
