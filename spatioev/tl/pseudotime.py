# ============================================================
# Section 1: Pseudotime feature matrix and branch annotation  (from archive/spatial/pseudotime.py)
# ============================================================

"""Reusable helpers for spatial pseudotime workflows.

The functions in this module collect the repeated analysis patterns that were
previously embedded in development notebooks:

- prepare robust niche-level feature matrices
- balance broad feature families before trajectory fitting
- score signed biological feature modules
- annotate branches on a fitted principal tree

They intentionally avoid importing heavy optional trajectory packages at module
import time. If a workflow needs ElPiGraph or UMAP, fit those models in the
notebook or script and pass the resulting coordinates/tree into these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureMatrixResult:
    """Container returned by :func:`prepare_pseudotime_feature_matrix`."""

    matrix: pd.DataFrame
    selected_features: list[str]
    diagnostics: pd.DataFrame


DEFAULT_FEATURE_BLOCK_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("histology_modules", ("histology__", "pdac_", "panin_validation__")),
    ("duct_architecture", ("duct_lumen__", "duct_continuity__")),
    ("epithelial_stromal_interface", ("interface__",)),
    (
        "microenvironment",
        ("surround_prop__", "surround__", "graph_surround__"),
    ),
    (
        "nuclear_morphology",
        ("state__dapi_", "state__nucleus_boundary"),
    ),
    ("architecture_topology", ("geometry__", "topology__")),
    ("epithelial_state", ("state__",)),
)


def zscore_series(values: np.ndarray, ddof: int = 0) -> pd.Series:
    """Return a NaN-aware z-score Series aligned to the input index."""

    series = pd.to_numeric(pd.Series(values), errors="coerce")
    mean = series.mean()
    std = series.std(ddof=ddof)
    if not np.isfinite(std) or np.isclose(std, 0.0):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - mean) / std


def minmax_scale(values: np.ndarray, feature_range: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """Scale numeric values into ``feature_range`` while preserving NaNs."""

    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)]
    lo, hi = feature_range
    if finite.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    src_min = finite.min()
    src_max = finite.max()
    if np.isclose(src_max, src_min):
        midpoint = lo + (hi - lo) / 2.0
        return pd.Series(midpoint, index=series.index, dtype=float).where(series.notna())

    scaled = (series - src_min) / (src_max - src_min)
    return scaled * (hi - lo) + lo


def score_signed_feature_module(
    data: pd.DataFrame,
    specs: Sequence[tuple[float, Sequence[str] | str]],
    min_features: int = 2,
    return_resolved: bool = False,
) -> pd.DataFrame:
    """Score a signed feature module from candidate feature columns.

    Parameters
    ----------
    data
        Feature table.
    specs
        Sequence of ``(sign, candidates)`` pairs. ``candidates`` may be one
        column name or an ordered list of fallback column names. The first
        available candidate is used.
    min_features
        Minimum number of resolved features required to compute a score.
    return_resolved
        When ``True``, return ``(score, resolved_features)``.

    Returns
    -------
    Series or tuple
        Z-scored signed module score aligned to ``data.index``. Missing scores
        are returned as NaN when too few features are available.
    """

    parts = []
    resolved = []
    for sign, candidates in specs:
        if isinstance(candidates, str):
            candidates = [candidates]
        column = next((col for col in candidates if col in data.columns), None)
        if column is None:
            continue

        z = zscore_series(data[column])
        if np.isfinite(z).sum() < 2:
            continue

        parts.append(float(sign) * z.to_numpy(dtype=float))
        resolved.append({"feature": column, "sign": float(sign)})

    if len(parts) < int(min_features):
        score = pd.Series(np.nan, index=data.index, dtype=float)
    else:
        stacked = np.vstack(parts)
        finite_counts = np.isfinite(stacked).sum(axis=0)
        score_values = np.nanmean(stacked, axis=0)
        score_values[finite_counts == 0] = np.nan
        score = pd.Series(score_values, index=data.index, dtype=float)

    if return_resolved:
        return score, pd.DataFrame(resolved)
    return score


def prepare_pseudotime_feature_matrix(
    data: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    priority_features: Sequence[str] | None = None,
    max_na_fraction: float = 0.35,
    correlation_threshold: float | None = 0.95,
    min_variance: float = 1e-8,
    impute: str | None = "median",
    standardize: bool = False,
) -> FeatureMatrixResult:
    """Prepare a clean numeric matrix for pseudotime or trajectory models.

    The routine follows the development-notebook workflow: select numeric
    features, remove sparse and near-constant columns, median-impute missing
    values, and optionally prune highly correlated features while preserving
    user-specified priority features.
    """

    if feature_cols is None:
        feature_cols = list(data.select_dtypes(include=[np.number]).columns)
    feature_cols = [col for col in dict.fromkeys(feature_cols) if col in data.columns]
    priority_features = [
        col for col in dict.fromkeys(priority_features or []) if col in feature_cols
    ]

    if not feature_cols:
        empty = pd.DataFrame(index=data.index)
        return FeatureMatrixResult(
            matrix=empty,
            selected_features=[],
            diagnostics=pd.DataFrame(
                columns=[
                    "feature",
                    "na_fraction",
                    "variance",
                    "selected",
                    "drop_reason",
                ]
            ),
        )

    X_raw = data.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    na_fraction = X_raw.isna().mean()
    diagnostics = pd.DataFrame(
        {
            "feature": feature_cols,
            "na_fraction": [float(na_fraction.get(col, np.nan)) for col in feature_cols],
            "variance": [
                float(X_raw[col].var(ddof=0)) if col in X_raw else np.nan
                for col in feature_cols
            ],
            "selected": False,
            "drop_reason": "",
        }
    ).set_index("feature", drop=False)

    keep_cols = [col for col in feature_cols if na_fraction[col] <= max_na_fraction]
    for col in set(feature_cols) - set(keep_cols):
        diagnostics.loc[col, "drop_reason"] = "high_missingness"

    X = X_raw.loc[:, keep_cols].copy()
    if impute == "median":
        X = X.apply(lambda col: col.fillna(col.median()), axis=0)
    elif impute == "mean":
        X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    elif impute is None:
        pass
    else:
        raise ValueError("impute must be 'median', 'mean', or None.")

    variance = X.var(ddof=0)
    keep_cols = [col for col in X.columns if variance[col] > min_variance]
    for col in set(X.columns) - set(keep_cols):
        diagnostics.loc[col, "drop_reason"] = "near_zero_variance"
    X = X.loc[:, keep_cols]

    priority_features = [col for col in priority_features if col in X.columns]
    candidate_cols = [col for col in X.columns if col not in priority_features]
    selected_features = list(priority_features)

    if correlation_threshold is None or X.shape[1] <= 1:
        selected_features = [*priority_features, *candidate_cols]
    else:
        corr_abs = X.corr().abs()
        for col in candidate_cols:
            if not selected_features:
                selected_features.append(col)
                continue
            col_corr = corr_abs.loc[col, selected_features].dropna()
            max_corr = col_corr.max() if len(col_corr) else np.nan
            if (not np.isfinite(max_corr)) or max_corr < correlation_threshold:
                selected_features.append(col)
            else:
                diagnostics.loc[col, "drop_reason"] = "high_correlation"

    X = X.loc[:, selected_features].copy()
    if standardize and X.shape[1] > 0:
        X.loc[:, :] = StandardScaler().fit_transform(X)

    diagnostics.loc[selected_features, "selected"] = True
    diagnostics = diagnostics.reset_index(drop=True).sort_values(
        ["selected", "drop_reason", "na_fraction", "feature"],
        ascending=[False, True, True, True],
    )

    return FeatureMatrixResult(
        matrix=X,
        selected_features=selected_features,
        diagnostics=diagnostics.reset_index(drop=True),
    )


def assign_feature_blocks(
    feature_names: Sequence[str],
    block_rules: Mapping[str, Sequence[str]] | None = None,
    other_label: str = "other",
) -> pd.DataFrame:
    """Assign each feature to a broad information block by name prefix."""

    if block_rules is None:
        rules = [(name, prefixes) for name, prefixes in DEFAULT_FEATURE_BLOCK_RULES]
    else:
        rules = [(name, tuple(prefixes)) for name, prefixes in block_rules.items()]

    rows = []
    for feature in feature_names:
        block = other_label
        for block_name, prefixes in rules:
            if any(str(feature).startswith(prefix) for prefix in prefixes):
                block = block_name
                break
        rows.append({"feature": feature, "feature_block": block})
    return pd.DataFrame(rows)


def block_balance_feature_matrix(
    matrix: pd.DataFrame,
    feature_blocks: pd.DataFrame | Mapping[str, str] | None = None,
    standardize: bool = True,
    return_blocks: bool = False,
) -> object:
    """Equalize broad feature families before trajectory fitting.

    Each feature is optionally z-scored, then divided by ``sqrt(n_features)`` in
    its block. This keeps over-represented blocks, such as many related shape
    or neighborhood descriptors, from dominating Euclidean models purely by
    column count.
    """

    X = matrix.copy()
    if standardize and X.shape[1] > 0:
        X.loc[:, :] = StandardScaler().fit_transform(X)

    if feature_blocks is None:
        block_df = assign_feature_blocks(X.columns)
    elif isinstance(feature_blocks, pd.DataFrame):
        if not {"feature", "feature_block"}.issubset(feature_blocks.columns):
            raise ValueError("feature_blocks DataFrame must include feature and feature_block.")
        block_df = feature_blocks.loc[:, ["feature", "feature_block"]].copy()
    else:
        block_df = pd.DataFrame(
            {
                "feature": list(feature_blocks),
                "feature_block": [feature_blocks[key] for key in feature_blocks],
            }
        )

    block_df = block_df[block_df["feature"].isin(X.columns)].copy()
    for _, sub in block_df.groupby("feature_block", dropna=False):
        cols = sub["feature"].tolist()
        if cols:
            X.loc[:, cols] = X.loc[:, cols] / np.sqrt(len(cols))

    if return_blocks:
        return X, block_df.reset_index(drop=True)
    return X


def sample_center_feature_matrix(
    matrix: pd.DataFrame,
    sample_ids: list[str],
    min_sample_size: int = 2,
) -> pd.DataFrame:
    """Z-center a feature matrix within each sample or image."""

    X = matrix.copy()
    sample_ids = pd.Series(sample_ids, index=X.index).astype(str)
    for idx in sample_ids.groupby(sample_ids).groups.values():
        if len(idx) < min_sample_size:
            continue
        mu = X.loc[idx].mean(axis=0)
        sd = X.loc[idx].std(axis=0, ddof=0).replace(0, np.nan)
        X.loc[idx] = (X.loc[idx] - mu) / sd
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _coerce_edge_array(raw_edges) -> np.ndarray:
    """
    Resolve an edge array from the slightly different containers used by
    principal-tree packages.

    ElPiGraph commonly stores edges as ``PG["Edges"][0]``, but some versions
    include additional edge metadata beside the edge list. This helper searches
    nested list/tuple containers and returns the first numeric ``(n_edges, 2)``
    array it can safely identify.
    """

    try:
        edges = np.asarray(raw_edges, dtype=int)
    except (TypeError, ValueError):
        edges = None

    if edges is not None:
        if edges.ndim == 2 and edges.shape[1] == 2:
            return edges
        if edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] != 2:
            return edges.T

    if isinstance(raw_edges, (list, tuple)):
        for item in raw_edges:
            try:
                return _coerce_edge_array(item)
            except ValueError:
                continue

    raise ValueError("principal_tree edges must be an array with shape (n_edges, 2).")


def tree_edges(principal_tree: object) -> list[tuple[int, int]]:
    """Return principal-tree edges as ``(source, target)`` integer tuples."""

    if isinstance(principal_tree, nx.Graph):
        return [(int(a), int(b)) for a, b in principal_tree.edges()]

    if isinstance(principal_tree, Mapping):
        raw_edges = principal_tree.get("Edges")
        edges = _coerce_edge_array(raw_edges)
    else:
        edges = _coerce_edge_array(principal_tree)
    return [tuple(map(int, edge)) for edge in edges]


def node_graph(principal_tree: object) -> nx.Graph:
    """Convert principal-tree edges to a NetworkX graph."""

    graph = nx.Graph()
    graph.add_edges_from(tree_edges(principal_tree))
    return graph


def infer_branch_labels(
    principal_tree: object,
    observations: pd.DataFrame,
    source_node: int,
    node_col: str = "node_id",
    trunk_label: str = "trunk",
    branch_prefix: str = "branch",
) -> object:
    """Assign observation-level branch labels from a principal tree.

    The source node anchors pseudotime. The first high-degree node encountered
    from the source is treated as the hub. Nodes on the source-to-hub path form
    the trunk, and each hub-to-leaf path becomes a branch.
    """

    if node_col not in observations.columns:
        raise ValueError(f"{node_col!r} not found in observations.")

    graph = node_graph(principal_tree)
    if source_node not in graph:
        raise ValueError("source_node is not present in the principal tree.")

    degrees = dict(graph.degree())
    branch_nodes = [node for node, degree in degrees.items() if degree >= 3]
    if branch_nodes:
        hub = min(
            branch_nodes,
            key=lambda node: nx.shortest_path_length(graph, source=source_node, target=node),
        )
    else:
        leaves = [node for node, degree in degrees.items() if degree == 1 and node != source_node]
        hub = max(
            leaves or [source_node],
            key=lambda node: nx.shortest_path_length(graph, source=source_node, target=node),
        )

    trunk_path = nx.shortest_path(graph, source=source_node, target=hub)
    branch_paths: dict[str, list[int]] = {}
    branch_i = 1

    if hub in graph:
        remaining = graph.copy()
        remaining.remove_node(hub)
        for neighbor in sorted(graph.neighbors(hub)):
            if neighbor in trunk_path:
                continue
            component = nx.node_connected_component(remaining, neighbor)
            leaves = sorted([node for node in component if graph.degree(node) == 1])
            if not leaves:
                leaves = [neighbor]
            for leaf in leaves:
                path = [hub] + nx.shortest_path(graph, source=neighbor, target=leaf)
                branch_paths[f"{branch_prefix} {branch_i}"] = path
                branch_i += 1

    node_to_branch = {node: trunk_label for node in trunk_path}
    for label, path in branch_paths.items():
        for node in path[1:]:
            node_to_branch[node] = label

    labels = observations[node_col].map(node_to_branch).fillna("other").astype(str)
    metadata = {
        "hub_node": int(hub),
        "source_node": int(source_node),
        "trunk_path": [int(node) for node in trunk_path],
        "branch_paths": {
            label: [int(node) for node in path] for label, path in branch_paths.items()
        },
    }
    return labels, metadata


def summarize_branches(
    data: pd.DataFrame,
    branch_col: str,
    pseudotime_col: str | None = None,
    score_cols: Sequence[str] | None = None,
    sample_col: str | None = None,
    disease_col: str | None = None,
) -> pd.DataFrame:
    """Summarize branch size, composition, pseudotime, and score enrichment."""

    if branch_col not in data.columns:
        raise ValueError(f"{branch_col!r} not found in data.")

    score_cols = [col for col in (score_cols or []) if col in data.columns]
    score_median_all = {
        col: pd.to_numeric(data[col], errors="coerce").median() for col in score_cols
    }
    score_sd_all = {
        col: pd.to_numeric(data[col], errors="coerce").std(ddof=0) for col in score_cols
    }

    rows = []
    for branch, sub in data.groupby(branch_col, observed=True, dropna=False):
        row = {
            "branch": branch,
            "n_observations": int(len(sub)),
            "fraction_of_all_observations": float(len(sub) / len(data)) if len(data) else np.nan,
        }
        if pseudotime_col and pseudotime_col in sub.columns:
            pt = pd.to_numeric(sub[pseudotime_col], errors="coerce")
            row["median_pseudotime"] = float(pt.median())
            row["min_pseudotime"] = float(pt.min())
            row["max_pseudotime"] = float(pt.max())

        for col_name, out_prefix in ((sample_col, "sample"), (disease_col, "disease")):
            if col_name and col_name in sub.columns and len(sub):
                counts = sub[col_name].astype(str).value_counts(normalize=True)
                row[f"dominant_{out_prefix}"] = counts.index[0]
                row[f"dominant_{out_prefix}_fraction"] = float(counts.iloc[0])

        for col in score_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            median_val = values.median()
            sd = score_sd_all[col]
            row[f"{col}__median"] = float(median_val)
            row[f"{col}__z_enrichment"] = (
                float((median_val - score_median_all[col]) / sd)
                if np.isfinite(sd) and not np.isclose(sd, 0.0)
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("branch").reset_index(drop=True)


def project_tree_nodes_to_embedding(
    observations: pd.DataFrame,
    node_col: str,
    embedding_cols: Sequence[str],
    reducer: Callable = np.nanmedian,
) -> pd.DataFrame:
    """Estimate tree-node positions in a plotted embedding from assigned cells."""

    required = [node_col, *embedding_cols]
    missing = [col for col in required if col not in observations.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    for node_id, sub in observations.groupby(node_col, observed=True):
        row = {"node_id": int(node_id), "n_observations": int(len(sub))}
        for col in embedding_cols:
            row[col] = float(reducer(pd.to_numeric(sub[col], errors="coerce")))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)



# ============================================================
# Section 2: Pseudotime dynamics  (from archive/spatial/pseudotime_dynamics.py)
# ============================================================

"""
Helpers for summarizing spatial interaction dynamics around epithelial pseudotime.

These utilities are designed to work after a pseudotime workflow has already
assigned a niche- or cell-level pseudotime value back onto ``adata.obs``.
They intentionally build on top of the local source-centered interaction
functions in ``spatial_stats.py`` rather than introducing a second interaction
framework.
"""

from .stats import cross_ripley_local_counts


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



# ============================================================
# Section 3: Pseudotime trends  (from archive/spatial/pseudotime_trends.py)
# ============================================================

"""Trend summaries for spatial pseudotime analyses."""

from collections.abc import Sequence

from scipy.stats import mannwhitneyu, spearmanr


def benjamini_hochberg(p_values: np.ndarray) -> pd.Series:
    """Benjamini-Hochberg FDR correction that preserves input order."""

    series = pd.to_numeric(pd.Series(p_values), errors="coerce")
    q_values = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if valid.empty:
        return q_values

    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return q_values


def compute_feature_trend_table(
    data: pd.DataFrame,
    feature_cols: Sequence[str] | Sequence[dict],
    pseudotime_col: str,
    group_name: str = "all",
    min_n: int = 30,
    early_quantile: float = 0.1,
    late_quantile: float = 0.9,
) -> pd.DataFrame:
    """Summarize monotonic and early/late changes along pseudotime.

    ``feature_cols`` may be a sequence of feature names or dictionaries with
    ``feature``, ``label``, and ``category`` keys. The output is designed for
    manuscript-ready ranking tables and heatmaps.
    """

    if pseudotime_col not in data.columns:
        raise ValueError(f"{pseudotime_col!r} not found in data.")

    specs = []
    for item in feature_cols:
        if isinstance(item, dict):
            if "feature" not in item:
                raise ValueError("Feature specs must include a 'feature' key.")
            specs.append(item)
        else:
            specs.append({"feature": str(item), "label": str(item), "category": ""})

    pt = pd.to_numeric(data[pseudotime_col], errors="coerce")
    rows = []
    for spec in specs:
        feature = spec["feature"]
        if feature not in data.columns:
            continue
        y = pd.to_numeric(data[feature], errors="coerce")
        tmp = pd.DataFrame({"pseudotime": pt, "value": y}).dropna()
        if len(tmp) < min_n or tmp["pseudotime"].nunique() < 5:
            continue

        rho, spearman_p = spearmanr(tmp["pseudotime"], tmp["value"])
        q_early = tmp["pseudotime"].quantile(early_quantile)
        q_late = tmp["pseudotime"].quantile(late_quantile)
        early = tmp.loc[tmp["pseudotime"] <= q_early, "value"]
        late = tmp.loc[tmp["pseudotime"] >= q_late, "value"]
        if early.empty or late.empty:
            mw_p = np.nan
            delta = np.nan
        else:
            delta = late.median() - early.median()
            try:
                mw_p = mannwhitneyu(early, late, alternative="two-sided").pvalue
            except ValueError:
                mw_p = np.nan

        rows.append(
            {
                "group": group_name,
                "category": spec.get("category", ""),
                "feature": feature,
                "label": spec.get("label", feature),
                "n": int(len(tmp)),
                "spearman_r": float(rho),
                "spearman_p": float(spearman_p),
                "early_quantile": float(early_quantile),
                "late_quantile": float(late_quantile),
                "early_median": float(early.median()) if not early.empty else np.nan,
                "late_median": float(late.median()) if not late.empty else np.nan,
                "late_minus_early_median": float(delta),
                "mannwhitney_p_early_vs_late": float(mw_p),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["spearman_fdr"] = benjamini_hochberg(out["spearman_p"]).to_numpy()
    out["mannwhitney_fdr"] = benjamini_hochberg(
        out["mannwhitney_p_early_vs_late"]
    ).to_numpy()
    return out.sort_values(["category", "spearman_r"], ascending=[True, False]).reset_index(
        drop=True
    )


def add_branch_time_bins(
    data: pd.DataFrame,
    branch_col: str,
    pseudotime_col: str,
    n_bins: int = 3,
    labels: Sequence[str] | None = None,
    output_col: str = "branch_time_bin",
) -> pd.DataFrame:
    """Add within-branch pseudotime bins to a copy of ``data``."""

    if branch_col not in data.columns:
        raise ValueError(f"{branch_col!r} not found in data.")
    if pseudotime_col not in data.columns:
        raise ValueError(f"{pseudotime_col!r} not found in data.")
    if labels is None:
        labels = [f"bin_{i + 1}" for i in range(n_bins)]
    if len(labels) != n_bins:
        raise ValueError("labels must have length n_bins.")

    out = data.copy()
    out[output_col] = pd.NA
    for _, idx in out.groupby(branch_col, observed=True).groups.items():
        values = pd.to_numeric(out.loc[idx, pseudotime_col], errors="coerce")
        valid = values.dropna()
        if valid.nunique() < 2:
            continue
        bins = min(n_bins, valid.nunique())
        try:
            assigned = pd.qcut(valid, q=bins, labels=labels[:bins], duplicates="drop")
        except ValueError:
            assigned = pd.cut(valid, bins=bins, labels=labels[:bins], include_lowest=True)
        out.loc[assigned.index, output_col] = assigned.astype(str)
    return out


def branch_time_feature_matrix(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    branch_col: str = "branch",
    time_bin_col: str = "branch_time_bin",
    aggfunc: str = "median",
    zscore_features: bool = True,
) -> pd.DataFrame:
    """Build a branch-by-time feature summary matrix."""

    required = [branch_col, time_bin_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [col for col in feature_cols if col in data.columns]
    if not feature_cols:
        return pd.DataFrame()

    numeric = data.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    tmp = pd.concat([data[[branch_col, time_bin_col]], numeric], axis=1)
    summary = tmp.groupby([branch_col, time_bin_col], observed=True)[feature_cols].agg(aggfunc)
    if zscore_features and not summary.empty:
        summary = (summary - summary.mean(axis=0)) / summary.std(axis=0, ddof=0).replace(
            0, np.nan
        )
    return summary.replace([np.inf, -np.inf], np.nan)


def find_branch_transition_features(
    branch_time_matrix: pd.DataFrame,
    min_abs_delta: float = 0.5,
) -> pd.DataFrame:
    """Rank features by the largest adjacent branch-time change."""

    if branch_time_matrix.empty:
        return pd.DataFrame(
            columns=[
                "branch",
                "feature",
                "from_bin",
                "to_bin",
                "delta",
                "abs_delta",
            ]
        )
    if not isinstance(branch_time_matrix.index, pd.MultiIndex):
        raise ValueError("branch_time_matrix must have a MultiIndex of branch and time bin.")

    rows = []
    for branch, sub in branch_time_matrix.groupby(level=0, observed=True):
        sub = sub.droplevel(0)
        if len(sub) < 2:
            continue
        for i in range(len(sub) - 1):
            from_bin = sub.index[i]
            to_bin = sub.index[i + 1]
            delta = sub.iloc[i + 1] - sub.iloc[i]
            for feature, value in delta.items():
                if np.isfinite(value) and abs(value) >= min_abs_delta:
                    rows.append(
                        {
                            "branch": branch,
                            "feature": feature,
                            "from_bin": from_bin,
                            "to_bin": to_bin,
                            "delta": float(value),
                            "abs_delta": float(abs(value)),
                        }
                    )
    return pd.DataFrame(rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)



__all__ = [
    "prepare_pseudotime_feature_matrix",
    "block_balance_feature_matrix",
    "sample_center_feature_matrix",
    "infer_branch_labels",
    "summarize_branches",
    "project_tree_nodes_to_embedding",
    "score_signed_feature_module",
    "assign_pseudotime_bins",
    "compute_epithelial_centered_interaction_dynamics",
    "summarize_epithelial_interaction_dynamics",
    "compute_feature_trend_table",
    "add_branch_time_bins",
    "branch_time_feature_matrix",
    "find_branch_transition_features",
    # helper / building-block functions also defined in this module
    "FeatureMatrixResult",
    "zscore_series",
    "minmax_scale",
    "assign_feature_blocks",
    "tree_edges",
    "node_graph",
    "benjamini_hochberg",
]
