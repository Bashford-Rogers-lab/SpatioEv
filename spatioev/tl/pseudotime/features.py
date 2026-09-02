"""Pseudotime feature matrices and branch annotation.

Builds the feature matrix a trajectory model is fitted on, and annotates the
resulting principal tree with branch labels.

Heavy optional trajectory packages (ElPiGraph, UMAP) are intentionally not
imported here; fit those models in the caller and pass the result in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:  # pragma: no cover
    pass




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


__all__ = [
    "FeatureMatrixResult",
    "zscore_series",
    "minmax_scale",
    "score_signed_feature_module",
    "prepare_pseudotime_feature_matrix",
    "assign_feature_blocks",
    "block_balance_feature_matrix",
    "sample_center_feature_matrix",
    "tree_edges",
    "node_graph",
    "infer_branch_labels",
    "summarize_branches",
    "project_tree_nodes_to_embedding",
]
