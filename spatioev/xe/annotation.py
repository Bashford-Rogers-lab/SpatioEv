"""Marker-rule annotation helpers for Xenium cell tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _as_expression_frame(expression) -> pd.DataFrame:
    if isinstance(expression, pd.DataFrame):
        return expression.copy()
    if hasattr(expression, "to_df"):
        return expression.to_df()
    raise TypeError("expression must be a pandas DataFrame or AnnData-like object.")


def compute_marker_set_scores(
    expression: pd.DataFrame,
    marker_sets: Mapping[str, Sequence[str]],
    zscore_genes: bool = True,
    min_markers: int = 1,
) -> pd.DataFrame:
    """Compute per-cell marker-set scores from expression values.

    Parameters
    ----------
    expression
        Gene-expression DataFrame with cells as rows, or an AnnData-like object
        with ``to_df``.
    marker_sets
        Mapping from score name to candidate marker genes.
    zscore_genes
        When ``True``, each gene is z-scored before set averaging.
    min_markers
        Minimum number of available markers required for a score.
    """

    expr = _as_expression_frame(expression)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    if zscore_genes and not expr.empty:
        expr = (expr - expr.mean(axis=0)) / expr.std(axis=0, ddof=0).replace(0, np.nan)

    out = pd.DataFrame(index=expr.index)
    for score_name, markers in marker_sets.items():
        available = [gene for gene in markers if gene in expr.columns]
        if len(available) < min_markers:
            out[score_name] = np.nan
            continue
        out[score_name] = expr[available].mean(axis=1)
    return out


def summarize_cluster_marker_scores(
    scores: pd.DataFrame,
    labels: pd.Series | list[str],
    aggfunc: str = "median",
) -> pd.DataFrame:
    """Summarize marker scores per cluster or annotation label."""

    labels = pd.Series(labels, index=scores.index, name="cluster").astype(str)
    tmp = pd.concat([labels, scores.apply(pd.to_numeric, errors="coerce")], axis=1)
    summary = tmp.groupby("cluster", observed=True)[scores.columns.tolist()].agg(aggfunc)
    summary["n_cells"] = labels.value_counts().reindex(summary.index).astype(int)
    return summary.reset_index()


def assign_labels_from_marker_rules(
    cluster_summary: pd.DataFrame,
    marker_rules: Mapping[str, Mapping[str, float]],
    cluster_col: str = "cluster",
    unknown_label: str = "unassigned",
    output_col: str = "suggested_label",
) -> pd.DataFrame:
    """Assign cluster labels using simple thresholded marker-score rules.

    ``marker_rules`` maps a label to required marker-score thresholds. A
    threshold is interpreted as ``score >= threshold`` when positive and
    ``score <= threshold`` when negative. Rules are evaluated in dictionary
    order, making high-confidence or specific labels easy to prioritize.
    """

    if cluster_col not in cluster_summary.columns:
        raise ValueError(f"{cluster_col!r} not found in cluster_summary.")

    out = cluster_summary.copy()
    labels = []
    matched_rules = []
    for _, row in out.iterrows():
        assigned = unknown_label
        matched = ""
        for label, thresholds in marker_rules.items():
            ok = True
            for score_name, threshold in thresholds.items():
                if score_name not in row or not np.isfinite(row[score_name]):
                    ok = False
                    break
                if threshold >= 0 and row[score_name] < threshold:
                    ok = False
                    break
                if threshold < 0 and row[score_name] > threshold:
                    ok = False
                    break
            if ok:
                assigned = label
                matched = ",".join(thresholds)
                break
        labels.append(assigned)
        matched_rules.append(matched)
    out[output_col] = labels
    out[f"{output_col}_rule"] = matched_rules
    return out


__all__ = [
    "compute_marker_set_scores",
    "summarize_cluster_marker_scores",
    "assign_labels_from_marker_rules",
]
