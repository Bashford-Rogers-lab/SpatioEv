"""Trend summaries for spatial pseudotime analyses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

if TYPE_CHECKING:  # pragma: no cover
    pass


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
    "benjamini_hochberg",
    "compute_feature_trend_table",
    "add_branch_time_bins",
    "branch_time_feature_matrix",
    "find_branch_transition_features",
]
