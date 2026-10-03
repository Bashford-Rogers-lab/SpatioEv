"""Compare per-sample spatial summaries between groups (e.g. HPV+ vs HPV-).

Spatial features are measured per image, but images are not independent
replicates: a patient with three TMA cores is still one patient. These
helpers therefore aggregate to one value per *unit* (patient, or sample when
there is one image per patient) before testing, never testing cell-level or
core-level values directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .pseudotime.trends import benjamini_hochberg

__all__ = ["aggregate_to_units", "compare_groups"]


def aggregate_to_units(
    table: pd.DataFrame,
    unit_key: str,
    feature_key: str = "feature",
    value_key: str = "value",
    group_key: str = "group",
    agg: str = "mean",
) -> pd.DataFrame:
    """Collapse a long feature table to one value per unit and feature.

    Raises if a unit maps to more than one group, which would mean the sample
    sheet assigns one patient to both arms.
    """
    groups_per_unit = table.groupby(unit_key)[group_key].nunique()
    conflicting = groups_per_unit[groups_per_unit > 1].index.tolist()
    if conflicting:
        raise ValueError(f"Units assigned to more than one group: {conflicting[:10]}")
    finite = table[np.isfinite(pd.to_numeric(table[value_key], errors="coerce"))]
    return (
        finite.groupby([unit_key, group_key, feature_key], observed=True)[value_key]
        .agg(agg)
        .reset_index()
    )


def compare_groups(
    table: pd.DataFrame,
    group_key: str = "group",
    feature_key: str = "feature",
    value_key: str = "value",
    unit_key: str | None = None,
    groups: Sequence[str] | None = None,
    agg: str = "mean",
    min_units_per_group: int = 2,
) -> pd.DataFrame:
    """Test every feature for a difference between groups.

    Parameters
    ----------
    table : DataFrame
        Long format: one row per unit (or image) x feature, with a group
        column, a feature-name column and a numeric value column.
    unit_key : str, optional
        Column identifying independent units (patients). Rows sharing a unit
        are averaged first with ``agg``. Omit when rows are already one per
        independent unit.
    groups : sequence of str, optional
        Groups to compare and their order; the first is the reference. All
        groups present are used when omitted.
    min_units_per_group : int
        Features with fewer units in any group are reported untested.

    Returns
    -------
    DataFrame
        One row per feature: units and median per group, the difference in
        medians (second minus first group, two-group case), the test used
        (Mann-Whitney U for two groups, Kruskal-Wallis for more), its p-value,
        the smallest p-value the group sizes allow (two-group case) and the
        Benjamini-Hochberg q-value across features.
    """
    from scipy.stats import kruskal, mannwhitneyu

    data = table.copy()
    data[group_key] = data[group_key].astype(str)
    if unit_key is not None:
        data = aggregate_to_units(data, unit_key, feature_key, value_key, group_key, agg)
    else:
        data = data[np.isfinite(pd.to_numeric(data[value_key], errors="coerce"))]
    order = [str(g) for g in groups] if groups is not None else sorted(data[group_key].unique())
    data = data[data[group_key].isin(order)]

    rows = []
    for feature, frame in data.groupby(feature_key, sort=True):
        samples = [frame.loc[frame[group_key] == g, value_key].to_numpy(float) for g in order]
        row = {feature_key: feature}
        for name, values in zip(order, samples):
            row[f"n_{name}"] = len(values)
            row[f"median_{name}"] = float(np.median(values)) if len(values) else np.nan
            row[f"mean_{name}"] = float(np.mean(values)) if len(values) else np.nan
        testable = len(order) >= 2 and all(len(values) >= min_units_per_group for values in samples)
        row["test"], row["statistic"], row["p_value"] = None, np.nan, np.nan
        if len(order) == 2:
            row["median_difference"] = row[f"median_{order[1]}"] - row[f"median_{order[0]}"]
            # The smallest two-sided Mann-Whitney p-value these group sizes
            # can produce; with 3 vs 3 it is 0.1, so nothing can reach 0.05.
            n1, n2 = len(samples[0]), len(samples[1])
            row["min_achievable_p"] = 2 / math.comb(n1 + n2, n1) if n1 and n2 else np.nan
        if testable:
            pooled = np.concatenate(samples)
            if np.ptp(pooled) == 0:
                row["test"], row["p_value"] = "constant", 1.0
            elif len(order) == 2:
                result = mannwhitneyu(samples[0], samples[1], alternative="two-sided")
                row["test"], row["statistic"], row["p_value"] = "mann_whitney_u", float(result.statistic), float(result.pvalue)
            else:
                result = kruskal(*samples)
                row["test"], row["statistic"], row["p_value"] = "kruskal_wallis", float(result.statistic), float(result.pvalue)
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value"] = benjamini_hochberg(result["p_value"].to_numpy()).to_numpy()
    return result.sort_values(["q_value", "p_value"], na_position="last").reset_index(drop=True)
