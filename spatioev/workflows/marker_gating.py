#!/usr/bin/env python3
"""Condition-driven marker autogating and QC for multiplexed images.

The script is intentionally conservative: visual marker-condition labels choose
the gate family, but all-cell 2-component GMM gates are rejected when they call
implausibly broad positivity.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import anndata as ad

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_marker_gating_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from sklearn.mixture import GaussianMixture

from spatioev.workflows.image_collection import channel_names

try:
    from skimage.filters import threshold_otsu
except Exception:  # pragma: no cover - optional dependency
    threshold_otsu = None


CHANNEL_ALIASES = {
    "HOECHST 2": "HOECHST2",
    "HOECHST": "HOECHST2",
    "LYVE-1": "LYVE1",
    "HNFalpha": "HNF4a",
    "pancytokeratin": "panCK",
    "pancyto": "panCK",
    "EPCAM": "EpCAM",
    "Cd16": "CD16",
    "alphaSMA": "aSMA",
    "TCRValpha": "TCRVa",
    "ILR18a": "IL18Ra",
    "FOXP3": "FoxP3",
    "CD4 good": "CD4",
}

DEFAULT_REVIEW_MARKERS = [
    "CD25",
    "ICOS",
    "CD62L",
    "CD45RO",
    "IL18Ra",
    "CD38",
    "CD279",
    "CD69",
    "CD40",
    "CD39",
    "CD56",
    "CD68",
    "FoxP3",
    "Ki67",
    "CD3",
    "CD4",
    "CD8",
]

DIRECT_METHOD_COLUMNS = {
    "otsu": ("otsu_gate", "otsu_pos_frac"),
    "2component_gmm": ("gmm2_gate", "gmm2_pos_frac"),
    "3component_gmm_high": ("gmm3_high_gate", "gmm3_high_pos_frac"),
    "upper_tail_gmm_q75": ("upper_tail_q75_gate", "upper_tail_q75_pos_frac"),
    "upper_tail_gmm_q90": ("upper_tail_q90_gate", "upper_tail_q90_pos_frac"),
    "upper_tail_gmm_q95": ("upper_tail_q95_gate", "upper_tail_q95_pos_frac"),
    "robust_q95_mad2": ("robust_q95_mad2_gate", "robust_q95_mad2_pos_frac"),
    "robust_q99_mad3": ("robust_q99_mad3_gate", "robust_q99_mad3_pos_frac"),
}


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    project_root: Path
    adata_path: Path
    image_path: Path
    marker_condition_path: Path
    output_dir: Path


def default_sample_paths(sample_id: str, project_root: Path) -> SamplePaths:
    return SamplePaths(
        sample_id=sample_id,
        project_root=project_root,
        adata_path=project_root / "data" / sample_id / f"{sample_id}_adata.h5ad",
        image_path=project_root / sample_id / "processed" / f"{sample_id}_combined.ome.tif",
        marker_condition_path=project_root / "data" / sample_id / "marker_condition_for_gating.csv",
        output_dir=project_root / "results" / f"{sample_id}_marker_gating_qc",
    )


def read_marker_conditions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "marker",
        "staining_condition",
        "compartment_pattern",
        "expression_condition",
        "artifact_level",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Marker condition CSV is missing columns: {sorted(missing)}")
    if df["marker"].duplicated().any():
        duplicated = sorted(df.loc[df["marker"].duplicated(), "marker"].astype(str).unique())
        raise ValueError(f"Marker condition CSV contains duplicated markers: {duplicated}")
    return df.copy()


def read_strategy_profile(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"marker", "preferred_method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strategy profile is missing columns: {sorted(missing)}")
    if df["marker"].duplicated().any():
        duplicated = sorted(df.loc[df["marker"].duplicated(), "marker"].astype(str).unique())
        raise ValueError(f"Strategy profile contains duplicated markers: {duplicated}")
    return df.copy()


def load_adata(adata_path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(adata_path)
    if adata.raw is None:
        adata.raw = adata.copy()
    return adata


def expression_matrix(adata: ad.AnnData, layer: str | None = None) -> np.ndarray:
    if layer is None:
        x = adata.X
    elif layer == "raw":
        if adata.raw is None:
            raise ValueError("Requested layer='raw' but adata.raw is None")
        x = adata.raw.X
    else:
        x = adata.layers[layer]
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=float)


def clean_log1p_values(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0, None)
    return np.log1p(x)


def finite_fraction(x: np.ndarray, gate: float) -> float:
    if not np.isfinite(gate) or len(x) == 0:
        return np.nan
    return float(np.mean(x >= gate))


def sample_for_fit(x: np.ndarray, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_n is None or max_n <= 0:
        return x
    if len(x) <= max_n:
        return x
    return x[rng.choice(len(x), size=max_n, replace=False)]


def robust_tail_gate(x: np.ndarray, quantile: float = 0.99, mad_k: float = 3.0) -> float:
    if len(x) == 0:
        return np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_sd = 1.4826 * mad
    return float(max(np.quantile(x, quantile), median + mad_k * robust_sd))


def gmm_crossing_gate(
    x: np.ndarray,
    n_components: int,
    *,
    random_state: int,
    max_fit_cells: int | None,
    rng: np.random.Generator,
) -> dict[str, object]:
    xs = sample_for_fit(x, max_fit_cells, rng)
    empty = {
        "gate": np.nan,
        "positive_fraction": np.nan,
        "gap": np.nan,
        "separation": np.nan,
        "high_weight": np.nan,
        "means": "",
    }
    if len(xs) < 50 or np.nanstd(xs) == 0:
        return empty

    try:
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            n_init=5,
            max_iter=300,
        ).fit(xs.reshape(-1, 1))
    except Exception:
        return empty

    order = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[order]
    sds = np.sqrt(gmm.covariances_.reshape(n_components)[order])
    weights = gmm.weights_.ravel()[order]

    grid = np.linspace(np.percentile(xs, 0.1), np.percentile(xs, 99.9), 2000).reshape(-1, 1)
    resp = gmm.predict_proba(grid)[:, order]
    j = n_components - 2
    diff = resp[:, j + 1] - resp[:, j]
    between = (grid[:, 0] >= means[j]) & (grid[:, 0] <= means[j + 1])
    crossing = np.where(np.diff(np.sign(diff)) != 0)[0]
    crossing_between = [i for i in crossing if between[i]]
    if crossing_between:
        gate = float(grid[crossing_between[0], 0])
    elif between.any():
        between_grid = grid[between, 0]
        between_diff = diff[between]
        gate = float(between_grid[np.argmin(np.abs(between_diff))])
    else:
        gate = float(np.mean([means[-2], means[-1]]))

    gap = float(means[-1] - means[-2])
    pooled_sd = float(np.sqrt(sds[-1] ** 2 + sds[-2] ** 2))
    separation = float(gap / pooled_sd) if pooled_sd > 0 else np.nan

    return {
        "gate": gate,
        "positive_fraction": finite_fraction(x, gate),
        "gap": gap,
        "separation": separation,
        "high_weight": float(weights[-1]),
        "means": ";".join(f"{m:.4f}" for m in means),
    }


def upper_tail_gmm_gate(
    x: np.ndarray,
    tail_quantile: float,
    *,
    random_state: int,
    max_fit_cells: int | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    empty = {
        "gate": np.nan,
        "positive_fraction": np.nan,
        "low_mean": np.nan,
        "high_mean": np.nan,
        "high_weight": np.nan,
    }
    if len(x) < 50 or np.nanstd(x) == 0:
        return empty

    cutoff = float(np.quantile(x, tail_quantile))
    tail = x[x >= cutoff]
    xs = sample_for_fit(tail, max_fit_cells, rng)
    if len(xs) < 50 or np.nanstd(xs) == 0:
        return empty

    try:
        gmm = GaussianMixture(
            n_components=2,
            random_state=random_state,
            n_init=5,
            max_iter=300,
        ).fit(xs.reshape(-1, 1))
    except Exception:
        return empty

    order = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[order]
    weights = gmm.weights_.ravel()[order]
    grid = np.linspace(np.percentile(xs, 0.1), np.percentile(xs, 99.9), 1500).reshape(-1, 1)
    resp = gmm.predict_proba(grid)[:, order]
    diff = resp[:, 1] - resp[:, 0]
    between = (grid[:, 0] >= means[0]) & (grid[:, 0] <= means[1])
    crossing = np.where(np.diff(np.sign(diff)) != 0)[0]
    crossing_between = [i for i in crossing if between[i]]
    if crossing_between:
        gate = float(grid[crossing_between[0], 0])
    elif between.any():
        between_grid = grid[between, 0]
        between_diff = diff[between]
        gate = float(between_grid[np.argmin(np.abs(between_diff))])
    else:
        gate = np.nan

    return {
        "gate": gate,
        "positive_fraction": finite_fraction(x, gate),
        "low_mean": float(means[0]),
        "high_mean": float(means[1]),
        "high_weight": float(weights[1]),
    }


def compute_gate_candidates(
    adata: ad.AnnData,
    marker_conditions: pd.DataFrame,
    *,
    layer: str | None = None,
    max_fit_cells: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    xmat = expression_matrix(adata, layer=layer)
    marker_to_index = {marker: i for i, marker in enumerate(adata.var_names)}
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for _, row in marker_conditions.iterrows():
        marker = str(row["marker"])
        if marker not in marker_to_index:
            raise ValueError(f"Marker {marker!r} not found in adata.var_names")

        x = clean_log1p_values(xmat[:, marker_to_index[marker]])
        quantiles = np.quantile(x, [0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99]) if len(x) else np.repeat(np.nan, 7)
        mean = float(np.mean(x)) if len(x) else np.nan
        standard_deviation = float(np.std(x)) if len(x) else np.nan
        skewness = (
            float(np.mean(((x - mean) / standard_deviation) ** 3))
            if len(x) and standard_deviation > 0
            else np.nan
        )

        otsu_gate = np.nan
        if threshold_otsu is not None and len(x) >= 50 and len(np.unique(x)) > 2:
            try:
                otsu_gate = float(threshold_otsu(sample_for_fit(x, max_fit_cells, rng)))
            except Exception:
                otsu_gate = np.nan

        gmm2 = gmm_crossing_gate(x, 2, random_state=random_state, max_fit_cells=max_fit_cells, rng=rng)
        gmm3 = gmm_crossing_gate(x, 3, random_state=random_state, max_fit_cells=max_fit_cells, rng=rng)
        u75 = upper_tail_gmm_gate(x, 0.75, random_state=random_state + 2, max_fit_cells=max_fit_cells, rng=rng)
        u90 = upper_tail_gmm_gate(x, 0.90, random_state=random_state + 2, max_fit_cells=max_fit_cells, rng=rng)
        u95 = upper_tail_gmm_gate(x, 0.95, random_state=random_state + 2, max_fit_cells=max_fit_cells, rng=rng)
        robust95 = robust_tail_gate(x, quantile=0.95, mad_k=2)
        robust99 = robust_tail_gate(x, quantile=0.99, mad_k=3)

        out = row.to_dict()
        out.update(
            {
                "n_cells": int(len(x)),
                "q25": quantiles[0],
                "q50": quantiles[1],
                "q75": quantiles[2],
                "q90": quantiles[3],
                "q95": quantiles[4],
                "q975": quantiles[5],
                "q99": quantiles[6],
                "distribution_iqr": quantiles[2] - quantiles[0],
                "distribution_dynamic_range": quantiles[6] - quantiles[1],
                "distribution_skewness": skewness,
                "otsu_gate": otsu_gate,
                "otsu_pos_frac": finite_fraction(x, otsu_gate),
                "gmm2_gate": gmm2["gate"],
                "gmm2_pos_frac": gmm2["positive_fraction"],
                "gmm2_sep": gmm2["separation"],
                "gmm2_gap": gmm2["gap"],
                "gmm2_high_weight": gmm2["high_weight"],
                "gmm2_means": gmm2["means"],
                "gmm3_high_gate": gmm3["gate"],
                "gmm3_high_pos_frac": gmm3["positive_fraction"],
                "gmm3_high_sep": gmm3["separation"],
                "gmm3_high_gap": gmm3["gap"],
                "gmm3_high_weight": gmm3["high_weight"],
                "gmm3_means": gmm3["means"],
                "upper_tail_q75_gate": u75["gate"],
                "upper_tail_q75_pos_frac": u75["positive_fraction"],
                "upper_tail_q90_gate": u90["gate"],
                "upper_tail_q90_pos_frac": u90["positive_fraction"],
                "upper_tail_q95_gate": u95["gate"],
                "upper_tail_q95_pos_frac": u95["positive_fraction"],
                "robust_q95_mad2_gate": robust95,
                "robust_q95_mad2_pos_frac": finite_fraction(x, robust95),
                "robust_q99_mad3_gate": robust99,
                "robust_q99_mad3_pos_frac": finite_fraction(x, robust99),
            }
        )
        candidate_gates = np.asarray(
            [
                out["otsu_gate"],
                out["gmm2_gate"],
                out["gmm3_high_gate"],
                out["upper_tail_q75_gate"],
                out["upper_tail_q90_gate"],
                out["upper_tail_q95_gate"],
                out["robust_q95_mad2_gate"],
                out["robust_q99_mad3_gate"],
            ],
            dtype=float,
        )
        candidate_fractions = np.asarray(
            [
                out["otsu_pos_frac"],
                out["gmm2_pos_frac"],
                out["gmm3_high_pos_frac"],
                out["upper_tail_q75_pos_frac"],
                out["upper_tail_q90_pos_frac"],
                out["upper_tail_q95_pos_frac"],
                out["robust_q95_mad2_pos_frac"],
                out["robust_q99_mad3_pos_frac"],
            ],
            dtype=float,
        )
        finite_gates = candidate_gates[np.isfinite(candidate_gates)]
        finite_fractions = candidate_fractions[np.isfinite(candidate_fractions)]
        out["candidate_gate_span"] = (
            float(np.ptp(finite_gates)) if len(finite_gates) > 1 else np.nan
        )
        out["candidate_positive_fraction_span"] = (
            float(np.ptp(finite_fractions)) if len(finite_fractions) > 1 else np.nan
        )
        rows.append(out)

    return add_distribution_diagnostics(pd.DataFrame(rows))


def infer_expression_condition(row: pd.Series) -> tuple[str, str, str]:
    """Suggest a visual expression label from distribution diagnostics.

    This is deliberately heuristic. It helps initialize review but does not
    attempt to infer staining specificity or image artifacts from intensities.
    """
    dynamic_range = float(row.get("distribution_dynamic_range", np.nan))
    gmm2_sep = float(row.get("gmm2_sep", np.nan))
    gmm2_frac = float(row.get("gmm2_pos_frac", np.nan))
    gmm3_sep = float(row.get("gmm3_high_sep", np.nan))
    gmm3_frac = float(row.get("gmm3_high_pos_frac", np.nan))
    skewness = float(row.get("distribution_skewness", np.nan))

    if not np.isfinite(dynamic_range) or dynamic_range < 0.08:
        return "broad_gradient", "low", "very limited dynamic range; inspect as possibly negative"
    if (
        np.isfinite(gmm2_sep)
        and gmm2_sep >= 1.5
        and np.isfinite(gmm2_frac)
        and 0.01 <= gmm2_frac <= 0.55
    ):
        confidence = "high" if gmm2_sep >= 2.25 else "medium"
        return "bimodal", confidence, "two separated all-cell components"
    if (
        np.isfinite(gmm3_sep)
        and gmm3_sep >= 0.9
        and np.isfinite(gmm3_frac)
        and 0.003 <= gmm3_frac <= 0.35
    ):
        confidence = "high" if gmm3_sep >= 1.5 else "medium"
        return "multi_level", confidence, "distinct high component within a multi-level distribution"
    if np.isfinite(skewness) and skewness >= 1.0:
        return "broad_gradient", "medium", "continuous right-skewed distribution without a stable valley"
    return "broad_gradient", "low", "no strongly separated automatic component"


def add_distribution_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out
    conservative_fraction_columns = [
        "upper_tail_q75_pos_frac",
        "upper_tail_q90_pos_frac",
        "upper_tail_q95_pos_frac",
        "robust_q95_mad2_pos_frac",
        "robust_q99_mad3_pos_frac",
    ]
    available = [column for column in conservative_fraction_columns if column in out]
    if available:
        out["conservative_candidate_positive_fraction_span"] = out[available].apply(
            lambda row: (
                float(np.ptp(row.dropna().astype(float).loc[lambda values: values <= 0.50]))
                if (row.dropna().astype(float) <= 0.50).sum() > 1
                else np.nan
            ),
            axis=1,
        )
    inferred = out.apply(infer_expression_condition, axis=1)
    out["inferred_expression_condition"] = [value[0] for value in inferred]
    out["expression_inference_confidence"] = [value[1] for value in inferred]
    out["distribution_note"] = [value[2] for value in inferred]
    return out


def condition_based_method(row: pd.Series) -> str:
    staining = str(row["staining_condition"])
    expression = str(row["expression_condition"])
    artifact = str(row["artifact_level"])
    gmm2_frac = float(row.get("gmm2_pos_frac", np.nan))
    gmm2_sep = float(row.get("gmm2_sep", np.nan))

    if staining in {"negative_or_absent", "failed_or_unusable", "artifact_dominated"}:
        return "negative_reference_controlled_or_robust_q99_mad3"
    if expression == "multi_level":
        if staining == "clear_specific" and artifact == "low":
            return "3component_gmm_high"
        return "upper_tail_gmm_q75"
    if expression == "broad_gradient":
        if staining != "clear_specific" or artifact != "low":
            return "upper_tail_gmm_q90_or_robust_q99_mad3"
        return "3component_gmm_high_or_upper_tail_q75"
    if expression == "bimodal":
        if (
            staining == "clear_specific"
            and artifact == "low"
            and np.isfinite(gmm2_frac)
            and gmm2_frac <= 0.70
            and np.isfinite(gmm2_sep)
            and gmm2_sep >= 0.75
        ):
            return "2component_gmm"
        if staining == "clear_specific" and artifact == "low":
            return "3component_gmm_high_or_negative_reference_controlled"
        if artifact == "medium" or staining == "diffuse_background":
            return "upper_tail_gmm_q90"
        return "upper_tail_gmm_q75"
    return "manual_review"


def review_flags(row: pd.Series) -> str:
    flags: list[str] = []
    gmm2_frac = float(row.get("gmm2_pos_frac", np.nan))
    if str(row["expression_condition"]) == "bimodal" and (not np.isfinite(gmm2_frac) or gmm2_frac > 0.70):
        flags.append("2GMM_all_cells_too_inclusive")
    if str(row["staining_condition"]) in {"diffuse_background", "high_background_specific_tail"}:
        flags.append("background_aware_gate")
    if str(row["artifact_level"]) in {"medium", "high", "severe"}:
        flags.append("visual_QC_required")
    inferred_expression = str(row.get("inferred_expression_condition", ""))
    inference_confidence = str(row.get("expression_inference_confidence", ""))
    if (
        inference_confidence == "high"
        and inferred_expression
        and inferred_expression != str(row["expression_condition"])
    ):
        flags.append("expression_shape_disagreement")
    candidate_span = float(row.get("conservative_candidate_positive_fraction_span", np.nan))
    if np.isfinite(candidate_span) and candidate_span >= 0.10:
        flags.append("candidate_methods_disagree")
    dynamic_range = float(row.get("distribution_dynamic_range", np.nan))
    if np.isfinite(dynamic_range) and dynamic_range < 0.08:
        flags.append("low_dynamic_range")
    return ";".join(flags)


def add_method_plan(candidates: pd.DataFrame, strategy_profile: pd.DataFrame | None = None) -> pd.DataFrame:
    out = candidates.copy()
    out["condition_based_method"] = out.apply(condition_based_method, axis=1)
    out["review_flags"] = out.apply(review_flags, axis=1)
    if strategy_profile is not None:
        strategy_cols = [c for c in strategy_profile.columns if c != "marker" and c not in out.columns]
        out = out.merge(strategy_profile[["marker", *strategy_cols]], on="marker", how="left")
        profile_mask = out["preferred_method"].notna() & out["preferred_method"].astype(str).str.strip().ne("")
        out.loc[profile_mask, "review_flags"] = out.loc[profile_mask, "review_flags"].fillna("").astype(str)
        out.loc[profile_mask, "review_flags"] = (
            out.loc[profile_mask, "review_flags"]
            .str.strip(";")
            .where(out.loc[profile_mask, "review_flags"].str.len() == 0, out.loc[profile_mask, "review_flags"] + ";")
            + "strategy_profile"
        )
    return out


def target_positive_fraction_gate(x_values: np.ndarray, target_fraction: float) -> float:
    target_fraction = float(np.clip(target_fraction, 0.0001, 0.9999))
    return float(np.quantile(x_values, 1 - target_fraction))


def choose_strategy_gate(row: pd.Series, x_values: np.ndarray | None = None) -> tuple[str, float, str] | None:
    preferred = row.get("preferred_method", "")
    if pd.isna(preferred) or str(preferred).strip() == "":
        return None

    preferred = str(preferred).strip()
    if preferred == "target_positive_fraction":
        if x_values is None:
            return None
        target = row.get("target_positive_fraction", np.nan)
        if not np.isfinite(float(target)):
            return None
        gate = target_positive_fraction_gate(x_values, float(target))
        return preferred, gate, f"strategy profile target positive fraction {float(target):.3%}"

    if preferred in DIRECT_METHOD_COLUMNS:
        gate_col, _ = DIRECT_METHOD_COLUMNS[preferred]
        gate = float(row.get(gate_col, np.nan))
        if np.isfinite(gate):
            return preferred, gate, f"strategy profile preferred method: {preferred}"

    return None


def choose_gate(row: pd.Series, x_values: np.ndarray | None = None) -> tuple[str, float, str]:
    strategy_gate = choose_strategy_gate(row, x_values=x_values)
    if strategy_gate is not None:
        return strategy_gate

    method = str(row["condition_based_method"])

    if method == "2component_gmm":
        if np.isfinite(row.get("gmm2_gate", np.nan)):
            return "2component_gmm", float(row["gmm2_gate"]), "2-component GMM accepted by fraction/separation guardrails"
        return "otsu", float(row["otsu_gate"]), "Otsu fallback because 2-component GMM gate was unavailable"

    if method == "3component_gmm_high":
        if np.isfinite(row.get("gmm3_high_gate", np.nan)):
            return "3component_gmm_high", float(row["gmm3_high_gate"]), "3-component GMM high-vs-rest gate"
        return "upper_tail_gmm_q90", float(row["upper_tail_q90_gate"]), "upper-tail q90 fallback because 3-component GMM was unavailable"

    if method == "3component_gmm_high_or_negative_reference_controlled":
        if np.isfinite(row.get("gmm3_high_gate", np.nan)):
            return "3component_gmm_high", float(row["gmm3_high_gate"]), "negative-reference-aware marker; using 3-component high gate as initial automatic gate"
        return "upper_tail_gmm_q90", float(row["upper_tail_q90_gate"]), "upper-tail q90 fallback"

    if method == "3component_gmm_high_or_upper_tail_q75":
        if np.isfinite(row.get("gmm3_high_gate", np.nan)):
            return "3component_gmm_high", float(row["gmm3_high_gate"]), "clear broad-gradient marker; using high-component gate"
        return "upper_tail_gmm_q75", float(row["upper_tail_q75_gate"]), "upper-tail q75 fallback"

    if method == "upper_tail_gmm_q75":
        if np.isfinite(row.get("upper_tail_q75_gate", np.nan)):
            return "upper_tail_gmm_q75", float(row["upper_tail_q75_gate"]), "upper-tail q75 GMM gate"
        return "robust_q95_mad2", float(row["robust_q95_mad2_gate"]), "robust q95/MAD fallback"

    if method == "upper_tail_gmm_q90":
        if np.isfinite(row.get("upper_tail_q90_gate", np.nan)):
            return "upper_tail_gmm_q90", float(row["upper_tail_q90_gate"]), "upper-tail q90 GMM gate"
        return "robust_q99_mad3", float(row["robust_q99_mad3_gate"]), "robust q99/MAD fallback"

    if method == "upper_tail_gmm_q90_or_robust_q99_mad3":
        q90_frac = float(row.get("upper_tail_q90_pos_frac", np.nan))
        robust_frac = float(row.get("robust_q99_mad3_pos_frac", np.nan))
        if np.isfinite(q90_frac) and q90_frac <= 0.08:
            return "upper_tail_gmm_q90", float(row["upper_tail_q90_gate"]), "diffuse/broad marker; q90 upper-tail fraction was plausible"
        return "robust_q99_mad3", float(row["robust_q99_mad3_gate"]), "diffuse/broad marker; conservative robust tail selected"

    if method == "negative_reference_controlled_or_robust_q99_mad3":
        return "robust_q99_mad3", float(row["robust_q99_mad3_gate"]), "negative/failed marker conservative robust tail"

    # Final fallback.
    if np.isfinite(row.get("gmm3_high_gate", np.nan)):
        return "3component_gmm_high", float(row["gmm3_high_gate"]), "manual-review fallback to 3-component high gate"
    return "robust_q99_mad3", float(row["robust_q99_mad3_gate"]), "manual-review fallback to robust q99/MAD"


def selected_gate_table(
    method_plan: pd.DataFrame,
    adata: ad.AnnData | None = None,
    *,
    layer: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    xmat = expression_matrix(adata, layer=layer) if adata is not None else None
    marker_to_index = {marker: i for i, marker in enumerate(adata.var_names)} if adata is not None else {}

    for _, row in method_plan.iterrows():
        marker = str(row["marker"])
        x_values = None
        if xmat is not None and marker in marker_to_index:
            x_values = clean_log1p_values(xmat[:, marker_to_index[marker]])
        selected_method, gate, source = choose_gate(row, x_values=x_values)
        pos_frac = np.nan
        if np.isfinite(gate) and x_values is not None:
            pos_frac = finite_fraction(x_values, gate)
        elif np.isfinite(gate):
            if selected_method == "2component_gmm":
                pos_frac = row.get("gmm2_pos_frac", np.nan)
            elif selected_method == "3component_gmm_high":
                pos_frac = row.get("gmm3_high_pos_frac", np.nan)
            elif selected_method == "upper_tail_gmm_q75":
                pos_frac = row.get("upper_tail_q75_pos_frac", np.nan)
            elif selected_method == "upper_tail_gmm_q90":
                pos_frac = row.get("upper_tail_q90_pos_frac", np.nan)
            elif selected_method == "robust_q95_mad2":
                pos_frac = row.get("robust_q95_mad2_pos_frac", np.nan)
            elif selected_method == "robust_q99_mad3":
                pos_frac = row.get("robust_q99_mad3_pos_frac", np.nan)
        out_row = {
            "marker": row["marker"],
            "selected_method": selected_method,
            "selected_log1p_gate": gate,
            "selected_raw_gate": float(np.expm1(gate)) if np.isfinite(gate) else np.nan,
            "selected_positive_fraction": pos_frac,
            "gate_source": source,
            "condition_based_method": row["condition_based_method"],
            "review_flags": row.get("review_flags", ""),
            "staining_condition": row["staining_condition"],
            "compartment_pattern": row["compartment_pattern"],
            "expression_condition": row["expression_condition"],
            "artifact_level": row["artifact_level"],
        }
        for col in [
            "preferred_method",
            "target_positive_fraction",
            "reference_positive_fraction",
            "reference_log1p_gate",
            "review_priority",
            "strategy_note",
            "learned_from",
            "inferred_expression_condition",
            "expression_inference_confidence",
            "distribution_dynamic_range",
            "distribution_skewness",
            "candidate_gate_span",
            "candidate_positive_fraction_span",
            "conservative_candidate_positive_fraction_span",
            "distribution_note",
        ]:
            if col in row.index:
                out_row[col] = row.get(col, np.nan)
        rows.append(out_row)
    return pd.DataFrame(rows)


def apply_selected_gates(
    adata: ad.AnnData,
    selected_gates: pd.DataFrame,
    *,
    layer: str | None = None,
    prefix: str = "gate",
) -> ad.AnnData:
    out = adata.copy()
    xmat = expression_matrix(out, layer=layer)
    marker_to_index = {marker: i for i, marker in enumerate(out.var_names)}
    new_obs: dict[str, np.ndarray] = {}
    for _, row in selected_gates.iterrows():
        marker = str(row["marker"])
        if marker not in marker_to_index:
            continue
        raw = np.asarray(xmat[:, marker_to_index[marker]], dtype=float)
        raw = np.where(np.isfinite(raw), raw, np.nan)
        log_values = np.log1p(np.clip(raw, 0, None))
        gate = float(row["selected_log1p_gate"])
        new_obs[f"{prefix}_{marker}_log1p"] = log_values
        new_obs[f"{prefix}_{marker}_positive"] = log_values >= gate
        new_obs[f"{prefix}_{marker}_margin"] = log_values - gate
    if new_obs:
        out.obs = pd.concat([out.obs, pd.DataFrame(new_obs, index=out.obs_names)], axis=1)
    return out


def write_scimap_manual_gates(selected_gates: pd.DataFrame, out_path: Path) -> None:
    scimap_gates = selected_gates[["marker", "selected_log1p_gate"]].rename(
        columns={"marker": "markers", "selected_log1p_gate": "gates"}
    )
    scimap_gates.to_csv(out_path, index=False)


def canonical_channel_names(
    image_path: Path, fallback_markers: list[str] | None = None
) -> list[str]:
    channels = channel_names(image_path, fallback=fallback_markers)
    return [CHANNEL_ALIASES.get(channel, channel) for channel in channels]


def nuclear_channel(channel_names: list[str]) -> str:
    for preferred in ["HOECHST2", "DNA_1", "DAPI"]:
        if preferred in channel_names:
            return preferred
    return channel_names[0]


def read_overview_array(
    tif: tifffile.TiffFile,
    *,
    level_index: int = -1,
    max_dimension: int = 1100,
) -> tuple[np.ndarray, float, float]:
    """Read a bounded CYX overview without loading a flat full-resolution stack."""
    level = tif.series[0].levels[level_index]
    _, level_y, level_x = level.shape
    stride = max(1, math.ceil(max(level_y, level_x) / max_dimension))
    store = level.aszarr()
    try:
        root = zarr.open(store, mode="r")
        array = root["0"] if isinstance(root, zarr.hierarchy.Group) else root
        overview = np.asarray(array[:, ::stride, ::stride])
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()
    _, full_y, full_x = tif.series[0].shape
    return overview, full_y / overview.shape[-2], full_x / overview.shape[-1]


def normalize_image(x: np.ndarray, lo_pct: float = 1, hi_pct: float = 99.8, gamma: float = 0.8) -> np.ndarray:
    x = x.astype(float)
    vals = x[np.isfinite(x)]
    if len(vals) == 0:
        return np.zeros_like(x, dtype=float)
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y**gamma


def save_image_overviews(
    image_path: Path,
    marker_conditions: pd.DataFrame,
    output_dir: Path,
    *,
    sample_id: str,
    overview_level: int = -1,
    channel_markers: list[str] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(image_path) as tif:
        channels = canonical_channel_names(image_path, channel_markers)
        idx = {marker: i for i, marker in enumerate(channels)}
        arr, _, _ = read_overview_array(tif, level_index=overview_level)

    markers = [m for m in marker_conditions["marker"].tolist() if m in idx]
    nuclear = nuclear_channel(channels)
    hoechst = normalize_image(arr[idx[nuclear]], 0.5, 99.7, 0.8)
    cond = marker_conditions.set_index("marker")
    ncols = 7
    nrows = math.ceil(len(markers) / ncols)

    grayscale_path = output_dir / f"{sample_id}_marker_grayscale_overview.png"
    overlay_path = output_dir / f"{sample_id}_marker_hoechst_overlay_overview.png"

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.65 * nrows), dpi=160)
    axes = np.ravel(axes)
    for ax, marker in zip(axes, markers):
        ax.imshow(normalize_image(arr[idx[marker]]), cmap="gray", interpolation="nearest")
        row = cond.loc[marker]
        ax.set_title(
            f"{marker}\n{row.staining_condition}; {row.expression_condition}; art={row.artifact_level}",
            fontsize=7,
        )
        ax.axis("off")
    for ax in axes[len(markers) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(grayscale_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.65 * nrows), dpi=160)
    axes = np.ravel(axes)
    for ax, marker in zip(axes, markers):
        marker_img = normalize_image(arr[idx[marker]], 1, 99.85, 0.75)
        rgb = np.zeros((*marker_img.shape, 3), dtype=float)
        rgb[..., 2] = 0.75 * hoechst
        rgb[..., 1] = 0.12 * hoechst
        rgb[..., 0] = marker_img
        row = cond.loc[marker]
        ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
        ax.set_title(
            f"{marker}\n{row.staining_condition}; {row.expression_condition}; art={row.artifact_level}",
            fontsize=7,
        )
        ax.axis("off")
    for ax in axes[len(markers) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(overlay_path, bbox_inches="tight")
    plt.close(fig)

    return grayscale_path, overlay_path


def choose_crop_windows(
    image_path: Path,
    *,
    crop_size: int = 1536,
    n_crops: int = 3,
    channel_markers: list[str] | None = None,
) -> tuple[list[tuple[int, int]], Path | None]:
    with tifffile.TiffFile(image_path) as tif:
        channels = canonical_channel_names(image_path, channel_markers)
        idx = {marker: i for i, marker in enumerate(channels)}
        low, sy, sx = read_overview_array(tif)

    hoechst_low = normalize_image(low[idx[nuclear_channel(channels)]], 1, 99.5, 1.0)
    low_y, low_x = hoechst_low.shape
    full_y = round(low_y * sy)
    full_x = round(low_x * sx)
    low_win = max(12, int(crop_size / sy / 2))
    bounds = np.linspace(20, max(21, low_y - 20), n_crops + 1, dtype=int)
    windows: list[tuple[int, int]] = []

    for start, stop in zip(bounds[:-1], bounds[1:]):
        best: tuple[float, int, int] | None = None
        for yy in range(max(20, start), min(low_y - 20, stop), 8):
            for xx in range(20, low_x - 20, 8):
                patch = hoechst_low[max(0, yy - low_win) : min(low_y, yy + low_win), max(0, xx - low_win) : min(low_x, xx + low_win)]
                score = float(np.mean(patch > 0.18)) + 0.2 * float(np.mean(patch))
                if best is None or score > best[0]:
                    best = (score, yy, xx)
        if best is None:
            continue
        _, yy, xx = best
        fy = int(np.clip(round(yy * sy - crop_size / 2), 0, full_y - crop_size))
        fx = int(np.clip(round(xx * sx - crop_size / 2), 0, full_x - crop_size))
        windows.append((fy, fx))
    return windows, None


def save_crop_box_overview(
    image_path: Path,
    windows: list[tuple[int, int]],
    output_path: Path,
    *,
    crop_size: int = 1536,
    channel_markers: list[str] | None = None,
) -> Path:
    with tifffile.TiffFile(image_path) as tif:
        channels = canonical_channel_names(image_path, channel_markers)
        idx = {marker: i for i, marker in enumerate(channels)}
        low, sy, sx = read_overview_array(tif)

    hoechst_low = normalize_image(low[idx[nuclear_channel(channels)]], 1, 99.5, 0.8)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(hoechst_low, cmap="gray")
    colors = ["cyan", "yellow", "lime", "orange", "magenta"]
    for i, (fy, fx) in enumerate(windows, start=1):
        color = colors[(i - 1) % len(colors)]
        ax.add_patch(
            plt.Rectangle(
                (fx / sx, fy / sy),
                crop_size / sx,
                crop_size / sy,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        ax.text(fx / sx, fy / sy, f"crop {i}", color=color, fontsize=9, weight="bold")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_review_marker_crops(
    image_path: Path,
    review_markers: list[str],
    output_dir: Path,
    *,
    sample_id: str,
    windows: list[tuple[int, int]],
    crop_size: int = 1536,
    channel_markers: list[str] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with tifffile.TiffFile(image_path) as tif:
        channels = canonical_channel_names(image_path, channel_markers)
        idx = {marker: i for i, marker in enumerate(channels)}
        rootz = zarr.open(tif.series[0].levels[0].aszarr(), mode="r")
        z = rootz["0"] if isinstance(rootz, zarr.hierarchy.Group) else rootz

        markers = [m for m in review_markers if m in idx]
        for crop_i, (fy, fx) in enumerate(windows, start=1):
            hoechst = z[idx[nuclear_channel(channels)], fy : fy + crop_size, fx : fx + crop_size]
            hoechst_n = normalize_image(hoechst, 0.5, 99.7, 0.8)
            ncols = 5
            nrows = math.ceil(len(markers) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), dpi=160)
            axes = np.ravel(axes)
            for ax, marker in zip(axes, markers):
                marker_img = z[idx[marker], fy : fy + crop_size, fx : fx + crop_size]
                marker_n = normalize_image(marker_img, 1, 99.85, 0.75)
                rgb = np.zeros((*marker_n.shape, 3), dtype=float)
                rgb[..., 2] = 0.78 * hoechst_n
                rgb[..., 1] = 0.10 * hoechst_n
                rgb[..., 0] = marker_n
                ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
                ax.set_title(marker, fontsize=9)
                ax.axis("off")
            for ax in axes[len(markers) :]:
                ax.axis("off")
            fig.suptitle(
                f"{sample_id} review markers crop {crop_i}: y={fy}, x={fx}",
                fontsize=11,
                y=0.995,
            )
            fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.975])
            path = output_dir / f"{sample_id}_review_markers_crop{crop_i}_overlay.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def save_histogram_qc(
    method_plan: pd.DataFrame,
    adata: ad.AnnData,
    output_path: Path,
    *,
    layer: str | None = None,
) -> Path:
    xmat = expression_matrix(adata, layer=layer)
    marker_to_index = {marker: i for i, marker in enumerate(adata.var_names)}
    ncols = 5
    nrows = math.ceil(len(method_plan) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 2.7 * nrows), dpi=150)
    axes = np.ravel(axes)

    for ax, (_, row) in zip(axes, method_plan.iterrows()):
        marker = row["marker"]
        x = clean_log1p_values(xmat[:, marker_to_index[marker]])
        ax.hist(x, bins=120, color="0.72", edgecolor="none")
        for label, gate, color in [
            ("2G", row.get("gmm2_gate", np.nan), "#1f77b4"),
            ("3G", row.get("gmm3_high_gate", np.nan), "#9467bd"),
            ("u75", row.get("upper_tail_q75_gate", np.nan), "#2ca02c"),
            ("u90", row.get("upper_tail_q90_gate", np.nan), "#ff7f0e"),
            ("rob", row.get("robust_q99_mad3_gate", np.nan), "#8c564b"),
        ]:
            if np.isfinite(gate):
                ax.axvline(gate, color=color, linewidth=1.0, label=label)
        selected_method, selected_gate, _ = choose_gate(row, x_values=x)
        if np.isfinite(selected_gate):
            ax.axvline(selected_gate, color="#111111", linewidth=1.8, linestyle="--", label="selected")
        ax.set_title(f"{marker}: {selected_method}", fontsize=7)
        ax.tick_params(labelsize=6, length=2)
        ax.legend(fontsize=5, frameon=False, loc="upper right")
    for ax in axes[len(method_plan) :]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_gate_summary_qc(selected_gates: pd.DataFrame, output_path: Path) -> Path:
    """Save a compact overview of positive fractions and review burden."""
    table = selected_gates.copy().iloc[::-1]
    fractions = pd.to_numeric(table["selected_positive_fraction"], errors="coerce").fillna(0.0)
    risk_flags = (
        table["review_flags"]
        .fillna("")
        .astype(str)
        .str.replace("strategy_profile", "", regex=False)
        .str.strip(";")
    )
    flagged = risk_flags.ne("")
    colors = np.where(flagged, "#c46b42", "#277c75")
    height = max(5.0, 0.28 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, height), dpi=160)
    positions = np.arange(len(table))
    ax.barh(positions, fractions * 100, color=colors, height=0.72)
    ax.set_yticks(positions, table["marker"].astype(str), fontsize=8)
    ax.set_xlabel("Selected positive cells (%)")
    ax.set_title("Selected gates: positive fraction and review status")
    ax.grid(axis="x", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    for position, fraction, method in zip(
        positions,
        fractions,
        table["selected_method"].fillna("").astype(str),
    ):
        ax.text(
            fraction * 100 + 0.25,
            position,
            f"{fraction:.1%}  {method}",
            va="center",
            fontsize=6.5,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_gate_spatial_overlays(
    gated_adata: ad.AnnData,
    selected_gates: pd.DataFrame,
    image_path: Path,
    review_markers: list[str],
    output_dir: Path,
    *,
    sample_id: str,
    windows: list[tuple[int, int]],
    crop_size: int = 1536,
    prefix: str = "gate",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    obs = gated_adata.obs
    channel_names = canonical_channel_names(image_path, list(gated_adata.var_names.astype(str)))
    idx = {marker: i for i, marker in enumerate(channel_names)}
    nuclear = nuclear_channel(channel_names)
    markers = [m for m in review_markers if m in idx and f"{prefix}_{m}_positive" in obs.columns]

    with tifffile.TiffFile(image_path) as tif:
        rootz = zarr.open(tif.series[0].levels[0].aszarr(), mode="r")
        z = rootz["0"] if isinstance(rootz, zarr.hierarchy.Group) else rootz
        for crop_i, (fy, fx) in enumerate(windows, start=1):
            ncols = 4
            nrows = math.ceil(len(markers) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), dpi=150)
            axes = np.ravel(axes)
            for ax, marker in zip(axes, markers):
                marker_img = z[idx[marker], fy : fy + crop_size, fx : fx + crop_size]
                hoechst = z[idx[nuclear], fy : fy + crop_size, fx : fx + crop_size]
                marker_n = normalize_image(marker_img, 1, 99.85, 0.75)
                hoechst_n = normalize_image(hoechst, 0.5, 99.7, 0.8)
                rgb = np.zeros((*marker_n.shape, 3), dtype=float)
                rgb[..., 2] = 0.72 * hoechst_n
                rgb[..., 0] = marker_n
                ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
                in_crop = (
                    (obs["Y_centroid"] >= fy)
                    & (obs["Y_centroid"] < fy + crop_size)
                    & (obs["X_centroid"] >= fx)
                    & (obs["X_centroid"] < fx + crop_size)
                )
                pos = in_crop & obs[f"{prefix}_{marker}_positive"].astype(bool)
                if pos.any():
                    ax.scatter(
                        obs.loc[pos, "X_centroid"] - fx,
                        obs.loc[pos, "Y_centroid"] - fy,
                        s=8,
                        c="cyan",
                        linewidths=0,
                        alpha=0.75,
                    )
                gate_row = selected_gates.loc[selected_gates["marker"].eq(marker)].iloc[0]
                ax.set_title(
                    f"{marker}: {gate_row['selected_method']} ({gate_row['selected_positive_fraction']:.1%})",
                    fontsize=8,
                )
                ax.axis("off")
            for ax in axes[len(markers) :]:
                ax.axis("off")
            fig.suptitle(
                f"{sample_id} selected-gate positive cells, crop {crop_i}",
                fontsize=11,
                y=0.995,
            )
            fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.975])
            path = output_dir / f"{sample_id}_selected_gate_positive_overlay_crop{crop_i}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def run_qc(
    paths: SamplePaths,
    *,
    layer: str | None = None,
    max_fit_cells: int | None = None,
    random_state: int = 0,
    strategy_profile_path: Path | None = None,
    review_markers: list[str] | None = None,
    make_image_qc: bool = True,
    make_gate_overlay_qc: bool = True,
    write_gated_h5ad: bool = False,
) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    review_markers = review_markers or DEFAULT_REVIEW_MARKERS
    marker_conditions = read_marker_conditions(paths.marker_condition_path)
    strategy_profile = read_strategy_profile(strategy_profile_path)
    adata = load_adata(paths.adata_path)

    candidates = compute_gate_candidates(
        adata,
        marker_conditions,
        layer=layer,
        max_fit_cells=max_fit_cells,
        random_state=random_state,
    )
    method_plan = add_method_plan(candidates, strategy_profile=strategy_profile)
    selected = selected_gate_table(method_plan, adata=adata, layer=layer)
    gated = apply_selected_gates(adata, selected, layer=layer)

    outputs: dict[str, Path] = {}
    outputs["candidate_table"] = paths.output_dir / f"{paths.sample_id}_marker_condition_evaluation_with_gate_candidates.csv"
    outputs["method_plan"] = paths.output_dir / f"{paths.sample_id}_marker_gating_method_plan.csv"
    outputs["selected_gates"] = paths.output_dir / f"{paths.sample_id}_selected_marker_gates.csv"
    outputs["scimap_manual_gates"] = paths.output_dir / f"{paths.sample_id}_scimap_manual_gates.csv"
    outputs["histograms"] = paths.output_dir / f"{paths.sample_id}_marker_histograms_candidate_gates.png"
    outputs["gate_summary"] = paths.output_dir / f"{paths.sample_id}_selected_gate_summary.png"

    method_plan.to_csv(outputs["candidate_table"], index=False)
    selected.to_csv(outputs["selected_gates"], index=False)
    method_summary_cols = [
        "marker",
        "selected_method",
        "gate_source",
        "selected_log1p_gate",
        "selected_raw_gate",
        "selected_positive_fraction",
        "condition_based_method",
        "review_flags",
    ]
    method_summary_cols.extend(
        [
            col
            for col in [
                "preferred_method",
                "target_positive_fraction",
                "reference_positive_fraction",
                "review_priority",
                "strategy_note",
                "learned_from",
            ]
            if col in selected.columns
        ]
    )
    selected[method_summary_cols].to_csv(outputs["method_plan"], index=False)
    write_scimap_manual_gates(selected, outputs["scimap_manual_gates"])
    save_histogram_qc(method_plan, adata, outputs["histograms"], layer=layer)
    save_gate_summary_qc(selected, outputs["gate_summary"])

    windows: list[tuple[int, int]] = []
    if make_image_qc:
        grayscale, overlay = save_image_overviews(paths.image_path, marker_conditions, paths.output_dir, sample_id=paths.sample_id)
        outputs["image_grayscale_overview"] = grayscale
        outputs["image_overlay_overview"] = overlay
        windows, _ = choose_crop_windows(paths.image_path)
        crop_boxes = paths.output_dir / f"{paths.sample_id}_selected_crop_boxes.png"
        save_crop_box_overview(paths.image_path, windows, crop_boxes)
        outputs["crop_boxes"] = crop_boxes
        save_review_marker_crops(paths.image_path, review_markers, paths.output_dir, sample_id=paths.sample_id, windows=windows)

    if make_gate_overlay_qc:
        if not windows:
            windows, _ = choose_crop_windows(paths.image_path)
        overlay_paths = save_gate_spatial_overlays(
            gated,
            selected,
            paths.image_path,
            review_markers,
            paths.output_dir,
            sample_id=paths.sample_id,
            windows=windows,
        )
        if overlay_paths:
            outputs["selected_gate_overlay_example"] = overlay_paths[0]

    if write_gated_h5ad:
        gated_path = paths.output_dir / f"{paths.sample_id}_autogated.h5ad"
        gated.write_h5ad(gated_path)
        outputs["gated_h5ad"] = gated_path

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--adata-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--marker-condition-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument(
        "--max-fit-cells",
        type=int,
        default=0,
        help="Maximum cells used to fit GMM/Otsu gates. Use 0 to fit with all cells.",
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--strategy-profile", default=None, help="CSV with per-marker preferred gate strategy.")
    parser.add_argument("--review-markers", default=None, help="Comma-separated marker list for crop overlays.")
    parser.add_argument("--skip-image-qc", action="store_true")
    parser.add_argument("--skip-gate-overlay-qc", action="store_true")
    parser.add_argument("--write-gated-h5ad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    paths = default_sample_paths(args.sample_id, project_root)
    paths = SamplePaths(
        sample_id=args.sample_id,
        project_root=project_root,
        adata_path=Path(args.adata_path) if args.adata_path else paths.adata_path,
        image_path=Path(args.image_path) if args.image_path else paths.image_path,
        marker_condition_path=Path(args.marker_condition_path) if args.marker_condition_path else paths.marker_condition_path,
        output_dir=Path(args.output_dir) if args.output_dir else paths.output_dir,
    )
    review_markers = (
        [m.strip() for m in args.review_markers.split(",") if m.strip()]
        if args.review_markers
        else DEFAULT_REVIEW_MARKERS
    )
    outputs = run_qc(
        paths,
        layer=args.layer,
        max_fit_cells=args.max_fit_cells,
        random_state=args.random_state,
        strategy_profile_path=Path(args.strategy_profile) if args.strategy_profile else None,
        review_markers=review_markers,
        make_image_qc=not args.skip_image_qc,
        make_gate_overlay_qc=not args.skip_gate_overlay_qc,
        write_gated_h5ad=args.write_gated_h5ad,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
