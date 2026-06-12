"""Xenium epithelial-niche feature helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from spatioev.tl.pseudotime import score_signed_feature_module


DEFAULT_XENIUM_HISTOLOGY_MODULES: dict[str, list[tuple[float, tuple[str, ...]]]] = {
    "histology__ductal_integrity_score": [
        (1.0, ("duct_lumen__fraction", "duct_lumen__mean")),
        (1.0, ("duct_continuity__score", "duct_continuity__mean")),
        (1.0, ("state__EPCAM_mean", "state__EPCAM")),
        (-1.0, ("interface__stromal_fraction", "surround_prop__fibroblast")),
        (-1.0, ("state__dapi_boundary_irregularity_mean",)),
    ],
    "histology__dysplasia_score": [
        (1.0, ("state__dapi_area_mean", "state__nucleus_area_mean")),
        (1.0, ("state__dapi_texture_entropy_mean",)),
        (1.0, ("state__nucleus_boundary_irregularity_mean",)),
        (1.0, ("interface__immune_fraction", "surround_prop__immune")),
        (-1.0, ("duct_continuity__score", "duct_lumen__fraction")),
    ],
    "histology__invasive_context_score": [
        (1.0, ("interface__stromal_fraction", "surround_prop__fibroblast")),
        (1.0, ("graph_surround__cross_edges_per_niche_cell",)),
        (1.0, ("surround__fibroblast__COL1A1_mean", "surround__COL1A1_mean")),
        (1.0, ("surround__myeloid__LYZ_mean", "surround__LYZ_mean")),
        (-1.0, ("histology__ductal_integrity_score",)),
    ],
}


def available_feature_map(
    columns: Sequence[str],
    candidates: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Resolve canonical feature names to the first available candidate column."""

    available = set(columns)
    resolved: dict[str, str] = {}
    for canonical, options in candidates.items():
        match = next((col for col in options if col in available), None)
        if match is not None:
            resolved[canonical] = match
    return resolved


def score_xenium_histology_modules(
    feature_table: pd.DataFrame,
    module_specs: Mapping[str, Sequence[tuple[float, Sequence[str] | str]]] | None = None,
    min_features: int = 2,
    return_resolved: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Score Xenium epithelial-niche histology modules.

    These scores are intentionally transparent signed averages of standardized
    features. They are useful as interpretable companion axes for pooled
    pseudotime analyses, while users can pass project-specific ``module_specs``
    to override the defaults.
    """

    module_specs = module_specs or DEFAULT_XENIUM_HISTOLOGY_MODULES
    out = pd.DataFrame(index=feature_table.index)
    resolved_rows = []
    for score_name, specs in module_specs.items():
        score, resolved = score_signed_feature_module(
            feature_table,
            specs,
            min_features=min_features,
            return_resolved=True,
        )
        out[score_name] = score
        if not resolved.empty:
            resolved = resolved.copy()
            resolved.insert(0, "score", score_name)
            resolved_rows.append(resolved)

    resolved_df = (
        pd.concat(resolved_rows, ignore_index=True)
        if resolved_rows
        else pd.DataFrame(columns=["score", "feature", "sign"])
    )
    if return_resolved:
        return out, resolved_df
    return out


def add_module_scores(
    feature_table: pd.DataFrame,
    scores: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Return ``feature_table`` with module-score columns appended."""

    out = feature_table.copy()
    for col in scores.columns:
        if col in out.columns and not overwrite:
            raise ValueError(f"{col!r} already exists. Set overwrite=True to replace it.")
        out[col] = scores[col]
    return out


__all__ = [
    "available_feature_map",
    "score_xenium_histology_modules",
    "add_module_scores",
]
