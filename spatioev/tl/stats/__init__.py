"""Spatial statistics.

Split by statistic family:

    ripley   Ripley's K, cross-type K, envelopes, local counts, scales
    moran    Moran's I -- global, local, and cross-variable

All names remain importable directly from this package, so
``from spatioev.tl.stats import morans_i`` is unchanged.
"""

from __future__ import annotations

from .moran import (
    add_local_cross_morans_i,
    add_local_cross_morans_i_between_phenotypes,
    add_local_cross_morans_i_quadrants,
    add_local_morans_i,
    add_local_morans_i_quadrants,
    classify_local_cross_morans_i,
    classify_local_morans_i,
    cross_morans_i,
    cross_morans_i_by_image,
    cross_morans_i_by_image_permutation_test,
    cross_morans_i_feature_matrix,
    cross_morans_i_permutation_test,
    local_cross_morans_i,
    local_morans_i,
    morans_i,
    morans_i_by_image,
    morans_i_by_image_permutation_test,
    morans_i_permutation_test,
    summarize_target_features_around_source_cells,
)
from .ripley import (
    cross_ripley_envelope,
    cross_ripley_envelope_by_phenotype,
    cross_ripley_local_counts,
    cross_ripley_permutation_envelope,
    cross_ripleys_curve,
    cross_ripleys_curve_by_phenotype,
    cross_ripleys_k,
    cross_ripleys_k_all_pairs,
    cross_ripleys_k_by_phenotype,
    ripley_envelope,
    ripley_interaction_scale,
    ripley_local_counts_by_phenotype,
    ripley_spatial_scales,
    ripleys_curve,
    ripleys_k,
    ripleys_k_by_image,
    ripleys_k_by_phenotype,
)

__all__ = [
    "ripleys_k",
    "ripleys_curve",
    "ripley_envelope",
    "ripleys_k_by_image",
    "ripleys_k_by_phenotype",
    "cross_ripleys_k",
    "cross_ripleys_k_by_phenotype",
    "cross_ripleys_k_all_pairs",
    "cross_ripleys_curve",
    "cross_ripley_envelope",
    "cross_ripleys_curve_by_phenotype",
    "cross_ripley_envelope_by_phenotype",
    "cross_ripley_permutation_envelope",
    "ripley_local_counts_by_phenotype",
    "cross_ripley_local_counts",
    "ripley_interaction_scale",
    "ripley_spatial_scales",
    "morans_i",
    "morans_i_permutation_test",
    "morans_i_by_image",
    "morans_i_by_image_permutation_test",
    "local_morans_i",
    "add_local_morans_i",
    "classify_local_morans_i",
    "add_local_morans_i_quadrants",
    "cross_morans_i",
    "cross_morans_i_by_image",
    "cross_morans_i_permutation_test",
    "cross_morans_i_by_image_permutation_test",
    "local_cross_morans_i",
    "add_local_cross_morans_i",
    "classify_local_cross_morans_i",
    "add_local_cross_morans_i_quadrants",
    "summarize_target_features_around_source_cells",
    "cross_morans_i_feature_matrix",
    "add_local_cross_morans_i_between_phenotypes",
]
