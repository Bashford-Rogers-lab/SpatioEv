"""The automatic gate must not call the main population positive.

A 3-component GMM always returns a "high" component. For a marker without a
separate positive peak, that component can be the bulk of the cells, and the
gate then calls most cells positive: on a real 28-FOV testis TMA, 69% of
cells were CD68-positive (0.65% by hand). The selection falls back to the
upper-tail q75 gate when the high gate would call most cells positive.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from spatioev.workflows.marker_gating import (
    GMM3_HIGH_MAX_POSITIVE_FRACTION,
    add_method_plan,
    choose_gate,
    compute_gate_candidates,
    selected_gate_table,
)

CLEAR = {
    "staining_condition": "clear_specific",
    "compartment_pattern": "cytoplasmic",
    "expression_condition": "broad_gradient",
    "artifact_level": "low",
}


def _row(gmm3_fraction: float, **extra) -> pd.Series:
    return pd.Series(
        {
            "marker": "CD68",
            **CLEAR,
            "condition_based_method": "3component_gmm_high_or_upper_tail_q75",
            "gmm3_high_gate": 1.0,
            "gmm3_high_pos_frac": gmm3_fraction,
            "upper_tail_q75_gate": 2.0,
            "robust_q95_mad2_gate": 2.5,
            **extra,
        }
    )


def test_a_high_component_with_most_cells_falls_back_to_upper_tail_q75():
    method, gate, source = choose_gate(_row(0.69))
    assert (method, gate) == ("upper_tail_gmm_q75", 2.0)
    assert "69% of cells" in source


def test_a_minority_high_component_keeps_the_3_component_gate():
    assert choose_gate(_row(0.30))[:2] == ("3component_gmm_high", 1.0)
    at_limit = choose_gate(_row(GMM3_HIGH_MAX_POSITIVE_FRACTION))
    assert at_limit[0] == "3component_gmm_high"


def test_every_plan_that_uses_the_3_component_gate_is_guarded():
    for plan in (
        "3component_gmm_high",
        "3component_gmm_high_or_negative_reference_controlled",
        "3component_gmm_high_or_upper_tail_q75",
        "manual_review",
    ):
        assert choose_gate(_row(0.8, condition_based_method=plan))[0] == "upper_tail_gmm_q75", plan


def test_without_an_upper_tail_gate_the_robust_tail_is_used():
    method, gate, _ = choose_gate(_row(0.8, upper_tail_q75_gate=np.nan))
    assert (method, gate) == ("robust_q95_mad2", 2.5)


def test_a_strategy_profile_that_names_the_3_component_gate_still_gets_it():
    method, gate, _ = choose_gate(_row(0.8, preferred_method="3component_gmm_high"))
    assert (method, gate) == ("3component_gmm_high", 1.0)


def test_the_fallback_is_flagged_for_review():
    candidates = pd.DataFrame([_row(0.8).drop("condition_based_method"), _row(0.3).drop("condition_based_method")])
    candidates["marker"] = ["CD68", "CD45"]
    flags = add_method_plan(candidates).set_index("marker")["review_flags"]
    assert "3GMM_high_is_majority" in flags["CD68"]
    assert "3GMM_high_is_majority" not in flags["CD45"]


def test_a_marker_without_a_positive_peak_is_not_called_mostly_positive():
    # Three levels of one stain, the brightest holding 70% of cells: the
    # 3-component high component is the bulk, not a positive group.
    rng = np.random.default_rng(0)
    log_values = np.concatenate([rng.normal(0.3, 0.1, 500), rng.normal(1.2, 0.15, 2500), rng.normal(2.2, 0.2, 7000)])
    adata = ad.AnnData(np.expm1(np.clip(log_values, 0, None)).reshape(-1, 1).astype(np.float32))
    adata.var_names = ["CD68"]
    conditions = pd.DataFrame([{"marker": "CD68", **CLEAR}])

    candidates = compute_gate_candidates(adata, conditions, random_state=0)
    assert candidates.loc[0, "gmm3_high_pos_frac"] > GMM3_HIGH_MAX_POSITIVE_FRACTION
    selected = selected_gate_table(add_method_plan(candidates), adata)

    assert selected.loc[0, "selected_method"] == "upper_tail_gmm_q75"
    assert selected.loc[0, "selected_positive_fraction"] <= 0.25
    assert "3GMM_high_is_majority" in selected.loc[0, "review_flags"]
