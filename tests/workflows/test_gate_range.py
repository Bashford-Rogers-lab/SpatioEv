"""The napari gate slider must offer useful travel in both directions.

The range is derived from empirical positivity around the calculated gate,
which keeps resolution where it matters. On its own that collapses for rare
markers: when the gate sits inside the dominant negative mode, the reference
quantiles all land nearby and the slider covered roughly a quarter of the
data, so the gate could not be pushed stricter at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatioev.workflows.marker_gating_review import (
    GATE_RANGE_MIN_SPAN_FRACTION,
    adaptive_gate_range,
    gate_to_slider,
    slider_to_gate,
)


def _rare_marker() -> np.ndarray:
    rng = np.random.default_rng(0)
    raw = np.concatenate([rng.lognormal(0, 0.3, 970), rng.lognormal(2.6, 0.3, 30)])
    return np.log1p(raw)


def _common_marker() -> np.ndarray:
    rng = np.random.default_rng(1)
    raw = np.concatenate([rng.lognormal(0, 0.3, 700), rng.lognormal(2.2, 0.4, 300)])
    return np.log1p(raw)


@pytest.mark.parametrize("values", [_rare_marker(), _common_marker()])
def test_slider_covers_a_useful_share_of_the_data(values):
    gate = float(np.quantile(values, 0.75))
    low, high = adaptive_gate_range(values, gate)
    span = float(values.max() - values.min())

    assert (high - low) / span >= 0.5, "slider should span at least half the data"
    assert low <= gate <= high, "the calculated gate must remain reachable"


def test_rare_marker_range_is_wider_than_positivity_alone_would_give():
    """The regression: the floor is what rescues rare markers."""
    values = _rare_marker()
    gate = float(np.quantile(values, 0.75))

    without_floor = adaptive_gate_range(values, gate, min_span_fraction=0.0)
    with_floor = adaptive_gate_range(values, gate)

    assert (with_floor[1] - with_floor[0]) > 2 * (without_floor[1] - without_floor[0])


def test_width_is_monotone_in_the_fraction():
    values = _rare_marker()
    gate = float(np.quantile(values, 0.75))

    widths = [
        adaptive_gate_range(values, gate, min_span_fraction=f)[1]
        - adaptive_gate_range(values, gate, min_span_fraction=f)[0]
        for f in (0.0, 0.3, 0.6, 0.9)
    ]
    assert widths == sorted(widths), "raising the fraction must not narrow the range"


def test_range_never_escapes_the_observed_data():
    values = _rare_marker()
    gate = float(np.quantile(values, 0.75))
    low, high = adaptive_gate_range(values, gate, min_span_fraction=5.0)

    assert low >= values.min()
    assert high <= values.max()


def test_slider_round_trip_preserves_the_gate():
    """Widening must not cost the ability to land back on the calculated gate."""
    values = _rare_marker()
    gate = float(np.quantile(values, 0.75))
    low, high = adaptive_gate_range(values, gate)

    recovered = slider_to_gate(gate_to_slider(gate, low, high), low, high)
    assert recovered == pytest.approx(gate, abs=(high - low) / 1000)


def test_degenerate_inputs_are_handled():
    assert adaptive_gate_range(np.array([]), 1.0) == (0.0, 2.0)

    flat = np.full(100, 2.0)
    low, high = adaptive_gate_range(flat, 2.0)
    assert high > low, "a constant marker must still yield a usable range"


def test_default_fraction_is_the_documented_one():
    assert GATE_RANGE_MIN_SPAN_FRACTION == 0.60
