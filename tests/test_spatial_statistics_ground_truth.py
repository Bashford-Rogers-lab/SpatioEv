"""Numerical ground-truth tests for the core spatial statistics.

The rest of the suite mostly asserts that results are non-empty or finite,
which cannot catch a sign error or a mis-normalised weight matrix. These tests
pin the statistics against analytically known values and known-sign
configurations, so that the optimisation work (precomputing and reusing the
spatial weight matrix) is verifiable rather than hopeful.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatioev.tl.stats import (
    morans_i,
    morans_i_permutation_test,
    ripleys_k,
)

# --------------------------------------------------------------------------- #
# Ripley's K
# --------------------------------------------------------------------------- #


def test_ripleys_k_expectation_is_exactly_pi_r_squared():
    """K_expected is the CSR closed form and must equal pi * r^2 exactly."""
    rng = np.random.default_rng(12345)
    coords = rng.uniform(0, 1000.0, size=(500, 2))

    for radius in (10.0, 40.0, 125.0):
        res = ripleys_k(coords, radius=radius)
        assert res["K_expected"] == pytest.approx(np.pi * radius**2, rel=1e-12)


def test_ripleys_k_on_poisson_process_is_close_to_csr():
    """For complete spatial randomness, K_observed ~ K_expected, so L - r ~ 0."""
    rng = np.random.default_rng(12345)
    side, radius = 1000.0, 40.0
    coords = rng.uniform(0, side, size=(3000, 2))

    res = ripleys_k(coords, radius=radius)

    assert res["K_observed"] == pytest.approx(res["K_expected"], rel=0.2), (
        f"CSR K_observed={res['K_observed']:.1f} vs expected={res['K_expected']:.1f}"
    )
    # L - r is the variance-stabilised form; near zero under CSR.
    assert abs(res["L_minus_r"]) < 0.15 * radius


def test_ripleys_k_separates_clustered_from_random_and_dispersed():
    rng = np.random.default_rng(7)
    side, radius = 1000.0, 40.0

    random_pts = rng.uniform(0, side, size=(1600, 2))

    # Tight clusters around 16 seeds -> strong clustering.
    seeds = rng.uniform(100, side - 100, size=(16, 2))
    clustered = np.repeat(seeds, 100, axis=0) + rng.normal(0, 12, size=(1600, 2))

    # A regular lattice -> dispersion.
    g = np.linspace(0, side, 40)
    xs, ys = np.meshgrid(g, g)
    dispersed = np.column_stack([xs.ravel(), ys.ravel()])

    k_random = ripleys_k(random_pts, radius=radius)["L_minus_r"]
    k_clustered = ripleys_k(clustered, radius=radius)["L_minus_r"]
    k_dispersed = ripleys_k(dispersed, radius=radius)["L_minus_r"]

    assert k_clustered > k_random, "clustered pattern must exceed CSR"
    assert k_dispersed < k_clustered, "lattice must be less clustered than clusters"


# --------------------------------------------------------------------------- #
# Moran's I
# --------------------------------------------------------------------------- #


def _grid(n_side: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    return np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])


def test_morans_i_positive_for_smooth_spatial_gradient():
    """A monotone gradient is maximally autocorrelated: I should be near +1."""
    coords = _grid(20)
    values = coords[:, 0] + coords[:, 1]

    i = morans_i(coords, values, k=4)
    assert i > 0.8, f"gradient should give strong positive autocorrelation, got {i}"


def test_morans_i_negative_for_checkerboard():
    """A checkerboard alternates high/low between neighbours: I must be negative."""
    coords = _grid(20)
    values = (coords[:, 0] + coords[:, 1]) % 2.0

    i = morans_i(coords, values, k=4)
    assert i < -0.5, f"checkerboard should give negative autocorrelation, got {i}"


def test_morans_i_near_expected_value_for_random_field():
    """Under no autocorrelation, E[I] = -1/(n-1), which is ~0 for large n."""
    rng = np.random.default_rng(3)
    coords = _grid(25)
    n = coords.shape[0]
    values = rng.normal(size=n)

    i = morans_i(coords, values, k=8)
    expected = -1.0 / (n - 1)
    assert abs(i - expected) < 0.1, f"random field I={i}, expected ~{expected}"


def test_morans_i_is_invariant_to_affine_rescaling_of_values():
    """Moran's I is standardised: a*x + b must not change it."""
    rng = np.random.default_rng(11)
    coords = _grid(15)
    values = rng.normal(size=coords.shape[0])

    base = morans_i(coords, values, k=6)
    scaled = morans_i(coords, 3.5 * values + 17.0, k=6)
    assert scaled == pytest.approx(base, rel=1e-10)


def test_morans_i_matches_explicit_formula():
    """Pin the implementation against a direct, independent computation."""
    from sklearn.neighbors import kneighbors_graph

    rng = np.random.default_rng(5)
    coords = _grid(12)
    values = rng.normal(size=coords.shape[0])
    k = 5

    got = morans_i(coords, values, k=k)

    n = len(values)
    W = kneighbors_graph(coords, k, mode="connectivity", include_self=False)
    x = values - values.mean()
    expected = (n / W.sum()) * float(x @ (W @ x)) / float((x**2).sum())

    assert got == pytest.approx(expected, rel=1e-12)


def test_morans_i_returns_nan_on_degenerate_input():
    coords = _grid(6)
    assert np.isnan(morans_i(coords, np.ones(coords.shape[0]), k=4))  # zero variance
    assert np.isnan(morans_i(coords[:2], np.array([1.0, 2.0]), k=4))  # n < 3


# --------------------------------------------------------------------------- #
# Permutation inference
# --------------------------------------------------------------------------- #


def test_permutation_p_value_is_a_valid_probability():
    rng = np.random.default_rng(2)
    coords = _grid(14)
    values = rng.normal(size=coords.shape[0])

    res = morans_i_permutation_test(coords, values, k=6, n_sim=199, random_state=0)

    assert 0.0 < res["p_value"] <= 1.0
    assert res["n_sim"] > 0
    assert np.isfinite(res["observed"])


def test_permutation_null_is_centred_on_theoretical_expectation():
    """The permutation null mean must sit near E[I] = -1/(n-1)."""
    rng = np.random.default_rng(4)
    coords = _grid(16)
    n = coords.shape[0]
    values = rng.normal(size=n)

    res = morans_i_permutation_test(coords, values, k=8, n_sim=399, random_state=1)

    assert res["null_mean"] == pytest.approx(-1.0 / (n - 1), abs=0.05)
    assert res["null_std"] > 0


def test_permutation_detects_strong_autocorrelation():
    """A gradient must be significant; a shuffled field must not be."""
    coords = _grid(16)
    gradient = coords[:, 0] + coords[:, 1]

    strong = morans_i_permutation_test(coords, gradient, k=6, n_sim=199, random_state=0)
    assert strong["p_value"] <= 0.01
    assert strong["z_score"] > 3

    rng = np.random.default_rng(9)
    noise = rng.normal(size=coords.shape[0])
    weak = morans_i_permutation_test(coords, noise, k=6, n_sim=199, random_state=0)
    assert weak["p_value"] > 0.05


def test_batched_and_scalar_morans_i_agree_exactly():
    """The vectorised permutation path must equal the one-at-a-time path.

    This is the safety net for the weight-matrix reuse optimisation: the
    permutation test evaluates blocks of shuffled value vectors through a
    single sparse product, and that must be indistinguishable from calling
    ``morans_i`` once per simulation.
    """
    from spatioev._core.neighbors import knn_weights, morans_i_batch

    rng = np.random.default_rng(17)
    coords = _grid(18)
    values = rng.normal(size=coords.shape[0])
    W = knn_weights(coords, 8)

    V = np.column_stack([rng.permutation(values) for _ in range(64)])
    batched = morans_i_batch(W, V)
    one_by_one = np.array([morans_i(coords, V[:, j], W=W) for j in range(V.shape[1])])

    np.testing.assert_allclose(batched, one_by_one, rtol=0, atol=1e-12)


def test_precomputed_weights_match_internally_built_ones():
    from spatioev._core.neighbors import knn_weights

    rng = np.random.default_rng(19)
    coords = _grid(14)
    values = rng.normal(size=coords.shape[0])

    W = knn_weights(coords, 6)
    assert morans_i(coords, values, W=W) == pytest.approx(
        morans_i(coords, values, k=6), rel=1e-12
    )


def test_knn_weights_row_normalisation():
    from spatioev._core.neighbors import knn_weights

    coords = _grid(10)
    W = knn_weights(coords, 4, normalize=True)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-12)

    binary = knn_weights(coords, 4, normalize=False)
    assert set(np.unique(binary.data)) == {1.0}


def test_permutation_test_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(6)
    coords = _grid(12)
    values = rng.normal(size=coords.shape[0])

    a = morans_i_permutation_test(coords, values, k=5, n_sim=99, random_state=42)
    b = morans_i_permutation_test(coords, values, k=5, n_sim=99, random_state=42)

    assert a["p_value"] == b["p_value"]
    assert a["null_mean"] == pytest.approx(b["null_mean"], rel=1e-12)
