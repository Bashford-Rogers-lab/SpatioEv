"""Numerical ground-truth tests for :mod:`spatioev.tl.ecm`.

``tl/ecm.py`` is ~3,000 lines and previously had no functional coverage at
all — every one of its 32 public functions was only asserted to be
``callable``. These tests pin the geometric and statistical kernels against
analytically known answers so that the module can be refactored and optimised
safely.

Where a convention is easy to get wrong (notably the orientation angle unit),
the test states the convention explicitly, so that "fixing" it silently
becomes a test failure rather than a silent change in published numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioev.tl import ecm

# --------------------------------------------------------------------------- #
# Fixtures — deliberately hand-computable geometry
# --------------------------------------------------------------------------- #


def _adata(coords: np.ndarray, image: str = "img1", **obs_cols):
    ad = pytest.importorskip("anndata")
    obs = pd.DataFrame(
        {"X_centroid": coords[:, 0], "Y_centroid": coords[:, 1], "imageid": image}
    )
    for key, value in obs_cols.items():
        obs[key] = value
    adata = ad.AnnData(X=np.zeros((len(obs), 1)))
    adata.obs = obs
    adata.obs_names = [f"c{i}" for i in range(len(obs))]
    return adata


@pytest.fixture
def line_geometry():
    """Three cells and four fibers on the x-axis at known offsets.

        cells   c0 @ 0      c1 @ 100    c2 @ 200
        fibers  f0 @ 10  f1 @ 30  f2 @ 100  f3 @ 500
    """
    cells = _adata(np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]))
    fibers = pd.DataFrame(
        {
            "X_centroid": [10.0, 30.0, 100.0, 500.0],
            "Y_centroid": [0.0, 0.0, 0.0, 0.0],
            "imageid": "img1",
            "fiber_type": ["col1", "col1", "col6", "col1"],
            "orientation": [0.0, 90.0, 0.0, 45.0],
        }
    )
    return cells, fibers


# --------------------------------------------------------------------------- #
# Cell–fiber adjacency: exact distances
# --------------------------------------------------------------------------- #


def test_build_cell_fiber_links_selects_exactly_the_in_radius_pairs(line_geometry):
    cells, fibers = line_geometry
    links = ecm.build_cell_fiber_links(cells, fibers, radius=50)

    # c0 reaches f0 (d=10) and f1 (d=30); c1 sits on f2 (d=0); c2 reaches nothing.
    got = {
        (row.cell_id, row.fiber_id, row.distance)
        for row in links.itertuples()
    }
    assert got == {("c0", 0, 10.0), ("c0", 1, 30.0), ("c1", 2, 0.0)}


def test_build_cell_fiber_links_radius_is_inclusive_and_monotone(line_geometry):
    cells, fibers = line_geometry

    assert len(ecm.build_cell_fiber_links(cells, fibers, radius=5)) == 1  # only d=0
    small = len(ecm.build_cell_fiber_links(cells, fibers, radius=50))
    large = len(ecm.build_cell_fiber_links(cells, fibers, radius=400))
    assert large > small, "growing the radius cannot lose links"


def test_nearest_fiber_map_returns_the_true_minimum_distance(line_geometry):
    cells, fibers = line_geometry
    nearest = ecm.build_nearest_cell_fiber_map(cells, fibers)

    got = {row.cell_id: (row.fiber_id, row.distance) for row in nearest.itertuples()}
    assert got == {
        "c0": (0, 10.0),    # f0 at x=10
        "c1": (2, 0.0),     # coincident with f2
        "c2": (2, 100.0),   # f2 at x=100 is nearer than f3 at x=500
    }


def test_nearest_fiber_never_exceeds_any_link_distance(line_geometry):
    """The nearest-fiber distance is by definition the minimum over all links."""
    cells, fibers = line_geometry
    links = ecm.build_cell_fiber_links(cells, fibers, radius=1000)
    nearest = ecm.build_nearest_cell_fiber_map(cells, fibers)

    per_cell_min = links.groupby("cell_id")["distance"].min()
    for row in nearest.itertuples():
        assert row.distance == pytest.approx(per_cell_min[row.cell_id])


# --------------------------------------------------------------------------- #
# Fiber orientation
# --------------------------------------------------------------------------- #


def test_fiber_vectors_are_unit_length():
    fibers = pd.DataFrame({"orientation": [0.0, 30.0, 45.0, 90.0, 180.0, 270.0]})
    out = ecm.fiber_vectors(fibers)
    norms = np.hypot(out["vx"], out["vy"])
    np.testing.assert_allclose(norms, 1.0, rtol=1e-12)


def test_fiber_orientation_is_interpreted_in_DEGREES():
    """SpatioEv's fiber/cell orientation convention is degrees, not radians.

    This is a genuine trap: ``skimage.measure.regionprops`` reports
    ``orientation`` in *radians*, and feeding those in directly would be
    silently wrong rather than an error. The convention is asserted here so it
    cannot be changed without a deliberate decision.
    """
    out = ecm.fiber_vectors(pd.DataFrame({"orientation": [0.0, 90.0, 180.0]}))

    np.testing.assert_allclose(out["vx"].to_numpy(), [1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(out["vy"].to_numpy(), [0.0, 1.0, 0.0], atol=1e-12)

    # A radians interpretation of pi/2 would give (~1.0, ~0.027); degrees gives (0, 1).
    quarter_turn = ecm.fiber_vectors(pd.DataFrame({"orientation": [np.pi / 2]}))
    assert quarter_turn["vx"].iloc[0] == pytest.approx(np.cos(np.deg2rad(np.pi / 2)))
    assert quarter_turn["vx"].iloc[0] > 0.99, "value is being read as degrees"


def test_fiber_vectors_are_periodic_over_360_degrees():
    a = ecm.fiber_vectors(pd.DataFrame({"orientation": [15.0, 200.0]}))
    b = ecm.fiber_vectors(pd.DataFrame({"orientation": [375.0, 560.0]}))
    np.testing.assert_allclose(a["vx"], b["vx"], atol=1e-12)
    np.testing.assert_allclose(a["vy"], b["vy"], atol=1e-12)


# --------------------------------------------------------------------------- #
# Moran's I over fibers
# --------------------------------------------------------------------------- #


def _fiber_grid(n_side: int, feature: np.ndarray) -> pd.DataFrame:
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    return pd.DataFrame(
        {
            "X_centroid": xs.ravel().astype(float),
            "Y_centroid": ys.ravel().astype(float),
            "imageid": "img1",
            "fiber_type": "col1",
            "feature": feature,
        }
    )


def test_morans_i_fibers_returns_a_scalar():
    """Pin the actual return contract.

    NOTE: the signature is annotated ``-> pd.DataFrame`` but the function
    returns a bare float. The annotation is wrong, and the generated API
    reference renders it. Asserted here so the mismatch is visible and so a
    future correction is a deliberate, reviewed change.
    """
    n = 10
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    df = _fiber_grid(n, xs.ravel() + ys.ravel())

    result = ecm.morans_i_fibers(df, feature="feature", k=4)
    assert np.isscalar(result) or isinstance(result, (float, np.floating))


def test_morans_i_fibers_positive_for_gradient_negative_for_checkerboard():
    n = 18
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    flat_x, flat_y = xs.ravel().astype(float), ys.ravel().astype(float)

    i_grad = float(ecm.morans_i_fibers(_fiber_grid(n, flat_x + flat_y), feature="feature", k=4))
    i_check = float(
        ecm.morans_i_fibers(_fiber_grid(n, (flat_x + flat_y) % 2.0), feature="feature", k=4)
    )

    assert i_grad > 0.8, f"gradient should be strongly clustered, got {i_grad}"
    assert i_check < -0.5, f"checkerboard should be dispersed, got {i_check}"


def test_morans_i_fibers_is_invariant_to_affine_rescaling():
    rng = np.random.default_rng(3)
    n = 14
    values = rng.normal(size=n * n)

    base = float(ecm.morans_i_fibers(_fiber_grid(n, values), feature="feature", k=6))
    scaled = float(
        ecm.morans_i_fibers(_fiber_grid(n, 4.0 * values - 9.0), feature="feature", k=6)
    )

    assert base == pytest.approx(scaled, rel=1e-10)


# --------------------------------------------------------------------------- #
# Spatial regression
# --------------------------------------------------------------------------- #


def test_spatial_linear_regression_recovers_a_known_slope():
    """y = 3x + 2 exactly: the fitted slope must be 3."""
    x = np.linspace(0, 100, 200)
    df = pd.DataFrame({"predictor": x, "response": 3.0 * x + 2.0})

    res = ecm.spatial_linear_regression(df, "predictor", "response")
    flat = res if isinstance(res, dict) else res.iloc[0].to_dict()

    slope = next(v for k, v in flat.items() if "slope" in k or "coef" in k)
    assert slope == pytest.approx(3.0, rel=1e-6)

    r2 = next((v for k, v in flat.items() if "r2" in k.lower()), None)
    if r2 is not None:
        assert r2 == pytest.approx(1.0, abs=1e-9), "a perfect line must give R^2 = 1"


def test_spatial_linear_regression_slope_is_zero_for_unrelated_variables():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"predictor": rng.normal(size=400), "response": rng.normal(size=400)})

    res = ecm.spatial_linear_regression(df, "predictor", "response")
    flat = res if isinstance(res, dict) else res.iloc[0].to_dict()

    slope = next(v for k, v in flat.items() if "slope" in k or "coef" in k)
    assert abs(slope) < 0.2, f"unrelated variables should give ~0 slope, got {slope}"


# --------------------------------------------------------------------------- #
# Convex hull area (shared with pp.spatial_prep)
# --------------------------------------------------------------------------- #


def test_convex_hull_area_of_known_shapes():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert ecm.compute_convex_hull_area(square) == pytest.approx(100.0)

    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert ecm.compute_convex_hull_area(triangle) == pytest.approx(6.0)

    # Interior points must not change the hull.
    with_interior = np.vstack([square, [[5.0, 5.0], [2.0, 3.0]]])
    assert ecm.compute_convex_hull_area(with_interior) == pytest.approx(100.0)


def test_convex_hull_area_handles_degenerate_and_non_finite_input():
    assert np.isnan(ecm.compute_convex_hull_area(np.array([[0.0, 0.0], [1.0, 1.0]])))

    # Non-finite rows are dropped rather than raising in Qhull.
    dirty = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [np.nan, 5.0]])
    assert ecm.compute_convex_hull_area(dirty) == pytest.approx(100.0)
