"""TWOMBLI-style architecture metrics on shapes with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spatioev.tl.ecm import architecture as arch


def test_straight_and_diagonal_line_length():
    horizontal = np.zeros((40, 220), bool)
    horizontal[20, 10:210] = True
    features = arch.skeleton_features(horizontal, 5, 10)
    assert features["length_weight"].sum() == pytest.approx(199)
    assert features["endpoints"].sum() == 2 and len(features["branchpoints"]) == 0
    assert np.allclose(features["curvature"][:, 2], 0)

    from skimage.draw import line

    diagonal = np.zeros((120, 120), bool)
    rr, cc = line(10, 10, 109, 109)
    diagonal[rr, cc] = True
    assert arch.skeleton_features(diagonal, 5, 10)["length_weight"].sum() == pytest.approx(99 * math.sqrt(2))


def test_cross_has_one_branchpoint_and_four_ends():
    cross = np.zeros((101, 101), bool)
    cross[50, 10:91] = True
    cross[10:91, 50] = True
    features = arch.skeleton_features(cross, 5, 10)
    assert features["endpoints"].sum() == 4
    assert len(features["branchpoints"]) == 1
    assert len(features["segments"]) == 4


def test_short_spur_is_pruned():
    shape = np.zeros((40, 220), bool)
    shape[25, 10:210] = True
    shape[22:25, 100] = True  # 3-pixel spur
    unpruned = arch.skeleton_features(shape, 0, 10)
    pruned = arch.skeleton_features(shape, 5, 10)
    assert unpruned["endpoints"].sum() == 3 and len(unpruned["branchpoints"]) == 1
    assert pruned["endpoints"].sum() == 2 and len(pruned["branchpoints"]) == 0


def test_circle_curvature_is_one_over_radius():
    from skimage.draw import circle_perimeter

    circle = np.zeros((200, 200), bool)
    rr, cc = circle_perimeter(100, 100, 60)
    circle[rr, cc] = True
    curvature = arch.skeleton_features(circle, 5, 15)["curvature"][:, 2]
    assert curvature.mean() == pytest.approx(1 / 60, rel=0.02)


def test_box_counting_dimension_orders_line_plane_and_fractal():
    line_image = np.zeros((900, 900), bool)
    line_image[450, :] = True
    carpet = np.ones((1, 1), bool)
    for _ in range(5):
        empty = np.zeros_like(carpet)
        carpet = np.block([[carpet, carpet, carpet], [carpet, empty, carpet], [carpet, carpet, carpet]])
    d_line = arch.box_counting_dimension(line_image)
    d_plane = arch.box_counting_dimension(np.ones((900, 900), bool))
    d_carpet = arch.box_counting_dimension(carpet)
    # AnaMorf's estimator reads slightly low; the ordering and rough values hold.
    assert d_line == pytest.approx(1.0, abs=0.06)
    assert d_plane == pytest.approx(2.0, abs=0.1)
    assert d_line < d_carpet < d_plane
    assert math.isnan(arch.box_counting_dimension(np.ones((30, 30), bool)))  # too small for the scale range


def test_lacunarity_definitions():
    quarter = np.zeros((100, 100), bool)
    quarter[:, :25] = True
    assert arch.anamorf_lacunarity(quarter) == pytest.approx(2.0)  # |1/0.25 - 2|

    from skimage.draw import disk

    rng = np.random.default_rng(0)
    scattered = rng.random((400, 400)) < 0.1
    clumped = np.zeros((400, 400), bool)
    for _ in range(40):
        centre = rng.integers(20, 380, 2)
        rr, cc = disk(tuple(centre), 9, shape=clumped.shape)
        clumped[rr, cc] = True
    assert arch.gliding_box_lacunarity(np.ones((100, 100), bool), 20) == pytest.approx(1.0)
    assert arch.gliding_box_lacunarity(scattered, 20) < 1.1
    assert arch.gliding_box_lacunarity(clumped, 20) > 3


def test_orientation_coherency_and_direction():
    horizontal = np.zeros((200, 200), bool)
    horizontal[::10, :] = True
    coherency, angle = arch.orientation_coherency(horizontal)
    assert coherency > 0.95 and min(angle, 180 - angle) < 1
    coherency, angle = arch.orientation_coherency(horizontal.T.copy())
    assert coherency > 0.95 and abs(angle - 90) < 1
    rng = np.random.default_rng(1)
    assert arch.orientation_coherency(rng.random((200, 200)) < 0.1)[0] < 0.05


def test_hdm_threshold_follows_twombli_scaling():
    intensity = np.zeros((100, 100))
    intensity[:, 50:] = 1000
    threshold = arch.hdm_threshold(intensity, max_display_hdm=200, saturation_pct=0)
    assert threshold == pytest.approx(1000 * (1 - 200 / 255))
    assert (intensity > threshold).mean() == pytest.approx(0.5)


def test_groups_and_tiling_agree():
    from skimage.draw import line

    rng = np.random.default_rng(2)
    mask = np.zeros((700, 700), bool)
    for _ in range(120):
        r0, c0 = rng.integers(0, 700, 2)
        angle, length = rng.uniform(0, np.pi), rng.integers(30, 120)
        rr, cc = line(int(r0), int(c0), int(np.clip(r0 + length * np.sin(angle), 0, 699)), int(np.clip(c0 + length * np.cos(angle), 0, 699)))
        mask[rr, cc] = True
    mask = np.asarray(arch.ndi.binary_dilation(mask, iterations=1))
    intensity = mask * 1000.0 + rng.normal(100, 10, mask.shape)
    groups = np.ones(mask.shape, dtype=int)
    groups[:, 350:] = 2
    names = {1: "left", 2: "right"}
    threshold = arch.hdm_threshold(intensity)
    single = arch.architecture_by_group(mask, groups, names, intensity, threshold, 0.5, 6, 10, 10).set_index("group")
    tiled = arch.matrix_architecture_tiled(
        mask, mask.shape, lambda y0, y1, x0, x1: groups[y0:y1, x0:x1], names, intensity, threshold, 0.5, 6, 10, 10, tile_size=256,
    ).set_index("group")
    for metric in ("total_length_mm", "endpoints", "branchpoints", "hdm_pct", "alignment_coherency", "fiber_area_fraction_pct"):
        assert tiled.loc["all_tissue", metric] == pytest.approx(single.loc["all_tissue", metric], rel=0.03)
    for group in ("left", "right", "all_tissue"):
        assert tiled.loc[group, "fractal_dimension"] == pytest.approx(single.loc[group, "fractal_dimension"])
    # The two halves add up to the whole.
    assert single.loc[["left", "right"], "total_length_mm"].sum() == pytest.approx(single.loc["all_tissue", "total_length_mm"])
    assert single.loc["all_tissue", "tissue_area_mm2"] == pytest.approx(700 * 700 * 0.0005**2)
