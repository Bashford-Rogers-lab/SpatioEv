"""Fiber segmentation (adapted from ark-analysis) and fiber summaries."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi

from spatioev.pp import fibers
from spatioev.tl.ecm import fiber_stats


def synthetic_matrix(shape=(320, 320), n_lines=40, seed=0) -> np.ndarray:
    """Bright line segments of varying width on a noisy background."""
    from skimage.draw import line

    rng = np.random.default_rng(seed)
    image = rng.normal(200, 30, shape).clip(0)
    for _ in range(n_lines):
        r0, c0 = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        angle, length = rng.uniform(0, np.pi), rng.integers(20, 90)
        r1 = int(np.clip(r0 + length * np.sin(angle), 0, shape[0] - 1))
        c1 = int(np.clip(c0 + length * np.cos(angle), 0, shape[1] - 1))
        rr, cc = line(r0, c0, r1, c1)
        image[rr, cc] += rng.uniform(2000, 6000)
    return ndi.gaussian_filter(image, 1.2).astype(np.uint16)


def ark_segment_fibers(data, blur=2, contrast_scaling_divisor=128, fiber_widths=range(1, 10, 2),
                       ridge_cutoff=0.1, sobel_blur=1, min_fiber_size=15):
    """ark.segmentation.fiber_segmentation.segment_fibers with its I/O removed."""
    from skimage.exposure import equalize_adapthist
    from skimage.filters import frangi, sobel, threshold_multiotsu
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    fov_len = data.shape[0]
    data = data.astype("float")
    blurred = ndi.gaussian_filter(data, sigma=blur)
    contrast_adjusted = equalize_adapthist(blurred / np.max(blurred), kernel_size=fov_len / contrast_scaling_divisor)
    ridges = frangi(contrast_adjusted, sigmas=fiber_widths, black_ridges=False) * 10000
    distance_transformed = ndi.gaussian_filter(ndi.distance_transform_edt(ridges > ridge_cutoff), sigma=1)
    threshed = np.zeros_like(distance_transformed)
    thresholds = threshold_multiotsu(distance_transformed, classes=3)
    threshed[distance_transformed < thresholds[0]] = 1
    threshed[distance_transformed > thresholds[1]] = 2
    elevation_map = sobel(ndi.gaussian_filter(distance_transformed, sigma=sobel_blur))
    segmentation = watershed(elevation_map.astype(np.int32), threshed.astype(np.int32)) - 1
    labeled, _ = ndi.label(segmentation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # min_size is deprecated in scikit-image 0.26
        try:
            kept = remove_small_objects(labeled, min_size=min_fiber_size)
        except TypeError:  # releases after min_size was removed
            kept = remove_small_objects(labeled, max_size=min_fiber_size - 1)
    return kept * segmentation


def test_segment_fibers_reproduces_ark_exactly():
    image = synthetic_matrix()
    expected = ark_segment_fibers(image)
    labels = fibers.segment_fibers(image)
    assert expected.max() > 5
    np.testing.assert_array_equal(labels, expected)


def test_blank_image_gives_no_fibers():
    assert fibers.segment_fibers(np.zeros((64, 64))).max() == 0
    labels, table = fibers.segment_fibers_tiled(np.zeros((64, 64)), tile_size=32, overlap=8)
    assert labels.max() == 0 and table.empty


def test_tiled_matches_whole_image_closely():
    image = synthetic_matrix((640, 640), n_lines=140, seed=3)
    kernel = fibers.resolve_clahe_kernel_size(image.shape)
    whole = fibers.segment_fibers(image, clahe_kernel_size=kernel)
    tiled, table = fibers.segment_fibers_tiled(image, tile_size=256, overlap=64, clahe_kernel_size=kernel)
    union = (whole > 0) | (tiled > 0)
    iou = ((whole > 0) & (tiled > 0)).sum() / union.sum()
    assert iou > 0.9
    assert abs((tiled > 0).mean() - (whole > 0).mean()) / (whole > 0).mean() < 0.05
    assert len(table) == tiled.max() == len(np.unique(tiled)) - 1
    assert table.index.is_unique


def test_single_tile_path_is_the_exact_ark_path():
    image = synthetic_matrix()
    labels, table = fibers.segment_fibers_tiled(image, tile_size=4096)
    np.testing.assert_array_equal(labels, fibers.segment_fibers(image))
    assert len(table) == labels.max()


def test_orientation_is_degrees_from_x_axis():
    labels = np.zeros((80, 80), dtype=int)
    labels[10:13, 5:60] = 1  # horizontal
    labels[20:75, 70:73] = 2  # vertical
    for i in range(40):  # down-right in image coordinates
        labels[30 + i, 10 + i : 12 + i] = 3
    table = fibers.fiber_table_from_labels(labels, "img", "COL1").set_index("label")
    assert min(table.loc[1, "orientation"], 180 - table.loc[1, "orientation"]) < 2
    assert abs(table.loc[2, "orientation"] - 90) < 2
    assert abs(table.loc[3, "orientation"] - 45) < 3
    # X is the column, Y the row.
    assert abs(table.loc[1, "X_centroid"] - 32) < 1 and abs(table.loc[1, "Y_centroid"] - 11) < 1


def test_alignment_reproduces_ark_formula_when_not_axial():
    table = pd.DataFrame({
        "imageid": "fov1", "fiber_type": "COL1", "label": [1, 2, 3, 4],
        "orientation_rowaxis_rad": [-30.0, -15.0, 15.0, 0.0],
        "Y_centroid": [0, 3, 1, 2], "X_centroid": [0, 3, 3, 2],
        "major_axis_length": [2, 2, 2, 1.5], "minor_axis_length": [1, 1, 1, 1],
    })
    closest = {1: [3, 2], 2: [3, 1], 3: [2, 1]}
    for k in (1, 2):
        scored = fibers.calculate_fiber_alignment(table, k=k, axial=False).set_index("label")
        assert math.isnan(scored.loc[4, "alignment_score"])  # ratio below 2
        for fiber, neighbours in closest.items():
            angle = table.set_index("label").loc[fiber, "orientation_rowaxis_rad"]
            others = table.set_index("label").loc[neighbours[:k], "orientation_rowaxis_rad"].to_numpy()
            assert scored.loc[fiber, "alignment_score"] == pytest.approx(np.sqrt(np.sum((others - angle) ** 2)) / k)


def test_axial_alignment_treats_near_horizontal_fibers_as_parallel():
    near = np.radians(89.0)
    table = pd.DataFrame({
        "imageid": "fov1", "fiber_type": "COL1", "label": [1, 2],
        "orientation_rowaxis_rad": [near, -near], "Y_centroid": [0, 0], "X_centroid": [0, 5],
        "major_axis_length": [10, 10], "minor_axis_length": [1, 1],
    })
    axial = fibers.calculate_fiber_alignment(table, k=1)["alignment_score"]
    raw = fibers.calculate_fiber_alignment(table, k=1, axial=False)["alignment_score"]
    assert (axial < np.radians(3)).all()
    assert (raw > np.radians(170)).all()


def test_density_and_tile_stats_follow_ark():
    areas = np.arange(100, 110)
    table = pd.DataFrame({"label": range(1, 11), "area": areas})
    assert fiber_stats.calculate_fiber_density(table, 50**2) == pytest.approx(
        (areas.sum() / 50**2 * 100, 10 / 50**2 * 100)
    )

    rng = np.random.default_rng(0)
    table = pd.DataFrame({
        "imageid": "fov1", "fiber_type": "COL1", "label": range(1, 7),
        "Y_centroid": [0, 1, 1, 0, 2, 9], "X_centroid": [0, 1, 0, 1, 2, 9],
        "major_axis_length": rng.uniform(1, 20, 6), "minor_axis_length": rng.uniform(1, 20, 6),
        "area": 1.0, "eccentricity": rng.uniform(0, 1, 6), "euler_number": 1,
        "alignment_score": rng.uniform(0, 1, 6), "orientation": rng.uniform(0, 180, 6),
    })
    tiles = fiber_stats.fiber_tile_stats(table, (16, 12), tile_size=8, min_fiber_num=5)
    assert len(tiles) == 4  # 2 rows x 2 cols, right column is a partial 4-px tile
    first = tiles[(tiles.tile_y == 0) & (tiles.tile_x == 0)].iloc[0]
    assert first["avg_major_axis_length"] == pytest.approx(table.major_axis_length[:5].mean())
    assert first["pixel_density"] == pytest.approx(5 / 64 * 100)
    partial = tiles[(tiles.tile_y == 8) & (tiles.tile_x == 8)].iloc[0]
    assert partial["tile_width"] == 4 and partial["n_fibers"] == 1
    assert math.isnan(partial["avg_major_axis_length"])  # below min_fiber_num, as in ark
    assert partial["area_fraction_pct"] == pytest.approx(1 / 32 * 100)


def test_bright_matrix_counts_as_area_but_not_as_fibers():
    table = pd.DataFrame({
        "imageid": "fov1", "fiber_type": "COL1", "label": range(1, 7),
        "object_type": ["fiber"] * 4 + ["bright_matrix"] * 2,
        "Y_centroid": [0, 0, 0, 0, 0, 0], "X_centroid": [0, 10, 20, 30, 5, 15],
        "orientation_rowaxis_rad": [0.0, 0.0, 0.0, 0.0, 1.5, 1.5], "orientation": [90.0] * 4 + [4.0] * 2,
        "major_axis_length": [20.0] * 4 + [60.0] * 2, "minor_axis_length": [2.0] * 4 + [10.0] * 2,
        "area": [10.0] * 4 + [100.0] * 2, "eccentricity": 0.9, "euler_number": 1,
    })
    scored = fibers.calculate_fiber_alignment(table, k=2)
    assert scored.loc[scored.object_type == "bright_matrix", "alignment_score"].isna().all()
    assert (scored.loc[scored.object_type == "fiber", "alignment_score"] == 0).all()  # bright pieces are not neighbours
    stats = fiber_stats.fiber_image_stats(scored, (100, 100)).iloc[0]
    assert stats["n_fibers"] == 4 and stats["n_bright_matrix"] == 2
    assert stats["fiber_area"] == 240 and stats["area_fraction_pct"] == pytest.approx(2.4)
    assert stats["avg_major_axis_length"] == 20 and stats["orientation_coherence"] == pytest.approx(1)
    assert stats["mean_orientation"] == pytest.approx(90)


def test_region_stats_count_fiber_pixels_inside_each_region():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[5:7, 2:18] = 1  # in region 1 (left half)
    labels[30:32, 22:38] = 2  # in region 2 (right half)
    raster = np.zeros((40, 40), dtype=np.uint8)
    raster[:, :20] = 1
    raster[:, 20:] = 2
    table = fibers.fiber_table_from_labels(labels, "img", "COL1")
    stats = fiber_stats.fiber_region_stats(table, raster, {1: "left", 2: "right"}, labels=labels).set_index("region")
    assert stats.loc["left", "fiber_area"] == 32 and stats.loc["right", "fiber_area"] == 32
    assert stats.loc["left", "area_fraction_pct"] == pytest.approx(32 / 800 * 100)
    assert stats.loc["all_tissue", "n_fibers"] == 2


def test_cell_matrix_proximity_and_enrichment():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 50] = True
    obs = pd.DataFrame({"X_centroid": [60.0, 50.0, 90.0], "Y_centroid": [50.0, 50.0, 50.0]})
    proximity = fiber_stats.cell_matrix_proximity(obs, mask, radius=5)
    assert proximity["distance_to_matrix"].tolist() == pytest.approx([10, 0, 40])
    assert proximity.loc[1, "matrix_fraction"] > 0 and proximity.loc[0, "matrix_fraction"] == 0

    rng = np.random.default_rng(1)
    values = pd.Series(rng.uniform(0, 1, 400))
    labels = pd.Series(np.where(values > 0.8, "near", "other"))
    result = fiber_stats.cell_matrix_enrichment(values, labels, ["near"], n_permutations=199).iloc[0]
    assert result["z_score"] > 5 and result["p_value"] <= 0.01


def test_orientation_coherence():
    assert fiber_stats.orientation_coherence([10, 10, 10])[0] == pytest.approx(1.0)
    coherence, mean = fiber_stats.orientation_coherence([179, 1])
    assert coherence > 0.99 and min(mean, 180 - mean) < 1
    assert fiber_stats.orientation_coherence([0, 45, 90, 135])[0] < 1e-9


def test_include_bright_adds_cross_sections_frangi_misses():
    from skimage.draw import disk

    image = synthetic_matrix((320, 320), n_lines=40, seed=5).astype(float)
    rr, cc = disk((160, 160), 22, shape=image.shape)
    image[rr, cc] += 5000  # a bright, filled cross-section
    image = image.astype(np.uint16)
    plain = fibers.segment_fibers(image)
    bright = fibers.segment_fibers(image, include_bright=True)
    disk_mask = np.zeros(image.shape, bool)
    disk_mask[rr, cc] = True
    assert (bright[disk_mask] > 0).mean() > 0.9 > (plain[disk_mask] > 0).mean()
    assert np.array_equal(bright[plain > 0], plain[plain > 0])  # ark's fibers are kept as they are
    assert bright.max() > plain.max()  # added matrix becomes new objects
    _, steps = fibers.segment_fibers(image, include_bright=True, return_steps=True)
    assert steps["bright"][disk_mask].mean() > 0.9


def test_include_bright_is_consistent_when_tiled():
    from skimage.draw import disk

    image = synthetic_matrix((640, 640), n_lines=140, seed=3).astype(float)
    for centre in ((150, 200), (420, 470)):
        rr, cc = disk(centre, 25, shape=image.shape)
        image[rr, cc] += 6000
    image = image.astype(np.uint16)
    kernel = fibers.resolve_clahe_kernel_size(image.shape)
    whole = fibers.segment_fibers(image, clahe_kernel_size=kernel, include_bright=True)
    tiled, table = fibers.segment_fibers_tiled(image, tile_size=256, overlap=64, clahe_kernel_size=kernel, include_bright=True)
    assert abs((tiled > 0).mean() - (whole > 0).mean()) / (whole > 0).mean() < 0.05
    assert len(table) == len(np.unique(tiled)) - 1


def test_local_window_fills_medium_bright_patches():
    from skimage.draw import disk

    image = synthetic_matrix((400, 400), n_lines=60, seed=7).astype(float)
    rr, cc = disk((120, 120), 15, shape=image.shape)
    image[rr, cc] += 9000  # small, very bright: brightest class
    rr, cc = disk((280, 280), 40, shape=image.shape)
    image[rr, cc] += 1500  # large, medium-bright patch beside dark background
    image = image.astype(np.uint16)
    patch = np.zeros(image.shape, bool)
    patch[rr, cc] = True
    global_only = fibers.segment_fibers(image, include_bright=True)
    local = fibers.segment_fibers(image, include_bright=True, bright_local_window=201)
    assert (local[patch] > 0).mean() > 0.8 > (global_only[patch] > 0).mean()
    background = np.ones(image.shape, bool)
    background[200:360, 200:360] = False
    assert (local[background] > 0).mean() < (global_only[background] > 0).mean() + 0.05  # no flooding elsewhere


def test_bright_cuts_ignore_a_few_saturated_spots():
    rng = np.random.default_rng(0)
    sample = np.concatenate([rng.normal(100, 10, 6000), rng.normal(400, 30, 3000), rng.normal(1200, 80, 990)])
    clean = fibers.bright_matrix_cuts(sample[None, :], sample_stride=1)
    saturated = fibers.bright_matrix_cuts(np.concatenate([sample, np.full(10, 60000.0)])[None, :], sample_stride=1)
    assert 400 < clean[1] < 1200
    assert abs(saturated[1] - clean[1]) < 0.15 * clean[1]  # the 0.1% of saturated pixels do not become the bright class


def test_object_type_marks_added_bright_matrix_and_qc_figure_renders():
    import matplotlib

    matplotlib.use("Agg")
    from skimage.draw import disk

    from spatioev.pl.fibers import plot_fiber_qc

    image = synthetic_matrix((640, 640), n_lines=140, seed=3).astype(float)
    rr, cc = disk((300, 300), 30, shape=image.shape)
    image[rr, cc] += 6000
    image = image.astype(np.uint16)
    plain, plain_table = fibers.segment_fibers_tiled(image, tile_size=4096)
    assert (plain_table["object_type"] == "fiber").all()

    labels, table = fibers.segment_fibers_tiled(image, tile_size=4096, include_bright=True)
    fiber_mask = np.isin(labels, table.loc[table["object_type"] == "fiber", "label"])
    bright_labels = table.loc[table["object_type"] == "bright_matrix", "label"].to_numpy()
    bright_mask = np.isin(labels, bright_labels)
    assert np.array_equal(fiber_mask, plain > 0)  # "fiber" = exactly ark's objects
    assert bright_mask.any() and not (bright_mask & (plain > 0)).any()  # added matrix lies outside them

    tiled, tiled_table = fibers.segment_fibers_tiled(image, tile_size=256, overlap=64, include_bright=True)
    assert set(tiled_table["object_type"]) == {"fiber", "bright_matrix"}

    figure = plot_fiber_qc(image, labels, bright_labels, title="test", crop_size=200)
    assert len(figure.axes) >= 4
    matplotlib.pyplot.close(figure)
