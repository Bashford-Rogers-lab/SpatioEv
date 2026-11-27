"""Averaging clustering markers over neighbouring cells (page 01)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatioev.workflows.clustering import (
    average_over_neighbours,
    neighbourhood_means,
    neighbourhood_radius_px,
)


def _cells(x, values, markers, imageid=None):
    obs = pd.DataFrame(
        {"X_centroid": np.asarray(x, dtype=float), "Y_centroid": 0.0},
        index=[f"c{i}" for i in range(len(x))],
    )
    if imageid is not None:
        obs["imageid"] = imageid
    adata = ad.AnnData(X=np.asarray(values, dtype=np.float32), obs=obs)
    adata.var_names = markers
    return adata


def test_neighbourhood_means_stay_within_each_fov():
    coords = np.array([[0, 0], [1, 0], [5, 0], [0, 0]], dtype=float)
    values = np.array([[0.0], [2.0], [10.0], [100.0]])
    # The fourth cell sits on top of the first, but in another FOV.
    groups = np.array(["fov1", "fov1", "fov1", "fov2"])

    means, counts = neighbourhood_means(values, coords, groups, radius=1.5)

    np.testing.assert_allclose(means[:, 0], [1.0, 1.0, 10.0, 100.0])
    assert counts.tolist() == [2, 2, 1, 1]


def test_neighbourhood_means_skip_missing_values():
    coords = np.array([[0, 0], [1, 0]], dtype=float)
    values = np.array([[np.nan], [2.0]])

    means, _ = neighbourhood_means(values, coords, np.zeros(2), radius=1.5)

    np.testing.assert_allclose(means[:, 0], [2.0, 2.0])


def test_markers_kept_per_cell_are_not_averaged():
    source = _cells(
        x=[0, 5, 50],
        values=[[0, 1, 7], [4, 2, 7], [8, 3, 7]],
        markers=["SOX9", "SMA", "CD45"],
    )
    config = {
        "markers": ["SOX9", "SMA"],
        "per_cell_markers": ["SMA"],
        # 3.25 µm at 0.325 µm/px is 10 px: the first two cells are neighbours.
        "neighbourhood_um": 3.25,
        "pixel_size_um": 0.325,
    }

    averaged, mean_cells = average_over_neighbours(source, config)

    assert list(averaged.var_names) == ["SOX9", "SMA"]
    np.testing.assert_allclose(averaged[:, "SOX9"].X.ravel(), [2, 2, 8])
    np.testing.assert_allclose(averaged[:, "SMA"].X.ravel(), [1, 2, 3])
    assert mean_cells == pytest.approx(5 / 3)
    np.testing.assert_allclose(source[:, "SOX9"].X.ravel(), [0, 4, 8])


def test_averaging_is_off_by_default():
    source = _cells(x=[0, 1], values=[[0], [2]], markers=["SOX9"])

    averaged, mean_cells = average_over_neighbours(source, {"markers": ["SOX9"]})

    assert averaged is source
    assert mean_cells is None
    assert neighbourhood_radius_px({"neighbourhood_um": 0, "pixel_size_um": 0.325}) is None


def test_averaging_needs_pixel_size_coordinates_and_known_markers():
    source = _cells(x=[0, 1], values=[[0, 1], [2, 3]], markers=["SOX9", "SMA"])
    config = {"markers": ["SOX9", "SMA"], "neighbourhood_um": 15}

    with pytest.raises(ValueError, match="pixel size"):
        average_over_neighbours(source, config)
    with pytest.raises(ValueError, match="MYH11"):
        average_over_neighbours(source, {**config, "pixel_size_um": 0.5, "per_cell_markers": ["MYH11"]})
    without_coordinates = ad.AnnData(X=source.X.copy(), var=source.var.copy())
    with pytest.raises(KeyError, match="X_centroid"):
        average_over_neighbours(without_coordinates, {**config, "pixel_size_um": 0.5})


def test_averaged_clusters_follow_tissue_regions_that_single_cells_miss(tmp_path):
    """Only some cells of each region carry its marker, as in testis interstitium.

    Per cell, the unmarked cells of both regions look alike and cluster
    together. Averaged over neighbours, every cell takes on its region's
    marker and the clusters follow the regions.
    """
    pytest.importorskip("scanpy")
    from spatioev.workflows.clustering import run_level0

    rng = np.random.default_rng(1)
    frames = []
    for fov in ["fov1", "fov2"]:
        n = 600
        x, y = rng.uniform(0, 300, n), rng.uniform(0, 300, n)
        left = x < 150
        marked = rng.random(n) < 0.4
        values = rng.normal(0, 0.3, (n, 3))
        values[:, 0] += np.where(left & marked, 3, 0)  # tubule marker
        values[:, 1] += np.where(~left & marked, 3, 0)  # interstitium marker
        frame = pd.DataFrame(values, columns=["T", "L", "N"])
        frame["X_centroid"], frame["Y_centroid"] = x, y
        frame["imageid"], frame["region"] = fov, np.where(left, "left", "right")
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.index = [f"c{i}" for i in range(len(table))]
    source = ad.AnnData(X=table[["T", "L", "N"]].to_numpy(np.float32), obs=table.drop(columns=["T", "L", "N"]))
    source.var_names = ["T", "L", "N"]
    adata_path = tmp_path / "cells.h5ad"
    source.write_h5ad(adata_path)

    def purity(radius_um):
        output_dir = tmp_path / f"r{radius_um:g}"
        config = {
            "sample_id": "S",
            "adata_path": str(adata_path),
            "output_dir": str(output_dir),
            "markers": ["T", "L", "N"],
            "resolution": 0.3,
            "n_neighbors": 15,
            "scale": True,
            "scale_within_image": True,
            "neighbourhood_um": radius_um,
            "pixel_size_um": 0.5,
        }
        outputs = run_level0(config, output_dir / "status.json")
        labels = ad.read_h5ad(outputs["clustered_h5ad"]).obs["leiden"]
        counts = pd.crosstab(labels, table.loc[labels.index, "region"])
        return counts.max(axis=1).sum() / counts.to_numpy().sum(), outputs

    per_cell, per_cell_outputs = purity(0)
    averaged, averaged_outputs = purity(10)  # 20 px: about eight cells

    assert per_cell < 0.8
    assert averaged > 0.9
    assert per_cell_outputs["mean_cells_averaged"] is None
    assert averaged_outputs["neighbourhood_um"] == 10
    assert 5 < averaged_outputs["mean_cells_averaged"] < 12
