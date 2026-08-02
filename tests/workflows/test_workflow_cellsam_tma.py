import anndata as ad
import numpy as np
import pandas as pd
import pytest

# spatioev.workflows.{module} imports tifffile at module level, so a
# missing image-IO stack must skip this file rather than abort collection
# for the whole session. These are declared in the "dev" extra, so CI has
# them and the tests actually run there.
tifffile = pytest.importorskip("tifffile")

from spatioev.workflows.cellsam_tma import (
    TMAConversionPlan,
    build_tma_anndata,
    inspect_tma,
)


def _write_batch(
    root,
    batch,
    fov,
    offset,
    primary_filename="cell_table_arcsinh_transformed.csv",
    secondary_filename="cell_table_size_normalized.csv",
):
    table_dir = root / batch / "segmentation" / "cell_table"
    table_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "cell_size": [100, 20, 110, 22],
            "M2": [2 + offset, 12 + offset, 4 + offset, 14 + offset],
            "M1": [1 + offset, 11 + offset, 3 + offset, 13 + offset],
            "label": [1, 1, 2, 2],
            "area": [100, 20, 110, 22],
            "centroid-0": [20, 20, 40, 40],
            "centroid-1": [30, 30, 50, 50],
            "fov": [fov] * 4,
            "mask_type": ["whole_cell", "nuclear", "whole_cell", "nuclear"],
        }
    )
    frame.to_csv(table_dir / primary_filename, index=False)
    normalized = frame.copy()
    normalized[["M1", "M2"]] += 100
    normalized.to_csv(table_dir / secondary_filename, index=False)


def test_build_tma_anndata_discovers_batches_and_maps_fovs(tmp_path):
    primary_filename = "transformed.csv"
    secondary_filename = "normalized.csv"
    _write_batch(
        tmp_path,
        "ark_wdir_1",
        "fov1",
        0,
        primary_filename,
        secondary_filename,
    )
    _write_batch(
        tmp_path,
        "ark_wdir_2",
        "fov2",
        20,
        primary_filename,
        secondary_filename,
    )
    image_dir = tmp_path / "dearray"
    image_dir.mkdir()
    for fov in [1, 2]:
        tifffile.imwrite(
            image_dir / f"{fov}.ome.tif",
            np.zeros((2, 5, 5), dtype=np.uint16),
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
    marker_manifest = image_dir / "markers.csv"
    pd.DataFrame({"channel_number": [1, 2], "marker_name": ["M1", "M2"]}).to_csv(
        marker_manifest, index=False
    )
    output_path = tmp_path / "tma.h5ad"
    plan = TMAConversionPlan(
        project_root=tmp_path,
        image_dir=image_dir,
        marker_manifest=marker_manifest,
        dataset_id="TMA1",
        output_path=output_path,
        primary_filename=primary_filename,
        secondary_filename=secondary_filename,
        make_qc=False,
    )

    report = inspect_tma(plan)
    assert report["n_batches"] == 2
    assert report["n_fovs"] == 2
    assert report["n_cells"] == 4
    assert report["marker_order"] == ["M1", "M2"]
    assert report["primary_filename"] == primary_filename
    assert report["secondary_filename"] == secondary_filename

    build_tma_anndata(plan)
    result = ad.read_h5ad(output_path)
    assert result.shape == (4, 2)
    assert list(result.var_names) == ["M1", "M2"]
    np.testing.assert_allclose(result.X, [[1, 2], [3, 4], [21, 22], [23, 24]])
    np.testing.assert_allclose(result.layers["size_normalized"], result.X + 100)
    assert list(result.obs["imageid"].astype(str).unique()) == ["fov1", "fov2"]
    assert list(result.obs_names) == [
        "TMA1_fov1_1",
        "TMA1_fov1_2",
        "TMA1_fov2_1",
        "TMA1_fov2_2",
    ]
    assert {"M1_nuclear", "M2_nuclear", "area_nuclear", "nc_ratio"}.issubset(
        result.obs.columns
    )
    np.testing.assert_allclose(result.obs["nc_ratio"], [0.2, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(
        result.obsm["spatial"], [[30, 20], [50, 40], [30, 20], [50, 40]]
    )
    assert list(result.uns["image_manifest"].index) == ["fov1", "fov2"]
