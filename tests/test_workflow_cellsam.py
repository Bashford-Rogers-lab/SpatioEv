import anndata as ad
import numpy as np
import pandas as pd
import tifffile

from spatioev.workflows.cellsam import (
    ROLE_MARKER,
    ROLE_OBSERVATION,
    ConversionPlan,
    build_anndata,
    inspect_inputs,
    preflight,
)


def test_build_anndata_reorders_markers_and_preserves_layer(tmp_path):
    image_path = tmp_path / "sample.ome.tif"
    tifffile.imwrite(
        image_path,
        np.zeros((3, 5, 6), dtype=np.uint16),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD8", "EpCAM"]}},
    )
    metadata = {
        "label": [2, 1],
        "area": [20.0, 10.0],
        "centroid-0": [4.0, 2.0],
        "centroid-1": [5.0, 3.0],
        "fov": ["fov0", "fov0"],
        "mask_type": ["whole_cell", "whole_cell"],
    }
    primary = pd.DataFrame(
        {
            "cell_size": [20, 10],
            "EPCAM": [30.0, 31.0],
            "DAPI": [10.0, 11.0],
            "CD8": [20.0, 21.0],
            **metadata,
        }
    )
    secondary = primary.copy()
    secondary[["EPCAM", "DAPI", "CD8"]] += 100
    primary_path = tmp_path / "cell_table_arcsinh_transformed.csv"
    secondary_path = tmp_path / "cell_table_size_normalized.csv"
    primary.to_csv(primary_path, index=False)
    secondary.to_csv(secondary_path, index=False)
    output_path = tmp_path / "sample.h5ad"

    plan = ConversionPlan(
        primary_csv=primary_path,
        secondary_csv=secondary_path,
        image_path=image_path,
        imageid="sample",
        output_path=output_path,
        make_qc=False,
    )
    report = preflight(plan)
    assert [row["marker"] for row in report["marker_mapping"]] == [
        "DAPI",
        "CD8",
        "EpCAM",
    ]

    build_anndata(plan)
    result = ad.read_h5ad(output_path)
    assert list(result.var_names) == ["DAPI", "CD8", "EpCAM"]
    np.testing.assert_allclose(result.X, [[10, 20, 30], [11, 21, 31]])
    np.testing.assert_allclose(result.layers["size_normalized"], result.X + 100)
    assert "cell_size" not in result.obs
    assert list(result.obs_names) == ["sample_2", "sample_1"]
    assert list(result.uns["all_markers"]) == ["DAPI", "CD8", "EpCAM"]
    assert (result.obs["imageid"] == "sample").all()
    np.testing.assert_allclose(result.obsm["spatial"], [[5, 4], [3, 2]])


def test_ome_driven_schema_preserves_unmatched_columns_in_obs(tmp_path):
    image_path = tmp_path / "sample.ome.tif"
    tifffile.imwrite(
        image_path,
        np.zeros((2, 4, 4), dtype=np.uint16),
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["DNA_1", "CD3"]}},
    )
    primary = pd.DataFrame(
        {
            "DNA_1": [1.0, 2.0],
            "DNA_1_nuclear": [11.0, 12.0],
            "CD3": [3.0, 4.0],
            "CD3_nuclear": [13.0, 14.0],
            "label": [10, 11],
            "centroid_y": [20.0, 21.0],
            "centroid_x": [30.0, 31.0],
            "cell_size": [100.0, 110.0],
            "cell_area_px2": [100.0, 110.0],
            "passes_size_qc": [True, False],
        }
    )
    secondary = primary.copy()
    secondary[["DNA_1", "CD3"]] += 100
    primary_path = tmp_path / "cell_table_arcsinh_transformed.csv"
    secondary_path = tmp_path / "cell_table_size_normalized.csv"
    primary.to_csv(primary_path, index=False)
    secondary.to_csv(secondary_path, index=False)
    output_path = tmp_path / "sample.h5ad"
    plan = ConversionPlan(
        primary_csv=primary_path,
        secondary_csv=secondary_path,
        image_path=image_path,
        imageid="sample",
        output_path=output_path,
        make_qc=False,
    )

    report = inspect_inputs(plan)
    roles = {row["column"]: row["role"] for row in report["column_roles"]}
    assert report["valid"]
    assert roles["DNA_1"] == ROLE_MARKER
    assert roles["CD3"] == ROLE_MARKER
    assert roles["DNA_1_nuclear"] == ROLE_OBSERVATION
    assert roles["CD3_nuclear"] == ROLE_OBSERVATION

    build_anndata(plan)
    result = ad.read_h5ad(output_path)
    assert list(result.var_names) == ["DNA_1", "CD3"]
    np.testing.assert_allclose(result.X, [[1, 3], [2, 4]])
    np.testing.assert_allclose(result.layers["size_normalized"], result.X + 100)
    assert {"DNA_1_nuclear", "CD3_nuclear", "passes_size_qc"}.issubset(
        result.obs.columns
    )
    assert "cell_size" not in result.obs
    assert "cell_area_px2" in result.obs
    assert "area" in result.obs
    np.testing.assert_allclose(result.obs["X_centroid"], [30, 31])
    np.testing.assert_allclose(result.obs["Y_centroid"], [20, 21])
    np.testing.assert_allclose(result.obsm["spatial"], [[30, 20], [31, 21]])


def test_column_role_and_marker_target_overrides(tmp_path):
    image_path = tmp_path / "sample.ome.tif"
    tifffile.imwrite(
        image_path,
        np.zeros((1, 3, 3), dtype=np.uint16),
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["CD8"]}},
    )
    frame = pd.DataFrame(
        {"signal": [1.0, 2.0], "object": [4, 5], "xpos": [8, 9], "ypos": [6, 7]}
    )
    primary_path = tmp_path / "primary.csv"
    secondary_path = tmp_path / "secondary.csv"
    frame.to_csv(primary_path, index=False)
    frame.assign(signal=frame["signal"] + 10).to_csv(secondary_path, index=False)
    plan = ConversionPlan(
        primary_csv=primary_path,
        secondary_csv=secondary_path,
        image_path=image_path,
        imageid="sample",
        output_path=tmp_path / "sample.h5ad",
        make_qc=False,
        column_roles={
            "signal": "marker",
            "object": "cell_id",
            "xpos": "x_coordinate",
            "ypos": "y_coordinate",
        },
        marker_targets={"signal": "CD8"},
    )

    assert preflight(plan)["valid"]
    result, _ = build_anndata(plan)
    assert list(result.var_names) == ["CD8"]
    assert list(result.obs_names) == ["sample_4", "sample_5"]
    np.testing.assert_allclose(result.obsm["spatial"], [[8, 6], [9, 7]])
