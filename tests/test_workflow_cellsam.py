import anndata as ad
import numpy as np
import pandas as pd
import tifffile

from spatioev.workflows.cellsam import ConversionPlan, build_anndata, preflight


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
    assert [row["marker"] for row in report["marker_mapping"]] == ["DAPI", "CD8", "EpCAM"]

    build_anndata(plan)
    result = ad.read_h5ad(output_path)
    assert list(result.var_names) == ["DAPI", "CD8", "EpCAM"]
    np.testing.assert_allclose(result.X, [[10, 20, 30], [11, 21, 31]])
    np.testing.assert_allclose(result.layers["size_normalized"], result.X + 100)
    assert "cell_size" not in result.obs
    assert list(result.obs_names) == ["sample_2", "sample_1"]
    assert list(result.uns["all_markers"]) == ["DAPI", "CD8", "EpCAM"]
    assert (result.obs["imageid"] == "sample").all()
