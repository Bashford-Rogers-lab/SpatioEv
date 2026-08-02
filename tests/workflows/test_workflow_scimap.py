import anndata as ad
import numpy as np
import pandas as pd
import pytest

# spatioev.workflows.{module} imports tifffile/zarr at module level, so a
# missing image-IO stack must skip this file rather than abort collection
# for the whole session. These are declared in the "dev" extra, so CI has
# them and the tests actually run there.
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("zarr")

from spatioev.workflows.scimap_phenotyping import inspect_inputs, validate_workflow


def make_inputs(tmp_path):
    obs = pd.DataFrame(
        {
            "imageid": ["sample"] * 4,
            "annotation_level2": ["immune", "immune", "tumour", "LSEC"],
            "gate_CD3_positive": [True, False, False, False],
            "X_centroid": [1.0, 2.0, 3.0, 4.0],
            "Y_centroid": [4.0, 3.0, 2.0, 1.0],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    adata = ad.AnnData(
        X=np.ones((4, 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["DAPI", "CD3", "CD8"]),
    )
    h5ad = tmp_path / "gated.h5ad"
    adata.write_h5ad(h5ad)
    annotations = tmp_path / "annotations.csv"
    obs[["annotation_level2"]].to_csv(annotations)
    gates = tmp_path / "gates.csv"
    pd.DataFrame({"markers": ["CD3", "CD8"], "gates": [1.0, 1.5]}).to_csv(gates, index=False)
    workflow = tmp_path / "workflow.csv"
    pd.DataFrame(
        {
            "parent": ["all", "T cells"],
            "phenotype": ["T cells", "CD8 T cells"],
            "CD3": ["pos", np.nan],
            "CD8": [np.nan, "pos"],
        }
    ).to_csv(workflow, index=False)
    image = tmp_path / "image.ome.tif"
    tifffile.imwrite(
        image,
        np.zeros((3, 6, 6), dtype=np.uint16),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3", "CD8"]}},
    )
    return h5ad, annotations, gates, workflow, image


def test_inspection_finds_broad_populations_and_validates_workflow(tmp_path):
    h5ad, annotations, gates, workflow, image = make_inputs(tmp_path)
    report = inspect_inputs(
        {
            "gated_h5ad": str(h5ad),
            "broad_annotations": str(annotations),
            "gate_csv": str(gates),
            "workflow_csv": str(workflow),
            "image_path": str(image),
        }
    )
    assert report["n_cells"] == 4
    assert list(report["candidate_columns"]) == ["annotation_level2"]
    assert report["candidate_columns"]["annotation_level2"][0] == {"value": "immune", "n_cells": 2}
    assert report["workflow"]["workflow_rows"] == 2
    assert report["workflow"]["top_level_phenotypes"] == ["T cells"]


def test_workflow_requires_a_gate_for_every_marker(tmp_path):
    _, _, gates, workflow, _ = make_inputs(tmp_path)
    pd.DataFrame({"markers": ["CD3"], "gates": [1.0]}).to_csv(gates, index=False)
    with pytest.raises(ValueError, match="absent from gate CSV"):
        validate_workflow(workflow, gates, ["DAPI", "CD3", "CD8"])
