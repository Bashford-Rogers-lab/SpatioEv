"""Step 00 finds ``background/markers_bs.csv`` by itself, in every mode.

The marker order CSV is the only record of which marker each unnamed image
plane holds, so every mode looks for it, ``adata.X`` follows its row order,
and ``uns["all_markers"]`` keeps one name per image plane for the steps that
read the image.
"""

import dataclasses

import anndata as ad
import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")

from spatioev.io.demo_testis import TESTIS_PANEL, make_demo_testis_tma  # noqa: E402
from spatioev.workflows.cellsam import (  # noqa: E402
    ConversionPlan,
    build_anndata,
    find_marker_csv,
    marker_order_warning,
    plan_marker_manifest,
    plane_display_names,
    preflight,
)
from spatioev.workflows.cellsam_tma import (  # noqa: E402
    TMAConversionPlan,
    build_tma_anndata,
    inspect_tma,
    resolve_tma_marker_manifest,
)
from spatioev.workflows.image_collection import plane_marker_names  # noqa: E402

PANEL = [marker for marker, _, _ in TESTIS_PANEL]


def _csv(path, markers, **columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"marker_name": markers, **columns}).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# find_marker_csv
# --------------------------------------------------------------------------


def test_found_from_the_cell_table_folder_the_project_and_an_image(tmp_path):
    root = tmp_path / "ST_TMA"
    wanted = _csv(root / "background" / "markers_bs.csv", ["DAPI", "CD3"]).resolve()
    _csv(root / "markers.csv", ["DAPI", "CD3"])  # same folder, but background/ wins
    cell_tables = root / "ark_wdir_1" / "segmentation" / "cell_table"
    cell_tables.mkdir(parents=True)

    assert find_marker_csv(cell_tables) == wanted
    assert find_marker_csv(root) == wanted
    assert find_marker_csv(root / "dearray" / "1.ome.tif") == wanted


def test_markers_bs_is_preferred_and_only_marker_csvs_count(tmp_path):
    _csv(tmp_path / "background" / "markers.csv", ["A"])
    preferred = _csv(tmp_path / "background" / "markers_bs.csv", ["A"])
    assert find_marker_csv(tmp_path) == preferred.resolve()

    other = tmp_path / "other"
    other.mkdir()
    (other / "markers_bs.csv").write_text("channel,name\n0,A\n")  # no marker_name column
    assert find_marker_csv(other, max_levels=0) is None


def test_search_stops_after_max_levels(tmp_path):
    _csv(tmp_path / "background" / "markers_bs.csv", ["A"])
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    assert find_marker_csv(deep) is None
    assert find_marker_csv(deep, max_levels=4) is not None


def test_plane_display_names_number_repeats():
    assert plane_display_names(["DAPI", "CD3", "DAPI"]) == ["DAPI", "CD3", "DAPI (2)"]


# --------------------------------------------------------------------------
# Single-image mode
# --------------------------------------------------------------------------


def _single_image(tmp_path, markers, *, planes=None, channel_names=None, columns=None):
    """A cell table whose marker columns are alphabetical, and an image."""
    columns = sorted(columns or markers, key=str.lower)
    frame = pd.DataFrame({"CellID": [1, 2], "X_centroid": [1.0, 3.0], "Y_centroid": [1.0, 3.0], "area": [5.0, 6.0]})
    for marker in columns:
        frame[marker] = float(markers.index(marker) + 1) * 10 if marker in markers else 0.0
    sample = tmp_path / "S1"
    (sample / "quantification").mkdir(parents=True)
    frame.to_csv(sample / "quantification" / "primary.csv", index=False)
    frame.to_csv(sample / "quantification" / "secondary.csv", index=False)
    metadata = {"axes": "CYX"}
    if channel_names:
        metadata["Channel"] = {"Name": channel_names}
    image = sample / "S1.ome.tif"
    tifffile.imwrite(
        image, np.zeros((planes or len(markers), 6, 6), np.uint16), ome=True, photometric="minisblack", metadata=metadata
    )
    return ConversionPlan(
        primary_csv=sample / "quantification" / "primary.csv",
        secondary_csv=sample / "quantification" / "secondary.csv",
        image_path=image,
        imageid="S1",
        output_path=tmp_path / "out.h5ad",
        make_qc=False,
    )


def test_single_image_uses_a_marker_csv_found_in_background(tmp_path):
    order = ["DAPI", "PanCK", "CD8", "CD3"]
    _csv(tmp_path / "background" / "markers_bs.csv", order)
    plan = _single_image(tmp_path, order)
    path, source = plan_marker_manifest(plan)
    assert source == "auto-detected" and path.name == "markers_bs.csv"

    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == order
    assert result.X[0].tolist() == [10.0, 20.0, 30.0, 40.0]  # values travel with names
    assert list(result.var["channel_index"]) == [0, 1, 2, 3]
    assert result.uns["cellsam_conversion"]["marker_manifest_source"] == "auto-detected"


def test_single_image_detection_can_be_switched_off(tmp_path):
    _csv(tmp_path / "background" / "markers_bs.csv", ["CD3", "DAPI", "PanCK", "CD8"])
    named = ["DAPI", "CD3", "CD8", "PanCK"]
    plan = dataclasses.replace(_single_image(tmp_path, named, channel_names=named), auto_marker_manifest=False)
    assert plan_marker_manifest(plan) == (None, "none")
    build_anndata(plan)
    assert list(ad.read_h5ad(plan.output_path).var_names) == named


def test_removed_planes_need_no_column_and_keep_their_plane_index(tmp_path):
    # Plane 2 is a blank/autofluorescence channel the cell table never measured.
    planes = ["DAPI", "CD3", "AF488_blank", "PanCK"]
    _csv(tmp_path / "background" / "markers_bs.csv", planes, remove=["", "", "TRUE", ""])
    plan = _single_image(tmp_path, planes, columns=["DAPI", "CD3", "PanCK"])
    report = preflight(plan)
    assert report["removed_channels"] == ["AF488_blank"]

    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == ["DAPI", "CD3", "PanCK"]
    assert list(result.var["channel_index"]) == [0, 1, 3]  # PanCK is still plane 3
    assert list(result.uns["all_markers"]) == planes
    assert plane_marker_names(result) == planes


# --------------------------------------------------------------------------
# TMA mode, on the synthetic testis TMA (unnamed planes, cycle-local channel
# numbers, alphabetical cell-table columns, nuclear rows)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def testis(tmp_path_factory):
    return make_demo_testis_tma(tmp_path_factory.mktemp("demo"), n_fovs=2, fovs_per_batch=1, size=240)


def _tma_plan(root, output, marker_manifest=None):
    return TMAConversionPlan(
        project_root=root,
        image_dir=root / "dearray",
        marker_manifest=marker_manifest,
        dataset_id=root.name,
        output_path=output,
        make_qc=False,
    )


def test_tma_follows_markers_bs_row_order_and_planes(testis, tmp_path):
    plan = _tma_plan(testis, tmp_path / "tma.h5ad")
    report = inspect_tma(plan)
    assert report["marker_manifest_source"] == "auto-detected"

    build_tma_anndata(plan)
    adata = ad.read_h5ad(plan.output_path)
    assert list(adata.var_names) == PANEL
    assert list(adata.uns["all_markers"]) == PANEL
    assert list(adata.var["channel_index"]) == list(range(len(PANEL)))
    assert "remove" not in adata.var
    assert marker_order_warning(adata) is None

    # Each column of X must be the brightness of its own image plane.
    image = tifffile.imread(testis / "dearray" / "1.ome.tif")
    fov = adata[adata.obs["imageid"] == "fov1"]
    x, y = fov.obs["X_centroid"].astype(int), fov.obs["Y_centroid"].astype(int)
    for marker in ["SOX9", "MYH11", "INSL3", "CD31"]:
        values = np.asarray(fov.layers["size_normalized"][:, PANEL.index(marker)]).ravel()
        pixels = image[PANEL.index(marker), y, x].astype(float)
        assert np.corrcoef(values, pixels)[0, 1] > 0.9, marker


def test_tma_explicit_and_missing_marker_csv(testis, tmp_path):
    explicit = _csv(tmp_path / "panel.csv", PANEL)
    assert resolve_tma_marker_manifest(_tma_plan(testis, tmp_path / "x.h5ad", explicit)) == (explicit.resolve(), "explicit")
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_tma_marker_manifest(_tma_plan(testis, tmp_path / "x.h5ad", tmp_path / "missing.csv"))

    bare = tmp_path / "bare"
    (bare / "dearray").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="background"):
        resolve_tma_marker_manifest(_tma_plan(bare, tmp_path / "y.h5ad"))


def test_an_adata_from_the_old_channel_number_sort_is_flagged(testis, tmp_path):
    plan = _tma_plan(testis, tmp_path / "tma.h5ad")
    build_tma_anndata(plan)
    adata = ad.read_h5ad(plan.output_path)

    # What SpatioEv 0.2.0 and earlier did: sort the panel by its cycle-local numbers.
    scrambled = adata[:, adata.var.sort_values("channel_number", kind="stable").index].copy()
    assert "older SpatioEv" in marker_order_warning(scrambled)

    reordered = adata.copy()
    reordered.uns["all_markers"] = list(reversed(PANEL))
    assert "differs from the row order" in marker_order_warning(reordered)


def test_plane_marker_names_falls_back_to_var_names():
    adata = ad.AnnData(np.zeros((1, 2)), var=pd.DataFrame(index=["A", "B"]))
    assert plane_marker_names(adata) == ["A", "B"]
    adata.uns["all_markers"] = ["DAPI", "A", "B"]
    assert plane_marker_names(adata) == ["DAPI", "A", "B"]
