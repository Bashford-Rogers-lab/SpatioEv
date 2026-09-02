"""The marker order CSV, not the image, defines the AnnData marker order.

TMA exports routinely lose their OME channel names, so an image cannot be
trusted to arbitrate its own marker order. Both conversion paths therefore take
the order from the marker order CSV when one is supplied, and its *row* order
is what counts.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# spatioev.workflows.{module} imports tifffile at module level, so a
# missing image-IO stack must skip this file rather than abort collection
# for the whole session. These are declared in the "dev" extra, so CI has
# them and the tests actually run there.
tifffile = pytest.importorskip("tifffile")

from spatioev.workflows.cellsam import (  # noqa: E402
    ConversionPlan,
    build_anndata,
    preflight,
    read_marker_manifest,
)
from spatioev.workflows.cellsam_tma import (  # noqa: E402
    TMAConversionPlan,
    build_tma_anndata,
    inspect_tma,
)

MARKERS = ["DAPI", "CD3", "CD8", "PanCK"]
N_PLANES = len(MARKERS)


def _marker_value(marker: str) -> int:
    """A distinctive value per marker, so a mis-mapping is visible in .X."""
    return (MARKERS.index(marker) + 1) * 10


def _decode(row) -> list[str]:
    """Recover which marker each .X column actually came from."""
    return [MARKERS[int(value) // 10 - 1] for value in row]


def _write_image(path, channel_names=None, planes=N_PLANES):
    metadata = {"axes": "CYX"}
    if channel_names is not None:
        metadata["Channel"] = {"Name": channel_names}
    tifffile.imwrite(
        path,
        np.zeros((planes, 5, 5), dtype=np.uint16),
        ome=True,
        photometric="minisblack",
        metadata=metadata,
    )


def _write_manifest(path, order, channel_numbers=None):
    frame = pd.DataFrame({"marker_name": order})
    if channel_numbers is not None:
        frame.insert(0, "channel_number", channel_numbers)
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# read_marker_manifest: row order wins, and self-contradiction is an error
# --------------------------------------------------------------------------


def test_manifest_row_order_is_preserved(tmp_path):
    path = _write_manifest(tmp_path / "m.csv", ["CD8", "DAPI", "PanCK", "CD3"])
    assert read_marker_manifest(path)["marker_name"].tolist() == [
        "CD8",
        "DAPI",
        "PanCK",
        "CD3",
    ]


def test_manifest_row_order_wins_when_channel_number_agrees(tmp_path):
    path = _write_manifest(tmp_path / "m.csv", MARKERS, channel_numbers=[1, 2, 3, 4])
    assert read_marker_manifest(path)["marker_name"].tolist() == MARKERS


def test_manifest_disagreeing_with_its_own_channel_number_is_rejected(tmp_path):
    # Row order says PanCK first; channel_number says DAPI first. Silently
    # picking either one is how the marker order drifts away from the CSV.
    path = _write_manifest(
        tmp_path / "m.csv",
        ["PanCK", "CD3", "DAPI", "CD8"],
        channel_numbers=[4, 2, 1, 3],
    )
    with pytest.raises(ValueError, match="disagrees with its own"):
        read_marker_manifest(path)


def test_manifest_rejects_duplicate_marker_names(tmp_path):
    path = _write_manifest(tmp_path / "m.csv", ["DAPI", "CD3", "DAPI"])
    with pytest.raises(ValueError, match="duplicated names"):
        read_marker_manifest(path)


# --------------------------------------------------------------------------
# Single-image path
# --------------------------------------------------------------------------


def _single_image_plan(tmp_path, manifest_order, channel_names, columns=MARKERS):
    frame = pd.DataFrame(
        {
            "CellID": [1, 2],
            "X_centroid": [10.0, 20.0],
            "Y_centroid": [10.0, 20.0],
            "area": [5.0, 6.0],
        }
    )
    for marker in columns:
        frame[marker] = [_marker_value(marker), _marker_value(marker)]
    primary = tmp_path / "primary.csv"
    secondary = tmp_path / "secondary.csv"
    frame.to_csv(primary, index=False)
    frame.to_csv(secondary, index=False)
    image = tmp_path / "image.ome.tif"
    _write_image(image, channel_names)
    manifest = (
        _write_manifest(tmp_path / "markers.csv", manifest_order)
        if manifest_order is not None
        else None
    )
    return ConversionPlan(
        primary_csv=primary,
        secondary_csv=secondary,
        image_path=image,
        imageid="S1",
        output_path=tmp_path / "out.h5ad",
        make_qc=False,
        marker_manifest=manifest,
    )


def test_single_image_unnamed_channels_uses_manifest_order(tmp_path):
    # Without a manifest this case cannot convert at all: the OME planes are
    # C0..C3 and no expression column matches them.
    plan = _single_image_plan(tmp_path, MARKERS, channel_names=None)
    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == MARKERS
    assert _decode(result.X[0]) == MARKERS
    assert result.uns["cellsam_conversion"]["channel_order_source"] == "marker order CSV"


def test_single_image_follows_manifest_row_order_not_alphabet_or_table(tmp_path):
    order = ["CD8", "DAPI", "PanCK", "CD3"]
    plan = _single_image_plan(tmp_path, order, channel_names=None)
    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == order
    # The values must travel with their names, not just the labels.
    assert _decode(result.X[0]) == order


def test_single_image_manifest_overrides_named_channels_with_a_warning(tmp_path):
    plan = _single_image_plan(
        tmp_path, MARKERS, channel_names=["CD3", "PanCK", "DAPI", "CD8"]
    )
    report = preflight(plan)
    assert any("overrides the channel names" in w for w in report["warnings"])
    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == MARKERS
    assert _decode(result.X[0]) == MARKERS


def test_single_image_without_manifest_keeps_ome_order(tmp_path):
    # Backwards compatibility: no manifest means the previous behaviour.
    channels = ["CD3", "PanCK", "DAPI", "CD8"]
    plan = _single_image_plan(tmp_path, None, channel_names=channels)
    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == channels
    assert result.uns["cellsam_conversion"]["channel_order_source"] == "OME metadata"


def test_single_image_manifest_length_must_match_the_image(tmp_path):
    plan = _single_image_plan(tmp_path, ["DAPI", "CD3"], channel_names=None)
    with pytest.raises(ValueError, match="2 markers but .* 4 image channels"):
        preflight(plan)


def test_all_markers_matches_var_names_under_a_manifest(tmp_path):
    # uns["all_markers"] is one name per image plane; a manifest supplies
    # exactly one name per plane, so the two must agree.
    plan = _single_image_plan(tmp_path, MARKERS, channel_names=None)
    build_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.uns["all_markers"]) == list(result.var_names)


# --------------------------------------------------------------------------
# TMA path
# --------------------------------------------------------------------------


def _write_tma_batch(root, batch, fov):
    table_dir = root / batch / "segmentation" / "cell_table"
    table_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "label": [1, 2],
            "area": [100.0, 110.0],
            "centroid-0": [20.0, 40.0],
            "centroid-1": [30.0, 50.0],
            "fov": [fov, fov],
            "mask_type": ["whole_cell", "whole_cell"],
        }
    )
    for marker in MARKERS:
        frame[marker] = [_marker_value(marker), _marker_value(marker)]
    frame.to_csv(table_dir / "primary.csv", index=False)
    frame.to_csv(table_dir / "secondary.csv", index=False)


def _tma_plan(tmp_path, manifest_order, channel_names, channel_numbers=None):
    _write_tma_batch(tmp_path, "ark_wdir_1", "fov1")
    image_dir = tmp_path / "dearray"
    image_dir.mkdir()
    _write_image(image_dir / "1.ome.tif", channel_names)
    manifest = _write_manifest(
        image_dir / "markers.csv", manifest_order, channel_numbers
    )
    return TMAConversionPlan(
        project_root=tmp_path,
        image_dir=image_dir,
        marker_manifest=manifest,
        dataset_id="TMA1",
        output_path=tmp_path / "tma.h5ad",
        primary_filename="primary.csv",
        secondary_filename="secondary.csv",
        make_qc=False,
    )


def test_tma_unnamed_image_follows_manifest_row_order(tmp_path):
    order = ["CD8", "DAPI", "PanCK", "CD3"]
    plan = _tma_plan(tmp_path, order, channel_names=None)
    build_tma_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == order
    assert _decode(result.X[0]) == order


def test_tma_manifest_overrides_named_channels_with_a_warning(tmp_path):
    # Previously a hard failure, which blocked conversion whenever the stored
    # channel names disagreed with the panel the user actually ran.
    plan = _tma_plan(tmp_path, MARKERS, channel_names=["CD3", "DAPI", "PanCK", "CD8"])
    report = inspect_tma(plan)
    assert any("overrides the channel names" in w for w in report["warnings"])
    build_tma_anndata(plan)
    result = ad.read_h5ad(plan.output_path)
    assert list(result.var_names) == MARKERS
    assert _decode(result.X[0]) == MARKERS


def test_tma_rejects_a_manifest_of_the_wrong_length(tmp_path):
    # An unnamed image trips the count check inside channel_names() first; a
    # named one reaches the panel-size check. Either way it must not build.
    plan = _tma_plan(tmp_path, ["DAPI", "CD3"], channel_names=None)
    with pytest.raises(ValueError, match="4 channels|2 markers"):
        inspect_tma(plan)


def test_tma_rejects_a_named_image_with_a_different_panel_size(tmp_path):
    plan = _tma_plan(
        tmp_path, ["DAPI", "CD3"], channel_names=["DAPI", "CD3", "CD8", "PanCK"]
    )
    with pytest.raises(ValueError, match="2 markers but"):
        inspect_tma(plan)


def test_tma_and_single_image_agree_on_the_same_manifest(tmp_path):
    """The two paths must not disagree about what the marker order CSV means."""
    order = ["PanCK", "CD8", "DAPI", "CD3"]

    tma_root = tmp_path / "tma"
    tma_root.mkdir()
    tma_plan = _tma_plan(tma_root, order, channel_names=None)
    build_tma_anndata(tma_plan)
    tma_result = ad.read_h5ad(tma_plan.output_path)

    single_root = tmp_path / "single"
    single_root.mkdir()
    single_plan = _single_image_plan(single_root, order, channel_names=None)
    build_anndata(single_plan)
    single_result = ad.read_h5ad(single_plan.output_path)

    assert list(tma_result.var_names) == list(single_result.var_names) == order
