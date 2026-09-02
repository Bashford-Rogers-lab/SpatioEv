"""Repeated OME channel names must not block AnnData conversion.

Reported by a user on the prepare-anndata page:

    Input schema is not valid: OME image contains ambiguous duplicate
    channels: ['DAPI_INIT']

Cyclic imaging platforms (CODEX, PhenoCycler, CyCIF) re-image the nuclear
stain every round, so several planes legitimately carry the same channel
name. Rejecting the image outright made those datasets unconvertible.

Markers come from CSV columns, which are unique, so a repeated channel name
does not make the expression matrix ambiguous. It only decides which plane a
marker is displayed against, which is now resolved deterministically to the
first occurrence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("zarr")

from spatioev.workflows.cellsam import (  # noqa: E402
    ConversionPlan,
    build_anndata,
    inspect_inputs,
)


def _project(tmp_path, channels: list[str], csv_markers: list[str]) -> ConversionPlan:
    rng = np.random.default_rng(0)
    image = tmp_path / "image.ome.tif"
    data = rng.integers(0, 255, (len(channels), 32, 32), dtype=np.uint16)
    tifffile.imwrite(
        image,
        data,
        metadata={"axes": "CYX", "Channel": {"Name": channels}},
        ome=True,
    )

    table = {
        "label": range(1, 11),
        "centroid-0": rng.random(10) * 100,
        "centroid-1": rng.random(10) * 100,
        "area": rng.random(10) * 100,
    }
    for marker in csv_markers:
        table[marker] = rng.random(10)
    frame = pd.DataFrame(table)

    primary = tmp_path / "cell_table_size_normalized.csv"
    secondary = tmp_path / "cell_table_arcsinh.csv"
    frame.to_csv(primary, index=False)
    frame.to_csv(secondary, index=False)

    return ConversionPlan(
        primary_csv=primary,
        secondary_csv=secondary,
        image_path=image,
        imageid="fov1",
        output_path=tmp_path / "out.h5ad",
        layer_name="arcsinh",
    )


def _as_adata(result):
    return result[0] if isinstance(result, tuple) else result


def test_repeated_channel_name_is_a_warning_not_an_error(tmp_path):
    """The exact case reported: two planes named DAPI_INIT."""
    plan = _project(
        tmp_path,
        channels=["DAPI_INIT", "DAPI_INIT", "CD8", "PANCK"],
        csv_markers=["DAPI_INIT", "CD8", "PANCK"],
    )
    report = inspect_inputs(plan)

    assert not report["errors"], report["errors"]
    assert any("repeats channel name" in w for w in report["warnings"])


def test_repeated_channel_name_still_builds_unique_markers(tmp_path):
    plan = _project(
        tmp_path,
        channels=["DAPI_INIT", "DAPI_INIT", "CD8", "PANCK"],
        csv_markers=["DAPI_INIT", "CD8", "PANCK"],
    )
    adata = _as_adata(build_anndata(plan))

    assert list(adata.var_names) == ["DAPI_INIT", "CD8", "PANCK"]
    assert adata.n_vars == len(set(adata.var_names)), "var_names must be unique"
    assert adata.n_obs == 10


@pytest.mark.parametrize("repeats", [2, 3, 4])
def test_any_number_of_repeats_collapses_to_one_marker(tmp_path, repeats):
    plan = _project(
        tmp_path,
        channels=["DAPI"] * repeats + ["CD8"],
        csv_markers=["DAPI", "CD8"],
    )
    report = inspect_inputs(plan)
    assert not report["errors"], report["errors"]

    adata = _as_adata(build_anndata(plan))
    assert list(adata.var_names) == ["DAPI", "CD8"]


def test_marker_mapping_has_one_row_per_distinct_channel(tmp_path):
    """A repeated plane must not add a second mapping row for the same column."""
    plan = _project(
        tmp_path,
        channels=["DAPI", "DAPI", "CD8"],
        csv_markers=["DAPI", "CD8"],
    )
    mapping = inspect_inputs(plan)["marker_mapping"]

    markers = [row["marker"] for row in mapping]
    sources = [row["source_column"] for row in mapping]
    assert markers == ["DAPI", "CD8"]
    assert len(sources) == len(set(sources)), "a CSV column must be read once"


def test_clean_image_produces_no_duplicate_warning(tmp_path):
    plan = _project(
        tmp_path,
        channels=["DAPI", "CD8", "PANCK"],
        csv_markers=["DAPI", "CD8", "PANCK"],
    )
    report = inspect_inputs(plan)

    assert not report["errors"]
    assert not any("repeats channel name" in w for w in report["warnings"])


def test_a_channel_with_no_expression_column_is_still_an_error(tmp_path):
    """Relaxing duplicates must not weaken the genuine completeness check."""
    plan = _project(
        tmp_path,
        channels=["DAPI", "CD8", "PANCK"],
        csv_markers=["DAPI", "CD8"],
    )
    report = inspect_inputs(plan)

    assert any("without an expression column" in e for e in report["errors"])
    assert any("PANCK" in e for e in report["errors"])


def test_all_markers_has_one_entry_per_image_plane(tmp_path):
    """scimap.pl.image_viewer asserts len(all_markers) == number of channels.

    var_names holds one entry per distinct marker, so with a repeated channel
    name the two diverge and napari review failed with:

        AssertionError: number of channel names (17) must match
        number of channels (18)
    """
    from spatioev.workflows.cellsam import image_channel_names

    plan = _project(
        tmp_path,
        channels=["DAPI_INIT", "DAPI_INIT", "CD8", "PANCK"],
        csv_markers=["DAPI_INIT", "CD8", "PANCK"],
    )
    adata = _as_adata(build_anndata(plan))

    n_planes = len(image_channel_names(plan.image_path))
    assert n_planes == 4
    assert adata.n_vars == 3, "markers are deduped"
    assert len(adata.uns["all_markers"]) == n_planes, (
        "all_markers must have one entry per image plane, not per marker"
    )


def test_repeated_channel_display_names_are_distinguishable(tmp_path):
    """napari layer names must not collide."""
    plan = _project(
        tmp_path,
        channels=["DAPI", "DAPI", "DAPI", "CD8"],
        csv_markers=["DAPI", "CD8"],
    )
    adata = _as_adata(build_anndata(plan))

    names = list(adata.uns["all_markers"])
    assert names == ["DAPI", "DAPI (2)", "DAPI (3)", "CD8"]
    assert len(names) == len(set(names))
