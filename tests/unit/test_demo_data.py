"""Synthetic demo TMA: layout, metadata, table/mask consistency and reader compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.ndimage as ndi

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("shapely")
pytest.importorskip("skimage")

from spatioev.io import qupath  # noqa: E402
from spatioev.io.demo import (  # noqa: E402
    CELL_CLASSES,
    PANEL,
    PIXEL_SIZE_UM,
    make_demo_tma,
)

SIZE = 600


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> tuple[Path, list[Path]]:
    root = tmp_path_factory.mktemp("demo")
    return root, make_demo_tma(root, n_cores=2, size=SIZE, seed=0, with_classes=True)


def read_table(core: Path, kind: str) -> pd.DataFrame:
    return pd.read_csv(core / "quantification" / f"cell_table_{kind}.csv")


def test_layout_sample_sheet_and_readme(demo):
    root, cores = demo
    assert [core.name for core in cores] == ["DEMO_01", "DEMO_02"]
    for core in cores:
        name = core.name
        for relative in (
            f"background/{name}.ome.tiff",
            f"segmentation/{name}_whole_cell.tiff",
            f"segmentation/{name}_nuclear.tiff",
            "quantification/cell_table_raw.csv",
            "quantification/cell_table_size_normalized.csv",
            "quantification/cell_table_arcsinh_transformed.csv",
            "quantification/channel_map.csv",
            f"qupath/{name}_cell_classes.csv",
            f"qupath/{name}_annotations.geojson",
        ):
            assert (core / relative).is_file(), relative
    assert [core.name for core in qupath.find_cores(root)] == ["DEMO_01", "DEMO_02"]

    sheet = pd.read_csv(root / "demo_sample_sheet.csv", dtype=str, keep_default_na=False)
    assert list(sheet.columns) == ["sample_id", "group", "imageid", "patient_id", "project_root"]
    assert sheet["sample_id"].tolist() == ["DEMO_01", "DEMO_02"]
    assert sheet["group"].tolist() == ["A", "B"]
    assert sheet["patient_id"].tolist() == ["P01", "P02"]
    assert (sheet["imageid"] == "").all() and (sheet["project_root"] == "").all()

    readme = (root / "README.txt").read_text(encoding="utf-8")
    assert "spatioev demo" in readme and "no real patient data" in readme
    assert "DEMO_01" in readme and "group B" in readme


def test_ome_tiff_metadata(demo):
    from spatioev.workflows.fiber_segmentation import OMEImage, physical_pixel_size
    from spatioev.workflows.image_collection import channel_names

    _, cores = demo
    path = cores[0] / "background" / "DEMO_01.ome.tiff"
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        assert tif.is_ome and series.axes == "CYX"
        assert series.shape == (len(PANEL), SIZE, SIZE) and series.dtype == np.uint16
        assert series.levels[0].shape == series.shape
        assert [level.shape[1:] for level in series.levels] == [(SIZE, SIZE), (SIZE // 2, SIZE // 2), (SIZE // 4, SIZE // 4)]
        assert tif.pages[0].is_tiled
    assert channel_names(path) == list(PANEL)
    assert physical_pixel_size(path) == pytest.approx(PIXEL_SIZE_UM)
    with OMEImage(path) as image:
        assert image.channels == list(PANEL) and image.shape == (SIZE, SIZE)
        dapi = image.plane("DAPI_INIT")[:, :]
        col1 = image.plane("COL1")[:, :]
    # Bright nuclei and matrix on a dim background, within uint16.
    assert np.percentile(dapi, 99.5) > 3000 and np.median(dapi) < 1000
    assert col1.max() > 2000


def test_tables_agree_with_image_and_masks(demo):
    _, cores = demo
    core = cores[1]
    name = core.name
    cells = tifffile.imread(core / "segmentation" / f"{name}_whole_cell.tiff")
    nuclei = tifffile.imread(core / "segmentation" / f"{name}_nuclear.tiff")
    stack = tifffile.imread(core / "background" / f"{name}.ome.tiff")
    assert cells.dtype == np.uint32 and nuclei.dtype == np.uint32 and cells.shape == (SIZE, SIZE)

    raw, normalized, arcsinh = (read_table(core, kind) for kind in ("raw", "size_normalized", "arcsinh_transformed"))
    expected_columns = [column for marker in PANEL for column in (marker, f"{marker}_nuclear")] + [
        "label", "centroid_y", "centroid_x", "eccentricity", "major_axis_length_px", "minor_axis_length_px",
        "perimeter_px", "solidity", "cell_size", "cell_area_px2", "passes_size_qc", "matched_nuclear_label",
        "nuclear_size", "nuclear_to_cell_area_ratio", "sample", "mask_type",
    ]
    for table in (raw, normalized, arcsinh):
        assert list(table.columns) == expected_columns
    assert len(raw) > 50 and (raw["sample"] == name).all() and (raw["mask_type"] == "whole_cell").all()

    labels = raw["label"].to_numpy()
    assert set(labels) == set(np.unique(cells)) - {0}
    matched = raw["matched_nuclear_label"].to_numpy()
    assert set(matched) == set(np.unique(nuclei)) - {0}
    assert (matched != labels).mean() > 0.9  # nuclei are numbered independently of cells
    # Each matched nucleus lies inside its own cell.
    owner = ndi.maximum(cells, labels=nuclei, index=matched)
    assert np.array_equal(owner, labels)
    assert np.array_equal(ndi.minimum(cells, labels=nuclei, index=matched), labels)

    areas = np.bincount(cells.ravel())[labels]
    nuclear_areas = np.bincount(nuclei.ravel())[matched]
    np.testing.assert_array_equal(raw["cell_size"], areas)
    np.testing.assert_array_equal(raw["cell_area_px2"], areas)
    np.testing.assert_array_equal(raw["nuclear_size"], nuclear_areas)
    np.testing.assert_allclose(raw["nuclear_to_cell_area_ratio"], nuclear_areas / areas)
    centres = np.array(ndi.center_of_mass(np.ones_like(cells), cells, labels))
    np.testing.assert_allclose(raw[["centroid_y", "centroid_x"]].to_numpy(), centres, atol=1e-6)

    for index, marker in enumerate(PANEL):
        plane = stack[index].astype(np.float64)
        whole = ndi.sum_labels(plane, cells, labels)
        nuclear = ndi.sum_labels(plane, nuclei, matched)
        np.testing.assert_allclose(raw[marker], whole, rtol=1e-12)
        np.testing.assert_allclose(raw[f"{marker}_nuclear"], nuclear, rtol=1e-12)
        np.testing.assert_allclose(normalized[marker], whole / areas, rtol=1e-12)
        np.testing.assert_allclose(normalized[f"{marker}_nuclear"], nuclear / nuclear_areas, rtol=1e-12)
        np.testing.assert_allclose(arcsinh[marker], np.arcsinh(normalized[marker] / 5), rtol=1e-12)
        np.testing.assert_allclose(arcsinh[f"{marker}_nuclear"], np.arcsinh(normalized[f"{marker}_nuclear"] / 5), rtol=1e-12)
    for column in ("label", "centroid_x", "cell_size", "matched_nuclear_label", "passes_size_qc"):
        pd.testing.assert_series_equal(raw[column], arcsinh[column])

    channel_map = pd.read_csv(core / "quantification" / "channel_map.csv")
    assert list(channel_map.columns) == ["channel_index", "marker_name", "quantification_column"]
    assert channel_map["channel_index"].tolist() == list(range(len(PANEL)))
    assert channel_map["marker_name"].tolist() == list(PANEL) == channel_map["quantification_column"].tolist()


def test_cell_types_are_distinguishable(demo):
    _, cores = demo
    core = cores[0]
    table = read_table(core, "size_normalized")
    classes = pd.read_csv(core / "qupath" / f"{core.name}_cell_classes.csv")["qupath_class"].to_numpy()
    tumour = classes == "Tumour"
    assert tumour.sum() > 20
    assert table.loc[tumour, "PCK"].median() > 10 * table.loc[~tumour, "PCK"].median()
    wnt5a = classes == "Fibroblast: WNT5A+"
    if wnt5a.any():
        assert table.loc[wnt5a, "WNT5A"].median() > 5 * table.loc[classes == "Fibroblast", "WNT5A"].median()


def test_write_qupath_cells(demo, tmp_path):
    _, cores = demo
    result = qupath.write_qupath_cells(cores[0], out_path=tmp_path / "cells.geojson")
    n_cells = len(read_table(cores[0], "arcsinh_transformed"))
    assert result["cells"] == n_cells
    assert result["nuclei_attached"] == n_cells and result["cells_without_nucleus"] == 0
    assert result["markers"] == list(PANEL)


def test_classes_give_phenotyped_anndata(demo):
    from spatioev.workflows import qupath_bridge

    root, cores = demo
    core = cores[0]
    classes = qupath.read_qupath_classes(core / "qupath" / f"{core.name}_cell_classes.csv")
    raw = read_table(core, "raw")
    assert list(classes["cell_id"]) == [f"{core.name}_{label}" for label in raw["label"]]
    np.testing.assert_allclose(classes["centroid_x_px"], raw["centroid_x"] + 0.5)
    np.testing.assert_allclose(classes["centroid_y_px"], raw["centroid_y"] + 0.5)
    assert set(classes["qupath_class"]) <= set(CELL_CLASSES)
    assert len(qupath.read_qupath_annotations(core / "qupath" / f"{core.name}_annotations.geojson")) == 0

    adata = qupath.qupath_phenotyped_anndata(core)
    assert adata.n_obs == len(raw) and list(adata.var_names) == list(PANEL)
    assert "qupath_class" in adata.obs and (adata.obs["qupath_class"].astype(str) != "Not in QuPath export").all()
    assert {"Tumour", "Fibroblast"} <= set(adata.obs["phenotype"].astype(str))
    assert "nuclear" in adata.layers

    summary, counts = qupath_bridge.collect({"roots": [str(root)]}, log=lambda message: None)
    assert summary["status"].tolist() == ["written", "written"]
    assert (core / "qupath" / f"{core.name}_phenotyped.h5ad").is_file()
    assert set(counts["qupath_class"]) <= set(CELL_CLASSES)


def test_refuses_to_overwrite_and_is_deterministic(tmp_path):
    first = make_demo_tma(tmp_path, n_cores=1, size=300, seed=5)
    assert [core.name for core in first] == ["DEMO_01"]
    assert not (first[0] / "qupath").exists()
    table = (first[0] / "quantification" / "cell_table_raw.csv").read_bytes()
    with pytest.raises(FileExistsError, match="DEMO_01"):
        make_demo_tma(tmp_path, n_cores=1, size=300, seed=6)
    assert (first[0] / "quantification" / "cell_table_raw.csv").read_bytes() == table
    again = make_demo_tma(tmp_path, n_cores=1, size=300, seed=5, overwrite=True)
    assert (again[0] / "quantification" / "cell_table_raw.csv").read_bytes() == table
    assert not any(path.name.startswith(".") for path in tmp_path.iterdir())
    with pytest.raises(ValueError):
        make_demo_tma(tmp_path / "other", n_cores=0)
