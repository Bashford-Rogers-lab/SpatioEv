"""QuPath bridge: geometry, GeoJSON writer, class/annotation readers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("shapely")

from spatioev.io import qupath  # noqa: E402

MARKERS = ["DAPI", "CD68", "PCK"]


def make_core(root: Path, name: str = "CORE_1-A") -> Path:
    """A synthetic core: 3 cells, nuclei numbered differently from the cells."""
    core = root / name
    (core / "segmentation").mkdir(parents=True)
    (core / "quantification").mkdir()
    cells = np.zeros((60, 60), dtype=np.uint32)
    cells[5:20, 5:25] = 1
    cells[30:50, 10:30] = 2
    cells[25:45, 40:55] = 3
    nuclei = np.zeros_like(cells)
    nuclei[10:15, 10:16] = 7  # inside cell 1
    nuclei[35:42, 15:22] = 2  # inside cell 2
    nuclei[2:6, 50:54] = 1  # matches no cell: "nucleus 1" is not cell 1's nucleus
    tifffile.imwrite(core / "segmentation" / f"{name}_whole_cell.tiff", cells)
    tifffile.imwrite(core / "segmentation" / f"{name}_nuclear.tiff", nuclei)
    rows = []
    for label, nucleus in ((1, 7), (2, 2), (3, 0)):
        ys, xs = np.nonzero(cells == label)
        row = {m: float(label) for m in MARKERS}
        row.update({f"{m}_nuclear": float(label) / 2 for m in MARKERS})
        row.update(label=label, centroid_y=ys.mean(), centroid_x=xs.mean(), cell_area_px2=len(ys), eccentricity=0.5,
                   solidity=0.9, matched_nuclear_label=nucleus, sample=name)
        rows.append(row)
    pd.DataFrame(rows).to_csv(core / "quantification" / "cell_table_arcsinh_transformed.csv", index=False)
    return core


def test_label_polygons_are_exact_pixel_edges():
    mask = np.zeros((20, 20), dtype=np.uint32)
    mask[2:6, 3:9] = 1  # rectangle
    mask[10:16, 10:16] = 2
    mask[12:14, 12:14] = 0  # ring with a hole
    mask[10:13, 2:4] = 3
    mask[12:15, 2:7] = 3  # L-shape
    polygons = qupath.label_polygons(mask)
    for label, geometry in polygons.items():
        assert geometry.area == pytest.approx((mask == label).sum())
    assert polygons[1].bounds == (3, 2, 9, 6)  # pixel (r, c) spans [c, c+1) x [r, r+1)
    assert len(polygons[2].interiors) == 1


def test_write_qupath_cells_ids_nuclei_and_measurements(tmp_path):
    core = make_core(tmp_path)
    result = qupath.write_qupath_cells(core)
    assert result["cells"] == 3 and result["nuclei_attached"] == 2 and result["cells_without_nucleus"] == 1
    payload = json.loads(Path(result["path"]).read_text())
    features = {f["properties"]["name"]: f for f in payload["features"]}
    assert set(features) == {"CORE_1-A_1", "CORE_1-A_2", "CORE_1-A_3"}
    first = features["CORE_1-A_1"]
    assert first["id"] == qupath.cell_uuid("CORE_1-A_1")
    assert first["properties"]["objectType"] == "cell"
    measurements = first["properties"]["measurements"]
    assert measurements["CD68"] == 1.0 and measurements["CD68 nucleus"] == 0.5 and measurements["Label"] == 1
    # Cell 1's nucleus is nucleus 7, not nucleus 1, and lies inside the cell.
    from shapely.geometry import shape

    cell, nucleus = shape(first["geometry"]), shape(first["nucleusGeometry"])
    assert cell.contains(nucleus) and nucleus.area == 30
    assert "nucleusGeometry" not in features["CORE_1-A_3"]
    status = qupath.core_status(core)
    assert status["cells_geojson"] and not status["classes"]


def write_export(core: Path, rows: list[tuple[str, str]], annotations: list[dict] | None = None) -> None:
    bridge = core / "qupath"
    bridge.mkdir(exist_ok=True)
    pd.DataFrame(
        [{"image": core.name, "cell_id": cid, "qupath_class": cls, "centroid_x_px": 0, "centroid_y_px": 0} for cid, cls in rows]
    ).to_csv(bridge / f"{core.name}_cell_classes.csv", index=False)
    if annotations is not None:
        (bridge / f"{core.name}_annotations.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": annotations}))


def rectangle(x0, y0, x1, y1, cls):
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
        "properties": {"objectType": "annotation", "classification": {"name": cls, "color": [255, 0, 0]}},
    }


def test_phenotyped_anndata_from_qupath_export(tmp_path):
    core = make_core(tmp_path)
    write_export(
        core,
        [("CORE_1-A_1", "Tumour"), ("CORE_1-A_2", "Fibroblast: WNT5A+"), ("CORE_1-A_3", ""), ("", "Macrophage")],
        [rectangle(0, 0, 30, 25, "Exclude"), rectangle(35, 20, 60, 50, "Stroma")],
    )
    flagged = qupath.qupath_phenotyped_anndata(core, drop_excluded=False)
    assert flagged.obs.loc["CORE_1-A_1", "excluded_by_annotation"]
    assert flagged.obs.loc["CORE_1-A_3", "qupath_annotation"] == "Stroma"
    adata = qupath.qupath_phenotyped_anndata(core)
    assert list(adata.obs_names) == ["CORE_1-A_2", "CORE_1-A_3"]
    obs = adata.obs
    assert obs.loc["CORE_1-A_2", "qupath_class"] == "Fibroblast: WNT5A+"
    assert obs.loc["CORE_1-A_2", "phenotype"] == "Fibroblast" and obs.loc["CORE_1-A_2", "qupath_subclass"] == "WNT5A+"
    assert obs.loc["CORE_1-A_3", "phenotype"] == "Unclassified"
    assert (obs["imageid"] == "CORE_1-A").all()
    assert list(adata.var_names) == MARKERS and "nuclear" in adata.layers
    assert adata.uns["qupath_bridge"]["n_excluded"] == 1


def test_qupath_export_problems_are_reported(tmp_path):
    core = make_core(tmp_path)
    write_export(core, [("CORE_1-A_1", "Tumour"), ("CORE_1-A_1", "Tumour")])
    with pytest.raises(ValueError, match="imported into QuPath twice"):
        qupath.qupath_phenotyped_anndata(core)
    write_export(core, [("OTHER_9", "Tumour")])
    with pytest.raises(ValueError, match="not in the cell table"):
        qupath.qupath_phenotyped_anndata(core)


def test_find_cores_scripts_and_cli(tmp_path):
    make_core(tmp_path, "A_1")
    make_core(tmp_path, "B_2")
    (tmp_path / "results").mkdir()
    assert [c.name for c in qupath.find_cores(tmp_path)] == ["A_1", "B_2"]
    written = qupath.write_qupath_scripts(tmp_path / "scripts")
    assert sorted(p.name for p in written) == sorted(qupath.QUPATH_SCRIPTS)
    assert "importObjectsFromFile" in written[0].read_text()

    from spatioev.cli import build_parser

    args = build_parser().parse_args(["qupath", "prepare", str(tmp_path), "--cores", "A_1", "--overwrite"])
    assert args.qupath_command == "prepare" and args.cores == ["A_1"] and args.overwrite

    from spatioev.workflows import qupath_bridge

    summary = qupath_bridge.prepare({"roots": [str(tmp_path)], "cores": ["A_1"]}, log=lambda message: None)
    assert summary["status"].tolist() == ["written"]
    assert qupath_bridge.scan([tmp_path])["cells_geojson"].tolist() == [True, False]
