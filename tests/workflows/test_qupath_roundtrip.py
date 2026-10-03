"""Round trip through a real QuPath with the shipped scripts.

Skipped unless a QuPath executable is found: set ``SPATIOEV_QUPATH`` to it, or
install QuPath under /Applications on macOS. Takes ~30 s (four QuPath runs).
"""

from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("shapely")

from spatioev.io import qupath  # noqa: E402


def qupath_executable() -> str | None:
    if os.environ.get("SPATIOEV_QUPATH"):
        return os.environ["SPATIOEV_QUPATH"]
    found = sorted(glob.glob("/Applications/QuPath*.app/Contents/MacOS/QuPath*"))
    return found[-1] if found else None


QUPATH = qupath_executable()
pytestmark = pytest.mark.skipif(QUPATH is None, reason="QuPath executable not found (set SPATIOEV_QUPATH)")

CREATE_PROJECT = """
import qupath.lib.projects.Projects
import qupath.lib.images.servers.ImageServerProvider
import java.awt.image.BufferedImage
def folder = new File(args[0]); folder.mkdirs()
def project = Projects.createProject(folder, BufferedImage.class)
def support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, args[1])
project.addImage(support.getBuilders()[0]).setImageName(new File(args[1]).getName())
project.syncChanges()
"""

CLASSIFY = """
getCellObjects().each { c ->
    def pck = c.getMeasurements().get("PCK")
    c.setPathClass(pck > 1.5 ? getPathClass("Tumour") : getDerivedPathClass(getPathClass("Fibroblast"), "WNT5A+"))
}
addObject(PathObjects.createAnnotationObject(ROIs.createRectangleROI(0, 0, 30, 25, ImagePlane.getDefaultPlane()), getPathClass("Exclude")))
"""


def run_qupath(*arguments) -> str:
    result = subprocess.run([QUPATH, "script", *map(str, arguments)], capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 0 and "Exception" not in output, output[-3000:]
    return output


def make_core(root: Path, name: str = "OXF999_1-A") -> Path:
    core = root / name
    for folder in ("background", "segmentation", "quantification"):
        (core / folder).mkdir(parents=True)
    cells = np.zeros((64, 64), dtype=np.uint32)
    cells[5:20, 5:25], cells[30:50, 10:30], cells[25:45, 40:55] = 1, 2, 3
    nuclei = np.zeros_like(cells)
    nuclei[10:15, 10:16], nuclei[35:42, 15:22] = 7, 2
    tifffile.imwrite(core / "segmentation" / f"{name}_whole_cell.tiff", cells)
    tifffile.imwrite(core / "segmentation" / f"{name}_nuclear.tiff", nuclei)
    image = np.random.default_rng(0).integers(0, 200, (3, 64, 64), dtype=np.uint16)
    tifffile.imwrite(
        core / "background" / f"{name}.ome.tiff", image, ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "CD68", "PCK"]}, "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5},
    )
    rows = []
    for label, nucleus in ((1, 7), (2, 2), (3, 0)):
        ys, xs = np.nonzero(cells == label)
        row = {"DAPI": 1.0, "CD68": 1.0, "PCK": float(label), "DAPI_nuclear": 1.0, "CD68_nuclear": 1.0, "PCK_nuclear": 1.0}
        row.update(label=label, centroid_y=ys.mean(), centroid_x=xs.mean(), cell_area_px2=len(ys), matched_nuclear_label=nucleus, sample=name)
        rows.append(row)
    pd.DataFrame(rows).to_csv(core / "quantification" / "cell_table_arcsinh_transformed.csv", index=False)
    return core


def test_shipped_scripts_round_trip(tmp_path):
    core = make_core(tmp_path / "TMA")
    qupath.write_qupath_cells(core)
    scripts = {p.name: p for p in qupath.write_qupath_scripts(tmp_path / "scripts")}
    (tmp_path / "create.groovy").write_text(CREATE_PROJECT)
    (tmp_path / "classify.groovy").write_text(CLASSIFY)
    project = tmp_path / "project" / "project.qpproj"
    image = core / "background" / f"{core.name}.ome.tiff"

    # Run without an open image (as from the script editor with nothing selected): a message, not a crash.
    for name in scripts:
        assert "no image is open" in run_qupath(scripts[name])

    run_qupath(tmp_path / "create.groovy", "-a", tmp_path / "project", "-a", image)
    assert "imported 3 cells" in run_qupath("-p", project, "-s", scripts["spatioev_import_cells.groovy"])
    assert "skipping" in run_qupath("-p", project, "-s", scripts["spatioev_import_cells.groovy"])
    run_qupath("-p", project, "-s", tmp_path / "classify.groovy")
    assert "exported 3 cells" in run_qupath("-p", project, scripts["spatioev_export_classes.groovy"])

    classes = qupath.read_qupath_classes(core / "qupath" / f"{core.name}_cell_classes.csv")
    assert set(classes.index) == {f"{core.name}_{i}" for i in (1, 2, 3)}
    # QuPath centroids are the table's pixel-centre centroids shifted by half a pixel.
    table = pd.read_csv(core / "quantification" / "cell_table_arcsinh_transformed.csv")
    shift = classes.loc[f"{core.name}_" + table["label"].astype(str), "centroid_x_px"].to_numpy() - table["centroid_x"].to_numpy()
    assert np.allclose(shift, 0.5, atol=1e-3)

    adata = qupath.qupath_phenotyped_anndata(core)
    assert list(adata.obs_names) == [f"{core.name}_2", f"{core.name}_3"]  # cell 1 is in the Exclude box
    assert adata.obs.loc[f"{core.name}_3", "qupath_class"] == "Tumour"
    assert adata.obs.loc[f"{core.name}_2", "qupath_subclass"] == ""  # PCK 2 > 1.5 -> Tumour
    assert adata.uns["qupath_bridge"]["n_excluded"] == 1
