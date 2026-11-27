"""Page 04 warns when the pixel size gives cells of an implausible size."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
anndata = pytest.importorskip("anndata")

from spatioev.workflows.fiber_segmentation import (  # noqa: E402
    implausible_cell_diameter_um,
    median_cell_diameter_px,
)

# A real testis TMA scanned at 0.325 µm/px: median cell area about 430 px²,
# so about 23 px across. Its OME metadata says 1 µm/px.
CELL_AREA_PX2 = 430.0


def _adata(path: Path, obs: dict) -> Path:
    frame = pd.DataFrame(obs, index=[f"c{i}" for i in range(len(next(iter(obs.values()))))])
    anndata.AnnData(X=np.zeros((len(frame), 1)), obs=frame).write_h5ad(path)
    return path


def test_median_cell_diameter_comes_from_cell_areas(tmp_path: Path):
    area = np.array([CELL_AREA_PX2, CELL_AREA_PX2, CELL_AREA_PX2, 0.0, np.nan])
    path = _adata(tmp_path / "cells.h5ad", {"imageid": ["fov1"] * 5, "area": area})

    # Empty or missing areas are ignored; the diameter is that of a circle of the same area.
    assert median_cell_diameter_px(str(path)) == pytest.approx(2 * np.sqrt(CELL_AREA_PX2 / np.pi))


def test_median_cell_diameter_needs_an_area_column(tmp_path: Path):
    path = _adata(tmp_path / "cells.h5ad", {"imageid": ["fov1", "fov1"]})

    assert median_cell_diameter_px(str(path)) is None
    assert median_cell_diameter_px(None) is None
    assert median_cell_diameter_px(str(tmp_path / "missing.h5ad")) is None


def test_cell_size_check_catches_a_pixel_size_three_times_off():
    diameter_px = 2 * np.sqrt(CELL_AREA_PX2 / np.pi)

    assert implausible_cell_diameter_um(0.325, diameter_px) is None
    assert implausible_cell_diameter_um(1.0, diameter_px) == pytest.approx(diameter_px)
    assert implausible_cell_diameter_um(0.1, diameter_px) == pytest.approx(0.1 * diameter_px)
    assert implausible_cell_diameter_um(None, diameter_px) is None
    assert implausible_cell_diameter_um(1.0, None) is None


@pytest.fixture
def project(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for fov in ("fov1", "fov2"):
        tifffile.imwrite(
            images / f"{fov}.ome.tif", rng.integers(0, 200, (2, 64, 64)).astype(np.uint16), ome=True,
            metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "COL1"]}, "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0},
        )
    _adata(tmp_path / "cells.h5ad", {"imageid": ["fov1", "fov2"] * 50, "area": np.full(100, CELL_AREA_PX2)})
    return tmp_path


def _inspected_page(project: Path, adata: str):
    pytest.importorskip("streamlit", minversion="1.40")
    from streamlit.testing.v1 import AppTest

    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    app = AppTest.from_file(str(root / "spatioev" / "apps" / "pages" / "04_fiber_segmentation.py"), default_timeout=60)
    app.session_state["spatioev_project_root"] = str(project)
    app.session_state["spatioev_sample_id"] = "S1"
    app.run()
    app.text_input(key="fiber_image").set_value(str(project / "images"))
    app.text_input(key="fiber_adata").set_value(adata)
    app.button(key="fiber_inspect").click().run()
    assert not app.exception and not app.error
    return app


def _pixel_warnings(app) -> list[str]:
    return [w.value for w in app.warning if "Check the pixel size" in w.value]


def test_page_warns_until_the_real_pixel_size_is_typed(project: Path):
    app = _inspected_page(project, str(project / "cells.h5ad"))

    assert app.number_input(key="fiber_pixel_size").value == pytest.approx(1.0)
    [warning] = _pixel_warnings(app)
    assert "23.4 µm across" in warning and "too large" in warning

    app.number_input(key="fiber_pixel_size").set_value(0.325).run()
    assert _pixel_warnings(app) == []

    app.number_input(key="fiber_pixel_size").set_value(0.1).run()
    [warning] = _pixel_warnings(app)
    assert "too small" in warning


def test_page_flags_one_micron_per_pixel_without_cell_sizes(project: Path):
    app = _inspected_page(project, "")

    [warning] = _pixel_warnings(app)
    assert "1 µm/px" in warning and "add the cell AnnData" in warning

    app.number_input(key="fiber_pixel_size").set_value(0.325).run()
    assert _pixel_warnings(app) == []
