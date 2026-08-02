"""The FOV selector must hold its value, including while a worker is running.

Regression test for a reported bug: in TMA mode, choosing a FOV for
single-FOV clustering always snapped back to the first FOV.

Cause: the job-status block polled with ``time.sleep()`` + ``st.rerun()``,
restarting the entire script every two seconds. A selection made in between
was discarded, so the selectbox reverted to its default. The status display is
now an ``st.fragment(run_every=...)``, which reruns in isolation and leaves
the rest of the page — including the FOV selector — untouched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit", minversion="1.40")
pytest.importorskip("tifffile")
tifffile = pytest.importorskip("tifffile")
anndata = pytest.importorskip("anndata")

from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
PAGE = REPO_ROOT / "spatioev" / "apps" / "pages" / "01_broad_clustering.py"

FOVS = ["fov1", "fov2", "fov3", "fov4", "fov5"]
MARKERS = ["DAPI", "CD8", "Ki67", "PANCK"]


@pytest.fixture
def tma_project(tmp_path: Path) -> Path:
    """A small multi-FOV project: one OME-TIFF per FOV plus a matching AnnData."""
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    images.mkdir()
    for fov in FOVS:
        data = rng.integers(0, 255, (len(MARKERS), 16, 16), dtype=np.uint16)
        tifffile.imwrite(
            images / f"{fov}.ome.tif",
            data,
            metadata={"axes": "CYX", "Channel": {"Name": MARKERS}},
            ome=True,
        )

    frames = []
    for index, fov in enumerate(FOVS):
        n = 20 + index * 5
        frames.append(
            pd.DataFrame(
                {
                    "imageid": fov,
                    "X_centroid": rng.uniform(0, 100, n),
                    "Y_centroid": rng.uniform(0, 100, n),
                    "area": rng.uniform(20, 200, n),
                    "nc_ratio": rng.uniform(0.1, 0.8, n),
                }
            )
        )
    obs = pd.concat(frames, ignore_index=True)
    adata = anndata.AnnData(X=rng.lognormal(0, 1, (len(obs), len(MARKERS))))
    adata.var_names = MARKERS
    adata.obs = obs
    adata.obs_names = [f"c{i}" for i in range(len(obs))]
    adata.write_h5ad(tmp_path / "tma.h5ad")

    (tmp_path / "out").mkdir()
    return tmp_path


def _write_status(out_dir: Path, scoped_id: str, state: str) -> None:
    (out_dir / f"{scoped_id}_level0_status.json").write_text(
        json.dumps(
            {
                "state": state,
                "stage": "cluster",
                "message": "clustering",
                "progress": 0.5,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "outputs": {},
            }
        )
    )


def _loaded_app(project: Path) -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    at.run()

    values = {
        "cluster_sample_id": "tma",
        "cluster_adata_input": str(project / "tma.h5ad"),
        "cluster_image_input": str(project / "images"),
        "cluster_output_input": str(project / "out"),
    }
    for widget in at.text_input:
        if widget.key in values:
            widget.set_value(values[widget.key])
    at.run()

    for button in at.button:
        if "Load sample" in (button.label or ""):
            button.click()
            break
    at.run()
    return at


def _fov(at: AppTest):
    matches = [
        w for w in at.selectbox if w.key and w.key.startswith("cluster_review_imageid")
    ]
    assert matches, "FOV selectbox was not rendered"
    return matches[0]


def _scope(at: AppTest):
    matches = [
        w for w in at.segmented_control if w.key and w.key.startswith("cluster_scope")
    ]
    assert matches, "clustering scope control was not rendered"
    return matches[0]


def test_fov_selectbox_lists_every_image(tma_project: Path):
    at = _loaded_app(tma_project)
    assert _fov(at).options == FOVS


@pytest.mark.parametrize("job_state", [None, "running", "complete"])
def test_fov_selection_persists(tma_project: Path, job_state: str | None):
    """Selecting a FOV must stick, whatever the worker is doing.

    The 'running' case is the regression: it previously left the page in a
    sleep/rerun loop that discarded the selection.
    """
    if job_state is not None:
        _write_status(tma_project / "out", "tma_fov1", job_state)

    at = _loaded_app(tma_project)
    _scope(at).set_value("Selected FOV only")
    at.run()

    for target in ["fov3", "fov5", "fov2"]:
        _fov(at).select(target)
        at.run()
        assert _fov(at).value == target, (
            f"FOV reset to {_fov(at).value!r} instead of holding {target!r} "
            f"(job_state={job_state})"
        )


def test_selected_fov_scopes_the_clustering(tma_project: Path):
    """The chosen FOV must actually drive the run, not just the label."""
    at = _loaded_app(tma_project)
    _scope(at).set_value("Selected FOV only")
    at.run()
    _fov(at).select("fov4")
    at.run()

    captions = " ".join(str(c.value) for c in at.caption)
    assert "Clustering only fov4" in captions
    # fov4 is the fourth block of the fixture: 20 + 3*5 = 35 cells.
    assert "35 cells" in captions


def test_running_job_does_not_block_the_page(tma_project: Path):
    """A running worker must not stall the script.

    Before the fix the page slept and reran forever, so AppTest timed out
    here rather than returning a rendered page.
    """
    _write_status(tma_project / "out", "tma_fov1", "running")
    at = _loaded_app(tma_project)
    _scope(at).set_value("Selected FOV only")
    at.run()

    assert not at.exception
    assert _fov(at).options == FOVS
