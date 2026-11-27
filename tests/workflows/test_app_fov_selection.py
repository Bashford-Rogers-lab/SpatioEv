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
from datetime import UTC, datetime
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


TESTIS_MARKERS = [
    "DAPI", "SOX9", "AMH", "DMRT1", "GFRA1", "PIWIL4", "NaKATPase", "MAGEA4", "SYCP3",
    "Acrosin", "SMA", "MYH11_2", "CYP17A1", "STAR", "CYP11A1", "CD68", "CD45",
]


@pytest.fixture
def tma_project(tmp_path: Path) -> Path:
    return _make_project(tmp_path, MARKERS)


@pytest.fixture
def testis_project(tmp_path: Path) -> Path:
    return _make_project(tmp_path, TESTIS_MARKERS)


def _make_project(tmp_path: Path, markers: list[str]) -> Path:
    """A small multi-FOV project: one OME-TIFF per FOV plus a matching AnnData."""
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    images.mkdir()
    for fov in FOVS:
        data = rng.integers(0, 255, (len(markers), 16, 16), dtype=np.uint16)
        tifffile.imwrite(
            images / f"{fov}.ome.tif",
            data,
            metadata={"axes": "CYX", "Channel": {"Name": markers}},
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
    adata = anndata.AnnData(X=rng.lognormal(0, 1, (len(obs), len(markers))))
    adata.var_names = markers
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
                "updated_at": datetime.now(UTC).isoformat(),
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


def test_within_fov_scaling_is_offered_only_for_several_fovs(tma_project: Path):
    at = _loaded_app(tma_project)

    def within():
        matches = [w for w in at.checkbox if w.label == "Z-score within each FOV"]
        assert matches, "within-FOV scaling checkbox was not rendered"
        return matches[0]

    assert within().value is False
    assert within().disabled is False
    _scope(at).set_value("Selected FOV only")
    at.run()
    assert within().disabled is True


def _checkbox(at: AppTest, label: str):
    matches = [w for w in at.checkbox if w.label == label]
    assert matches, f"checkbox {label!r} was not rendered"
    return matches[0]


def _multiselect(at: AppTest, label: str):
    matches = [w for w in at.multiselect if w.label == label]
    assert matches, f"multiselect {label!r} was not rendered"
    return matches[0]


def _run_button(at: AppTest):
    return next(b for b in at.button if b.label == "Run Level-0 clustering")


def test_neighbourhood_averaging_needs_a_pixel_size(tma_project: Path):
    at = _loaded_app(tma_project)
    _multiselect(at, "Markers used for broad clustering").set_value(MARKERS)
    at.run()
    average = "Average markers over neighbouring cells"

    assert _checkbox(at, average).value is False
    assert not [w for w in at.number_input if w.label == "Pixel size (µm / px)"]
    assert _run_button(at).disabled is False

    _checkbox(at, average).check()
    at.run()
    assert _run_button(at).disabled is True
    assert any("pixel size" in str(w.value) for w in at.warning)

    next(w for w in at.number_input if w.label == "Pixel size (µm / px)").set_value(0.325)
    at.run()
    assert _run_button(at).disabled is False
    assert not at.exception


def test_testis_panel_starts_from_the_testis_structure_markers(testis_project: Path):
    at = _loaded_app(testis_project)

    preset = next(w for w in at.selectbox if w.label == "Marker preset")
    assert preset.value == "Testis structures"
    # Panel suffixes such as MYH11_2 are matched; broad stains stay out.
    # SYCP3, Acrosin and DMRT1 are in the panel but stay out: no meiotic cells
    # before puberty, and DMRT1 stains Sertoli and germ cells alike.
    assert _multiselect(at, "Markers used for broad clustering").value == [
        "GFRA1", "PIWIL4", "NaKATPase", "AMH", "SOX9", "MAGEA4", "MYH11_2", "SMA", "CYP17A1", "STAR",
    ]
    average = _checkbox(at, "Average markers over neighbouring cells")
    assert average.value is False
    average.check()
    at.run()
    assert _multiselect(at, "Markers kept per cell").value == ["MYH11_2", "SMA"]


def _complete_level0(project: Path, annotations: dict[str, str]) -> None:
    """A finished Level-0 run on the whole sample, with saved cluster labels."""
    out = project / "out"
    plt = pytest.importorskip("matplotlib.pyplot")
    images = {}
    for name in ("umap", "heatmap"):
        images[name] = out / f"tma_level0_{name}.png"
        plt.imsave(images[name], np.zeros((4, 4)))
    clusters = list(annotations)
    summary = out / "tma_level0_cluster_summary.csv"
    pd.DataFrame(
        {"cluster": clusters, "n_cells": 10, "fraction": 1 / len(clusters), "top_markers": "SOX9"}
    ).to_csv(summary, index=False)
    pd.DataFrame(
        {"cluster": clusters, "annotation": list(annotations.values()), "custom_annotation": ""}
    ).to_csv(out / "tma_level0_annotation_mapping.csv", index=False)
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / "tma_level0_status.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "stage": "complete",
                "message": "Level-0 clustering complete",
                "progress": 1.0,
                "updated_at": datetime.now(UTC).isoformat(),
                "outputs": {
                    "clustered_h5ad": str(out / "tma_level0_clustered.h5ad"),
                    "summary_csv": str(summary),
                    "umap_png": str(images["umap"]),
                    "heatmap_png": str(images["heatmap"]),
                    "n_cells": 100,
                    "n_clusters": len(clusters),
                },
            }
        )
    )


def _refinement_widgets(at: AppTest, parent: str):
    markers = next(w for w in at.multiselect if w.key == f"refine_markers_tma_{parent}")
    resolution = next(w for w in at.number_input if w.key == f"refine_resolution_tma_{parent}")
    brightness = next(w for w in at.checkbox if w.key == f"refine_brightness_tma_{parent}")
    return markers, resolution, brightness


def test_testis_refinements_start_from_their_own_markers(testis_project: Path):
    _complete_level0(testis_project, {"0": "Tubule", "1": "Interstitium", "2": "artifact"})
    at = _loaded_app(testis_project)
    _multiselect(at, "Broad or mixed populations to recluster").set_value(
        ["Interstitium", "Tubule", "artifact"]
    )
    at.run()
    assert not at.exception

    markers, resolution, brightness = _refinement_widgets(at, "Tubule")
    assert markers.value == ["SOX9", "MAGEA4", "GFRA1", "PIWIL4", "AMH", "DMRT1"]
    assert resolution.value == 0.5
    assert brightness.value is True

    markers, resolution, brightness = _refinement_widgets(at, "Interstitium")
    assert markers.value == ["SMA", "MYH11_2", "CYP17A1", "STAR", "CYP11A1", "CD68"]
    assert resolution.value == 0.5
    assert brightness.value is False

    # Any other label starts from the Level-0 markers, as before.
    markers, resolution, brightness = _refinement_widgets(at, "artifact")
    assert markers.value == _multiselect(at, "Markers used for broad clustering").value
    assert resolution.value == 0.8
    assert brightness.value is False


def _page_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_clustering_page", PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_testis_labels_are_offered_for_annotation():
    page = _page_module()
    testis = ["Tubule", "Interstitium", "Sertoli", "Spermatogonia", "Spermatocyte", "Spermatid",
              "Myoid", "Leydig", "Other interstitium", "artifact"]
    for options in (page.FINAL_LABELS, page.LEVEL0_LABELS):
        assert set(testis) <= set(options)


def test_refinement_defaults_need_most_of_the_testis_markers():
    page = _page_module()
    level0 = ["A", "B", "C"]
    # Only two interstitium markers in the panel: fall back to the Level-0 set.
    assert page.refinement_defaults("Interstitium", ["SMA", "STAR", "A"], level0) == (level0, 0.8, False)
    assert page.refinement_defaults(" tubule ", ["sox9", "MAGEA4_1", "AMH", "X"], level0) == (
        ["sox9", "MAGEA4_1", "AMH"], 0.5, True,
    )
