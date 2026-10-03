"""End-to-end run of stages 04-07 on a small synthetic two-core TMA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("zarr")
anndata = pytest.importorskip("anndata")

from spatioev.workflows import (  # noqa: E402
    cohort_comparison,
    colocalization,
    fiber_segmentation,
    tissue_regions,
)

CHANNELS = ["DAPI", "COL1", "PCK"]
SIZE = 480


def _matrix_channel(rng) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    from skimage.draw import line

    image = rng.normal(150, 25, (SIZE, SIZE)).clip(0)
    for _ in range(70):
        r0, c0 = rng.integers(0, SIZE, 2)
        angle, length = rng.uniform(0, np.pi), rng.integers(20, 80)
        r1 = int(np.clip(r0 + length * np.sin(angle), 0, SIZE - 1))
        c1 = int(np.clip(c0 + length * np.cos(angle), 0, SIZE - 1))
        rr, cc = line(int(r0), int(c0), r1, c1)
        image[rr, cc] += 4000
    return gaussian_filter(image, 1.2).astype(np.uint16)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    images.mkdir()
    frames = []
    for fov, has_tumour in (("fov1", True), ("fov2", False)):
        stack = np.stack([rng.integers(0, 200, (SIZE, SIZE)).astype(np.uint16), _matrix_channel(rng), np.zeros((SIZE, SIZE), np.uint16)])
        tifffile.imwrite(
            images / f"{fov}.ome.tif", stack, ome=True,
            metadata={"axes": "CYX", "Channel": {"Name": CHANNELS}, "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5},
        )
        n = 900
        xy = rng.uniform(10, SIZE - 10, (n, 2))
        d = np.linalg.norm(xy - SIZE / 2, axis=1)
        phenotype = rng.choice(["Fibroblast", "Macrophage"], n, p=[0.6, 0.4])
        if has_tumour:
            phenotype = np.where(d < 110, "Tumour", phenotype)
        frames.append(pd.DataFrame({
            "imageid": fov, "X_centroid": xy[:, 0], "Y_centroid": xy[:, 1], "phenotype": phenotype,
        }, index=[f"{fov}_{i}" for i in range(n)]))
    obs = pd.concat(frames)
    obs["phenotype"] = obs["phenotype"].astype("category")
    anndata.AnnData(X=np.zeros((len(obs), 1)), obs=obs).write_h5ad(tmp_path / "S1_phenotyped.h5ad")
    return tmp_path


def _run(module, config: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    status = folder / "status.json"
    outputs = module.run_workflow(config, status)
    assert json.loads(status.read_text())["state"] == "complete"
    return outputs


def test_stages_04_to_07_run_end_to_end(project: Path):
    results = project / "results"
    fibers = _run(fiber_segmentation, {
        "sample_id": "S1", "image_path": str(project / "images"), "adata_path": str(project / "S1_phenotyped.h5ad"),
        "channels": ["COL1"], "output_dir": str(results / "S1_fiber_segmentation"),
    }, results / "S1_fiber_segmentation")
    assert fibers["pixel_size_um"] == pytest.approx(0.5)
    assert set(fibers["image_shapes"]) == {"fov1", "fov2"}
    table = fiber_segmentation.read_fiber_table(fibers["fiber_table"])
    assert len(table) > 10 and set(table["imageid"]) == {"fov1", "fov2"}
    assert table["alignment_score"].notna().any()
    architecture = pd.read_csv(fibers["architecture"])
    assert set(architecture["imageid"]) == {"fov1", "fov2"}
    assert architecture["length_density_mm_per_mm2"].gt(0).all() and architecture["hdm_pct"].between(0, 100).all()
    assert "hdm_pct" in pd.read_csv(fibers["tile_stats"]).columns

    regions = _run(tissue_regions, {
        "sample_id": "S1", "adata_path": str(project / "S1_phenotyped.h5ad"), "phenotype_key": "phenotype",
        "tumour_labels": ["Tumour"], "envelope_width_um": 15, "output_dir": str(results / "S1_tissue_regions"),
        "fiber_manifest": fibers["manifest"],
    }, results / "S1_tissue_regions")
    assert regions["n_nests"] >= 1
    labelled = anndata.read_h5ad(regions["h5ad"])
    fov2 = labelled.obs[labelled.obs["imageid"] == "fov2"]
    assert (fov2["tissue_region"] == "stroma").all()  # the core without tumour
    stats = pd.read_csv(regions["fiber_region_stats"])
    assert {"tumour", "envelope", "stroma", "all_tissue"} <= set(stats["region"])
    assert {"alignment_coherency", "branchpoints_per_mm2", "fractal_dimension"} <= set(stats.columns)
    assert not regions["warnings"]

    coloc = _run(colocalization, {
        "sample_id": "S1", "adata_path": regions["h5ad"], "phenotype_key": "phenotype",
        "output_dir": str(results / "S1_colocalization"), "pairs": [{"source": "Macrophage", "target": "Fibroblast"}],
        "radii_um": [10, 20, 40], "n_permutations": 19, "fiber_manifest": fibers["manifest"],
        "matrix_phenotypes": ["Macrophage", "Fibroblast"], "tile_size_um": 60,
    }, results / "S1_colocalization")
    curves = pd.read_csv(coloc["cell_cell_curves"])
    assert set(curves["imageid"]) == {"fov1", "fov2"} and len(curves) == 6
    summary = pd.read_csv(coloc["cell_matrix_summary"])
    assert {"tumour", "stroma", "all"} <= set(summary["region"])
    association = pd.read_csv(coloc["matrix_tile_association"])
    assert {"hdm_pct", "alignment_coherency"} <= set(association["metric"])

    sheet = project / "sheet.csv"
    pd.DataFrame([
        {"sample_id": "S1", "imageid": "fov1", "group": "A", "patient_id": "p1"},
        {"sample_id": "S1", "imageid": "fov2", "group": "B", "patient_id": "p2"},
    ]).to_csv(sheet, index=False)
    cohort = _run(cohort_comparison, {
        "sample_sheet": str(sheet), "project_root": str(project), "output_dir": str(results / "cohort"),
        "comparison_name": "test", "min_units_per_group": 1,
    }, results / "cohort")
    comparison = pd.read_csv(cohort["comparison"])
    assert set(comparison["family"]) == {"composition", "matrix", "cell_cell", "cell_matrix"}
    assert comparison["feature"].str.endswith("| hdm_pct").any()
    assert not comparison["feature"].str.endswith("| lacunarity_anamorf").any()
    assert cohort["units_per_group"] == {"A": 1, "B": 1}


@pytest.mark.parametrize("page", ["04_fiber_segmentation", "05_tissue_regions", "06_colocalization", "07_cohort_comparison"])
def test_new_pages_load(page: str, project: Path):
    pytest.importorskip("streamlit", minversion="1.40")
    from streamlit.testing.v1 import AppTest

    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    app = AppTest.from_file(str(root / "spatioev" / "apps" / "pages" / f"{page}.py"), default_timeout=60)
    app.session_state["spatioev_project_root"] = str(project)
    app.session_state["spatioev_sample_id"] = "S1"
    app.run()
    assert not app.exception
    if page == "04_fiber_segmentation":
        app.text_input(key="fiber_image").set_value(str(project / "images"))
        app.text_input(key="fiber_adata").set_value(str(project / "S1_phenotyped.h5ad"))
        app.button(key="fiber_inspect").click().run()
        assert not app.exception and not app.error
        assert app.multiselect(key="fiber_channels").value == ["COL1"]
    elif page == "05_tissue_regions":
        app.text_input(key="regions_adata").set_value(str(project / "S1_phenotyped.h5ad"))
        app.button(key="regions_inspect").click().run()
        assert not app.exception and not app.error
        assert app.multiselect(key="regions_tumour_phenotype").value == ["Tumour"]
