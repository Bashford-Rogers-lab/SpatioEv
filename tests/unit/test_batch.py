from __future__ import annotations

import json

import pytest

from spatioev.workflows import batch


def _template(root, core="CORE_A", **extra):
    return {
        "sample_id": core,
        "adata_path": str(root / core / "qupath" / f"{core}_phenotyped.h5ad"),
        "fiber_manifest": str(root / "results" / f"{core}_fiber_segmentation" / f"{core}_fiber_manifest.json"),
        "output_dir": str(root / "results" / f"{core}_tissue_regions"),
        "phenotype_key": "qupath_class",
        "tumour_labels": ["Tumour"],
        **extra,
    }


def test_config_for_core_changes_only_the_core_name_below_the_root(tmp_path):
    root = tmp_path / "CORE_A_study"  # the root itself contains the core name and must not change
    template = _template(root)
    config = batch.config_for_core(template, "CORE_B", root)
    assert config["sample_id"] == "CORE_B"
    assert config["adata_path"] == str(root / "CORE_B" / "qupath" / "CORE_B_phenotyped.h5ad")
    assert config["fiber_manifest"] == str(root / "results" / "CORE_B_fiber_segmentation" / "CORE_B_fiber_manifest.json")
    assert config["output_dir"] == str(root / "results" / "CORE_B_tissue_regions")
    assert config["tumour_labels"] == ["Tumour"]  # settings are copied, not renamed
    assert template["sample_id"] == "CORE_A"  # the template is left alone


def test_detect_stage_and_find_template(tmp_path):
    assert batch.detect_stage({"image_path": "x", "channels": ["COL1"]}) == "fiber"
    assert batch.detect_stage({"tumour_labels": ["Tumour"]}) == "regions"
    assert batch.detect_stage({"pairs": [], "radii_um": [10]}) == "coloc"
    with pytest.raises(ValueError):
        batch.detect_stage({"sample_id": "x"})
    run = tmp_path / "results" / "CORE_A_tissue_regions"
    run.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No run settings"):
        batch.find_template(run)
    config = run / "CORE_A_tissue_regions_config.json"
    config.write_text(json.dumps(_template(tmp_path)))
    assert batch.find_template(run) == config
    assert batch.find_template(config) == config


def _core(root, name):
    for sub in ("segmentation", "quantification"):
        (root / name / sub).mkdir(parents=True)


def test_run_batch_skips_finished_cores_and_reports_missing_inputs(tmp_path):
    for name in ("CORE_A", "CORE_B", "CORE_C"):
        _core(tmp_path, name)
    template = _template(tmp_path)
    run = tmp_path / "results" / "CORE_A_tissue_regions"
    run.mkdir(parents=True)
    (run / "CORE_A_tissue_regions_config.json").write_text(json.dumps(template))
    (run / "CORE_A_tissue_regions_manifest.json").write_text("{}")  # CORE_A is done
    # CORE_B has its cell types but not its fiber result; CORE_C has neither.
    (tmp_path / "CORE_B" / "qupath").mkdir()
    (tmp_path / "CORE_B" / "qupath" / "CORE_B_phenotyped.h5ad").write_text("")
    summary = batch.run_batch(run, log=lambda message: None).set_index("core")
    assert summary.loc["CORE_A", "status"] == "skipped"
    assert summary.loc["CORE_B", "status"] == "missing input"
    assert "fiber segmentation" in summary.loc["CORE_B", "message"]
    assert "cell AnnData" in summary.loc["CORE_C", "message"]
    assert list(summary.index) == ["CORE_A", "CORE_B", "CORE_C"]


def test_run_batch_refuses_paths_without_the_core_name(tmp_path):
    _core(tmp_path, "CORE_A")
    template = _template(tmp_path, output_dir=str(tmp_path / "results" / "my_regions"))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="without the core name"):
        batch.run_batch(config, roots=[tmp_path], log=lambda message: None)


def test_fiber_batch_runs_every_demo_core_and_resumes(tmp_path):
    pytest.importorskip("tifffile")
    pytest.importorskip("zarr")
    from spatioev.io.demo import make_demo_tma

    cores = make_demo_tma(tmp_path, n_cores=2, size=300)
    run = tmp_path / "results" / "DEMO_01_fiber_segmentation"
    template = {
        "sample_id": "DEMO_01",
        "image_path": str(cores[0] / "background" / "DEMO_01.ome.tiff"),
        "adata_path": None,
        "channels": ["COL1"],
        "output_dir": str(run),
        "parameters": {"min_fiber_size": 50, "include_bright": True, "architecture": False},
    }
    run.mkdir(parents=True)
    (run / "DEMO_01_fiber_config.json").write_text(json.dumps(template))

    first = batch.run_batch(run, log=lambda message: None).set_index("core")
    assert list(first["status"]) == ["done", "done"], first["message"].tolist()
    second_core = tmp_path / "results" / "DEMO_02_fiber_segmentation"
    assert (second_core / "DEMO_02_fiber_manifest.json").exists()
    assert json.loads((second_core / "DEMO_02_fiber_config.json").read_text())["image_path"].endswith("DEMO_02/background/DEMO_02.ome.tiff")

    again = batch.run_batch(run, log=lambda message: None)
    assert set(again["status"]) == {"skipped"}
