"""Stages 00-03 end to end on the synthetic testis TMA, checked against its ground truth.

Runs the same worker functions the app starts: conversion with the marker CSV
found automatically, broad clustering into tubule / myoid / interstitium,
calculated gates, then two chained scimap runs (tubule, then interstitium)
with hand-made per-FOV gates. It also checks that every stage keeps its
output folder tidy.
"""

import json

import anndata as ad
import pandas as pd
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scanpy")
pytest.importorskip("leidenalg")
pytest.importorskip("sklearn")

from spatioev.io.demo_testis import make_demo_testis_tma  # noqa: E402
from spatioev.workflows import clustering, scimap_phenotyping  # noqa: E402
from spatioev.workflows._io import run_files  # noqa: E402
from spatioev.workflows.cellsam_tma import TMAConversionPlan, build_tma_anndata  # noqa: E402
from spatioev.workflows.marker_gating import SamplePaths, run_qc, stage_output_dirs  # noqa: E402

S = "DEMO_TESTIS"
BROAD_MARKERS = ["SOX9", "AMH", "CLDN11_2", "KRT18", "MAGEA4", "DMRT1", "UTF1_2", "GFRA1", "SMA", "MYH11",
                 "CD31", "CD45", "CD68", "Vimentin_2", "INSL3", "CYP17A1", "STAR", "NF2F2"]
TUBULE = {"SOX9", "AMH", "CLDN11_2", "KRT18", "MAGEA4", "DMRT1", "UTF1_2", "GFRA1"}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = make_demo_testis_tma(tmp_path_factory.mktemp("testis"), n_fovs=4, size=300)
    dirs = stage_output_dirs(S, root)
    adata_path = root / "data" / f"{S}_adata.h5ad"
    build_tma_anndata(TMAConversionPlan(project_root=root, image_dir=root / "dearray", marker_manifest=None,
                                        dataset_id=S, output_path=adata_path, make_qc=False))

    # 01: cluster, name each cluster by its top marker, export
    out = dirs["clustering"]
    _, status, _ = run_files(out, f"{S}_level0")
    level0 = clustering.run_level0(
        {"sample_id": S, "adata_path": str(adata_path), "output_dir": str(out), "markers": BROAD_MARKERS,
         "resolution": 0.6, "n_neighbors": 15, "n_pcs": 12}, status)
    summary = pd.read_csv(level0["summary_csv"])
    first = summary["top_markers"].str.split(", ").str[0]
    names = first.map(lambda m: "Myoid" if m in {"SMA", "MYH11"} else "Tubule" if m in TUBULE else "Interstitium")
    mapping = out / f"{S}_level0_annotation_mapping.csv"
    pd.DataFrame({"cluster": summary["cluster"].astype(str), "annotation": names}).to_csv(mapping, index=False)
    _, status, _ = run_files(out, f"{S}_export")
    exported = clustering.run_export(
        {"sample_id": S, "output_dir": str(out), "adata_path": str(adata_path), "level0_h5ad": level0["clustered_h5ad"],
         "level0_mapping_path": str(mapping), "image_path": str(root / "dearray"), "refinements": []}, status)

    # 02: calculated gates on the size-normalised layer, image QC on one FOV
    gating = run_qc(
        SamplePaths(S, root, exported["phenotyped_h5ad"], root / "dearray",
                    root / "data" / f"{S}_marker_condition_for_gating.csv", dirs["gating"]),
        layer="size_normalized", review_markers=["SOX9", "SMA"], make_image_qc=False)

    # 03: tubule run, then the interstitium run on its result
    def phenotype(name, population, source, broad, final):
        _, status, _ = run_files(dirs["phenotyping"], f"{S}_{name}_scimap_phenotyping")
        return scimap_phenotyping.run_workflow(
            {"sample_id": S, "selected_populations": [population], "subset_name": name,
             "output_dir": str(dirs["phenotyping"]), "broad_label_column": broad, "gated_h5ad": str(source),
             "image_path": str(root / "dearray"), "gate_csv": str(root / "manual_gates.csv"),
             "workflow_csv": str(root / f"{name}_phenotype_workflow.csv"), "gate_layer": "size_normalized",
             "missing_gate_policy": "median", "phenotype_label": f"{name}_phenotype", "final_label": final,
             "pheno_threshold_abs": 0, "plot_min_cells": 5, "overlay_crop_size": 300, "overlay_n_crops": 1}, status)

    tubule = phenotype("tubule", "Tubule", exported["phenotyped_h5ad"], "annotation", "final_phenotype")
    interstitium = phenotype("interstitium", "Interstitium", tubule["full_h5ad"], "final_phenotype", "cell_type")
    return {"root": root, "dirs": dirs, "adata": adata_path, "exported": exported, "gating": gating,
            "tubule": tubule, "interstitium": interstitium}


def _scored(pipeline):
    final = ad.read_h5ad(pipeline["interstitium"]["full_h5ad"])
    truth = pd.read_csv(pipeline["root"] / "ground_truth.csv", keep_default_na=False)
    truth = truth.rename(columns={"cell_type": "true_type"})
    obs = final.obs[["fov", "label", "annotation", "cell_type"]].astype({"label": int})
    return obs.merge(truth, on=["fov", "label"], validate="one_to_one")


def test_adata_x_follows_markers_bs(pipeline):
    adata = ad.read_h5ad(pipeline["adata"])
    panel = pd.read_csv(pipeline["root"] / "background" / "markers_bs.csv")["marker_name"].tolist()
    assert list(adata.var_names) == panel
    assert adata.uns["tma_conversion"]["marker_manifest_source"] == "auto-detected"


def test_broad_compartments_and_final_cell_types_match_the_truth(pipeline):
    obs = _scored(pipeline)
    assert (obs["annotation"] == obs["compartment"]).mean() >= 0.98
    expected = obs["true_type"].where(obs["true_type"] != "Spermatogonia", "SPG " + obs["spg_state"])
    called = obs["cell_type"].astype(str).replace(
        {"Spermatogonia": "SPG state3-4", "Spermatocytes": "Spermatocyte", "Spermatids": "Spermatid",
         "CD8 T cell": "T cell"})
    assert (called == expected).mean() >= 0.95


def test_workflow_aliases_and_per_fov_gates_are_reported(pipeline):
    notes = pipeline["tubule"]["workflow"]["notes"]
    assert "Workflow column 'CLDN11_1' uses AnnData marker 'CLDN11_2'" in notes
    assert any("median" in note for note in notes)
    assert pipeline["tubule"]["workflow"]["gate_layout"] == "per-image"
    assert pipeline["tubule"]["warnings"] == []


def test_gating_writes_the_manifest_step_03_reads(pipeline):
    manifest = json.loads(pipeline["gating"]["manifest"].read_text())
    assert manifest["gate_layer"] == "size_normalized"
    assert manifest["review_imageid"] == "fov1"


def test_stage_folders_are_tidy(pipeline):
    dirs, s = pipeline["dirs"], S

    def top(folder):
        return {path.name for path in folder.iterdir()}

    assert top(dirs["clustering"]) == {
        "logs", "qc", "checkpoints", f"{s}_level0_annotation_mapping.csv", f"{s}_phenotyped.h5ad",
        f"{s}_phenotyping_annotations.csv", f"{s}_clustering_manifest.json"}
    assert top(dirs["gating"]) == {
        "qc", "details", f"{s}_marker_conditions.csv", f"{s}_selected_marker_gates.csv",
        f"{s}_scimap_manual_gates.csv", f"{s}_gating_manifest.json"}
    phenotyping = top(dirs["phenotyping"])
    assert {"qc", "details", "logs"} <= phenotyping
    assert all(name.endswith((".h5ad", ".csv", ".json")) for name in phenotyping - {"qc", "details", "logs"})
    assert not any(path.suffix == ".png" for folder in dirs.values() for path in folder.iterdir())
