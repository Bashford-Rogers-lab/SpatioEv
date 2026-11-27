"""Where each stage reads and writes, so results stay findable and folders tidy."""

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spatioev.workflows._io import find_run_file, run_files  # noqa: E402
from spatioev.workflows.marker_gating import (  # noqa: E402
    choose_crop_windows,
    default_sample_paths,
    gate_table_paths,
    keep_previous,
    latest_scimap_gates,
    stage_output_dirs,
)


def test_run_files_live_in_logs_and_old_runs_are_still_found(tmp_path):
    config, status, log = run_files(tmp_path, "S1_level0")
    assert {config.parent, status.parent, log.parent} == {tmp_path / "logs"}
    assert (config.name, status.name, log.name) == ("S1_level0_config.json", "S1_level0_status.json", "S1_level0.log")

    legacy = tmp_path / "S1_level0_status.json"  # written by an earlier version
    legacy.write_text("{}")
    assert find_run_file(tmp_path, status.name) == legacy
    status.parent.mkdir()
    status.write_text("{}")
    assert find_run_file(tmp_path, status.name) == status


def test_default_paths_for_a_tma_and_a_single_image_project(tmp_path):
    tma = tmp_path / "tma"
    (tma / "dearray").mkdir(parents=True)
    paths = default_sample_paths("ST_TMA", tma)
    assert paths.adata_path == tma / "data" / "ST_TMA_adata.h5ad"
    assert paths.image_path == tma / "dearray"
    assert paths.marker_condition_path == tma / "data" / "ST_TMA_marker_condition_for_gating.csv"

    single = tmp_path / "single"
    single.mkdir()
    paths = default_sample_paths("S1", single)
    assert paths.adata_path == single / "data" / "S1" / "S1_adata.h5ad"
    assert paths.image_path == single / "S1" / "processed" / "S1_combined.ome.tif"

    # Conditions saved by the last gating run are offered next time.
    saved = gate_table_paths(paths.output_dir, "S1")["conditions"]
    saved.parent.mkdir(parents=True)
    saved.write_text("marker\n")
    assert default_sample_paths("S1", single).marker_condition_path == saved


def test_stage_folders_keep_their_established_names(tmp_path):
    assert {key: path.name for key, path in stage_output_dirs("S1", tmp_path).items()} == {
        "clustering": "S1_clustering_workflow",
        "gating": "S1_marker_gating_qc",
        "phenotyping": "S1_scimap_phenotyping_interface",
    }


def test_latest_gates_and_one_level_of_undo(tmp_path):
    canonical = gate_table_paths(tmp_path, "S1")["scimap"]
    assert latest_scimap_gates(tmp_path, "S1") == canonical  # nothing yet: where it will be

    older = tmp_path / "S1_scimap_manual_gates_web.csv"
    newer = tmp_path / "S1_scimap_manual_gates_interactive.csv"
    older.write_text("markers,gates\n")
    newer.write_text("markers,gates\n")
    os.utime(older, (1, 1))
    assert latest_scimap_gates(tmp_path, "S1") == newer

    canonical.write_text("markers,gates\nSOX9,5\n")
    assert latest_scimap_gates(tmp_path, "S1") == canonical
    keep_previous(canonical, tmp_path / "details")
    assert (tmp_path / "details" / "S1_scimap_manual_gates_previous.csv").read_text() == canonical.read_text()


def test_a_core_smaller_than_the_crop_gives_one_crop_at_the_origin(tmp_path):
    image = tmp_path / "1.ome.tif"
    tifffile.imwrite(image, np.random.default_rng(0).integers(0, 500, (2, 300, 300), dtype=np.uint16),
                     ome=True, photometric="minisblack", metadata={"axes": "CYX"})
    windows, _ = choose_crop_windows(image, crop_size=1536, channel_markers=["DNA_1", "SOX9"])
    assert windows == [(0, 0)]


def test_cluster_cells_handles_a_panel_smaller_than_n_pcs():
    sc = pytest.importorskip("scanpy")  # noqa: F841
    pytest.importorskip("leidenalg")
    import anndata as ad

    from spatioev.config import ClusteringConfig
    from spatioev.tl.phenotype import cluster_cells

    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(5, 1, (40, 4))])
    adata = ad.AnnData(x.astype(np.float32))
    adata.var_names = ["A", "B", "C", "D"]
    clustered = cluster_cells(adata, ClusteringConfig(markers=["A", "B", "C", "D"], n_pcs=15, n_neighbors=10))
    assert clustered.uns["clustering_parameters"]["n_pcs_used"] == 3
    assert clustered.obs["leiden"].nunique() >= 2
