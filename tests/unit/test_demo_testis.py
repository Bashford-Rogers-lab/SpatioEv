"""The synthetic testis TMA reproduces the awkward parts of real ARK output."""

import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")

from spatioev.cli import build_parser, run_demo  # noqa: E402
from spatioev.io.demo_testis import TESTIS_PANEL, make_demo_testis_tma  # noqa: E402
from spatioev.workflows.image_collection import raw_channel_names  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo_testis_tma(tmp_path_factory.mktemp("demo"), n_fovs=2, fovs_per_batch=1, size=240)


def test_layout(demo):
    names = {path.relative_to(demo).as_posix() for path in demo.rglob("*") if path.is_file()}
    assert {
        "background/markers_bs.csv", "dearray/1.ome.tif", "dearray/2.ome.tif",
        "ark_wdir_1/segmentation/cell_table/cell_table_size_normalized.csv",
        "ark_wdir_2/segmentation/cell_table/cell_table_arcsinh_transformed.csv",
        "ark_wdir_1/segmentation/deepcell_output/fov1_whole_cell.tiff",
        "fov_metadata.csv", "ground_truth.csv", "manual_gates.csv",
        "data/DEMO_TESTIS_marker_condition_for_gating.csv",
        "tubule_phenotype_workflow.csv", "interstitium_phenotype_workflow.csv",
    } <= names


def test_panel_is_cycle_based_and_planes_are_unnamed(demo):
    panel = pd.read_csv(demo / "background" / "markers_bs.csv")
    assert panel["marker_name"].tolist() == [marker for marker, _, _ in TESTIS_PANEL]
    assert panel["channel_number"].duplicated().any()  # numbered within each cycle
    names = raw_channel_names(demo / "dearray" / "1.ome.tif")
    assert names == [f"C{index}" for index in range(len(TESTIS_PANEL))]


def test_cell_tables_are_alphabetical_with_nuclear_rows(demo):
    table = pd.read_csv(demo / "ark_wdir_1" / "segmentation" / "cell_table" / "cell_table_size_normalized.csv")
    markers = table.columns[1 : table.columns.get_loc("label")].tolist()
    assert markers == sorted(markers, key=str.lower)
    assert set(table["mask_type"]) == {"whole_cell", "nuclear"}
    truth = pd.read_csv(demo / "ground_truth.csv", keep_default_na=False)
    assert len(truth) == (table["mask_type"] == "whole_cell").sum() + len(
        pd.read_csv(demo / "ark_wdir_2" / "segmentation" / "cell_table" / "cell_table_size_normalized.csv")
        .query("mask_type == 'whole_cell'"))


def test_gates_cover_only_some_fovs(demo):
    gates = pd.read_csv(demo / "manual_gates.csv")
    assert gates.columns.tolist() == ["markers", "fov1"]


def test_overwrite_guard_and_size_check(demo, tmp_path):
    with pytest.raises(FileExistsError):
        make_demo_testis_tma(demo.parent, n_fovs=2, size=240)
    with pytest.raises(ValueError, match="at least"):
        make_demo_testis_tma(tmp_path, size=100)


def test_cli_kind_testis(tmp_path, capsys):
    args = build_parser().parse_args(["demo", str(tmp_path), "--kind", "testis", "--cores", "2", "--size", "200"])
    assert run_demo(args) == 0
    assert (tmp_path / "DEMO_TESTIS" / "background" / "markers_bs.csv").exists()
    assert "spatioev ui --project-root" in capsys.readouterr().out
