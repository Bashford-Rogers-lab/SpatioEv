import builtins
import tomllib
from pathlib import Path

import pytest

from spatioev.apps._common import module_command, resource_path
from spatioev.cli import build_parser, launch_ui
from spatioev.pl.spatial import _require_image_viewer_dependencies


def test_packaged_templates_exist():
    assert resource_path("hcc_phenocycler_consensus_strategy.csv").is_file()
    assert resource_path("hcc_immune_phenotype_workflow_example.csv").is_file()


def test_ui_dependencies_follow_scimap_compatibility_ranges():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    ui_dependencies = metadata["project"]["optional-dependencies"]["ui"]
    assert "dask[array]>=2023.11,<2024.0" in ui_dependencies
    assert "numcodecs<0.16" in ui_dependencies
    assert "numpy>=1.23,<2" in ui_dependencies
    assert "zarr==2.10.3" in ui_dependencies
    app_dependencies = metadata["project"]["optional-dependencies"]["apps"]
    viewer_dependencies = metadata["project"]["optional-dependencies"]["viewer"]
    assert not any(dependency.startswith("spatioev[") for dependency in app_dependencies)
    assert "scimap[napari]>=2.3,<2.4" in app_dependencies
    assert "setuptools>=68,<82" in app_dependencies
    assert "setuptools>=68,<82" in viewer_dependencies
    assert set(ui_dependencies).issubset(app_dependencies)


def test_module_command_uses_current_interpreter():
    command = module_command("spatioev.workflows.cellsam", "--help")
    assert command[1:] == ["-m", "spatioev.workflows.cellsam", "--help"]


def test_image_viewer_reports_removed_pkg_resources(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "napari":
            return object()
        if name == "scimap":
            raise ModuleNotFoundError(
                "No module named 'pkg_resources'", name="pkg_resources"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="setuptools<82"):
        _require_image_viewer_dependencies()


def test_ui_cli_defaults_to_launch_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["ui"])
    assert args.project_root == Path.cwd()
    assert args.port == 8501


def test_ui_cli_handles_ctrl_c(monkeypatch):
    args = build_parser().parse_args(["ui", "--no-browser"])
    monkeypatch.setattr("spatioev.cli.importlib.util.find_spec", lambda name: object())

    def interrupted(command):
        raise KeyboardInterrupt

    monkeypatch.setattr("spatioev.cli.subprocess.call", interrupted)
    assert launch_ui(args) == 130
