import builtins
from pathlib import Path

import pytest
import tomllib

from spatioev.apps._common import module_command, resource_path
from spatioev.cli import build_parser, launch_ui
from spatioev.pl.spatial import _require_image_viewer_dependencies


def test_packaged_templates_exist():
    assert resource_path("hcc_phenocycler_consensus_strategy.csv").is_file()
    assert resource_path("hcc_immune_phenotype_workflow_example.csv").is_file()


def _extras() -> dict[str, list[str]]:
    # Walk up to the repository root rather than assuming a fixed depth, so
    # this keeps working if the test is moved between directories.
    here = Path(__file__).resolve()
    root = next(p for p in here.parents if (p / "pyproject.toml").is_file())
    pyproject = root / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return metadata["project"]["optional-dependencies"]


def test_ui_extra_is_not_constrained_by_scimap():
    """The UI stack must not inherit scimap's pins.

    scimap pins zarr==2.10.3 and an old dask, which previously leaked into the
    ``ui``/``apps`` extras and contradicted the documented working environment.
    The algorithmic scimap functions are now vendored, so the UI stack is free.
    """
    ui = _extras()["ui"]
    joined = " ".join(ui)
    assert "zarr==2.10.3" not in joined
    assert "scimap" not in joined
    assert not any(dep.startswith("numpy") for dep in ui), (
        "ui must not re-pin numpy; it inherits the core numpy>=1.23 bound"
    )


def test_apps_extra_composes_rather_than_duplicating():
    extras = _extras()
    assert extras["apps"] == ["spatioev[scanpy,ui]"]
    # scimap is reachable only through the dedicated gating extra.
    assert "scimap" not in " ".join(extras["apps"])


def test_gating_extra_owns_the_scimap_dependency():
    extras = _extras()
    assert "scimap[napari]>=2.3,<2.4" in extras["gating"]
    assert "setuptools>=68,<82" in extras["gating"]
    # 'viewer' is retained as a backwards-compatible alias.
    assert extras["viewer"] == ["spatioev[gating]"]


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
