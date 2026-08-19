"""A blank path field must mean 'not set', not the current directory.

Reported from the marker-autogating page, step 3:

    IsADirectoryError: [Errno 21] Is a directory: '.'

``Path("")`` is ``PosixPath(".")``. Clearing the optional gating-strategy box
therefore produced a path to the working directory rather than ``None``. That
slipped through validation because the check used ``.exists()``, and ``"."``
exists, so the mistake only surfaced deep inside ``pd.read_csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spatioev.workflows.marker_gating import read_strategy_profile


def test_empty_string_path_is_the_current_directory():
    """The language behaviour this whole module guards against."""
    assert Path("") == Path(".")
    assert Path("").exists() is True, "which is why .exists() was the wrong check"
    assert Path("").is_file() is False


@pytest.mark.parametrize("blank", [None, Path(""), Path("."), Path("   ")])
def test_blank_strategy_profile_is_treated_as_absent(blank):
    """No profile is a valid choice; it must not raise."""
    assert read_strategy_profile(blank) is None


def test_directory_path_raises_a_clear_error(tmp_path):
    """A real directory is a mistake, and should say so."""
    with pytest.raises(FileNotFoundError) as excinfo:
        read_strategy_profile(tmp_path)

    message = str(excinfo.value)
    assert "not a file" in message
    assert "directory" in message
    assert "blank" in message, "the message should say how to opt out"


def test_valid_strategy_profile_still_loads(tmp_path):
    path = tmp_path / "strategy.csv"
    pd.DataFrame(
        {"marker": ["CD8", "CD4"], "preferred_method": ["otsu", "2component_gmm"]}
    ).to_csv(path, index=False)

    profile = read_strategy_profile(path)

    assert profile is not None
    assert list(profile["marker"]) == ["CD8", "CD4"]


def test_missing_required_column_is_still_rejected(tmp_path):
    """Hardening the blank case must not weaken schema validation."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"marker": ["CD8"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        read_strategy_profile(path)


def test_optional_path_helper_maps_blank_to_none():
    pytest.importorskip("streamlit", minversion="1.40")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_autogating_page",
        Path(__file__).resolve().parents[2]
        / "spatioev"
        / "apps"
        / "pages"
        / "02_marker_autogating.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.optional_path("") is None
    assert module.optional_path("   ") is None
    assert module.optional_path(None) is None
    assert module.optional_path("/tmp/x.csv") == Path("/tmp/x.csv")


def test_validate_paths_rejects_unset_and_directory_paths(tmp_path):
    pytest.importorskip("streamlit", minversion="1.40")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_autogating_page2",
        Path(__file__).resolve().parents[2]
        / "spatioev"
        / "apps"
        / "pages"
        / "02_marker_autogating.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    adata = tmp_path / "a.h5ad"
    adata.write_bytes(b"")
    strategy = tmp_path / "s.csv"
    strategy.write_text("marker,preferred_method\n")

    ok = module.validate_paths(
        {
            "adata": adata,
            "image": tmp_path,
            "conditions": None,
            "output": tmp_path,
            "strategy": strategy,
        },
        require_condition_source=False,
    )
    assert ok == []

    # strategy left blank -> reported as unset, not silently the cwd
    errors = module.validate_paths(
        {
            "adata": adata,
            "image": tmp_path,
            "conditions": None,
            "output": tmp_path,
            "strategy": None,
        },
        require_condition_source=False,
    )
    assert any("strategy" in e and "not set" in e for e in errors)

    # strategy pointing at a directory -> named as a directory
    errors = module.validate_paths(
        {
            "adata": adata,
            "image": tmp_path,
            "conditions": None,
            "output": tmp_path,
            "strategy": tmp_path,
        },
        require_condition_source=False,
    )
    assert any("directory" in e for e in errors)
