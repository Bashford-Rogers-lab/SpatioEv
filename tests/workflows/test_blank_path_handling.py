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

    with pytest.raises(ValueError, match="missing required column"):
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

    # A blank AnnData path is a real problem and must be reported as unset,
    # rather than silently becoming the working directory.
    errors = module.validate_paths(
        {
            "adata": None,
            "image": tmp_path,
            "conditions": None,
            "output": tmp_path,
            "strategy": strategy,
        },
        require_condition_source=False,
    )
    assert any("AnnData" in e and "not set" in e for e in errors)

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


# --------------------------------------------------------------------------- #
# Swapping the two CSV inputs
# --------------------------------------------------------------------------- #


def _condition_csv(path: Path) -> Path:
    pd.DataFrame(
        {
            "marker": ["CD8", "CD4"],
            "staining_condition": ["clear_specific"] * 2,
            "compartment_pattern": ["membrane"] * 2,
            "expression_condition": ["bimodal"] * 2,
            "artifact_level": ["low"] * 2,
        }
    ).to_csv(path, index=False)
    return path


def _strategy_csv(path: Path) -> Path:
    pd.DataFrame(
        {"marker": ["CD8", "CD4"], "preferred_method": ["otsu", "2component_gmm"]}
    ).to_csv(path, index=False)
    return path


def test_condition_csv_given_as_strategy_names_the_mixup(tmp_path):
    """The reported error: missing 'preferred_method'.

    The two CSVs sit in adjacent fields and are both keyed by 'marker', so a
    bare 'missing columns' message does not reveal what went wrong.
    """
    from spatioev.workflows.marker_gating import read_strategy_profile

    path = _condition_csv(tmp_path / "marker_condition_for_gating.csv")
    with pytest.raises(ValueError) as excinfo:
        read_strategy_profile(path)

    message = str(excinfo.value)
    assert "preferred_method" in message
    assert "marker condition CSV" in message, "should name the likely mix-up"
    assert "swapped" in message
    assert str(path) in message, "should name the offending file"
    assert "blank" in message, "should say the field is optional"


def test_strategy_given_as_condition_csv_names_the_mixup(tmp_path):
    from spatioev.workflows.marker_gating import read_marker_conditions

    path = _strategy_csv(tmp_path / "strategy.csv")
    with pytest.raises(ValueError) as excinfo:
        read_marker_conditions(path)

    message = str(excinfo.value)
    assert "gating strategy profile" in message
    assert "swapped" in message


def test_unrelated_csv_gets_no_misleading_hint(tmp_path):
    """A genuinely malformed file must not be blamed on a swap."""
    from spatioev.workflows.marker_gating import read_strategy_profile

    path = tmp_path / "junk.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)

    with pytest.raises(ValueError) as excinfo:
        read_strategy_profile(path)

    message = str(excinfo.value)
    assert "swapped" not in message
    assert "Columns found" in message


def test_packaged_strategy_resource_is_valid():
    """The shipped default must satisfy its own reader."""
    from spatioev.apps._common import resource_path
    from spatioev.workflows.marker_gating import read_strategy_profile

    profile = read_strategy_profile(
        resource_path("hcc_phenocycler_consensus_strategy.csv")
    )
    assert profile is not None
    assert {"marker", "preferred_method"} <= set(profile.columns)


def _page_module(name: str):
    pytest.importorskip("streamlit", minversion="1.40")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).resolve().parents[2]
        / "spatioev"
        / "apps"
        / "pages"
        / "02_marker_autogating.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blank_strategy_is_accepted_by_default(tmp_path):
    """A gating strategy profile is optional; blank must not block loading.

    Regression: an earlier fix for blank paths made the strategy field
    mandatory, so leaving it empty reported 'strategy path is not set'.
    """
    module = _page_module("_pg_optional")
    adata = tmp_path / "a.h5ad"
    adata.write_bytes(b"")

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
    assert errors == [], errors


def test_blank_strategy_is_rejected_only_for_the_hcc_template(tmp_path):
    """The template seeds the questionnaire from the profile, so it needs one."""
    module = _page_module("_pg_required")
    adata = tmp_path / "a.h5ad"
    adata.write_bytes(b"")

    errors = module.validate_paths(
        {
            "adata": adata,
            "image": tmp_path,
            "conditions": None,
            "output": tmp_path,
            "strategy": None,
        },
        require_condition_source=False,
        require_strategy=True,
    )
    assert any("HCC Phenocycler template" in e for e in errors)
    assert any("another condition source" in e for e in errors)
