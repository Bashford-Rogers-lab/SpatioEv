"""Step 03: gate tables, marker aliases, missing gates and scimap's rule quirks."""

import pandas as pd
import pytest

from spatioev.workflows.scimap_phenotyping import (
    apply_min_cells,
    fill_missing_gates,
    normalise_gate_table,
    resolve_marker_aliases,
    workflow_rule_notes,
)

MARKERS = ["SOX9", "CLDN11_2", "Vimentin_2", "NaKATPase", "MAGEA4"]


def test_marker_aliases():
    mapping, unresolved = resolve_marker_aliases(
        ["SOX9", "CLDN11_1", "Vimentin_1", "NAKATPASE", "PIWIL4"], MARKERS
    )
    assert mapping == {
        "SOX9": "SOX9",
        "CLDN11_1": "CLDN11_2",
        "Vimentin_1": "Vimentin_2",
        "NAKATPASE": "NaKATPase",
    }
    assert unresolved == ["PIWIL4"]


def test_per_image_gate_table_with_a_global_fallback():
    raw = pd.DataFrame(
        {"markers": ["SOX9", "CLDN11_1", "OLD"], "fov1": [5.0, 6.0, 1.0], "gates": [4.0, 4.5, 1.0]}
    )
    gates, info = normalise_gate_table(raw, MARKERS, ["fov1", "fov2"])
    assert info["layout"] == "per-image"
    assert info["renamed_markers"] == {"CLDN11_1": "CLDN11_2"}
    assert info["ignored_markers"] == ["OLD"]
    assert gates.loc["SOX9"].tolist() == [5.0, 4.0]  # fov2 has no column: global gate
    assert gates.loc["CLDN11_2"].tolist() == [6.0, 4.5]


def test_global_gate_table():
    gates, info = normalise_gate_table(pd.DataFrame({"markers": ["SOX9"], "gates": [5.0]}), MARKERS)
    assert info["layout"] == "global"
    assert list(gates.columns) == ["gates"]


def test_missing_gate_policies():
    raw = pd.DataFrame({"markers": ["SOX9", "MAGEA4"], "fov1": [5.0, 6.0], "fov3": [5.4, None]})
    gates, _ = normalise_gate_table(raw, MARKERS, ["fov1", "fov2", "fov3"])

    with pytest.raises(ValueError, match="SOX9 in fov2"):
        fill_missing_gates(gates.copy(), ["SOX9", "MAGEA4"], "error")

    filled, notes = fill_missing_gates(gates.copy(), ["SOX9", "MAGEA4"], "median")
    assert filled.loc["SOX9", "fov2"] == pytest.approx(5.2)
    assert filled.loc["MAGEA4"].tolist() == [6.0, 6.0, 6.0]
    assert "median" in notes[0]

    left, notes = fill_missing_gates(gates.copy(), ["SOX9", "MAGEA4"], "gmm")
    assert left.isna().sum().sum() == 3
    assert "Gaussian mixture" in notes[0]

    with pytest.raises(ValueError, match="absent from gate CSV"):
        fill_missing_gates(gates.copy(), ["SOX9", "CLDN11_2"], "median")


def test_single_pos_or_neg_markers_are_pointed_out():
    workflow = pd.DataFrame(
        [
            ["Germ cells", "Spermatogonia", "pos", "neg", None],
            ["Germ cells", "Spermatogonia (strict)", "allpos", "allneg", None],
            ["all", "Fibroblast", None, None, "pos"],
            ["all", "Two negatives", "pos", "neg", "neg"],
        ],
        columns=["parent", "phenotype", "MAGEA4", "SYCP3", "Vimentin_2"],
    )
    notes = workflow_rule_notes(workflow)
    assert len(notes) == 3
    assert any("Spermatogonia: the single 'neg' marker SYCP3" in note and "allneg" in note for note in notes)
    assert any("Spermatogonia: the single 'pos' marker MAGEA4" in note for note in notes)
    assert any(note.startswith("Two negatives: the single 'pos'") for note in notes)


def test_minimum_cells_per_phenotype_is_per_image_and_reported():
    labels = pd.Series(["A"] * 12 + ["B"] * 3 + ["A"] * 2 + ["Unknown"] * 2)
    images = pd.Series(["fov1"] * 15 + ["fov2"] * 4)
    relabelled, moved = apply_min_cells(labels, images, 10)
    assert moved == {"B": 3, "A": 2}  # A is rare only in fov2
    assert relabelled.tolist() == ["A"] * 12 + ["Unknown"] * 7

    unchanged, moved = apply_min_cells(labels, images, 0)
    assert moved == {} and unchanged.tolist() == labels.tolist()
