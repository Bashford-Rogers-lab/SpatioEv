#!/usr/bin/env python3
"""Streamlit interface for comparing spatial summaries between sample groups."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root
from spatioev.apps._worker import (
    path_value,
    remember_worker,
    render_worker,
    show_images,
    start_worker,
)
from spatioev.io.qupath import find_cores
from spatioev.workflows.cohort_comparison import FAMILIES, read_sample_sheet

PREFIX = "cohort"
FAMILY_LABELS = {
    "composition": "Cell composition",
    "matrix": "Matrix per region",
    "cell_cell": "Cell–cell",
    "cell_matrix": "Cell–matrix",
}
SHEET_COLUMNS = ["sample_id", "group", "imageid", "patient_id", "project_root"]


def initialize_state() -> None:
    root = Path(st.session_state.get("spatioev_project_root", str(default_project_root()))).expanduser()
    st.session_state.setdefault("cohort_project_root", str(root))
    st.session_state.setdefault("cohort_sheet", str(root / "sample_sheet.csv"))
    st.session_state.setdefault("cohort_name", "hpv_comparison")
    st.session_state.setdefault("cohort_output", str(root / "results" / "cohort_comparison"))


def render_outputs(outputs: dict) -> None:
    units = outputs.get("units_per_group", {})
    metric1, metric2, metric3 = st.columns(3)
    groups = outputs.get("groups", [])
    metric1.metric(f"Units ({' vs '.join(groups)})", " vs ".join(str(units.get(g, 0)) for g in groups))
    metric2.metric("Features tested", f"{outputs.get('n_features', 0):,}")
    metric3.metric("q < 0.05", outputs.get("n_significant_q05", 0))

    comparison = pd.read_csv(outputs["comparison"]) if outputs.get("comparison") and Path(outputs["comparison"]).exists() else pd.DataFrame()
    if not comparison.empty and "min_achievable_p" in comparison:
        floor = comparison["min_achievable_p"].min()
        if floor > 0.05:
            st.warning(
                f"With these group sizes the smallest possible Mann–Whitney p-value is {floor:.3g}, so no feature can reach "
                "p < 0.05 however large the difference. Treat results as descriptive, or add patients.",
                icon=":material/warning:",
            )
    show_images([outputs.get("top_features_png")], columns=1)
    if not comparison.empty:
        families = st.pills(
            "Show feature families", list(FAMILY_LABELS), default=list(FAMILY_LABELS), selection_mode="multi",
            format_func=FAMILY_LABELS.get, key="cohort_show_families",
        )
        shown = comparison[comparison["family"].isin(families or [])]
        st.dataframe(shown, hide_index=True, width="stretch", height=480)
        st.caption("p: Mann–Whitney U (two groups) or Kruskal–Wallis (more), on one value per unit. q: Benjamini–Hochberg across all features.")
    if outputs.get("inputs_found") and Path(outputs["inputs_found"]).exists():
        with st.expander("Which per-sample outputs were found"):
            st.dataframe(pd.read_csv(outputs["inputs_found"]), hide_index=True, width="stretch")
    st.code(outputs.get("manifest", ""), language=None)


def main() -> None:
    st.set_page_config(page_title="Cohort comparison", page_icon=":material/compare_arrows:", layout="wide")
    initialize_state()
    st.title("Cohort comparison")
    st.caption(
        "Compare composition, matrix and co-localisation summaries between groups such as HPV+ and HPV−. "
        "Run Tissue regions and Co-localisation for every sample first; values are averaged per patient before testing."
    )

    with st.container(border=True):
        left, right = st.columns(2)
        left.text_input("Sample sheet CSV", key="cohort_sheet")
        right.text_input("Default project root (where each sample's results/ folder is)", key="cohort_project_root")
        left.text_input("Comparison name", key="cohort_name")
        right.text_input("Output folder", key="cohort_output")

    sheet_path = Path(path_value("cohort_sheet")).expanduser()
    with st.expander("Create or edit the sample sheet", icon=":material/edit_note:", expanded=not sheet_path.exists()):
        st.caption(
            "One row per sample (for a TMA, one row per core). A new sheet starts with every core folder found in the "
            "project root: fill in group and patient_id, or edit the CSV in Excel. Rows sharing a patient_id are "
            "averaged into one unit."
        )
        if sheet_path.exists():
            current = read_sample_sheet(sheet_path)
        else:
            cores = [core.name for core in find_cores(path_value("cohort_project_root"))] if path_value("cohort_project_root") else []
            current = pd.DataFrame({"sample_id": cores}, columns=SHEET_COLUMNS)
        current = current.reindex(columns=SHEET_COLUMNS).fillna("")
        edited = st.data_editor(current, num_rows="dynamic", width="stretch", key=f"cohort_sheet_editor_{sheet_path}")
        if st.button("Save sample sheet", icon=":material/save:", key="cohort_save_sheet"):
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            edited.to_csv(sheet_path, index=False)
            st.toast(f"Saved {sheet_path.name}", icon=":material/check:")
            st.rerun()

    if not sheet_path.exists():
        return
    try:
        sheet = read_sample_sheet(sheet_path)
    except Exception as error:
        st.error(str(error))
        return
    groups = sorted(sheet["group"].unique())
    left, right = st.columns(2)
    order = left.multiselect("Groups to compare (first = reference)", groups, default=groups, key="cohort_groups")
    families = right.pills(
        "Feature families", list(FAMILY_LABELS), default=list(FAMILY_LABELS), selection_mode="multi",
        format_func=FAMILY_LABELS.get, key="cohort_families",
    )
    left, right = st.columns(2)
    min_units = left.number_input("Minimum units per group to test a feature", 2, 1000, 3, 1, key="cohort_min_units")
    min_cells = right.number_input("Minimum cells for a region's composition to count", 1, 100000, 20, 1, key="cohort_min_cells")
    if "patient_id" in sheet and (sheet["patient_id"] == "").all() and (sheet["imageid"] != "").any():
        st.warning("Cores have no patient_id, so every core counts as an independent unit. Add patient IDs if patients have several cores.", icon=":material/warning:")

    if st.button("Run comparison", type="primary", icon=":material/play_arrow:", key="cohort_run"):
        if len(order) < 2:
            st.error("Choose at least two groups")
        elif not families:
            st.error("Choose at least one feature family")
        else:
            config = {
                "comparison_name": st.session_state["cohort_name"].strip() or "cohort",
                "sample_sheet": str(sheet_path),
                "project_root": path_value("cohort_project_root"),
                "output_dir": path_value("cohort_output"),
                "groups": order,
                "families": [f for f in families if f in FAMILIES],
                "min_units_per_group": int(min_units),
                "min_region_cells": int(min_cells),
            }
            try:
                output_dir = Path(config["output_dir"]).expanduser().resolve()
                remember_worker(PREFIX, *start_worker("spatioev.workflows.cohort_comparison", config, output_dir, "cohort"))
                st.rerun()
            except Exception as error:
                st.error(str(error))

    if st.session_state.get("cohort_status_path"):
        st.subheader("Comparison run")
        render_worker(PREFIX, render_outputs)


if __name__ == "__main__":
    main()
