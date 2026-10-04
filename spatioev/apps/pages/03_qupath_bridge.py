#!/usr/bin/env python3
"""Streamlit interface for classifying cells in QuPath and bringing the classes back."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root
from spatioev.apps._worker import (
    path_value,
    remember_worker,
    render_worker,
    start_worker,
)
from spatioev.io.qupath import QUPATH_SCRIPTS, write_qupath_scripts
from spatioev.workflows.qupath_bridge import scan

PREFIX = "qp"
TABLES = {
    "arcsinh_transformed": "Arcsinh (default)",
    "size_normalized": "Size normalised",
    "raw": "Raw",
}


def initialize_state() -> None:
    st.session_state.setdefault("qp_root", st.session_state.get("spatioev_project_root", str(default_project_root())))
    default_scripts = Path.home() / "QuPath" / "v0.7" / "scripts"
    st.session_state.setdefault("qp_scripts_dir", str(default_scripts) if default_scripts.parent.exists() else "")


def render_outputs(outputs: dict) -> None:
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Cores", outputs.get("cores", 0))
    metric2.metric("Written", outputs.get("written", 0))
    metric3.metric("Failed", outputs.get("failed", 0))
    if outputs.get("summary") and Path(outputs["summary"]).exists():
        summary = pd.read_csv(outputs["summary"])
        if "status" in summary and (summary["status"] == "failed").any():
            st.error("Some cores failed; see the error column.", icon=":material/error:")
        # Files sit at <core>/qupath/; the full path only pushes the counts off screen.
        st.dataframe(summary.drop(columns=["path"], errors="ignore"), hide_index=True, width="stretch", height=min(400, 38 * len(summary) + 40))
    counts_path = outputs.get("class_counts")
    if counts_path and Path(counts_path).exists() and Path(counts_path).stat().st_size > 30:
        counts = pd.read_csv(counts_path)
        st.markdown("**Cells per QuPath class and core**")
        pivot = counts.pivot_table(index="core", columns="qupath_class", values="count", aggfunc="sum", fill_value=0).astype(int)
        st.dataframe(pivot, width="stretch")
        st.caption(
            "Each core now has <core>/qupath/<core>_phenotyped.h5ad. In stages 04-06 set the project root to this folder and "
            "the sample ID to the core name. Use the qupath_class column to keep derived classes such as "
            "'Fibroblast: WNT5A+' separate, or phenotype for the base class only."
        )


def main() -> None:
    st.set_page_config(page_title="QuPath bridge", page_icon=":material/swap_horiz:", layout="wide")
    initialize_state()
    st.title("QuPath bridge")
    st.caption(
        "Classify cells in QuPath instead of stages 01-03, using SpatioEv's segmentation. Cells move by cell_id: "
        "one import into QuPath and one export back per core, never matched by position."
    )

    with st.container(border=True):
        left, right = st.columns([0.7, 0.3])
        left.text_input("TMA or project folder (one sub-folder per core)", key="qp_root")
        table = right.selectbox("Measurements for QuPath", list(TABLES), format_func=TABLES.get, key="qp_table")
        if st.button("Scan cores", icon=":material/search:", key="qp_scan"):
            st.session_state["qp_scan_result"] = scan([path_value("qp_root")], table)

    found = st.session_state.get("qp_scan_result")
    if isinstance(found, pd.DataFrame):
        if found.empty:
            st.warning("No core folders found. A core folder holds segmentation/ and quantification/.", icon=":material/warning:")
        else:
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Cores", len(found))
            metric2.metric("For QuPath", int(found["cells_geojson"].sum()))
            metric3.metric("From QuPath", int(found["classes"].sum()))
            metric4.metric("Collected", int(found["phenotyped"].sum()))
            with st.expander("Files per core"):
                st.dataframe(found.drop(columns="path"), hide_index=True, width="stretch")
            missing = found[~(found["cell_mask"] & found["table"])]
            if len(missing):
                st.warning(f"{len(missing)} cores lack a whole-cell mask or cell table and will be skipped.", icon=":material/warning:")

    st.subheader("1 · Prepare cells for QuPath")
    with st.container(border=True):
        left, right = st.columns(2)
        include_nuclear = left.checkbox("Include nuclear means ('<marker> nucleus')", value=True, key="qp_nuclear")
        overwrite = right.checkbox("Rewrite cell files that already exist", value=False, key="qp_overwrite")
        if st.button("Write QuPath cell files", type="primary", icon=":material/upload_file:", key="qp_prepare"):
            config = {
                "sample_id": "qupath", "mode": "prepare", "roots": [path_value("qp_root")], "table": table,
                "include_nuclear": bool(include_nuclear), "overwrite": bool(overwrite),
                "output_dir": str(Path(path_value("qp_root")).expanduser() / "qupath_bridge"),
            }
            remember_worker(PREFIX, *start_worker("spatioev.workflows.qupath_bridge", config, Path(config["output_dir"]), "prepare"))
            st.rerun()

    st.subheader("2 · Classify in QuPath")
    with st.container(border=True):
        st.markdown(
            "1. Create a QuPath project and add each core's **background/&lt;core&gt;.ome.tiff** (the same image the fiber "
            "segmentation uses, so coordinates line up).\n"
            "2. Run **spatioev_import_cells.groovy** with *Automate › Run for project*. Running it twice is safe: "
            "images that already hold the cells are skipped.\n"
            "3. Train and apply your object classifier on the imported measurements. Derived classes such as "
            "*Fibroblast: WNT5A+* are kept. Optionally draw annotations: classes starting with *Exclude*, *Ignore* or "
            "*Artefact* remove the cells inside them; other classes (e.g. *Tumour*) are recorded per cell.\n"
            "4. Run **spatioev_export_classes.groovy** for the project. Re-run it whenever the classifier changes."
        )
        left, right = st.columns([0.7, 0.3])
        left.text_input("Save the scripts to (e.g. your QuPath scripts folder)", key="qp_scripts_dir")
        if right.button("Save QuPath scripts", icon=":material/save:", width="stretch", key="qp_save_scripts"):
            try:
                written = write_qupath_scripts(path_value("qp_scripts_dir"))
                st.success("Saved " + ", ".join(p.name for p in written), icon=":material/check_circle:")
            except Exception as error:
                st.error(str(error))
        st.caption("Scripts: " + ", ".join(QUPATH_SCRIPTS))

    st.subheader("3 · Collect QuPath classes")
    with st.container(border=True):
        drop = st.checkbox("Drop cells inside Exclude / Ignore / Artefact annotations", value=True, key="qp_drop")
        if st.button("Collect classes", type="primary", icon=":material/download:", key="qp_collect"):
            config = {
                "sample_id": "qupath", "mode": "collect", "roots": [path_value("qp_root")], "table": table,
                "drop_excluded": bool(drop),
                "output_dir": str(Path(path_value("qp_root")).expanduser() / "qupath_bridge"),
            }
            remember_worker(PREFIX, *start_worker("spatioev.workflows.qupath_bridge", config, Path(config["output_dir"]), "collect"))
            st.rerun()

    if st.session_state.get("qp_status_path"):
        st.subheader("Last run")
        render_worker(PREFIX, render_outputs)


if __name__ == "__main__":
    main()
