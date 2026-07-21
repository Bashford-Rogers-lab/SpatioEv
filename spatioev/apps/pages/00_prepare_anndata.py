#!/usr/bin/env python3
"""Streamlit setup and review interface for CellSAM table-to-AnnData conversion."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from spatioev.apps._common import default_project_root, module_command
from spatioev.workflows.cellsam import ConversionPlan, preflight, write_json


EXAMPLE_ROOT = default_project_root()
EXAMPLE_TABLE_DIR = EXAMPLE_ROOT / "cell_table"
EXAMPLE_IMAGE = EXAMPLE_ROOT / "sample.ome.tif"
AUTO_REFRESH_SECONDS = 2.0


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def table_path(directory: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else directory / path


def start_conversion(plan: ConversionPlan, status_path: Path, log_path: Path) -> int:
    status_path.unlink(missing_ok=True)
    write_json(
        status_path,
        {
            "state": "queued",
            "stage": "queued",
            "message": "Starting conversion worker",
            "progress": 0.01,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    command = module_command(
        "spatioev.workflows.cellsam",
        "--primary-csv",
        str(plan.primary_csv),
        "--secondary-csv",
        str(plan.secondary_csv),
        "--image",
        str(plan.image_path),
        "--imageid",
        plan.imageid,
        "--output",
        str(plan.output_path),
        "--layer-name",
        plan.layer_name,
        "--status",
        str(status_path),
    )
    if not plan.make_qc:
        command.append("--no-qc")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process.pid


def render_status(status_path: Path, log_path: Path) -> None:
    status = read_json(status_path)
    if status is None:
        return
    state = status.get("state")
    message = status.get("message", "")
    if state == "failed":
        st.error(message)
        with st.expander("Error details"):
            st.code(status.get("traceback", "No traceback recorded"))
        return
    if state == "complete":
        st.success(message)
        outputs = status.get("outputs", {})
        if outputs:
            col1, col2, col3 = st.columns(3)
            col1.metric("Cells", f"{outputs.get('n_cells', 0):,}")
            col2.metric("Markers", outputs.get("n_markers", 0))
            col3.metric("Layer", outputs.get("layer_name", ""))
            st.code(outputs.get("output_path", ""), language=None)
            qc_outputs = outputs.get("qc_outputs", {})
            images = [
                qc_outputs.get("marker_histograms_png"),
                qc_outputs.get("spatial_qc_png"),
            ]
            image_columns = st.columns(sum(bool(path) for path in images) or 1)
            for column, image_path in zip(image_columns, (path for path in images if path)):
                column.image(image_path, width="stretch")
        return

    progress = float(status.get("progress", 0.02))
    st.progress(progress, text=message)
    st.caption(f"Stage: {status.get('stage', 'queued')}")
    if log_path.exists():
        with st.expander("Worker log"):
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-6000:])
    time.sleep(AUTO_REFRESH_SECONDS)
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="CellSAM to AnnData", page_icon=":material/table_view:", layout="wide")
    st.title("CellSAM quantification to AnnData")

    with st.container(border=True):
        left, right = st.columns([1, 1])
        table_dir_text = left.text_input("Cell table folder", value=str(EXAMPLE_TABLE_DIR))
        image_text = right.text_input("Source OME-TIFF", value=str(EXAMPLE_IMAGE))
        table_dir = Path(table_dir_text).expanduser()

        left, middle, right = st.columns([1, 1, 1])
        imageid = left.text_input("Image ID", value="sample")
        output_text = middle.text_input("Output H5AD", value=str(EXAMPLE_ROOT / "sample_adata.h5ad"))
        layer_name = right.text_input("Secondary layer name", value="size_normalized")

        with st.expander("Table assignment"):
            primary_text = st.text_input("adata.X source file", value="cell_table_arcsinh_transformed.csv")
            secondary_text = st.text_input("Layer source file", value="cell_table_size_normalized.csv")
            st.caption("Default: arcsinh-transformed expression in adata.X; size-normalized expression in the layer.")
        make_qc = st.checkbox("Generate conversion QC", value=True)

    plan = ConversionPlan(
        primary_csv=table_path(table_dir, primary_text),
        secondary_csv=table_path(table_dir, secondary_text),
        image_path=Path(image_text).expanduser(),
        imageid=imageid,
        output_path=Path(output_text).expanduser(),
        layer_name=layer_name,
        make_qc=make_qc,
    )

    inspect_column, build_column, _ = st.columns([0.18, 0.22, 0.60])
    if inspect_column.button("Inspect inputs", type="secondary", icon=":material/search:", width="stretch"):
        try:
            with st.spinner("Reading table headers and OME channel metadata"):
                st.session_state["cellsam_preflight"] = preflight(plan)
                st.session_state["cellsam_preflight_signature"] = repr(plan)
        except Exception as error:
            st.session_state.pop("cellsam_preflight", None)
            st.error(str(error))

    if build_column.button("Build AnnData", type="primary", icon=":material/play_arrow:", width="stretch"):
        try:
            preflight_report = preflight(plan)
            output_path = Path(plan.output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            status_path = output_path.with_suffix(".conversion_status.json")
            log_path = output_path.with_suffix(".conversion.log")
            pid = start_conversion(plan, status_path, log_path)
            st.session_state["cellsam_status_path"] = str(status_path)
            st.session_state["cellsam_log_path"] = str(log_path)
            st.session_state["cellsam_preflight"] = preflight_report
            st.session_state["cellsam_preflight_signature"] = repr(plan)
            st.toast(f"Conversion worker started (PID {pid})")
            st.rerun()
        except Exception as error:
            st.error(str(error))

    report = st.session_state.get("cellsam_preflight")
    if report and st.session_state.get("cellsam_preflight_signature") == repr(plan):
        st.subheader("Input check")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Markers", report["n_markers"])
        metric2.metric("Observation columns", report["n_obs_columns"])
        metric3.metric("Marker mismatches", 0)
        st.dataframe(report["marker_mapping"], hide_index=True, width="stretch", height=360)

    status_text = st.session_state.get("cellsam_status_path")
    log_text = st.session_state.get("cellsam_log_path")
    if status_text and log_text:
        st.subheader("Conversion")
        render_status(Path(status_text), Path(log_text))


if __name__ == "__main__":
    main()
