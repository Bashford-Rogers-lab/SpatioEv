#!/usr/bin/env python3
"""Streamlit interface for SCIMAP phenotyping within selected broad populations."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root, module_command, resource_path
from spatioev.workflows._io import read_json
from spatioev.workflows.scimap_phenotyping import inspect_inputs, write_json

AUTO_REFRESH_SECONDS = 2.0


def standard_defaults(sample_id: str, project_root: Path) -> dict[str, str]:
    sample_id = sample_id.strip()
    gating_dir = project_root / "results" / f"{sample_id}_marker_gating_qc"
    manifest_path = gating_dir / f"{sample_id}_interactive_gate_review_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    return {
        "scimap_gated_h5ad": str(
            manifest.get("gated_h5ad", gating_dir / f"{sample_id}_autogated_interactive.h5ad")
        ),
        "scimap_annotations": str(project_root / "results" / f"{sample_id}_phenotyping_annotations.csv"),
        "scimap_image": str(
            manifest.get(
                "image_path",
                project_root / sample_id / "processed" / f"{sample_id}_combined.ome.tif",
            )
        ),
        "scimap_gates": str(
            manifest.get(
                "scimap_manual_gates",
                gating_dir / f"{sample_id}_scimap_manual_gates_interactive.csv",
            )
        ),
        "scimap_workflow": str(resource_path("hcc_immune_phenotype_workflow_example.csv")),
        "scimap_output": str(project_root / "results" / f"{sample_id}_scimap_phenotyping_interface"),
    }


def initialize_state() -> None:
    if "scimap_sample_id" not in st.session_state:
        st.session_state["scimap_sample_id"] = "sample"
    if "scimap_project_root" not in st.session_state:
        st.session_state["scimap_project_root"] = str(default_project_root())
    defaults = standard_defaults(
        st.session_state["scimap_sample_id"],
        Path(st.session_state["scimap_project_root"]).expanduser(),
    )
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)




def inspection_config() -> dict:
    return {
        "gated_h5ad": st.session_state["scimap_gated_h5ad"],
        "broad_annotations": st.session_state["scimap_annotations"],
        "image_path": st.session_state["scimap_image"],
        "gate_csv": st.session_state["scimap_gates"],
        "workflow_csv": st.session_state["scimap_workflow"],
    }


def start_worker(config: dict, output_dir: Path) -> tuple[int, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / f"{config['sample_id']}_scimap_interface_config.json"
    status_path = output_dir / f"{config['sample_id']}_scimap_interface_status.json"
    log_path = output_dir / f"{config['sample_id']}_scimap_interface_worker.log"
    write_json(config_path, config)
    write_json(
        status_path,
        {
            "state": "queued",
            "stage": "queued",
            "message": "Starting SCIMAP worker",
            "progress": 0.01,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    command = module_command(
        "spatioev.workflows.scimap_phenotyping",
        "--config",
        str(config_path),
        "--status",
        str(status_path),
    )
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process.pid, status_path, log_path


def launch_napari(
    adata_path: str,
    image_path: str,
    label: str,
    output_dir: Path,
    imageid: str | None = None,
) -> tuple[int, Path]:
    log_path = output_dir / "scimap_napari_review.log"
    command = module_command(
        "spatioev.workflows.clustering_review",
        "--adata",
        adata_path,
        "--image",
        image_path,
        "--label",
        label,
    )
    if imageid is not None:
        command.extend(["--imageid", imageid])
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process.pid, log_path


@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def _render_running_worker(status_path: Path, log_path: Path) -> None:
    """Live worker progress, refreshed in isolation.

    Polling with ``time.sleep()`` + ``st.rerun()`` restarts the whole script
    every couple of seconds, which discards widget interactions made in
    between -- selectors snap back to their defaults while a job runs.
    """
    status = read_json(status_path)
    if status is None:
        return
    if status.get("state") in {"complete", "failed"}:
        # Job finished; refresh the whole page so the results render.
        st.rerun()

    st.progress(float(status.get("progress", 0.02)), text=status.get("message", "Running"))
    st.caption(f"Stage: {status.get('stage', 'queued')}")
    if log_path.exists():
        with st.expander("Worker log"):
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-8000:])


def render_status(status_path: Path, log_path: Path, image_path: str, output_dir: Path) -> None:
    status = read_json(status_path)
    if status is None:
        return
    state = status.get("state")
    if state == "failed":
        st.error(status.get("message", "SCIMAP worker failed"))
        with st.expander("Error details"):
            st.code(status.get("traceback", "No traceback recorded"))
        return
    if state != "complete":
        _render_running_worker(status_path, log_path)
        return

    outputs = status.get("outputs", {})
    st.success(status.get("message", "SCIMAP phenotyping complete"))
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Selected cells", f"{outputs.get('n_subset_cells', 0):,}")
    metric2.metric("Phenotypes", outputs.get("n_phenotypes", 0))
    metric3.metric("Full tissue cells", f"{outputs.get('n_full_cells', 0):,}")

    button_column, _ = st.columns([0.25, 0.75])
    if button_column.button("Open napari review", icon=":material/open_in_new:", width="stretch"):
        pid, napari_log = launch_napari(
            outputs["subset_h5ad"],
            outputs.get("review_image_path", image_path),
            outputs["phenotype_label"],
            output_dir,
            outputs.get("review_imageid"),
        )
        st.toast(f"Napari started (PID {pid})")
        st.caption(f"Napari log: {napari_log}")

    st.code(outputs.get("subset_h5ad", ""), language=None)
    image_paths = [
        outputs.get("count_png"),
        outputs.get("heatmap_png"),
        outputs.get("subset_spatial_png"),
        outputs.get("full_spatial_png"),
        *outputs.get("image_overlay_paths", []),
    ]
    image_paths = [path for path in image_paths if path and Path(path).exists()]
    for start in range(0, len(image_paths), 2):
        columns = st.columns(2)
        for column, path in zip(columns, image_paths[start : start + 2]):
            column.image(path, width="stretch")
            column.caption(Path(path).name)


def main() -> None:
    st.set_page_config(page_title="SCIMAP subset phenotyping", page_icon=":material/account_tree:", layout="wide")
    initialize_state()
    st.title("SCIMAP subset phenotyping")

    sample_column, root_column, fill_column = st.columns([0.22, 0.50, 0.28])
    sample_column.text_input("Sample ID", key="scimap_sample_id")
    root_column.text_input("Project root", key="scimap_project_root")
    if fill_column.button("Fill standard sample paths", icon=":material/auto_fix_high:", width="stretch"):
        root = Path(st.session_state["scimap_project_root"]).expanduser()
        for key, value in standard_defaults(st.session_state["scimap_sample_id"], root).items():
            st.session_state[key] = value
        st.session_state.pop("scimap_inspection", None)
        st.session_state.pop("scimap_status_path", None)
        st.session_state.pop("scimap_log_path", None)
        st.rerun()

    with st.container(border=True):
        left, right = st.columns(2)
        left.text_input("Gated full-marker H5AD", key="scimap_gated_h5ad")
        right.text_input("Broad annotation CSV", key="scimap_annotations")
        left.text_input("Source OME-TIFF or FOV image folder", key="scimap_image")
        right.text_input("Reviewed SCIMAP gate CSV", key="scimap_gates")
        left.text_input("Phenotype workflow CSV", key="scimap_workflow")
        right.text_input("Output folder", key="scimap_output")

    inspect_column, _ = st.columns([0.22, 0.78])
    if inspect_column.button("Inspect inputs", icon=":material/search:", width="stretch"):
        try:
            with st.spinner("Validating cells, markers, gates, and phenotype hierarchy"):
                report = inspect_inputs(inspection_config())
            st.session_state["scimap_inspection"] = report
            st.session_state["scimap_inspection_signature"] = json.dumps(inspection_config(), sort_keys=True)
        except Exception as error:
            st.session_state.pop("scimap_inspection", None)
            st.error(str(error))

    report = st.session_state.get("scimap_inspection")
    signature = json.dumps(inspection_config(), sort_keys=True)
    if report and st.session_state.get("scimap_inspection_signature") == signature:
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Cells", f"{report['n_cells']:,}")
        metric2.metric("Markers", report["n_markers"])
        metric3.metric("Workflow phenotypes", report["workflow"]["workflow_rows"])
        # A key is required: without one the widget has no session-state entry,
        # so the recomputed `index` forces the value back to the default on
        # every rerun and the selection can never stick.
        review_imageid = st.selectbox(
            "FOV used for original-image overlays and napari review",
            report["imageids"],
            key="scimap_review_imageid",
        )
        st.caption(
            f"The SCIMAP model uses all selected cells; image review uses {review_imageid}."
        )

        st.subheader("Population selection")
        candidate_columns = list(report["candidate_columns"])
        default_column = "annotation_level2" if "annotation_level2" in candidate_columns else candidate_columns[0]
        # Keyed so the choice persists; a recomputed `index` with no key resets
        # the widget to the default on every rerun.
        st.session_state.setdefault("scimap_broad_column", default_column)
        broad_column = st.selectbox(
            "Broad annotation column",
            candidate_columns,
            key="scimap_broad_column",
        )
        population_rows = report["candidate_columns"][broad_column]
        population_options = [row["value"] for row in population_rows]
        default_populations = ["immune"] if "immune" in population_options else population_options[:1]
        selected_populations = st.multiselect(
            "Populations to phenotype",
            population_options,
            default=default_populations,
        )
        counts = pd.DataFrame(population_rows)
        counts["selected"] = counts["value"].isin(selected_populations)
        st.dataframe(counts, hide_index=True, width="stretch", height=min(360, 38 * len(counts) + 40))

        left, middle, right = st.columns(3)
        subset_name = left.text_input("Subset output name", value="", placeholder="Automatic from selected populations")
        phenotype_label = middle.text_input("SCIMAP phenotype column", value="scimap_phenotype")
        final_label = right.text_input("Combined final column", value="final_hierarchical_phenotype")

        st.subheader("Phenotype workflow")
        workflow_summary = report["workflow"]
        summary1, summary2, summary3 = st.columns(3)
        summary1.metric("Hierarchy rows", workflow_summary["workflow_rows"])
        summary2.metric("Workflow markers", len(workflow_summary["workflow_markers"]))
        summary3.metric("Reviewed gates", workflow_summary["gate_markers"])
        st.caption("Top-level phenotypes: " + ", ".join(workflow_summary["top_level_phenotypes"]))
        with st.expander("Preview phenotype workflow"):
            st.dataframe(report["workflow_preview"], hide_index=True, width="stretch", height=420)

        with st.expander("Run settings"):
            setting1, setting2, setting3 = st.columns(3)
            phenotype_gate = setting1.number_input("Phenotype positivity threshold", 0.0, 1.0, 0.5, 0.01)
            pheno_threshold_abs = setting2.number_input("Minimum cells per phenotype", 1, 10000, 10, 1)
            plot_min_cells = setting3.number_input("Minimum cells shown as separate QC label", 1, 10000, 50, 1)
            write_full_h5ad = st.checkbox("Write full-tissue combined H5AD", value=True)
            make_image_overlays = st.checkbox("Generate original-image crop overlays", value=True)
            crop1, crop2 = st.columns(2)
            overlay_crop_size = crop1.number_input("Overlay crop size", 512, 4096, 1536, 128)
            overlay_n_crops = crop2.number_input("Overlay crop count", 1, 6, 3, 1)

        run_column, _ = st.columns([0.26, 0.74])
        if run_column.button("Run SCIMAP phenotyping", type="primary", icon=":material/play_arrow:", width="stretch"):
            if not selected_populations:
                st.error("Select at least one population")
            else:
                config = {
                    **inspection_config(),
                    "sample_id": st.session_state["scimap_sample_id"].strip(),
                    "output_dir": st.session_state["scimap_output"],
                    "broad_label_column": broad_column,
                    "selected_populations": selected_populations,
                    "subset_name": subset_name,
                    "phenotype_label": phenotype_label,
                    "final_label": final_label,
                    "phenotype_gate": float(phenotype_gate),
                    "pheno_threshold_abs": int(pheno_threshold_abs),
                    "plot_min_cells": int(plot_min_cells),
                    "write_full_h5ad": bool(write_full_h5ad),
                    "make_image_overlays": bool(make_image_overlays),
                    "overlay_crop_size": int(overlay_crop_size),
                    "overlay_n_crops": int(overlay_n_crops),
                    "review_imageid": review_imageid,
                }
                output_dir = Path(config["output_dir"]).expanduser().resolve()
                try:
                    pid, status_path, log_path = start_worker(config, output_dir)
                    st.session_state["scimap_status_path"] = str(status_path)
                    st.session_state["scimap_log_path"] = str(log_path)
                    st.session_state["scimap_run_image"] = config["image_path"]
                    st.session_state["scimap_run_output"] = str(output_dir)
                    st.toast(f"SCIMAP worker started (PID {pid})")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

    status_text = st.session_state.get("scimap_status_path")
    log_text = st.session_state.get("scimap_log_path")
    if status_text and log_text:
        st.subheader("SCIMAP run")
        render_status(
            Path(status_text),
            Path(log_text),
            st.session_state.get("scimap_run_image", st.session_state["scimap_image"]),
            Path(st.session_state.get("scimap_run_output", st.session_state["scimap_output"])),
        )


if __name__ == "__main__":
    main()
