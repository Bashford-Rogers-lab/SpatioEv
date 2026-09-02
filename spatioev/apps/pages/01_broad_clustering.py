#!/usr/bin/env python3
"""Streamlit interface for unsupervised clustering and broad phenotyping."""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "spatioev_clustering_app_matplotlib"),
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "spatioev_clustering_app_numba"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root, module_command
from spatioev.workflows import marker_gating as mgq
from spatioev.workflows._io import read_json, write_json
from spatioev.workflows.cellsam import describe_unreadable_h5ad
from spatioev.workflows.image_collection import natural_key, resolve_image

PROJECT_ROOT_DEFAULT = default_project_root()
AUTO_REFRESH_SECONDS = 2.0
STALE_JOB_SECONDS = 30 * 60
JOB_STAGE_PROGRESS = {
    "queued": 0.02,
    "load": 0.08,
    "normalize": 0.20,
    "cluster": 0.55,
    "write": 0.90,
    "complete": 1.0,
}
CORE_MARKERS = [
    "HOECHST2",
    "CD39",
    "CD34",
    "LYVE1",
    "CD45",
    "CD68",
    "aSMA",
    "panCK",
    "EpCAM",
    "HNF4a",
]
FINAL_LABELS = [
    "",
    "immune",
    "LSEC",
    "vascular_cell",
    "aSMA_stroma",
    "hepatocyte",
    "cholangiocyte",
    "tumour",
    "epithelial",
    "artifact",
    "necrosis",
    "nerve",
]
LEVEL0_LABELS = FINAL_LABELS + [
    "fibro_immune",
    "immune_fibro",
    "immune_LSEC",
    "LSEC_immune",
    "vas_immune",
    "vascular_immune",
    "fibro_vas",
    "aSMA_vas",
    "tumour_hep",
    "immune_tumour",
    "artifact_tumour",
    "vas_tumour",
    "other",
]


def standard_paths(sample_id: str, project_root: Path) -> dict[str, Path]:
    paths = mgq.default_sample_paths(sample_id, project_root)
    return {
        "adata": paths.adata_path,
        "image": paths.image_path,
        "output": project_root / "results" / f"{sample_id}_clustering_workflow",
    }


@st.cache_data(show_spinner=False)
def sample_metadata(adata_path: str, image_path: str) -> dict[str, object]:
    adata = ad.read_h5ad(adata_path, backed="r")
    imageids = []
    if "imageid" in adata.obs:
        imageids = sorted(
            adata.obs["imageid"].dropna().astype(str).unique().tolist(),
            key=natural_key,
        )
        imageid_counts = {
            str(imageid): int(count)
            for imageid, count in adata.obs["imageid"]
            .astype(str)
            .value_counts()
            .items()
        }
    else:
        imageid_counts = {}
    metadata = {
        "cells": int(adata.n_obs),
        "markers": [str(marker) for marker in adata.var_names],
        "obs_columns": list(adata.obs.columns),
        "layers": list(adata.layers.keys()),
    }
    adata.file.close()
    resolved_image = resolve_image(Path(image_path), imageids[0] if imageids else None)
    channels = mgq.canonical_channel_names(resolved_image, metadata["markers"])
    metadata["image_channels"] = channels
    metadata["missing_image_channels"] = sorted(
        set(metadata["markers"]) - set(channels)
    )
    metadata["has_coordinates"] = {"X_centroid", "Y_centroid"}.issubset(
        metadata["obs_columns"]
    )
    metadata["imageids"] = imageids
    metadata["imageid_counts"] = imageid_counts
    metadata["default_image_path"] = str(resolved_image)
    return metadata


def recommended_defaults(sample_id: str, markers: list[str]) -> tuple[list[str], float]:
    """Choose a conservative panel-aware starting point for broad clustering."""

    selected = [marker for marker in CORE_MARKERS if marker in markers]
    return selected, 0.4


def start_worker(
    action: str, config_path: Path, status_path: Path, log_path: Path
) -> int:
    status_path.unlink(missing_ok=True)
    write_json(
        status_path,
        {
            "state": "queued",
            "message": f"Starting {action} worker",
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": "queued",
        },
    )
    command = module_command(
        "spatioev.workflows.clustering",
        "--action",
        action,
        "--config",
        str(config_path),
        "--status",
        str(status_path),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return process.pid


def launch_napari(
    adata_path: Path,
    image_path: Path,
    output_dir: Path,
    name: str,
    imageid: str | None = None,
) -> tuple[int, Path]:
    log_path = output_dir / f"{name}_napari.log"
    command = module_command(
        "spatioev.workflows.clustering_review",
        "--adata",
        str(adata_path),
        "--image",
        str(image_path),
        "--label",
        "leiden",
    )
    if imageid is not None:
        command.extend(["--imageid", imageid])
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return process.pid, log_path


@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def _render_running_job(status_path: Path, *, label: str) -> None:
    """Live progress for an in-flight worker, refreshed in isolation.

    This is a fragment so that polling reruns only this block. Sleeping and
    calling ``st.rerun()`` on the whole page instead would restart the entire
    script every couple of seconds, which discards widget interactions the
    user makes in between -- the FOV selector would snap back to its default
    while a job was running.
    """
    status = read_json(status_path)
    if status is None:
        return

    state = status.get("state")
    if state in {"complete", "failed"}:
        # The job finished while we were polling; refresh the whole page so
        # the downstream sections (QC images, mapping editor) render.
        st.rerun()

    message = status.get("message", "")
    stage = str(status.get("stage", "queued"))
    progress = JOB_STAGE_PROGRESS.get(stage, 0.05)
    st.progress(progress, text=f"{label}: {message}")
    if stage == "cluster":
        st.caption(
            "PCA, neighbor graph, UMAP, and Leiden are running. This is usually the longest stage; "
            "large slides can take several minutes."
        )

    updated_at = status.get("updated_at")
    age_seconds = 0.0
    if updated_at:
        try:
            timestamp = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
            age_seconds = max(
                0.0, (datetime.now(UTC) - timestamp).total_seconds()
            )
            st.caption(
                f"Last worker update: {int(age_seconds):,} seconds ago. Refreshing automatically."
            )
        except ValueError:
            pass

    if age_seconds > STALE_JOB_SECONDS:
        st.warning(
            f"No worker update for more than {STALE_JOB_SECONDS // 60} minutes. "
            "Check the worker log before rerunning the job."
        )
        st.button(
            "Refresh status",
            key=f"refresh_{status_path.name}",
            icon=":material/refresh:",
        )


def render_job(status_path: Path, *, label: str) -> dict | None:
    status = read_json(status_path)
    if status is None:
        return None
    state = status.get("state")
    message = status.get("message", "")
    if state == "complete":
        st.success(f"{label}: {message}")
    elif state == "failed":
        st.error(f"{label}: {message}")
        with st.expander("Worker error details"):
            st.code(status.get("traceback", "No traceback available"))
    else:
        _render_running_job(status_path, label=label)
    return status


def mapping_editor(
    summary: pd.DataFrame,
    *,
    key_prefix: str,
    options: list[str],
    existing_path: Path | None = None,
) -> pd.DataFrame:
    base = summary[["cluster", "n_cells", "fraction", "top_markers"]].copy()
    base["annotation"] = ""
    base["custom_annotation"] = ""
    if existing_path is not None and existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing["cluster"] = existing["cluster"].astype(str)
        base["cluster"] = base["cluster"].astype(str)
        base = base.drop(columns=["annotation", "custom_annotation"]).merge(
            existing[
                [
                    column
                    for column in ["cluster", "annotation", "custom_annotation"]
                    if column in existing
                ]
            ],
            on="cluster",
            how="left",
        )
        if "custom_annotation" not in base:
            base["custom_annotation"] = ""
        base["annotation"] = base["annotation"].fillna("")
        base["custom_annotation"] = base["custom_annotation"].fillna("")
    return st.data_editor(
        base,
        key=f"{key_prefix}_mapping_editor",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "cluster": st.column_config.TextColumn(
                "Cluster", disabled=True, width="small"
            ),
            "n_cells": st.column_config.NumberColumn(
                "Cells", disabled=True, format="%d", width="small"
            ),
            "fraction": st.column_config.NumberColumn(
                "Fraction", disabled=True, format="percent", width="small"
            ),
            "top_markers": st.column_config.TextColumn(
                "Top markers", disabled=True, width="medium"
            ),
            "annotation": st.column_config.SelectboxColumn(
                "Annotation", options=options, required=True, width="medium"
            ),
            "custom_annotation": st.column_config.TextColumn(
                "Custom label",
                help="Used instead of Annotation when non-empty",
                width="medium",
            ),
        },
    )


def final_mapping(editor: pd.DataFrame) -> pd.DataFrame:
    output = editor[["cluster", "annotation", "custom_annotation"]].copy()
    output["annotation"] = (
        output["custom_annotation"]
        .fillna("")
        .astype(str)
        .str.strip()
        .where(
            output["custom_annotation"].fillna("").astype(str).str.strip().ne(""),
            output["annotation"].fillna("").astype(str).str.strip(),
        )
    )
    return output


def incomplete_mapping(mapping: pd.DataFrame) -> list[str]:
    missing = mapping["annotation"].fillna("").astype(str).str.strip().eq("")
    return mapping.loc[missing, "cluster"].astype(str).tolist()


def main() -> None:
    st.set_page_config(page_title="01 Clustering and broad phenotyping", layout="wide")
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1500px; }
    h1 { font-size: 1.8rem !important; letter-spacing: 0 !important; }
    h2, h3 { letter-spacing: 0 !important; }
    [data-testid="stMetric"] { border-left: 3px solid #277c75; padding-left: 0.8rem; }
    button[kind="primary"] { background: #277c75 !important; border-color: #277c75 !important; }
    button[kind="primary"]:hover { background: #1f665f !important; border-color: #1f665f !important; }
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("01 Unsupervised clustering and broad phenotyping")
    st.caption(
        "Identify broad tissue populations before marker autogating and prior-knowledge phenotyping."
    )

    st.subheader("1. Sample setup")
    if "cluster_sample_id" not in st.session_state:
        st.session_state.cluster_sample_id = "sample"
    if "cluster_project_root" not in st.session_state:
        st.session_state.cluster_project_root = str(PROJECT_ROOT_DEFAULT)
    a, b, c = st.columns([1, 2.2, 1.2])
    with a:
        sample_id = st.text_input("Sample ID", key="cluster_sample_id")
    with b:
        project_root_text = st.text_input("Project root", key="cluster_project_root")
    project_root = Path(project_root_text).expanduser()
    defaults = standard_paths(sample_id, project_root)
    with c:
        fill_paths = st.button("Fill standard sample paths", icon=":material/refresh:")

    initial_paths = {
        "cluster_adata_input": str(defaults["adata"]),
        "cluster_image_input": str(defaults["image"]),
        "cluster_output_input": str(defaults["output"]),
    }
    for key, value in initial_paths.items():
        if key not in st.session_state:
            st.session_state[key] = value
        if fill_paths:
            st.session_state[key] = value
    p1, p2 = st.columns(2)
    with p1:
        adata_text = st.text_input("AnnData (.h5ad)", key="cluster_adata_input")
        output_text = st.text_input(
            "Workflow output directory", key="cluster_output_input"
        )
    with p2:
        image_text = st.text_input(
            "OME-TIFF image or FOV image folder", key="cluster_image_input"
        )

    adata_path = Path(adata_text).expanduser()
    image_path = Path(image_text).expanduser()
    output_dir = Path(output_text).expanduser()
    if st.button("Load sample", type="primary", icon=":material/folder_open:"):
        errors = []
        # is_file(), not exists(): a blank box yields Path("") == Path("."),
        # which exists as a directory and would pass an existence check only to
        # fail later inside read_h5ad.
        if not adata_text.strip():
            errors.append("AnnData path is not set")
        elif not adata_path.is_file():
            errors.append(
                f"AnnData is not a file: {adata_path}"
                + (" (this is a directory)" if adata_path.is_dir() else "")
            )
        else:
            # A H5AD left behind by an interrupted step 1 still opens far
            # enough to reach the root group, then fails with h5py's "unable to
            # determine object type". Say what is wrong instead.
            unreadable = describe_unreadable_h5ad(adata_path)
            if unreadable:
                errors.append(unreadable)
        if not image_text.strip():
            errors.append("Image path is not set")
        elif not image_path.exists():
            # The image may be a single OME-TIFF or a folder of per-FOV images.
            errors.append(f"Image does not exist: {image_path}")
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                metadata = sample_metadata(str(adata_path), str(image_path))
                if not metadata["has_coordinates"]:
                    st.error("AnnData is missing X_centroid or Y_centroid")
                elif metadata["missing_image_channels"]:
                    st.error(
                        f"Markers missing from image: {', '.join(metadata['missing_image_channels'])}"
                    )
                else:
                    st.session_state.cluster_metadata = metadata
                    st.session_state.cluster_loaded_sample = sample_id
                    st.session_state.cluster_loaded_paths = {
                        "adata": str(adata_path),
                        "image": str(image_path),
                        "output": str(output_dir),
                        "project_root": str(project_root),
                    }
                    st.session_state.pop("level0_mapping_saved", None)
                    st.success(
                        f"Loaded {sample_id}: {metadata['cells']:,} cells and {len(metadata['markers'])} markers."
                    )
            except Exception as exc:
                st.exception(exc)

    if "cluster_metadata" not in st.session_state:
        st.info("Load a sample to configure Level-0 clustering.")
        return

    metadata = st.session_state.cluster_metadata
    loaded_paths = {
        key: Path(value) for key, value in st.session_state.cluster_loaded_paths.items()
    }
    sample_id = st.session_state.cluster_loaded_sample
    output_dir = loaded_paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cells", f"{metadata['cells']:,}")
    m2.metric("Markers", len(metadata["markers"]))
    m3.metric("Image channels", len(metadata["image_channels"]))
    m4.metric("Missing channels", len(metadata["missing_image_channels"]))
    review_imageid = None
    review_image = Path(metadata["default_image_path"])
    if metadata["imageids"]:
        review_imageid = st.selectbox(
            "FOV selected for clustering or image review",
            metadata["imageids"],
            key=f"cluster_review_imageid_{sample_id}",
        )
        review_image = resolve_image(loaded_paths["image"], review_imageid)
        st.caption(f"Image review uses {review_imageid}: {review_image}")

    scope_options = ["All FOVs jointly"]
    if metadata["imageids"]:
        scope_options.append("Selected FOV only")
    cluster_scope = st.segmented_control(
        "Clustering scope",
        scope_options,
        default="All FOVs jointly",
        key=f"cluster_scope_{sample_id}",
    )
    cluster_imageid = review_imageid if cluster_scope == "Selected FOV only" else None
    scope_suffix = f"_{cluster_imageid}" if cluster_imageid is not None else ""
    scoped_id = f"{sample_id}{scope_suffix}"
    if cluster_imageid is None:
        st.caption(f"Clustering all {metadata['cells']:,} cells jointly.")
    else:
        scoped_cells = metadata["imageid_counts"].get(cluster_imageid, 0)
        st.caption(
            f"Clustering only {cluster_imageid}: {scoped_cells:,} cells. "
            "Normalization, PCA, neighbors, UMAP, Leiden, refinement, and export remain FOV-specific."
        )

    st.subheader("2. Level-0 clustering")
    recommended_markers, recommended_resolution = recommended_defaults(
        sample_id, metadata["markers"]
    )
    preset = st.selectbox(
        "Marker preset",
        ["Notebook-informed recommendation", "Core lineage panel", "Custom"],
        help="The recommendation reproduces the marker choices used in the reference notebooks.",
    )
    if preset == "Core lineage panel":
        default_markers = [
            marker for marker in CORE_MARKERS if marker in metadata["markers"]
        ]
    else:
        default_markers = recommended_markers
    marker_key = f"level0_markers_{sample_id}_{preset}"
    selected_markers = st.multiselect(
        "Markers used for broad clustering",
        options=metadata["markers"],
        default=default_markers,
        key=marker_key,
    )
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        resolution = st.number_input(
            "Leiden resolution", 0.1, 2.0, recommended_resolution, 0.1
        )
    with q2:
        n_neighbors = st.number_input("Neighbors", 3, 100, 10, 1)
    with q3:
        n_pcs = st.number_input("Requested PCs", 2, 50, 15, 1)
    with q4:
        scale = st.toggle("Z-score markers", value=True)

    level0_config_path = output_dir / f"{scoped_id}_level0_config.json"
    level0_status_path = output_dir / f"{scoped_id}_level0_status.json"
    level0_log_path = output_dir / f"{scoped_id}_level0_worker.log"
    run_disabled = len(selected_markers) < 3
    if st.button(
        "Run Level-0 clustering",
        type="primary",
        disabled=run_disabled,
        icon=":material/hub:",
    ):
        config = {
            "sample_id": sample_id,
            "adata_path": str(loaded_paths["adata"]),
            "image_path": str(loaded_paths["image"]),
            "output_dir": str(output_dir),
            "markers": selected_markers,
            "resolution": float(resolution),
            "n_neighbors": int(n_neighbors),
            "n_pcs": int(n_pcs),
            "scale": bool(scale),
            "cluster_imageid": cluster_imageid,
        }
        write_json(level0_config_path, config)
        pid = start_worker(
            "level0", level0_config_path, level0_status_path, level0_log_path
        )
        st.success(f"Level-0 worker started (process {pid}).")

    level0_status = render_job(level0_status_path, label="Level-0")
    if not level0_status or level0_status.get("state") != "complete":
        return
    level0_outputs = {
        key: Path(value) if isinstance(value, str) else value
        for key, value in level0_status["outputs"].items()
    }
    st.caption(
        f"Checkpoint: {level0_outputs['n_cells']:,} cells, {level0_outputs['n_clusters']} clusters. "
        f"Worker log: {level0_log_path}"
    )
    qc1, qc2 = st.columns(2)
    with qc1:
        st.image(
            str(level0_outputs["umap_png"]), caption="Level-0 UMAP", width="stretch"
        )
    with qc2:
        st.image(
            str(level0_outputs["heatmap_png"]),
            caption="Cluster marker heatmap",
            width="stretch",
        )
    if st.button("Launch Level-0 napari review", icon=":material/open_in_new:"):
        pid, log_path = launch_napari(
            level0_outputs["clustered_h5ad"],
            review_image,
            output_dir,
            f"{scoped_id}_level0",
            review_imageid,
        )
        st.success(f"Napari launched (process {pid}). Log: {log_path}")

    st.subheader("3. Assign broad cluster labels")
    st.caption(
        "Use UMAP, the marker heatmap, and napari spatial context together. Mixed labels can be selected for later refinement."
    )
    level0_mapping_path = output_dir / f"{scoped_id}_level0_annotation_mapping.csv"
    level0_summary = pd.read_csv(level0_outputs["summary_csv"], dtype={"cluster": str})
    level0_editor = mapping_editor(
        level0_summary,
        key_prefix=f"{scoped_id}_level0",
        options=LEVEL0_LABELS,
        existing_path=level0_mapping_path,
    )
    level0_mapping = final_mapping(level0_editor)
    missing_level0 = incomplete_mapping(level0_mapping)
    if missing_level0:
        st.warning(
            f"Annotations still required for clusters: {', '.join(missing_level0)}"
        )
    if st.button(
        "Save Level-0 annotations",
        disabled=bool(missing_level0),
        icon=":material/save:",
    ):
        level0_mapping.to_csv(level0_mapping_path, index=False)
        st.session_state.level0_mapping_saved = True
        st.success(f"Saved: {level0_mapping_path}")
    if not level0_mapping_path.exists():
        return

    saved_level0 = pd.read_csv(level0_mapping_path, dtype={"cluster": str})
    parent_labels = sorted(saved_level0["annotation"].dropna().astype(str).unique())
    st.subheader("4. Optional cluster refinement")
    refine_parents = st.multiselect(
        "Broad or mixed populations to recluster",
        options=parent_labels,
        help="Leave empty when Level-0 labels are already satisfactory.",
    )
    refinements = []
    for parent in refine_parents:
        with st.expander(f"Refinement: {parent}", expanded=True):
            r1, r2 = st.columns([3, 1])
            with r1:
                markers = st.multiselect(
                    "Refinement markers",
                    options=metadata["markers"],
                    default=selected_markers,
                    key=f"refine_markers_{scoped_id}_{parent}",
                )
            with r2:
                parent_resolution = st.number_input(
                    "Resolution",
                    0.1,
                    2.0,
                    0.8,
                    0.1,
                    key=f"refine_resolution_{scoped_id}_{parent}",
                )
            refinements.append(
                {
                    "annotation": parent,
                    "markers": markers,
                    "resolution": float(parent_resolution),
                    "n_neighbors": int(n_neighbors),
                    "n_pcs": int(n_pcs),
                    "scale": bool(scale),
                }
            )

    refinement_config_path = output_dir / f"{scoped_id}_refinement_config.json"
    refinement_status_path = output_dir / f"{scoped_id}_refinement_status.json"
    refinement_log_path = output_dir / f"{scoped_id}_refinement_worker.log"
    if refine_parents and st.button(
        "Run selected refinements", type="primary", icon=":material/account_tree:"
    ):
        config = {
            "sample_id": sample_id,
            "adata_path": str(loaded_paths["adata"]),
            "image_path": str(loaded_paths["image"]),
            "output_dir": str(output_dir),
            "level0_h5ad": str(level0_outputs["clustered_h5ad"]),
            "level0_mapping_path": str(level0_mapping_path),
            "refinements": refinements,
            "cluster_imageid": cluster_imageid,
        }
        write_json(refinement_config_path, config)
        pid = start_worker(
            "refine",
            refinement_config_path,
            refinement_status_path,
            refinement_log_path,
        )
        st.success(f"Refinement worker started (process {pid}).")

    refinement_status = (
        render_job(refinement_status_path, label="Refinement")
        if refine_parents
        else None
    )
    refinement_outputs = {}
    if refine_parents:
        if not refinement_status or refinement_status.get("state") != "complete":
            return
        refinement_outputs = refinement_status["outputs"]

    reviewed_refinements = []
    all_refinement_mappings_ready = True
    for parent in refine_parents:
        if parent not in refinement_outputs:
            st.error(f"No refinement output found for {parent}")
            all_refinement_mappings_ready = False
            continue
        result = refinement_outputs[parent]
        st.markdown(f"#### Review refinement: {parent}")
        x1, x2 = st.columns(2)
        with x1:
            st.image(result["umap_png"], caption=f"{parent} UMAP", width="stretch")
        with x2:
            st.image(
                result["heatmap_png"],
                caption=f"{parent} marker heatmap",
                width="stretch",
            )
        if st.button(
            f"Launch napari: {parent}", key=f"napari_refine_{scoped_id}_{parent}"
        ):
            pid, log_path = launch_napari(
                Path(result["clustered_h5ad"]),
                review_image,
                output_dir,
                f"{scoped_id}_refine_{parent}",
                review_imageid,
            )
            st.success(f"Napari launched (process {pid}). Log: {log_path}")
        mapping_path = (
            output_dir / f"{scoped_id}_refine_{parent}_annotation_mapping.csv"
        )
        summary = pd.read_csv(result["summary_csv"], dtype={"cluster": str})
        editor = mapping_editor(
            summary,
            key_prefix=f"{scoped_id}_refine_{parent}",
            options=FINAL_LABELS,
            existing_path=mapping_path,
        )
        mapping = final_mapping(editor)
        missing = incomplete_mapping(mapping)
        if missing:
            st.warning(f"{parent}: labels required for clusters {', '.join(missing)}")
            all_refinement_mappings_ready = False
        if st.button(
            f"Save {parent} annotations",
            disabled=bool(missing),
            key=f"save_refine_{scoped_id}_{parent}",
        ):
            mapping.to_csv(mapping_path, index=False)
            st.success(f"Saved: {mapping_path}")
        if not mapping_path.exists():
            all_refinement_mappings_ready = False
        else:
            reviewed_refinements.append(
                {
                    "parent": parent,
                    "clustered_h5ad": result["clustered_h5ad"],
                    "mapping_path": str(mapping_path),
                }
            )

    st.subheader("5. Merge and export")
    export_config_path = output_dir / f"{scoped_id}_export_config.json"
    export_status_path = output_dir / f"{scoped_id}_export_status.json"
    export_log_path = output_dir / f"{scoped_id}_export_worker.log"
    export_disabled = bool(refine_parents) and not all_refinement_mappings_ready
    if st.button(
        "Export broad phenotyping",
        type="primary",
        disabled=export_disabled,
        icon=":material/save:",
    ):
        level0_config = read_json(level0_config_path)
        config = {
            "sample_id": sample_id,
            "adata_path": str(loaded_paths["adata"]),
            "image_path": str(loaded_paths["image"]),
            "output_dir": str(output_dir),
            "level0_h5ad": str(level0_outputs["clustered_h5ad"]),
            "level0_mapping_path": str(level0_mapping_path),
            "level0_config": level0_config,
            "refinements": reviewed_refinements,
            "cluster_imageid": cluster_imageid,
        }
        write_json(export_config_path, config)
        pid = start_worker(
            "export", export_config_path, export_status_path, export_log_path
        )
        st.success(f"Export worker started (process {pid}).")

    export_status = render_job(export_status_path, label="Export")
    if export_status and export_status.get("state") == "complete":
        outputs = export_status["outputs"]
        st.success(
            f"Completed {sample_id}: {outputs['n_cells']:,} cells and {outputs['n_final_labels']} final broad labels."
        )
        st.code(
            f"Phenotyped h5ad: {outputs['phenotyped_h5ad']}\n"
            f"Annotations: {outputs['annotation_csv']}\n"
            f"Manifest: {outputs['manifest_json']}"
        )
        st.info(
            "Next workflow: 02 Marker autogating. Use the original expression matrix with these broad labels in obs."
        )


if __name__ == "__main__":
    main()
