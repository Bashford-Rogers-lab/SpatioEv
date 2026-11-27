#!/usr/bin/env python3
"""Streamlit interface for unsupervised clustering and broad phenotyping."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from spatioev.workflows._io import configure_caches

configure_caches()

import anndata as ad
import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root, module_command
from spatioev.workflows import clustering
from spatioev.workflows import marker_gating as mgq
from spatioev.workflows._io import (
    find_run_file,
    read_json,
    run_files,
    start_logged_process,
    write_json,
)
from spatioev.workflows.cellsam import describe_unreadable_h5ad, marker_order_warning
from spatioev.workflows.image_collection import (
    imageids_with_images,
    natural_key,
    plane_marker_names,
    resolve_image,
)

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
# Markers that separate the testis structures, chosen from the images of a
# prepubertal (8-14 year) TMA: GFRA1, PIWIL4, NaKATPase and AMH stain only
# inside tubules in every core, SOX9 marks Sertoli cells and MAGEA4
# spermatogonia; SMA and MYH11 mark the peritubular wall; CYP17A1 and STAR
# Leydig cells. DMRT1 is left out because it stains Sertoli and germ cells
# alike. SYCP3 (meiotic) and Acrosin (post-meiotic) are left out because no
# such cells exist before puberty, so they would add only noise: add them for
# pubertal or adult testes. Broad or nonspecific stains (Vimentin, NF2F2,
# CD45, CD68) are left out on purpose.
TESTIS_PRESET = "Testis structures"
TESTIS_STRUCTURE_MARKERS = [
    "GFRA1",
    "PIWIL4",
    "NaKATPase",
    "AMH",
    "SOX9",
    "MAGEA4",
    "MYH11",
    "SMA",
    "CYP17A1",
    "STAR",
]
# Starting points for refining the two broad testis labels. Leydig cells and
# the peritubular myoid cells fall into separate clusters on these markers;
# clusters with none of them are the other interstitial cells. For Sertoli
# cells and spermatogonia, removing each cell's overall brightness first
# helped: the two are packed tightly, so many cells carry some of both signals.
TESTIS_REFINEMENTS = {
    "TUBULE": {
        "markers": ["SOX9", "MAGEA4", "GFRA1", "PIWIL4", "AMH", "DMRT1"],
        "remove_brightness": True,
    },
    "INTERSTITIUM": {
        "markers": ["SMA", "MYH11", "CYP17A1", "STAR", "CYP11A1", "CD68"],
        "remove_brightness": False,
    },
}
TESTIS_REFINEMENT_RESOLUTION = 0.5
DEFAULT_REFINEMENT_RESOLUTION = 0.8
# Smooth-muscle markers: they mark structures one cell thick, which averaging
# over neighbours would blur, so they are suggested to stay per cell.
PER_CELL_MARKER_NAMES = {"SMA", "ASMA", "ACTA2", "MYH11"}
FINAL_LABELS = [
    "",
    "Tubule",
    "Interstitium",
    "Sertoli",
    "Spermatogonia",
    "Spermatocyte",
    "Spermatid",
    "Myoid",
    "Leydig",
    "Other interstitium",
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
        "marker_order_warning": marker_order_warning(adata),
    }
    planes = plane_marker_names(adata)
    adata.file.close()
    # A TMA image folder may hold images for only some FOVs.
    with_image = imageids_with_images(image_path, imageids)
    resolved_image = resolve_image(Path(image_path), with_image[0] if with_image else None)
    channels = mgq.canonical_channel_names(resolved_image, planes)
    metadata["image_channels"] = channels
    metadata["missing_image_channels"] = sorted(
        set(metadata["markers"]) - set(channels)
    )
    metadata["has_coordinates"] = {"X_centroid", "Y_centroid"}.issubset(
        metadata["obs_columns"]
    )
    metadata["imageids"] = imageids
    metadata["imageids_with_image"] = with_image
    metadata["imageid_counts"] = imageid_counts
    metadata["default_image_path"] = str(resolved_image)
    return metadata


def recommended_defaults(sample_id: str, markers: list[str]) -> tuple[list[str], float]:
    """Choose a conservative panel-aware starting point for broad clustering."""

    selected = [marker for marker in CORE_MARKERS if marker in markers]
    return selected, 0.4


def marker_base(name: str) -> str:
    """``CLDN11_2`` -> ``CLDN11``: panels number repeated stains."""
    return re.sub(r"_\d+$", "", str(name)).upper()


def panel_markers(wanted: list[str], available: list[str]) -> list[str]:
    """The markers of ``available`` that match ``wanted``, ignoring case and ``_1``/``_2``."""
    by_base: dict[str, str] = {}
    for marker in available:
        by_base.setdefault(marker_base(marker), marker)
    return [by_base[marker_base(name)] for name in wanted if marker_base(name) in by_base]


def suggested_per_cell_markers(markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker_base(marker) in PER_CELL_MARKER_NAMES]


def refinement_defaults(
    parent: str, available: list[str], level0_markers: list[str]
) -> tuple[list[str], float, bool]:
    """Starting markers, resolution and brightness removal for refining ``parent``.

    The broad testis labels (Tubule, Interstitium) start from their own
    markers when the panel has at least three of them; any other label starts
    from the Level-0 markers.
    """
    preset = TESTIS_REFINEMENTS.get(str(parent).strip().upper())
    if preset is not None:
        markers = panel_markers(preset["markers"], available)
        if len(markers) >= 3:
            return markers, TESTIS_REFINEMENT_RESOLUTION, bool(preset["remove_brightness"])
    return list(level0_markers), DEFAULT_REFINEMENT_RESOLUTION, False


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
    return start_logged_process(command, log_path)


def launch_napari(
    adata_path: Path,
    image_path: Path,
    output_dir: Path,
    name: str,
    imageid: str | None = None,
) -> tuple[int, Path]:
    _, _, log_path = run_files(output_dir, f"{name}_napari")
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
    return start_logged_process(command, log_path), log_path


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
                    if metadata.get("marker_order_warning"):
                        st.warning(
                            metadata["marker_order_warning"], icon=":material/warning:"
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
        with_image = metadata.get("imageids_with_image") or metadata["imageids"]
        review_imageid = st.selectbox(
            "FOV selected for clustering or image review",
            metadata["imageids"],
            index=metadata["imageids"].index(with_image[0]) if with_image[0] in metadata["imageids"] else 0,
            key=f"cluster_review_imageid_{sample_id}",
        )
        if review_imageid in with_image:
            review_image = resolve_image(loaded_paths["image"], review_imageid)
            st.caption(f"Image review uses {review_imageid}: {review_image}")
        else:
            review_image = None
            st.caption(
                f"{review_imageid} has no image in {loaded_paths['image']}, so napari "
                f"review is off for it. FOVs with an image: {', '.join(with_image)}."
            )

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
    testis_markers = panel_markers(TESTIS_STRUCTURE_MARKERS, metadata["markers"])
    presets = ["Notebook-informed recommendation", "Core lineage panel", TESTIS_PRESET, "Custom"]
    preset = st.selectbox(
        "Marker preset",
        presets,
        # A panel with most of the testis markers is a testis panel.
        index=presets.index(TESTIS_PRESET) if len(testis_markers) >= 6 else 0,
        help=(
            "The first two reproduce the liver (HCC) reference notebooks. "
            f"{TESTIS_PRESET} picks the markers that separate tubules, the "
            "peritubular wall and the interstitium: GFRA1, PIWIL4, NaKATPase, "
            "AMH, SOX9, MAGEA4, MYH11, SMA, CYP17A1 and STAR. For pubertal or "
            "adult testes add SYCP3 and Acrosin, so spermatocytes and "
            "spermatids have a marker too."
        ),
    )
    if preset == "Core lineage panel":
        default_markers = [
            marker for marker in CORE_MARKERS if marker in metadata["markers"]
        ]
    elif preset == TESTIS_PRESET:
        default_markers = testis_markers
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
    several_fovs = cluster_imageid is None and len(metadata["imageids"]) > 1
    scale_within_image = st.checkbox(
        "Z-score within each FOV",
        value=False,
        disabled=not (scale and several_fovs),
        help=(
            "Scale every marker separately in each FOV. Use it when cores were "
            "stained or imaged unevenly: clusters made of one or two FOVs in the "
            "summary table are the sign. Each FOV should contain several cell "
            "types, since a FOV that is all one type is scaled to look like the rest."
        ),
    )
    scale_within_image = bool(scale and several_fovs and scale_within_image)

    average = st.checkbox(
        "Average markers over neighbouring cells",
        value=False,
        help=(
            "Give every cell the mean of the cells around it before clustering. "
            "Clusters then follow tissue structures (tubules, interstitium) "
            "instead of single cells. Try it when cells without a reliable "
            "marker, such as most interstitial cells, end up in the wrong "
            "structure's clusters, and compare both runs on several FOVs. "
            "Refinement and export still use each cell's own values."
        ),
    )
    neighbourhood_um, pixel_size_um, per_cell_markers = 0.0, 0.0, []
    if average:
        n1, n2, n3 = st.columns([1, 1, 2])
        with n1:
            neighbourhood_um = st.number_input(
                "Neighbourhood radius (µm)",
                1.0,
                200.0,
                15.0,
                1.0,
                help=(
                    "Cells whose centre lies within this distance are averaged "
                    "together. One to two cell diameters; 15 µm suits testis cores."
                ),
            )
        with n2:
            st.session_state.setdefault("cluster_pixel_size", 0.0)
            pixel_size_um = st.number_input(
                "Pixel size (µm / px)",
                0.0,
                100.0,
                step=0.001,
                format="%.4f",
                key="cluster_pixel_size",
                help=(
                    "The size of one image pixel. It is not read from the image, "
                    "because some scanners record a wrong value there."
                ),
            )
        with n3:
            per_cell_markers = st.multiselect(
                "Markers kept per cell",
                options=selected_markers,
                default=suggested_per_cell_markers(selected_markers),
                help=(
                    "Markers of structures one cell thick, such as the smooth-muscle "
                    "wall (SMA, MYH11) around tubules and vessels. Averaging would "
                    "spread them onto the cells on both sides."
                ),
            )
        if not pixel_size_um:
            st.warning(
                "Type the pixel size of your scan to average markers over "
                "neighbouring cells.",
                icon=":material/straighten:",
            )

    level0_config_path, level0_status_path, level0_log_path = run_files(
        output_dir, f"{scoped_id}_level0"
    )
    run_disabled = len(selected_markers) < 3 or (average and not pixel_size_um)
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
            "scale_within_image": scale_within_image,
            "cluster_imageid": cluster_imageid,
            "neighbourhood_um": float(neighbourhood_um) if average else 0.0,
            "pixel_size_um": float(pixel_size_um) if average else None,
            "per_cell_markers": list(per_cell_markers) if average else [],
        }
        write_json(level0_config_path, config)
        pid = start_worker(
            "level0", level0_config_path, level0_status_path, level0_log_path
        )
        st.success(f"Level-0 worker started (process {pid}).")

    level0_status = render_job(
        find_run_file(output_dir, level0_status_path.name), label="Level-0"
    )
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
    mean_cells_averaged = level0_outputs.get("mean_cells_averaged")
    if mean_cells_averaged is not None:
        st.caption(
            f"Markers were averaged within {level0_outputs['neighbourhood_um']:g} µm: "
            f"{mean_cells_averaged:.1f} cells per neighbourhood on average."
        )
        if mean_cells_averaged < clustering.MIN_MEAN_CELLS_AVERAGED:
            st.warning(
                "Most cells had no neighbour within the radius, so markers were "
                "barely averaged. Check the pixel size: a typical nucleus is "
                "5–10 µm across.",
                icon=":material/warning:",
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
    if st.button(
        "Launch Level-0 napari review",
        icon=":material/open_in_new:",
        disabled=review_image is None,
    ):
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
        default_refine_markers, default_refine_resolution, default_remove_brightness = (
            refinement_defaults(parent, metadata["markers"], selected_markers)
        )
        with st.expander(f"Refinement: {parent}", expanded=True):
            r1, r2 = st.columns([3, 1])
            with r1:
                markers = st.multiselect(
                    "Refinement markers",
                    options=metadata["markers"],
                    default=default_refine_markers,
                    key=f"refine_markers_{scoped_id}_{parent}",
                )
            with r2:
                parent_resolution = st.number_input(
                    "Resolution",
                    0.1,
                    2.0,
                    default_refine_resolution,
                    0.1,
                    key=f"refine_resolution_{scoped_id}_{parent}",
                )
            remove_brightness = st.checkbox(
                "Remove each cell's overall brightness",
                value=default_remove_brightness,
                disabled=not scale,
                key=f"refine_brightness_{scoped_id}_{parent}",
                help=(
                    "Subtract each cell's mean over the refinement markers after "
                    "z-scoring. Big, bright or tightly packed cells raise every "
                    "marker together, which gives clusters that are high in "
                    "everything; this leaves which markers stand out in each cell. "
                    "It helped separate Sertoli cells from spermatogonia. Needs "
                    "Z-score markers on."
                ),
            )
            refinements.append(
                {
                    "annotation": parent,
                    "markers": markers,
                    "resolution": float(parent_resolution),
                    "n_neighbors": int(n_neighbors),
                    "n_pcs": int(n_pcs),
                    "scale": bool(scale),
                    "remove_brightness": bool(scale and remove_brightness),
                }
            )

    refinement_config_path, refinement_status_path, refinement_log_path = run_files(
        output_dir, f"{scoped_id}_refinement"
    )
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
            "scale_within_image": scale_within_image,
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
        render_job(
            find_run_file(output_dir, refinement_status_path.name), label="Refinement"
        )
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
            f"Launch napari: {parent}",
            key=f"napari_refine_{scoped_id}_{parent}",
            disabled=review_image is None,
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
    export_config_path, export_status_path, export_log_path = run_files(
        output_dir, f"{scoped_id}_export"
    )
    export_disabled = bool(refine_parents) and not all_refinement_mappings_ready
    if st.button(
        "Export broad phenotyping",
        type="primary",
        disabled=export_disabled,
        icon=":material/save:",
    ):
        level0_config = read_json(find_run_file(output_dir, level0_config_path.name))
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

    export_status = render_job(
        find_run_file(output_dir, export_status_path.name), label="Export"
    )
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
