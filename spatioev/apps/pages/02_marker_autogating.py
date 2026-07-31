#!/usr/bin/env python3
"""Streamlit setup page for condition-driven marker autogating."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_marker_gating_app_matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spatioev_marker_gating_app_numba"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import numpy as np
import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root, module_command, resource_path
from spatioev.workflows import marker_gating as mgq
from spatioev.workflows.image_collection import natural_key, resolve_image


PROJECT_ROOT_DEFAULT = default_project_root()
REQUIRED_CONDITION_COLUMNS = [
    "staining_condition",
    "compartment_pattern",
    "expression_condition",
    "artifact_level",
]
STAINING_OPTIONS = [
    "",
    "clear_specific",
    "diffuse_background",
    "high_background_specific_tail",
    "negative_or_absent",
    "artifact_dominated",
    "failed_or_unusable",
]
COMPARTMENT_OPTIONS = [
    "",
    "membrane",
    "cytoplasmic",
    "nuclear",
    "membrane_cytoplasmic",
    "extracellular",
    "mixed_or_uncertain",
]
EXPRESSION_OPTIONS = ["", "bimodal", "multi_level", "broad_gradient"]
ARTIFACT_OPTIONS = ["", "low", "medium", "high", "severe"]
CONDITION_START_OPTIONS = [
    "HCC Phenocycler template (example)",
    "Current sample CSV",
    "Blank questionnaire",
]
DISTRIBUTION_DISPLAY_COLUMNS = [
    "marker",
    "inferred_expression_condition",
    "expression_inference_confidence",
    "distribution_dynamic_range",
    "distribution_skewness",
    "gmm2_sep",
    "gmm3_high_sep",
    "conservative_candidate_positive_fraction_span",
    "distribution_note",
]


def standard_paths(sample_id: str, project_root: Path) -> dict[str, Path]:
    paths = mgq.default_sample_paths(sample_id, project_root)
    return {
        "adata": paths.adata_path,
        "image": paths.image_path,
        "conditions": paths.marker_condition_path,
        "output": paths.output_dir,
        "strategy": resource_path("hcc_phenocycler_consensus_strategy.csv"),
    }


def validate_paths(paths: dict[str, Path], *, require_condition_source: bool) -> list[str]:
    errors = []
    for key in ["adata", "image", "strategy"]:
        if not paths[key].exists():
            errors.append(f"{key} path does not exist: {paths[key]}")
    if require_condition_source and not paths["conditions"].exists():
        errors.append(f"condition CSV does not exist: {paths['conditions']}")
    return errors


@st.cache_data(show_spinner=False)
def sample_metadata(adata_path: str, image_path: str) -> dict[str, object]:
    adata = ad.read_h5ad(adata_path, backed="r")
    markers = [str(marker) for marker in adata.var_names]
    cells = int(adata.n_obs)
    layers = list(adata.layers.keys())
    imageids = []
    if "imageid" in adata.obs:
        imageids = sorted(
            adata.obs["imageid"].dropna().astype(str).unique().tolist(),
            key=natural_key,
        )
    adata.file.close()
    resolved_image = resolve_image(Path(image_path), imageids[0] if imageids else None)
    channels = mgq.canonical_channel_names(resolved_image, markers)
    nuclear_markers = {"HOECHST", "HOECHST2", "DAPI", "DNA_1"}
    review_markers = [
        marker for marker in markers if marker.upper() not in nuclear_markers
    ]
    return {
        "cells": cells,
        "markers": markers,
        "review_markers": review_markers,
        "layers": layers,
        "channels": channels,
        "missing_in_image": sorted(set(review_markers) - set(channels)),
        "imageids": imageids,
        "default_image_path": str(resolved_image),
    }


def blank_condition_table(markers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "marker": markers,
            "staining_condition": "",
            "compartment_pattern": "",
            "expression_condition": "",
            "artifact_level": "",
            "expected_positive_min_pct": np.nan,
            "expected_positive_max_pct": np.nan,
            "expected_parent": "",
        }
    )


def prepare_condition_table(
    markers: list[str],
    *,
    source: str,
    condition_path: Path,
    project_root: Path,
    strategy_path: Path,
) -> pd.DataFrame:
    table = blank_condition_table(markers)
    if source == "Blank questionnaire":
        return table

    source_path = condition_path
    if source == "HCC Phenocycler template (example)":
        source_path = strategy_path
    if source == "HCC Phenocycler template (example)":
        loaded = mgq.read_strategy_profile(source_path)
        if loaded is None:
            raise ValueError(f"Could not read consensus strategy profile: {source_path}")
    else:
        loaded = mgq.read_marker_conditions(source_path)
    optional_columns = [
        "expected_positive_min",
        "expected_positive_max",
        "expected_parent",
    ]
    columns = ["marker", *REQUIRED_CONDITION_COLUMNS]
    columns.extend(column for column in optional_columns if column in loaded.columns)
    table = table.drop(columns=REQUIRED_CONDITION_COLUMNS).merge(
        loaded[columns], on="marker", how="left"
    )
    for column in REQUIRED_CONDITION_COLUMNS:
        table[column] = table[column].fillna("").astype(str)
    if "expected_positive_min" in table:
        table["expected_positive_min_pct"] = pd.to_numeric(
            table.pop("expected_positive_min"), errors="coerce"
        ) * 100
    if "expected_positive_max" in table:
        table["expected_positive_max_pct"] = pd.to_numeric(
            table.pop("expected_positive_max"), errors="coerce"
        ) * 100
    if "expected_parent" not in table:
        table["expected_parent"] = ""
    table["expected_parent"] = table["expected_parent"].fillna("").astype(str)
    return table[
        [
            "marker",
            *REQUIRED_CONDITION_COLUMNS,
            "expected_positive_min_pct",
            "expected_positive_max_pct",
            "expected_parent",
        ]
    ]


def table_hash(table: pd.DataFrame) -> str:
    normalized = table.fillna("").to_csv(index=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_condition_table(table: pd.DataFrame) -> list[str]:
    errors = []
    for column in REQUIRED_CONDITION_COLUMNS:
        missing = table[column].fillna("").astype(str).str.strip().eq("")
        if missing.any():
            markers = ", ".join(table.loc[missing, "marker"].astype(str).tolist()[:8])
            suffix = "..." if int(missing.sum()) > 8 else ""
            errors.append(f"{column}: {int(missing.sum())} markers incomplete ({markers}{suffix})")

    minimum = pd.to_numeric(table["expected_positive_min_pct"], errors="coerce")
    maximum = pd.to_numeric(table["expected_positive_max_pct"], errors="coerce")
    invalid_bounds = minimum.notna() & maximum.notna() & minimum.gt(maximum)
    if invalid_bounds.any():
        markers = ", ".join(table.loc[invalid_bounds, "marker"].astype(str))
        errors.append(f"Expected positivity minimum exceeds maximum for: {markers}")
    return errors


def conditions_for_engine(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    output["expected_positive_min"] = pd.to_numeric(
        output.pop("expected_positive_min_pct"), errors="coerce"
    ) / 100.0
    output["expected_positive_max"] = pd.to_numeric(
        output.pop("expected_positive_max_pct"), errors="coerce"
    ) / 100.0
    output["expected_parent"] = output["expected_parent"].fillna("").astype(str)
    return output


def merge_conditions_with_diagnostics(
    condition_table: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach current editable conditions to condition-independent metrics."""
    conditions = conditions_for_engine(condition_table)
    condition_columns = set(conditions.columns) - {"marker"}
    metrics = diagnostics.drop(columns=[column for column in condition_columns if column in diagnostics], errors="ignore")
    return conditions.merge(metrics, on="marker", how="left", validate="one_to_one")


def analyze_distributions(
    *,
    sample_id: str,
    adata_path: Path,
    condition_table: pd.DataFrame,
    output_dir: Path,
    layer: str | None,
) -> tuple[pd.DataFrame, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = mgq.load_adata(adata_path)
    diagnostics = mgq.compute_gate_candidates(
        adata,
        conditions_for_engine(condition_table),
        layer=layer,
        max_fit_cells=None,
        random_state=0,
    )
    output_path = output_dir / f"{sample_id}_marker_distribution_diagnostics_web.csv"
    diagnostics.to_csv(output_path, index=False)
    return diagnostics, output_path


def calculate_gates(
    *,
    sample_id: str,
    adata_path: Path,
    condition_table: pd.DataFrame,
    strategy_path: Path,
    output_dir: Path,
    layer: str | None,
    diagnostics: pd.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_path = output_dir / f"{sample_id}_marker_conditions_web.csv"
    candidate_path = output_dir / f"{sample_id}_marker_gate_candidates_web.csv"
    selected_path = output_dir / f"{sample_id}_selected_marker_gates_web.csv"
    scimap_path = output_dir / f"{sample_id}_scimap_manual_gates_web.csv"
    histogram_path = output_dir / f"{sample_id}_marker_histograms_candidate_gates_web.png"
    summary_path = output_dir / f"{sample_id}_selected_gate_summary_web.png"

    conditions = conditions_for_engine(condition_table)
    conditions.to_csv(condition_path, index=False)
    adata = mgq.load_adata(adata_path)
    if diagnostics is None:
        candidates = mgq.compute_gate_candidates(
            adata,
            conditions,
            layer=layer,
            max_fit_cells=None,
            random_state=0,
        )
    else:
        candidates = merge_conditions_with_diagnostics(condition_table, diagnostics)
    strategy = mgq.read_strategy_profile(strategy_path)
    method_plan = mgq.add_method_plan(candidates, strategy_profile=strategy)
    selected = mgq.selected_gate_table(method_plan, adata=adata, layer=layer)
    method_plan.to_csv(candidate_path, index=False)
    selected.to_csv(selected_path, index=False)
    mgq.write_scimap_manual_gates(selected, scimap_path)
    mgq.save_histogram_qc(method_plan, adata, histogram_path, layer=layer)
    mgq.save_gate_summary_qc(selected, summary_path)
    return {
        "conditions": condition_path,
        "candidates": candidate_path,
        "selected": selected_path,
        "scimap": scimap_path,
        "histograms": histogram_path,
        "gate_summary": summary_path,
    }


def generate_spatial_qc(
    *,
    sample_id: str,
    adata_path: Path,
    image_path: Path,
    condition_path: Path,
    selected_path: Path,
    output_dir: Path,
    layer: str | None,
    imageid: str | None = None,
) -> dict[str, Path]:
    conditions = mgq.read_marker_conditions(condition_path)
    selected = pd.read_csv(selected_path)
    adata = mgq.load_adata(adata_path)
    gated = mgq.apply_selected_gates(adata, selected, layer=layer)
    if imageid is not None:
        gated = gated[gated.obs["imageid"].astype(str).eq(str(imageid))].copy()
    grayscale, overlay = mgq.save_image_overviews(
        image_path,
        conditions,
        output_dir,
        sample_id=sample_id,
        channel_markers=list(adata.var_names.astype(str)),
    )
    channel_markers = list(adata.var_names.astype(str))
    windows, _ = mgq.choose_crop_windows(image_path, channel_markers=channel_markers)
    crop_boxes = output_dir / f"{sample_id}_selected_crop_boxes_web.png"
    mgq.save_crop_box_overview(
        image_path, windows, crop_boxes, channel_markers=channel_markers
    )
    overlay_paths = mgq.save_gate_spatial_overlays(
        gated,
        selected,
        image_path,
        selected["marker"].astype(str).tolist(),
        output_dir,
        sample_id=sample_id,
        windows=windows,
    )
    outputs = {
        "image_grayscale_overview": grayscale,
        "image_overlay_overview": overlay,
        "crop_boxes": crop_boxes,
    }
    for index, path in enumerate(overlay_paths, start=1):
        outputs[f"selected_gate_overlay_crop{index}"] = path
    return outputs


def launch_napari(
    *,
    sample_id: str,
    project_root: Path,
    adata_path: Path,
    image_path: Path,
    condition_path: Path,
    gate_table: Path,
    output_dir: Path,
    layer: str | None,
    imageid: str | None = None,
) -> tuple[int, Path]:
    command = module_command(
        "spatioev.workflows.marker_gating_review",
        "--sample-id",
        sample_id,
        "--project-root",
        str(project_root),
        "--adata-path",
        str(adata_path),
        "--image-path",
        str(image_path),
        "--condition-path",
        str(condition_path),
        "--gate-table",
        str(gate_table),
        "--output-dir",
        str(output_dir),
    )
    if layer:
        command.extend(["--layer", layer])
    if imageid is not None:
        command.extend(["--imageid", imageid])
    log_path = output_dir / f"{sample_id}_napari_gate_review.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid, log_path


def render_option_guide() -> None:
    with st.expander("Condition options and what the software can infer"):
        st.markdown(
            """
**Staining condition** is a slide-level visual judgment.

| Option | Use when |
|---|---|
| `clear_specific` | Signal is localized and distinct from background |
| `diffuse_background` | Nonspecific background spans much of the tissue or intensity range |
| `high_background_specific_tail` | Background is high, but a convincing bright positive tail remains |
| `negative_or_absent` | No convincing biological positive population is visible |
| `artifact_dominated` | Folds, edge effects, precipitate, saturation, or other artifacts dominate |
| `failed_or_unusable` | The channel should not be interpreted or phenotyped |

**Compartment pattern** is usually stable marker biology and is initialized by the panel template:
`membrane`, `cytoplasmic`, `nuclear`, `membrane_cytoplasmic`, `extracellular`, or `mixed_or_uncertain`.

**Expression condition** can be suggested from the current cell distribution:

| Option | Distribution |
|---|---|
| `bimodal` | Two reasonably separated negative and positive populations |
| `multi_level` | More than two intensity states or a distinct high-positive component |
| `broad_gradient` | Continuous expression without a stable valley |

**Artifact level** records the visual burden: `low`, `medium`, `high`, or `severe`.

The software calculates expression shape, confidence, dynamic range, skewness, GMM separation, and candidate-method agreement. It does **not** infer staining specificity or artifacts from the histogram alone. Expected positivity is optional; enter percentages and separate multiple expected parent populations with semicolons.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Marker autogating setup", layout="wide")
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px; }
    h1 { font-size: 1.8rem !important; letter-spacing: 0 !important; }
    h2 { font-size: 1.2rem !important; letter-spacing: 0 !important; }
    [data-testid="stMetric"] { border-left: 3px solid #277c75; padding-left: 0.8rem; }
    [data-testid="stDataEditor"] { border: 1px solid #d7dcdf; }
    button[kind="primary"] { background: #277c75 !important; border-color: #277c75 !important; }
    button[kind="primary"]:hover { background: #1f665f !important; border-color: #1f665f !important; }
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Marker autogating setup")
    st.caption("Configure staining assumptions, calculate initial gates, then review them on the original image in napari.")

    if "sample_id" not in st.session_state:
        st.session_state.sample_id = "sample"
    if "project_root" not in st.session_state:
        st.session_state.project_root = str(PROJECT_ROOT_DEFAULT)

    st.subheader("1. Sample and inputs")
    top_a, top_b, top_c = st.columns([1, 2.2, 1.2])
    with top_a:
        sample_id = st.text_input("Sample ID", key="sample_id")
    with top_b:
        project_root_text = st.text_input("Project root", key="project_root")
    project_root = Path(project_root_text).expanduser()
    defaults = standard_paths(sample_id, project_root)
    with top_c:
        source = st.selectbox(
            "Condition starting point",
            CONDITION_START_OPTIONS,
            help="The bundled HCC template is a worked example. Every value remains editable, and other panels can start from the questionnaire or their own CSV.",
        )
        fill_standard_paths = st.button("Fill standard sample paths", icon=":material/refresh:")

    path_state = {
        "adata_input": str(defaults["adata"]),
        "condition_input": str(defaults["conditions"]),
        "output_input": str(defaults["output"]),
        "image_input": str(defaults["image"]),
        "strategy_input": str(defaults["strategy"]),
    }
    for key, value in path_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
        if fill_standard_paths:
            st.session_state[key] = value

    paths_a, paths_b = st.columns(2)
    with paths_a:
        adata_text = st.text_input("AnnData (.h5ad)", key="adata_input")
        condition_text = st.text_input("Existing condition CSV", key="condition_input")
        output_text = st.text_input("Output directory", key="output_input")
    with paths_b:
        image_text = st.text_input("OME-TIFF image or FOV image folder", key="image_input")
        strategy_text = st.text_input("Gating strategy profile", key="strategy_input")
        layer_choice = st.text_input(
            "Expression layer",
            value="",
            placeholder="Blank uses adata.X; enter raw for adata.raw",
        )

    paths = {
        "adata": Path(adata_text).expanduser(),
        "image": Path(image_text).expanduser(),
        "conditions": Path(condition_text).expanduser(),
        "output": Path(output_text).expanduser(),
        "strategy": Path(strategy_text).expanduser(),
    }
    require_source = source == "Current sample CSV"
    if st.button("Load sample", type="primary", icon=":material/folder_open:"):
        errors = validate_paths(paths, require_condition_source=require_source)
        if errors:
            for error in errors:
                st.error(error)
        else:
            try:
                metadata = sample_metadata(str(paths["adata"]), str(paths["image"]))
                if metadata["missing_in_image"]:
                    st.error(f"Markers missing from image: {', '.join(metadata['missing_in_image'])}")
                else:
                    table = prepare_condition_table(
                        metadata["review_markers"],
                        source=source,
                        condition_path=paths["conditions"],
                        project_root=project_root,
                        strategy_path=paths["strategy"],
                    )
                    st.session_state.loaded_metadata = metadata
                    st.session_state.condition_table = table
                    st.session_state.loaded_paths = {key: str(value) for key, value in paths.items()}
                    st.session_state.loaded_sample_id = sample_id
                    st.session_state.loaded_project_root = str(project_root)
                    st.session_state.loaded_condition_source = source
                    st.session_state.editor_version = st.session_state.get("editor_version", 0) + 1
                    st.session_state.pop("calculated_outputs", None)
                    st.session_state.pop("distribution_diagnostics", None)
                    st.session_state.pop("distribution_context", None)
                    st.session_state.pop("spatial_qc_outputs", None)
                    diagnostics_path = (
                        paths["output"] / f"{sample_id}_marker_distribution_diagnostics_web.csv"
                    )
                    if diagnostics_path.exists():
                        cached = pd.read_csv(diagnostics_path)
                        if set(DISTRIBUTION_DISPLAY_COLUMNS).issubset(cached.columns):
                            st.session_state.distribution_diagnostics = cached
                            st.session_state.distribution_diagnostics_path = str(diagnostics_path)
                            st.session_state.distribution_context = {
                                "adata": str(paths["adata"]),
                                "layer": layer_choice.strip() or None,
                            }
                    st.success(f"Loaded {sample_id}: {metadata['cells']:,} cells and {len(metadata['review_markers'])} markers.")
            except Exception as exc:
                st.exception(exc)

    if "condition_table" not in st.session_state:
        st.info("Load a sample to create the marker questionnaire.")
        render_option_guide()
        return

    metadata = st.session_state.loaded_metadata
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Cells", f"{metadata['cells']:,}")
    metric_b.metric("Markers to gate", len(metadata["review_markers"]))
    metric_c.metric("Image channels", len(metadata["channels"]))
    metric_d.metric("Missing channels", len(metadata["missing_in_image"]))
    loaded_paths = {
        key: Path(value) for key, value in st.session_state.loaded_paths.items()
    }
    review_imageid = None
    review_image = Path(metadata["default_image_path"])
    if metadata["imageids"]:
        review_imageid = st.selectbox(
            "FOV used for image QC and gate review",
            metadata["imageids"],
            key=f"gating_review_imageid_{st.session_state.loaded_sample_id}",
        )
        review_image = resolve_image(loaded_paths["image"], review_imageid)
        st.caption(f"Image review uses {review_imageid}: {review_image}")

    st.subheader("2. Marker staining questionnaire")
    initialized_rows = int(
        st.session_state.condition_table[REQUIRED_CONDITION_COLUMNS]
        .fillna("")
        .astype(str)
        .apply(lambda column: column.str.strip().ne(""))
        .all(axis=1)
        .sum()
    )
    st.caption(
        f"Starting point: {st.session_state.get('loaded_condition_source', 'custom')} "
        f"({initialized_rows}/{len(st.session_state.condition_table)} markers fully initialized). "
        "Expected positivity and parent populations are optional."
    )
    render_option_guide()
    st.download_button(
        "Download current condition template",
        data=st.session_state.condition_table.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state.loaded_sample_id}_marker_condition_template.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    editor_key = f"condition_editor_{st.session_state.editor_version}"
    edited = st.data_editor(
        st.session_state.condition_table,
        key=editor_key,
        hide_index=True,
        width="stretch",
        height=780,
        num_rows="fixed",
        column_config={
            "marker": st.column_config.TextColumn("Marker", disabled=True, width="small"),
            "staining_condition": st.column_config.SelectboxColumn(
                "Staining condition", options=STAINING_OPTIONS, required=True, width="medium"
            ),
            "compartment_pattern": st.column_config.SelectboxColumn(
                "Compartment", options=COMPARTMENT_OPTIONS, required=True, width="small"
            ),
            "expression_condition": st.column_config.SelectboxColumn(
                "Expression", options=EXPRESSION_OPTIONS, required=True, width="medium"
            ),
            "artifact_level": st.column_config.SelectboxColumn(
                "Artifact", options=ARTIFACT_OPTIONS, required=True, width="small"
            ),
            "expected_positive_min_pct": st.column_config.NumberColumn(
                "Expected + min (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"
            ),
            "expected_positive_max_pct": st.column_config.NumberColumn(
                "Expected + max (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f"
            ),
            "expected_parent": st.column_config.TextColumn(
                "Expected parent(s)", help="Separate multiple populations with semicolons", width="medium"
            ),
        },
    )
    st.session_state.condition_table = edited

    distribution_a, distribution_b = st.columns([1, 2])
    with distribution_a:
        analyze_clicked = st.button(
            "Analyze expression distributions",
            type="primary",
            icon=":material/monitoring:",
        )
    with distribution_b:
        st.caption(
            "Uses all cells to suggest expression shape and quantify dynamic range, component separation, and candidate-method agreement."
        )
    if analyze_clicked:
        loaded_paths = {key: Path(value) for key, value in st.session_state.loaded_paths.items()}
        with st.status("Analyzing all marker distributions", expanded=True) as status:
            try:
                diagnostics, diagnostics_path = analyze_distributions(
                    sample_id=st.session_state.loaded_sample_id,
                    adata_path=loaded_paths["adata"],
                    condition_table=edited,
                    output_dir=loaded_paths["output"],
                    layer=layer_choice.strip() or None,
                )
                st.session_state.distribution_diagnostics = diagnostics
                st.session_state.distribution_diagnostics_path = str(diagnostics_path)
                st.session_state.distribution_context = {
                    "adata": str(loaded_paths["adata"]),
                    "layer": layer_choice.strip() or None,
                }
                status.update(label="Distribution analysis complete", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Distribution analysis failed", state="error", expanded=True)
                st.exception(exc)

    if "distribution_diagnostics" in st.session_state:
        diagnostics = st.session_state.distribution_diagnostics
        diagnostic_view = edited[["marker", "expression_condition"]].merge(
            diagnostics[DISTRIBUTION_DISPLAY_COLUMNS],
            on="marker",
            how="left",
        )
        st.dataframe(
            diagnostic_view,
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "expression_condition": "Current expression label",
                "inferred_expression_condition": "Automatic suggestion",
                "expression_inference_confidence": "Confidence",
                "distribution_dynamic_range": st.column_config.NumberColumn("Dynamic range", format="%.3f"),
                "distribution_skewness": st.column_config.NumberColumn("Skewness", format="%.2f"),
                "gmm2_sep": st.column_config.NumberColumn("2-GMM separation", format="%.2f"),
                "gmm3_high_sep": st.column_config.NumberColumn("3-GMM high separation", format="%.2f"),
                "conservative_candidate_positive_fraction_span": st.column_config.NumberColumn(
                    "Tail candidate + span", format="%.1%%"
                ),
                "distribution_note": "Automatic interpretation",
            },
        )
        apply_mode = st.segmented_control(
            "Apply automatic expression suggestions",
            ["Fill blanks only", "Replace all"],
            default="Fill blanks only",
        )
        include_low_confidence = st.checkbox(
            "Include low-confidence suggestions",
            value=False,
            help="Leave off to apply only medium- and high-confidence distribution calls.",
        )
        if st.button("Apply expression suggestions", icon=":material/auto_fix_high:"):
            suggestions = diagnostics.set_index("marker")["inferred_expression_condition"]
            if not include_low_confidence:
                accepted = diagnostics.set_index("marker")["expression_inference_confidence"].isin(
                    ["medium", "high"]
                )
                suggestions = suggestions.where(accepted)
            updated = edited.copy()
            mapped = updated["marker"].map(suggestions)
            if apply_mode == "Replace all":
                updated["expression_condition"] = mapped.fillna(updated["expression_condition"])
            else:
                blank = updated["expression_condition"].fillna("").astype(str).str.strip().eq("")
                updated.loc[blank, "expression_condition"] = mapped.loc[blank]
            st.session_state.condition_table = updated
            st.session_state.editor_version += 1
            st.session_state.pop("calculated_outputs", None)
            st.session_state.pop("spatial_qc_outputs", None)
            st.rerun()

    errors = validate_condition_table(edited)
    complete_rows = int(
        edited[REQUIRED_CONDITION_COLUMNS]
        .fillna("")
        .astype(str)
        .apply(lambda column: column.str.strip().ne(""))
        .all(axis=1)
        .sum()
    )
    st.progress(complete_rows / len(edited), text=f"Required fields complete for {complete_rows}/{len(edited)} markers")
    if errors:
        with st.expander(f"{len(errors)} readiness issue(s)", expanded=True):
            for error in errors:
                st.warning(error)

    st.subheader("3. Calculate initial gates")
    st.caption("All cells are used. The consensus profile selects the learned method family after candidate gates are calculated.")
    calculate_disabled = bool(errors)
    if st.button(
        "Calculate gates",
        type="primary",
        disabled=calculate_disabled,
        icon=":material/calculate:",
    ):
        loaded_paths = {key: Path(value) for key, value in st.session_state.loaded_paths.items()}
        with st.status("Calculating gates from all cells", expanded=True) as status:
            try:
                context = st.session_state.get("distribution_context", {})
                expected_context = {
                    "adata": str(loaded_paths["adata"]),
                    "layer": layer_choice.strip() or None,
                }
                cached_diagnostics = (
                    st.session_state.get("distribution_diagnostics")
                    if context == expected_context
                    else None
                )
                if cached_diagnostics is not None:
                    st.write("Reusing the completed all-cell distribution analysis")
                else:
                    st.write("Calculating all-cell candidate distributions")
                outputs = calculate_gates(
                    sample_id=st.session_state.loaded_sample_id,
                    adata_path=loaded_paths["adata"],
                    condition_table=edited,
                    strategy_path=loaded_paths["strategy"],
                    output_dir=loaded_paths["output"],
                    layer=layer_choice.strip() or None,
                    diagnostics=cached_diagnostics,
                )
                st.write("Writing gates, histograms, and summary QC")
                st.session_state.calculated_outputs = {key: str(value) for key, value in outputs.items()}
                st.session_state.calculated_table_hash = table_hash(edited)
                st.session_state.pop("spatial_qc_outputs", None)
                status.update(label="Initial gates calculated", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Gate calculation failed", state="error", expanded=True)
                st.exception(exc)

    if "calculated_outputs" not in st.session_state:
        return

    outputs = {key: Path(value) for key, value in st.session_state.calculated_outputs.items()}
    changed_since_calculation = table_hash(edited) != st.session_state.calculated_table_hash
    if changed_since_calculation:
        st.warning("The questionnaire changed after calculation. Recalculate before launching image review.")
    selected = pd.read_csv(outputs["selected"])
    summary_columns = [
        "marker",
        "selected_method",
        "selected_log1p_gate",
        "selected_positive_fraction",
        "inferred_expression_condition",
        "expression_inference_confidence",
        "conservative_candidate_positive_fraction_span",
        "review_flags",
    ]
    summary_columns = [column for column in summary_columns if column in selected]
    st.dataframe(
        selected[summary_columns],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "selected_log1p_gate": st.column_config.NumberColumn("Gate (log1p)", format="%.4f"),
            "selected_positive_fraction": st.column_config.NumberColumn("Positive fraction", format="%.2%%"),
            "inferred_expression_condition": "Suggested expression",
            "expression_inference_confidence": "Suggestion confidence",
            "conservative_candidate_positive_fraction_span": st.column_config.NumberColumn(
                "Tail candidate + span", format="%.1%%"
            ),
        },
    )

    st.subheader("4. Gate QC")
    qc_a, qc_b = st.columns(2)
    with qc_a:
        st.image(str(outputs["histograms"]), caption="Candidate gate histograms", width="stretch")
    with qc_b:
        st.image(str(outputs["gate_summary"]), caption="Selected positive fractions and review status", width="stretch")

    st.caption(
        "Spatial QC is generated separately because reading image crops is slower than calculating the expression gates."
    )
    if st.button(
        "Generate spatial QC images",
        disabled=changed_since_calculation,
        icon=":material/image_search:",
    ):
        loaded_paths = {key: Path(value) for key, value in st.session_state.loaded_paths.items()}
        with st.status("Generating image and positive-cell overlays", expanded=True) as status:
            try:
                spatial_outputs = generate_spatial_qc(
                    sample_id=st.session_state.loaded_sample_id,
                    adata_path=loaded_paths["adata"],
                    image_path=review_image,
                    condition_path=outputs["conditions"],
                    selected_path=outputs["selected"],
                    output_dir=loaded_paths["output"],
                    layer=layer_choice.strip() or None,
                    imageid=review_imageid,
                )
                st.session_state.spatial_qc_outputs = {
                    key: str(value) for key, value in spatial_outputs.items()
                }
                status.update(label="Spatial QC images generated", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Spatial QC generation failed", state="error", expanded=True)
                st.exception(exc)

    if "spatial_qc_outputs" in st.session_state:
        spatial_outputs = {key: Path(value) for key, value in st.session_state.spatial_qc_outputs.items()}
        overview_a, overview_b = st.columns(2)
        with overview_a:
            st.image(
                str(spatial_outputs["image_overlay_overview"]),
                caption="Marker and Hoechst overview",
                width="stretch",
            )
        with overview_b:
            st.image(str(spatial_outputs["crop_boxes"]), caption="Selected spatial QC crops", width="stretch")
        overlay_keys = sorted(key for key in spatial_outputs if key.startswith("selected_gate_overlay_crop"))
        for key in overlay_keys:
            st.image(
                str(spatial_outputs[key]),
                caption=key.replace("selected_gate_overlay_", "Positive-cell overlay "),
                width="stretch",
            )

    st.subheader("5. Review on the original image")
    st.caption("Launches the napari viewer with the newly calculated table. Slider adjustments are saved separately from the calculated values.")
    if st.button(
        "Launch napari gate review",
        type="primary",
        disabled=changed_since_calculation,
        icon=":material/open_in_new:",
    ):
        loaded_paths = {key: Path(value) for key, value in st.session_state.loaded_paths.items()}
        try:
            pid, log_path = launch_napari(
                sample_id=st.session_state.loaded_sample_id,
                project_root=Path(st.session_state.loaded_project_root),
                adata_path=loaded_paths["adata"],
                image_path=review_image,
                condition_path=outputs["conditions"],
                gate_table=outputs["selected"],
                output_dir=loaded_paths["output"],
                layer=layer_choice.strip() or None,
                imageid=review_imageid,
            )
            st.success(f"Napari review launched (process {pid}). Log: {log_path}")
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
