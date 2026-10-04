#!/usr/bin/env python3
"""Streamlit interface for ECM fiber segmentation and quantification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root
from spatioev.apps._worker import (
    parse_numbers,
    path_value,
    remember_worker,
    render_batch,
    render_worker,
    show_images,
    start_worker,
)
from spatioev.workflows.fiber_segmentation import (
    DEFAULT_PARAMETERS,
    inspect_inputs,
    preview_steps,
)

PREFIX = "fiber"
ARCHITECTURE_COLUMNS = [
    "imageid", "fiber_type", "hdm_pct", "alignment_coherency", "length_density_mm_per_mm2",
    "branchpoints_per_mm2", "endpoints_per_mm2", "mean_branch_length_um", "curvature_per_um",
    "fractal_dimension", "lacunarity_gliding", "lacunarity_anamorf", "fiber_thickness_um",
]
ARCHITECTURE_CAPTION = (
    "Values follow TWOMBLI's definitions on SpatioEv's segmentation, so they compare between SpatioEv samples, "
    "not with Fiji TWOMBLI output. lacunarity_anamorf is TWOMBLI's value and depends only on coverage; "
    "lacunarity_gliding measures how clumped the matrix is at the chosen box size. Fractal dimension depends on the "
    "size of the area measured: compare it only between images or regions of similar size."
)


def standard_defaults(sample_id: str, project_root: Path) -> dict[str, str]:
    sample_id = sample_id.strip()
    candidates = [
        project_root / sample_id / "background" / f"{sample_id}.ome.tiff",
        project_root / "background" / f"{sample_id}.ome.tiff",
        project_root / sample_id / "processed" / f"{sample_id}_combined.ome.tif",
    ]
    image = next((path for path in candidates if path.exists()), candidates[0])
    adata_candidates = [
        project_root / sample_id / "qupath" / f"{sample_id}_phenotyped.h5ad",  # QuPath bridge
        project_root / f"{sample_id}_adata.h5ad",
        project_root / sample_id / f"{sample_id}_adata.h5ad",
    ]
    adata = next((path for path in adata_candidates if path.exists()), None)
    return {
        "fiber_image": str(image),
        "fiber_adata": str(adata) if adata else "",  # optional; an empty box beats a path that does not exist
        "fiber_output": str(project_root / "results" / f"{sample_id}_fiber_segmentation"),
    }


def initialize_state() -> None:
    st.session_state.setdefault("fiber_sample_id", st.session_state.get("spatioev_sample_id", "sample"))
    st.session_state.setdefault("fiber_project_root", st.session_state.get("spatioev_project_root", str(default_project_root())))
    for key, value in standard_defaults(st.session_state["fiber_sample_id"], Path(st.session_state["fiber_project_root"]).expanduser()).items():
        st.session_state.setdefault(key, value)


def inspection_config() -> dict:
    return {
        "sample_id": st.session_state["fiber_sample_id"].strip(),
        "image_path": path_value("fiber_image"),
        "adata_path": path_value("fiber_adata") or None,
    }


def parameter_inputs(report: dict) -> dict:
    with st.expander("Segmentation and quantification settings", icon=":material/tune:"):
        st.caption(
            "Defaults are ark-analysis's. Keep the same settings, and the same CLAHE window in pixels, "
            "for every image you intend to compare."
        )
        left, middle, right = st.columns(3)
        blur = left.number_input("Gaussian blur (px)", 0.0, 20.0, float(DEFAULT_PARAMETERS["blur"]), 0.5, key="fiber_blur")
        divisor = middle.number_input(
            "Contrast scaling divisor", 1.0, 2048.0, float(DEFAULT_PARAMETERS["contrast_scaling_divisor"]), 1.0, key="fiber_divisor",
            help="ark sets the CLAHE window to image rows / divisor. Ignored when a fixed window is given.",
        )
        rows = report["first_shape"][0]
        window = right.number_input(
            "CLAHE window (px, 0 = rows / divisor)", 0, 4096, 0, 1, key="fiber_window",
            help=f"rows / divisor = {int(rows / divisor)} px for the first image. Set a fixed value when images differ in size "
            "(e.g. TMA cores and whole slides) so the contrast step behaves the same on all of them.",
        )
        widths = left.text_input("Fiber widths, Frangi scales (px)", ", ".join(str(w) for w in DEFAULT_PARAMETERS["fiber_widths"]), key="fiber_widths")
        ridge = middle.number_input("Ridge cutoff", 0.0, 10000.0, float(DEFAULT_PARAMETERS["ridge_cutoff"]), 0.05, key="fiber_ridge")
        sobel = right.number_input("Sobel blur (px)", 0.0, 20.0, float(DEFAULT_PARAMETERS["sobel_blur"]), 0.5, key="fiber_sobel")
        min_size = left.number_input("Minimum fiber area (px)", 1, 100000, int(DEFAULT_PARAMETERS["min_fiber_size"]), 1, key="fiber_min_size")
        k = middle.number_input("Alignment neighbours (k)", 1, 50, int(DEFAULT_PARAMETERS["alignment_k"]), 1, key="fiber_k")
        axis = right.number_input("Min length / width ratio for alignment", 1.0, 20.0, float(DEFAULT_PARAMETERS["axis_thresh"]), 0.5, key="fiber_axis")
        axial = st.checkbox(
            "Axial angle differences (recommended)", value=True, key="fiber_axial",
            help="Fibers are undirected, so +89 and -89 degrees are nearly parallel. ark's original score treats them as "
            "opposite, which makes horizontal fibers look misaligned. Untick only to reproduce ark's numbers.",
        )
        left, middle, right = st.columns(3)
        tile_um = left.number_input("Summary tile size (µm)", 10.0, 2000.0, 150.0, 10.0, key="fiber_tile_um")
        min_fibers = middle.number_input("Min fibers per tile", 1, 100, int(DEFAULT_PARAMETERS["min_fiber_num"]), 1, key="fiber_min_fibers")
        processing_tile = right.number_input(
            "Processing tile (px)", 1024, 16384, int(DEFAULT_PARAMETERS["tile_size"]), 512, key="fiber_proc_tile",
            help="Images up to this size are segmented in one pass, exactly as ark does. Larger images (whole slides) are "
            "processed in tiles with shared normalisation and thresholds.",
        )
        include_bright = st.checkbox(
            "Also count bright matrix the ridge filter misses (cross-cut bundles, dense patches)", value=False,
            key="fiber_include_bright",
            help="The Frangi ridge filter suppresses round structures, so matrix cut across is only outlined. This adds "
            "pixels no fiber covers that are in the image's brightest intensity class, or brighter than their "
            "neighbourhood. Not part of ark.",
        )
        left, middle, _ = st.columns(3)
        bright_window = left.number_input(
            "Bright matrix: local window (µm, 0 = brightest class only)", 0.0, 500.0,
            float(DEFAULT_PARAMETERS["bright_local_window_um"]), 5.0, key="fiber_bright_window", disabled=not include_bright,
            help="Large medium-bright patches next to dark gaps are caught by comparing each pixel with its "
            "neighbourhood. 65 µm suited COL1 cross-sections in the HNSCC TMAs.",
        )
        bright_floor = middle.number_input(
            "Bright matrix: minimum level (0-1)", 0.0, 1.0, float(DEFAULT_PARAMETERS["bright_floor"]), 0.05,
            key="fiber_bright_floor", disabled=not include_bright,
            help="Locally bright pixels must also be this far from the tissue cut towards the bright cut "
            "(0 = any tissue, 1 = brightest class only).",
        )
        st.markdown("**TWOMBLI-style architecture**")
        architecture = st.checkbox(
            "Measure matrix architecture", value=True, key="fiber_architecture",
            help="Total fiber length, branchpoints, endpoints, curvature, fractal dimension, lacunarity, alignment and "
            "% high-density matrix, following TWOMBLI's definitions on this segmentation's skeleton.",
        )
        left, middle, right = st.columns(3)
        branch_um = left.number_input(
            "Minimum branch length (µm)", 0.0, 100.0, float(DEFAULT_PARAMETERS["min_branch_length_um"]), 0.5, key="fiber_branch_um",
            help="Skeleton spurs and isolated pieces shorter than this are pruned before counting (TWOMBLI's minimum branch length).",
        )
        curvature_um = middle.number_input(
            "Curvature window (µm)", 1.0, 200.0, float(DEFAULT_PARAMETERS["curvature_window_um"]), 1.0, key="fiber_curvature_um",
            help="Distance along a fiber on each side of a point used to measure its curvature. Fibers shorter than twice this are skipped.",
        )
        lacunarity_um = right.number_input(
            "Lacunarity box (µm)", 2.0, 500.0, float(DEFAULT_PARAMETERS["lacunarity_box_um"]), 1.0, key="fiber_lacunarity_um",
            help="Box size for the gliding-box lacunarity: how unevenly matrix and gaps are spread at this scale.",
        )
        hdm_display = left.number_input(
            "Maximum display HDM", 1.0, 255.0, float(DEFAULT_PARAMETERS["max_display_hdm"]), 5.0, key="fiber_hdm_display",
            help="TWOMBLI's HDM setting (default 200). Pixels above lo + (1 - value/255) × (hi - lo) of the channel count as high-density matrix.",
        )
        saturation = middle.number_input(
            "HDM intensity range clip (%)", 0.0, 10.0, float(DEFAULT_PARAMETERS["hdm_saturation_pct"]), 0.05, key="fiber_hdm_saturation",
            help="Percent of pixels clipped when finding the channel's intensity range for HDM (TWOMBLI's contrast saturation, 0.35 %).",
        )
    try:
        fiber_widths = parse_numbers(widths)
    except ValueError:
        st.error("Fiber widths must be numbers separated by commas")
        fiber_widths = list(DEFAULT_PARAMETERS["fiber_widths"])
    pixel_size = float(st.session_state.get("fiber_pixel_size") or 0) or None
    return {
        "blur": float(blur),
        "contrast_scaling_divisor": float(divisor),
        "clahe_kernel_size": int(window) or None,
        "fiber_widths": fiber_widths or list(DEFAULT_PARAMETERS["fiber_widths"]),
        "ridge_cutoff": float(ridge),
        "sobel_blur": float(sobel),
        "min_fiber_size": int(min_size),
        "alignment_k": int(k),
        "axis_thresh": float(axis),
        "axial": bool(axial),
        "stats_tile_size_px": int(round(tile_um / pixel_size)) if pixel_size else 512,
        "min_fiber_num": int(min_fibers),
        "tile_size": int(processing_tile),
        "include_bright": bool(include_bright),
        "bright_local_window_um": float(bright_window),
        "bright_floor": float(bright_floor),
        "architecture": bool(architecture),
        "min_branch_length_um": float(branch_um),
        "curvature_window_um": float(curvature_um),
        "lacunarity_box_um": float(lacunarity_um),
        "max_display_hdm": float(hdm_display),
        "hdm_saturation_pct": float(saturation),
    }


def render_preview(report: dict, channels: list[str], parameters: dict) -> None:
    st.subheader("Preview on a crop")
    images = [row["imageid"] for row in report["images"]]
    left, middle, right = st.columns([0.4, 0.3, 0.3])
    imageid = left.selectbox("Image", images, key="fiber_preview_image")
    channel = middle.selectbox("Channel", channels or report["channels"], key="fiber_preview_channel")
    size = right.select_slider("Crop size (px)", [512, 768, 1024, 1536], value=768, key="fiber_preview_size")
    if st.button("Preview segmentation steps", icon=":material/visibility:", key="fiber_preview_button"):
        config = {**inspection_config(), "parameters": parameters}
        with st.spinner("Running every segmentation step on the crop"):
            try:
                st.session_state["fiber_preview"] = (imageid, channel, preview_steps(config, imageid, channel, size=size))
            except Exception as error:
                st.session_state.pop("fiber_preview", None)
                st.error(str(error))
    stored = st.session_state.get("fiber_preview")
    if stored:
        from spatioev.pl.fibers import plot_fiber_segmentation_steps

        shown_image, shown_channel, preview = stored
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Fibers in crop", f"{preview['n_fibers']:,}")
        metric2.metric("Fiber area", f"{preview['area_fraction_pct']:.1f}%")
        metric3.metric("CLAHE window", f"{preview['kernel']} px")
        figure = plot_fiber_segmentation_steps(preview["steps"], title=f"{shown_image} {shown_channel}, crop at row {preview['origin'][0]}, col {preview['origin'][1]}")
        st.pyplot(figure, width="stretch")


def render_outputs(outputs: dict) -> None:
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Fibers", f"{outputs.get('n_fibers', 0):,}")
    metric2.metric("Images", len(outputs.get("image_shapes", {})))
    metric3.metric("Channels", len(outputs.get("channels", [])))
    if outputs.get("image_stats") and Path(outputs["image_stats"]).exists():
        stats = pd.read_csv(outputs["image_stats"])
        columns = [c for c in ["imageid", "fiber_type", "n_fibers", "n_bright_matrix", "area_fraction_pct", "fibers_per_mm2", "avg_alignment_score", "orientation_coherence", "avg_major_axis_length"] if c in stats]
        st.dataframe(stats[columns], hide_index=True, width="stretch")
        st.caption(
            "Whole-image values include any empty glass around the tissue. Run Tissue regions with this manifest "
            "to get matrix per tissue area, and per tumour / envelope / stroma region. Area counts all matrix; fiber counts, "
            "shape and alignment use the ridge fibers only, not the bright-matrix pieces. Lower alignment score = more aligned; "
            "orientation coherence runs from 0 (random) to 1 (parallel)."
        )
    if outputs.get("architecture") and Path(outputs["architecture"]).exists():
        st.markdown("**TWOMBLI-style architecture (whole image)**")
        architecture = pd.read_csv(outputs["architecture"])
        columns = [c for c in ARCHITECTURE_COLUMNS if c in architecture]
        st.dataframe(architecture[columns], hide_index=True, width="stretch")
        st.caption(ARCHITECTURE_CAPTION)
    st.code(outputs.get("manifest", ""), language=None)
    show_images(outputs.get("overlays", [])[:6])


def main() -> None:
    st.set_page_config(page_title="Fiber segmentation", page_icon=":material/grain:", layout="wide")
    initialize_state()
    st.title("Fiber segmentation")
    st.caption("ECM fiber segmentation and quantification adapted from ark-analysis (Angelo Lab, MIT licence).")

    sample_column, root_column, fill_column = st.columns([0.22, 0.50, 0.28])
    sample_column.text_input("Sample ID", key="fiber_sample_id")
    root_column.text_input("Project root", key="fiber_project_root")
    if fill_column.button("Fill standard sample paths", icon=":material/auto_fix_high:", width="stretch", key="fiber_fill"):
        root = Path(st.session_state["fiber_project_root"]).expanduser()
        for key, value in standard_defaults(st.session_state["fiber_sample_id"], root).items():
            st.session_state[key] = value
        for key in ("fiber_inspection", "fiber_preview", "fiber_status_path", "fiber_log_path"):
            st.session_state.pop(key, None)
        st.rerun()

    with st.container(border=True):
        left, right = st.columns(2)
        left.text_input("Source OME-TIFF or FOV image folder", key="fiber_image")
        right.text_input("Cell AnnData (optional; supplies image IDs)", key="fiber_adata")
        left.text_input("Output folder", key="fiber_output")

    if st.button("Inspect inputs", icon=":material/search:", key="fiber_inspect"):
        try:
            with st.spinner("Reading image metadata"):
                report = inspect_inputs(inspection_config())
            st.session_state["fiber_inspection"] = report
            st.session_state["fiber_inspection_signature"] = json.dumps(inspection_config(), sort_keys=True)
            if report.get("pixel_size_um"):
                st.session_state["fiber_pixel_size"] = float(report["pixel_size_um"])
        except Exception as error:
            st.session_state.pop("fiber_inspection", None)
            st.error(str(error))

    report = st.session_state.get("fiber_inspection")
    if report and st.session_state.get("fiber_inspection_signature") == json.dumps(inspection_config(), sort_keys=True):
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Images", report["n_images"])
        metric2.metric("Channels", len(report["channels"]))
        metric3.metric("First image (px)", f"{report['first_shape'][0]}×{report['first_shape'][1]}")
        if not report["imageids_from_adata"] and report["n_images"] > 1:
            st.warning("No AnnData given: image IDs come from file names and may not match the cell table's imageid.", icon=":material/warning:")

        collagen_i = {"COL1", "COL1A1", "COLLAGEN1", "COLLAGEN_I", "COLLAGENI"}
        default_channels = [c for c in report["matrix_channels"] if c.upper().replace(" ", "") in collagen_i] or report["matrix_channels"][:1]
        st.session_state.setdefault("fiber_channels", default_channels)
        left, right = st.columns([0.7, 0.3])
        channels = left.multiselect(
            "Matrix channels to segment", report["channels"], key="fiber_channels",
            help="Each channel is segmented separately. Detected matrix-like channels: " + (", ".join(report["matrix_channels"]) or "none"),
        )
        st.session_state.setdefault("fiber_pixel_size", report.get("pixel_size_um") or 0.0)
        right.number_input("Pixel size (µm / px)", 0.0, 100.0, step=0.001, format="%.4f", key="fiber_pixel_size")
        parameters = parameter_inputs(report)
        render_preview(report, channels, parameters)

        st.divider()
        if st.button("Run fiber segmentation", type="primary", icon=":material/play_arrow:", key="fiber_run"):
            if not channels:
                st.error("Select at least one matrix channel")
            else:
                config = {
                    **inspection_config(),
                    "channels": channels,
                    "output_dir": path_value("fiber_output"),
                    "pixel_size_um": float(st.session_state.get("fiber_pixel_size") or 0) or None,
                    "parameters": parameters,
                }
                try:
                    output_dir = Path(config["output_dir"]).expanduser().resolve()
                    remember_worker(PREFIX, *start_worker("spatioev.workflows.fiber_segmentation", config, output_dir, "fiber"))
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

    if st.session_state.get("fiber_status_path"):
        st.subheader("Fiber segmentation run")
        render_worker(PREFIX, render_outputs)
        render_batch(PREFIX)


if __name__ == "__main__":
    main()
