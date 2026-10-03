#!/usr/bin/env python3
"""Streamlit interface for tumour / envelope / stroma regions."""

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
from spatioev.workflows._io import read_json
from spatioev.workflows.tissue_regions import inspect_inputs

PREFIX = "regions"


def standard_defaults(sample_id: str, project_root: Path) -> dict[str, str]:
    sample_id = sample_id.strip()
    phenotyping = project_root / "results" / f"{sample_id}_scimap_phenotyping_interface"
    phenotyped = sorted(phenotyping.glob(f"{sample_id}_broad_plus_*_phenotyped.h5ad"), key=lambda p: p.stat().st_mtime)
    fiber_manifest = project_root / "results" / f"{sample_id}_fiber_segmentation" / f"{sample_id}_fiber_manifest.json"
    qupath = project_root / sample_id / "qupath" / f"{sample_id}_phenotyped.h5ad"  # QuPath bridge
    if qupath.exists():
        adata = qupath
    else:
        adata = phenotyped[-1] if phenotyped else phenotyping / f"{sample_id}_broad_plus_<subset>_phenotyped.h5ad"
    return {
        "regions_adata": str(adata),
        "regions_fiber_manifest": str(fiber_manifest) if fiber_manifest.exists() else "",
        "regions_output": str(project_root / "results" / f"{sample_id}_tissue_regions"),
    }


def initialize_state() -> None:
    st.session_state.setdefault("regions_sample_id", st.session_state.get("spatioev_sample_id", "sample"))
    st.session_state.setdefault("regions_project_root", st.session_state.get("spatioev_project_root", str(default_project_root())))
    for key, value in standard_defaults(st.session_state["regions_sample_id"], Path(st.session_state["regions_project_root"]).expanduser()).items():
        st.session_state.setdefault(key, value)


def manifest_pixel_size(path: str) -> float | None:
    manifest = read_json(Path(path).expanduser()) if path else None
    return (manifest or {}).get("outputs", {}).get("pixel_size_um")


def render_outputs(outputs: dict) -> None:
    counts = outputs.get("cells_per_region", {})
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Tumour nests", outputs.get("n_nests", 0))
    metric2.metric("Tumour region cells", f"{counts.get('tumour', 0):,}")
    metric3.metric("Envelope cells", f"{counts.get('envelope', 0):,}")
    metric4.metric("Stroma cells", f"{counts.get('stroma', 0):,}")
    show_images([*outputs.get("region_plots", [])[:2], outputs.get("composition_png")])

    if outputs.get("composition") and Path(outputs["composition"]).exists():
        composition = pd.read_csv(outputs["composition"])
        phenotype_key = outputs.get("phenotype_key")
        st.markdown("**Cell composition (proportion of cells in each region)**")
        imageids = sorted(composition["imageid"].astype(str).unique())
        imageid = st.selectbox("Image", imageids, key="regions_composition_image") if len(imageids) > 1 else imageids[0]
        table = composition[composition["imageid"].astype(str) == imageid].pivot_table(
            index=phenotype_key, columns="tissue_region", values="proportion"
        )
        table = table.reindex(columns=[c for c in ["all", "tumour", "envelope", "stroma"] if c in table.columns])
        st.dataframe(table.style.format("{:.1%}"), width="stretch")

    if outputs.get("fiber_region_stats") and Path(outputs["fiber_region_stats"]).exists():
        st.markdown("**Matrix inside each region**")
        stats = pd.read_csv(outputs["fiber_region_stats"])
        columns = [c for c in ["imageid", "fiber_type", "region", "area_fraction_pct", "fibers_per_mm2", "avg_alignment_score", "orientation_coherence", "n_fibers"] if c in stats]
        st.dataframe(stats[columns], hide_index=True, width="stretch")
        st.caption("area_fraction_pct = matrix pixels per region pixel. Lower alignment score = more aligned; coherence 1 = parallel fibers.")
        architecture_columns = [c for c in [
            "imageid", "fiber_type", "region", "hdm_pct", "alignment_coherency", "length_density_mm_per_mm2",
            "branchpoints_per_mm2", "endpoints_per_mm2", "curvature_per_um", "fractal_dimension",
            "lacunarity_gliding", "fiber_thickness_um",
        ] if c in stats]
        if "hdm_pct" in stats:
            st.markdown("**TWOMBLI-style architecture inside each region**")
            st.dataframe(stats[architecture_columns], hide_index=True, width="stretch")
            st.caption(
                "Measured on one skeleton of the whole image, so region edges do not create fiber ends. "
                "Fractal dimension and lacunarity depend on region size and shape: the envelope is a thin ring, so compare "
                "each region with the same region in other samples rather than regions with each other."
            )
    for warning in outputs.get("warnings", []):
        st.warning(warning, icon=":material/warning:")
    st.code(outputs.get("h5ad", ""), language=None)


def main() -> None:
    st.set_page_config(page_title="Tissue regions", page_icon=":material/layers:", layout="wide")
    initialize_state()
    st.title("Tissue regions")
    st.caption(
        "Groups tumour cells into nests, draws each nest's boundary and an envelope around it, and labels every cell "
        "tumour, envelope or stroma by where it sits."
    )

    sample_column, root_column, fill_column = st.columns([0.22, 0.50, 0.28])
    sample_column.text_input("Sample ID", key="regions_sample_id")
    root_column.text_input("Project root", key="regions_project_root")
    if fill_column.button("Fill standard sample paths", icon=":material/auto_fix_high:", width="stretch", key="regions_fill"):
        root = Path(st.session_state["regions_project_root"]).expanduser()
        for key, value in standard_defaults(st.session_state["regions_sample_id"], root).items():
            st.session_state[key] = value
        for key in ("regions_inspection", "regions_status_path", "regions_log_path"):
            st.session_state.pop(key, None)
        st.rerun()

    with st.container(border=True):
        left, right = st.columns(2)
        left.text_input("Phenotyped AnnData", key="regions_adata")
        right.text_input("Fiber manifest (optional; adds matrix per region)", key="regions_fiber_manifest")
        left.text_input("Output folder", key="regions_output")

    if st.button("Inspect inputs", icon=":material/search:", key="regions_inspect"):
        try:
            with st.spinner("Reading phenotype columns"):
                st.session_state["regions_inspection"] = inspect_inputs({"adata_path": path_value("regions_adata")})
            st.session_state["regions_inspection_signature"] = path_value("regions_adata")
        except Exception as error:
            st.session_state.pop("regions_inspection", None)
            st.error(str(error))

    report = st.session_state.get("regions_inspection")
    if report and st.session_state.get("regions_inspection_signature") == path_value("regions_adata"):
        metric1, metric2 = st.columns(2)
        metric1.metric("Cells", f"{report['n_cells']:,}")
        metric2.metric("Images", len(report["imageids"]))
        columns = list(report["candidate_columns"])
        if not columns:
            st.error("No categorical phenotype column found in the AnnData")
            return
        preferred = next((c for c in ("final_hierarchical_phenotype", "qupath_class", "phenotype", "annotation_level2") if c in columns), columns[0])
        st.session_state.setdefault("regions_phenotype_key", preferred)
        left, right = st.columns(2)
        phenotype_key = left.selectbox("Phenotype column", columns, key="regions_phenotype_key")
        values = report["candidate_columns"][phenotype_key]
        options = [row["value"] for row in values]
        guess = [v for v in options if any(t in v.lower() for t in ("tumour", "tumor", "epithel", "pck", "cancer"))][:1]
        tumour_labels = right.multiselect("Tumour phenotype(s)", options, default=guess, key=f"regions_tumour_{phenotype_key}")
        with st.expander(f"Cells per phenotype in {phenotype_key}"):
            st.dataframe(pd.DataFrame(values), hide_index=True, width="stretch")

        left, middle, right = st.columns(3)
        envelope = left.number_input(
            "Envelope width (µm)", 1.0, 1000.0, 30.0, 5.0, key="regions_envelope",
            help="Width of the ring outside each nest boundary that counts as tumour envelope.",
        )
        default_pixel = manifest_pixel_size(path_value("regions_fiber_manifest")) or 0.0
        st.session_state.setdefault("regions_pixel_size", float(default_pixel))
        middle.number_input("Pixel size (µm / px)", 0.0, 100.0, step=0.001, format="%.4f", key="regions_pixel_size")
        tissue_radius = right.number_input(
            "Tissue extent around cells (µm)", 5.0, 500.0, 50.0, 5.0, key="regions_tissue_radius",
            help="Pixels within this distance of a cell count as tissue when measuring region areas and matrix.",
        )
        with st.expander("Nest detection settings", icon=":material/tune:"):
            st.caption("Check the region map after a run: nests should be solid, not outlines around small clusters.")
            left, middle, right = st.columns(3)
            min_cells = left.number_input("Minimum cells per nest", 2, 100000, 30, 1, key="regions_min_cells")
            knn_k = middle.number_input("Neighbours used for linking radius", 1, 50, 5, 1, key="regions_knn")
            radius_scale = right.number_input(
                "Linking radius scale", 0.1, 10.0, 1.0, 0.1, key="regions_radius_scale",
                help="Raise to merge nearby nests, lower to split them.",
            )
            grid = left.number_input(
                "Boundary grid (µm, 0 = auto)", 0.0, 200.0, 0.0, 1.0, key="regions_grid",
                help="Auto uses half the median cell spacing. Larger values give smoother, more solid nests.",
            )
            sigma = middle.number_input("Boundary smoothing (grid cells)", 0.0, 20.0, 2.0, 0.5, key="regions_sigma")
            threshold = right.number_input("Boundary threshold", 0.01, 0.9, 0.1, 0.01, key="regions_threshold")
            closing = left.number_input("Gap closing (grid cells)", 0, 50, 5, 1, key="regions_closing")

        if st.button("Assign tissue regions", type="primary", icon=":material/play_arrow:", key="regions_run"):
            if not tumour_labels:
                st.error("Select at least one tumour phenotype")
            elif not st.session_state.get("regions_pixel_size"):
                st.error("Enter the pixel size")
            else:
                config = {
                    "sample_id": st.session_state["regions_sample_id"].strip(),
                    "adata_path": path_value("regions_adata"),
                    "fiber_manifest": path_value("regions_fiber_manifest") or None,
                    "output_dir": path_value("regions_output"),
                    "phenotype_key": phenotype_key,
                    "tumour_labels": tumour_labels,
                    "envelope_width_um": float(envelope),
                    "pixel_size_um": float(st.session_state["regions_pixel_size"]),
                    "tissue_radius_um": float(tissue_radius),
                    "min_nest_cells": int(min_cells),
                    "knn_k": int(knn_k),
                    "radius_scale": float(radius_scale),
                    "mask_resolution_um": float(grid) or None,
                    "mask_sigma": float(sigma),
                    "mask_threshold": float(threshold),
                    "mask_closing_size": int(closing),
                }
                try:
                    output_dir = Path(config["output_dir"]).expanduser().resolve()
                    remember_worker(PREFIX, *start_worker("spatioev.workflows.tissue_regions", config, output_dir, "tissue_regions"))
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

    if st.session_state.get("regions_status_path"):
        st.subheader("Tissue regions run")
        render_worker(PREFIX, render_outputs)


if __name__ == "__main__":
    main()
