#!/usr/bin/env python3
"""Streamlit interface for cell-cell and cell-matrix co-localisation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from spatioev.apps._common import default_project_root
from spatioev.apps._worker import (
    parse_numbers,
    path_value,
    remember_worker,
    render_worker,
    show_images,
    start_worker,
)
from spatioev.workflows._io import read_json
from spatioev.workflows.tissue_regions import inspect_inputs

PREFIX = "coloc"


def standard_defaults(sample_id: str, project_root: Path) -> dict[str, str]:
    sample_id = sample_id.strip()
    regions = project_root / "results" / f"{sample_id}_tissue_regions"
    fiber_manifest = project_root / "results" / f"{sample_id}_fiber_segmentation" / f"{sample_id}_fiber_manifest.json"
    return {
        "coloc_adata": str(regions / f"{sample_id}_tissue_regions.h5ad"),
        "coloc_fiber_manifest": str(fiber_manifest) if fiber_manifest.exists() else "",
        "coloc_output": str(project_root / "results" / f"{sample_id}_colocalization"),
    }


def initialize_state() -> None:
    st.session_state.setdefault("coloc_sample_id", st.session_state.get("spatioev_sample_id", "sample"))
    st.session_state.setdefault("coloc_project_root", st.session_state.get("spatioev_project_root", str(default_project_root())))
    for key, value in standard_defaults(st.session_state["coloc_sample_id"], Path(st.session_state["coloc_project_root"]).expanduser()).items():
        st.session_state.setdefault(key, value)


def regions_manifest(adata_path: str) -> dict:
    path = Path(adata_path).expanduser()
    manifest = read_json(path.with_name(path.name.replace("_tissue_regions.h5ad", "_tissue_regions_manifest.json")))
    return (manifest or {}).get("outputs", {})


def render_outputs(outputs: dict) -> None:
    if outputs.get("cell_cell_curves"):
        st.markdown("**Cell–cell co-localisation**")
        st.caption(
            "L(r) − r above the shaded 95% envelope means the target phenotype is found around the source phenotype "
            "more than expected if phenotype labels were shuffled among the same cell positions."
        )
        show_images([outputs.get("cell_cell_plot")], columns=1)
        local_path = outputs.get("cell_cell_local")
        if local_path and Path(local_path).exists():
            local = pd.read_csv(local_path)
            st.dataframe(
                local[["imageid", "source", "target", "region", "radius_um", "n_source_cells", "mean_neighbour_ratio", "fraction_source_with_excess"]],
                hide_index=True, width="stretch",
            )
            st.caption("Neighbour ratio = observed / expected target cells around each source cell; 1 = no preference.")

    if outputs.get("cell_matrix_summary"):
        st.markdown("**Cell–matrix co-localisation**")
        show_images([outputs.get("cell_matrix_plot")], columns=1)
        enrichment_path = outputs.get("cell_matrix_enrichment")
        if enrichment_path and Path(enrichment_path).exists() and Path(enrichment_path).stat().st_size > 2:
            enrichment = pd.read_csv(enrichment_path)
            st.dataframe(
                enrichment[["imageid", "fiber_type", "metric", "phenotype", "n_cells", "observed_mean", "null_mean", "z_score", "p_value"]],
                hide_index=True, width="stretch",
            )
            st.caption("Positive z for matrix_fraction (or negative for distance) = the phenotype sits in more matrix-rich places than cells in general.")
        association_path = outputs.get("matrix_tile_association")
        if association_path and Path(association_path).exists() and Path(association_path).stat().st_size > 2:
            st.markdown("**Dense / aligned matrix tiles**")
            association = pd.read_csv(association_path)
            st.dataframe(
                association[["imageid", "fiber_type", "metric", "phenotype", "n_tiles", "spearman_rho", "spearman_p", "share_in_high_tiles", "share_in_other_tiles", "enrichment_ratio"]],
                hide_index=True, width="stretch",
            )
            st.caption(
                "High tiles = top quartile of the metric within the image. Enrichment ratio > 1 = the phenotype makes up more of the cells "
                "in dense (area_fraction_pct) or aligned (orientation_coherence high, avg_alignment_score low) matrix tiles."
            )
    st.code(outputs.get("manifest", ""), language=None)


def main() -> None:
    st.set_page_config(page_title="Co-localisation", page_icon=":material/hub:", layout="wide")
    initialize_state()
    st.title("Co-localisation")
    st.caption("Which cells sit together, and which cells sit in dense or aligned matrix. Every image is analysed separately.")

    sample_column, root_column, fill_column = st.columns([0.22, 0.50, 0.28])
    sample_column.text_input("Sample ID", key="coloc_sample_id")
    root_column.text_input("Project root", key="coloc_project_root")
    if fill_column.button("Fill standard sample paths", icon=":material/auto_fix_high:", width="stretch", key="coloc_fill"):
        root = Path(st.session_state["coloc_project_root"]).expanduser()
        for key, value in standard_defaults(st.session_state["coloc_sample_id"], root).items():
            st.session_state[key] = value
        for key in ("coloc_inspection", "coloc_status_path", "coloc_log_path"):
            st.session_state.pop(key, None)
        st.rerun()

    with st.container(border=True):
        left, right = st.columns(2)
        left.text_input("Tissue-regions AnnData (or any phenotyped AnnData)", key="coloc_adata")
        right.text_input("Fiber manifest (optional; enables cell–matrix analysis)", key="coloc_fiber_manifest")
        left.text_input("Output folder", key="coloc_output")

    if st.button("Inspect inputs", icon=":material/search:", key="coloc_inspect"):
        try:
            with st.spinner("Reading phenotype columns"):
                st.session_state["coloc_inspection"] = inspect_inputs({"adata_path": path_value("coloc_adata")})
            st.session_state["coloc_inspection_signature"] = path_value("coloc_adata")
        except Exception as error:
            st.session_state.pop("coloc_inspection", None)
            st.error(str(error))

    report = st.session_state.get("coloc_inspection")
    if not (report and st.session_state.get("coloc_inspection_signature") == path_value("coloc_adata")):
        if st.session_state.get("coloc_status_path"):
            st.subheader("Co-localisation run")
            render_worker(PREFIX, render_outputs)
        return

    previous = regions_manifest(path_value("coloc_adata"))
    columns = [c for c in report["candidate_columns"] if c != "tissue_region"]
    preferred = previous.get("phenotype_key") if previous.get("phenotype_key") in columns else columns[0]
    st.session_state.setdefault("coloc_phenotype_key", preferred)
    left, right = st.columns(2)
    phenotype_key = left.selectbox("Phenotype column", columns, key="coloc_phenotype_key")
    has_regions = "tissue_region" in report["candidate_columns"]
    right.metric("Tissue regions", "found" if has_regions else "not found", help="Run Tissue regions first to get per-region results.")
    phenotypes = [row["value"] for row in report["candidate_columns"][phenotype_key]]
    fiber_pixel = None
    if path_value("coloc_fiber_manifest"):
        fiber_pixel = (read_json(Path(path_value("coloc_fiber_manifest")).expanduser()) or {}).get("outputs", {}).get("pixel_size_um")
    st.session_state.setdefault("coloc_pixel_size", float(previous.get("pixel_size_um") or fiber_pixel or 0.0))

    with st.container(border=True):
        st.markdown("**Cell–cell**")
        st.caption("Each row asks: are target cells found around source cells more than chance?")
        pairs_key = f"coloc_pairs_{phenotype_key}"
        if pairs_key not in st.session_state:
            myeloid = next((p for p in phenotypes if any(t in p.lower() for t in ("macrophage", "myeloid", "cd68"))), phenotypes[0])
            fibro = next((p for p in phenotypes if "fibro" in p.lower()), phenotypes[min(1, len(phenotypes) - 1)])
            st.session_state[pairs_key] = pd.DataFrame([{"source": myeloid, "target": fibro}])
        pairs = st.data_editor(
            st.session_state[pairs_key],
            num_rows="dynamic",
            width="stretch",
            column_config={
                "source": st.column_config.SelectboxColumn("Source phenotype", options=phenotypes, required=True),
                "target": st.column_config.SelectboxColumn("Target phenotype", options=phenotypes, required=True),
            },
            key=f"{pairs_key}_editor",
        )
        left, middle, right, fourth = st.columns(4)
        radii = left.text_input("Radii (µm)", "10, 20, 30, 50, 75, 100", key="coloc_radii")
        local_radius = middle.number_input("Neighbourhood radius by region (µm)", 5.0, 500.0, 30.0, 5.0, key="coloc_local_radius")
        permutations = right.number_input("Permutations", 19, 9999, 199, 10, key="coloc_permutations")
        fourth.number_input("Pixel size (µm / px)", 0.0, 100.0, step=0.001, format="%.4f", key="coloc_pixel_size")

    matrix_config = {}
    fiber_manifest = path_value("coloc_fiber_manifest")
    if fiber_manifest:
        manifest = read_json(Path(fiber_manifest).expanduser()) or {}
        channels = manifest.get("outputs", {}).get("channels", [])
        with st.container(border=True):
            st.markdown("**Cell–matrix**")
            left, right = st.columns(2)
            fiber_types = left.multiselect("Matrix channels", channels, default=channels, key="coloc_fiber_types")
            defaults = [p for p in phenotypes if any(t in p.lower() for t in ("macrophage", "myeloid", "fibro", "wnt5a", "tumour", "tumor"))]
            matrix_phenotypes = right.multiselect("Phenotypes to test", phenotypes, default=defaults or phenotypes[:4], key=f"coloc_matrix_pheno_{phenotype_key}")
            left, middle, right, fourth = st.columns(4)
            matrix_radius = left.number_input("Matrix neighbourhood radius (µm)", 2.0, 500.0, 20.0, 1.0, key="coloc_matrix_radius")
            contact = middle.number_input("Contact distance (µm)", 0.5, 100.0, 5.0, 0.5, key="coloc_contact")
            tile_um = right.number_input("Tile size for dense / aligned matrix (µm)", 20.0, 2000.0, 150.0, 10.0, key="coloc_tile")
            quantile = fourth.number_input("High-matrix tile quantile", 0.5, 0.99, 0.75, 0.05, key="coloc_quantile")
            matrix_config = {
                "fiber_manifest": fiber_manifest,
                "fiber_types": fiber_types,
                "matrix_phenotypes": matrix_phenotypes,
                "matrix_radius_um": float(matrix_radius),
                "contact_distance_um": float(contact),
                "tile_size_um": float(tile_um),
                "high_quantile": float(quantile),
            }
    else:
        st.info("Add a fiber manifest to also test which cells sit in dense or aligned matrix.", icon=":material/info:")

    if st.button("Run co-localisation", type="primary", icon=":material/play_arrow:", key="coloc_run"):
        pair_rows = pairs.dropna().to_dict("records")
        try:
            radii_um = parse_numbers(radii)
        except ValueError:
            radii_um = []
        if not pair_rows and not matrix_config:
            st.error("Add at least one cell pair or a fiber manifest")
        elif pair_rows and not radii_um:
            st.error("Radii must be numbers separated by commas")
        elif not st.session_state.get("coloc_pixel_size"):
            st.error("Enter the pixel size")
        else:
            config = {
                "sample_id": st.session_state["coloc_sample_id"].strip(),
                "adata_path": path_value("coloc_adata"),
                "output_dir": path_value("coloc_output"),
                "phenotype_key": phenotype_key,
                "region_key": "tissue_region" if has_regions else None,
                "pixel_size_um": float(st.session_state["coloc_pixel_size"]),
                "pairs": pair_rows,
                "radii_um": radii_um,
                "local_radius_um": float(local_radius),
                "n_permutations": int(permutations),
                **matrix_config,
            }
            try:
                output_dir = Path(config["output_dir"]).expanduser().resolve()
                remember_worker(PREFIX, *start_worker("spatioev.workflows.colocalization", config, output_dir, "colocalization"))
                st.rerun()
            except Exception as error:
                st.error(str(error))

    if st.session_state.get("coloc_status_path"):
        st.subheader("Co-localisation run")
        render_worker(PREFIX, render_outputs)


if __name__ == "__main__":
    main()
