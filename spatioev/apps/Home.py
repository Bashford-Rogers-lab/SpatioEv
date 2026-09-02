"""Entry page for the staged SpatioEv image-analysis workflows."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from spatioev.apps._common import default_project_root


def _apply_context(project_root: str, sample_id: str) -> None:
    root = str(Path(project_root).expanduser().resolve())
    os.environ["SPATIOEV_PROJECT_ROOT"] = root
    st.session_state["spatioev_project_root"] = root
    st.session_state["spatioev_sample_id"] = sample_id
    st.session_state["cluster_project_root"] = root
    st.session_state["cluster_sample_id"] = sample_id
    st.session_state["project_root"] = root
    st.session_state["sample_id"] = sample_id
    st.session_state["scimap_project_root"] = root
    st.session_state["scimap_sample_id"] = sample_id


def _artifact_state(path: Path) -> str:
    return "Ready" if path.exists() else "Not started"


def main() -> None:
    st.set_page_config(
        page_title="SpatioEv workflows",
        page_icon=":material/hub:",
        layout="wide",
    )
    st.markdown(
        """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1400px; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    button[kind="primary"] { background: #277c75 !important; border-color: #277c75 !important; }
    button[kind="primary"]:hover { background: #1f665f !important; border-color: #1f665f !important; }
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("SpatioEv image analysis")

    default_root = st.session_state.get("spatioev_project_root", str(default_project_root()))
    default_sample = st.session_state.get("spatioev_sample_id", "sample")
    root_column, sample_column, apply_column = st.columns([0.56, 0.24, 0.20])
    root_text = root_column.text_input("Project root", value=default_root)
    sample_id = sample_column.text_input("Sample ID", value=default_sample)
    if apply_column.button("Apply", type="primary", icon=":material/check:", width="stretch"):
        _apply_context(root_text, sample_id.strip() or "sample")
        st.rerun()

    root = Path(root_text).expanduser()
    sample = sample_id.strip() or "sample"
    stages = [
        (
            "00 Prepare AnnData",
            "pages/00_prepare_anndata.py",
            root / f"{sample}_adata.h5ad",
            ":material/table_view:",
        ),
        (
            "01 Broad clustering",
            "pages/01_broad_clustering.py",
            root / "results" / f"{sample}_clustering_workflow",
            ":material/scatter_plot:",
        ),
        (
            "02 Marker autogating",
            "pages/02_marker_autogating.py",
            root / "results" / f"{sample}_marker_gating_qc",
            ":material/tune:",
        ),
        (
            "03 Subset phenotyping",
            "pages/03_subset_phenotyping.py",
            root / "results" / f"{sample}_scimap_phenotyping_interface",
            ":material/account_tree:",
        ),
    ]

    st.divider()
    for index, (title, page, artifact, icon) in enumerate(stages):
        name_column, state_column, action_column = st.columns([0.54, 0.20, 0.26])
        name_column.subheader(title)
        state_column.caption("Status")
        state_column.write(f"**{_artifact_state(artifact)}**")
        action_column.page_link(page, label="Open", icon=icon, width="stretch")
        if index < len(stages) - 1:
            st.divider()


if __name__ == "__main__":
    main()
