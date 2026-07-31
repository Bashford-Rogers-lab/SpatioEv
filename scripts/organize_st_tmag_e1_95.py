"""Organize and rewrite the ST_TMAG_E1_95 analysis assets."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from textwrap import dedent

import anndata as ad
import nbformat as nbf
import numpy as np
import pandas as pd


SAMPLE_ID = "ST_TMAG_E1_95"
PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_95")
SPATIOEV_REPO = Path("/Users/shihongwu/SpatioEv")

MARKER_ORDER = [
    "DNA_1",
    "FMRP",
    "AMH",
    "GFRA1",
    "SOX9",
    "CREM",
    "MAGEA4",
    "INSL3",
    "KRT18",
    "PIWIL4",
    "NaKATPase",
    "panCad",
    "DMRT1",
    "STAR",
    "CYP17A1",
    "SYCP3",
    "CYP11A1",
    "PCNA",
    "CD45",
    "CD68",
    "NF2F2",
    "CD31",
    "UTF1_2",
    "Vimentin_2",
    "CX43_2",
    "Acrosin",
    "MYH11",
    "SMA",
    "CD8",
    "CLDN11_2",
]

MARKER_ALIASES = {
    "CLDN11_1": "CLDN11_2",
    "Vimentin_1": "Vimentin_2",
}


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


def project_paths(project_dir: Path = PROJECT_DIR) -> dict[str, Path]:
    return {
        "data": project_dir / "data",
        "figures": project_dir / "figures",
        "notebooks": project_dir / "notebooks",
        "background": project_dir / "background",
        "dearray": project_dir / "dearray",
        "qc_segmentation": project_dir / "qc" / "segmentation_qc",
        "qc_figures": project_dir / "qc" / "segmentation_qc" / "figures",
        "qc_tables": project_dir / "qc" / "segmentation_qc" / "tables",
        "qc_adata": project_dir / "qc" / "segmentation_qc" / "adata",
        "phenotyping": project_dir / "phenotyping" / "manual_gating",
        "phenotyping_figures": project_dir / "phenotyping" / "manual_gating" / "figures",
        "phenotyping_tables": project_dir / "phenotyping" / "manual_gating" / "tables",
        "phenotyping_adata": project_dir / "phenotyping" / "manual_gating" / "adata",
        "legacy_notebooks": project_dir / "notebooks" / "legacy",
    }


def ensure_layout(project_dir: Path = PROJECT_DIR) -> dict[str, Path]:
    paths = project_paths(project_dir)
    for key, path in paths.items():
        if key == "background":
            continue
        path.mkdir(parents=True, exist_ok=True)

    if not paths["dearray"].exists():
        raise FileNotFoundError(f"Expected dearray folder is missing: {paths['dearray']}")

    if not paths["background"].exists():
        paths["background"].symlink_to("dearray", target_is_directory=True)
    elif paths["background"].is_symlink():
        target = os.readlink(paths["background"])
        if target != "dearray":
            raise RuntimeError(f"Existing background symlink points to {target!r}")

    return paths


def write_marker_manifest(paths: dict[str, Path]) -> Path:
    marker_path = paths["background"] / f"{SAMPLE_ID}_markers.csv"
    marker_df = pd.DataFrame(
        {
            "channel_number": np.arange(1, len(MARKER_ORDER) + 1, dtype=int),
            "marker_name": MARKER_ORDER,
            "source": "legacy_notebook_marker_order",
        }
    )
    marker_df.to_csv(marker_path, index=False)
    return marker_path


def copy_legacy_inputs(paths: dict[str, Path], project_dir: Path = PROJECT_DIR) -> None:
    for filename in [
        f"1_anndata_creation_from_quantification_csv_{SAMPLE_ID}.ipynb",
        f"2_gating_rescaling_phenotyping_{SAMPLE_ID}.ipynb",
    ]:
        source = project_dir / filename
        if source.exists():
            shutil.copy2(source, paths["legacy_notebooks"] / filename)

    manual_source = project_dir / "manual_gates.csv"
    if manual_source.exists():
        shutil.copy2(manual_source, paths["phenotyping_tables"] / f"{SAMPLE_ID}_manual_gates.csv")
        shutil.copy2(manual_source, paths["phenotyping_tables"] / f"{SAMPLE_ID}_manual_gates_legacy.csv")

    workflow_source = project_dir / "phenotype_workflow.csv"
    if workflow_source.exists():
        workflow = pd.read_csv(workflow_source)
        workflow.to_csv(paths["phenotyping_tables"] / f"{SAMPLE_ID}_phenotype_workflow_legacy.csv", index=False)
        workflow = workflow.rename(columns=MARKER_ALIASES)
        duplicate_cols = workflow.columns[workflow.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ValueError(f"Duplicate columns after marker alias correction: {duplicate_cols}")
        workflow.to_csv(paths["phenotyping_tables"] / f"{SAMPLE_ID}_phenotype_workflow.csv", index=False)


def read_ark_tables(project_dir: Path = PROJECT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    ark_wdirs = sorted(
        path
        for path in project_dir.iterdir()
        if path.is_dir() and path.name.startswith("ark_wdir")
    )
    if not ark_wdirs:
        raise FileNotFoundError(f"No ark_wdir folders found in {project_dir}")

    arcsinh_tables: list[pd.DataFrame] = []
    size_tables: list[pd.DataFrame] = []
    for ark_dir in ark_wdirs:
        cell_table_dir = ark_dir / "segmentation" / "cell_table"
        arcsinh_path = cell_table_dir / "cell_table_arcsinh_transformed.csv"
        size_path = cell_table_dir / "cell_table_size_normalized.csv"
        if not arcsinh_path.exists() or not size_path.exists():
            raise FileNotFoundError(f"Missing cell-table CSVs under {cell_table_dir}")

        arcsinh = pd.read_csv(arcsinh_path)
        size = pd.read_csv(size_path)
        if len(arcsinh) != len(size):
            raise ValueError(f"Row-count mismatch in {ark_dir.name}")
        if list(arcsinh.columns) != list(size.columns):
            raise ValueError(f"Column mismatch between arcsinh and size-normalized tables in {ark_dir.name}")

        arcsinh["source_ark_wdir"] = ark_dir.name
        size["source_ark_wdir"] = ark_dir.name
        arcsinh_tables.append(arcsinh)
        size_tables.append(size)

    return pd.concat(arcsinh_tables, ignore_index=True), pd.concat(size_tables, ignore_index=True), ark_wdirs


def build_adata(paths: dict[str, Path], marker_path: Path, project_dir: Path = PROJECT_DIR) -> Path:
    arcsinh, size, ark_wdirs = read_ark_tables(project_dir)
    markers_df = pd.read_csv(marker_path)
    marker_order = markers_df["marker_name"].astype(str).tolist()

    if "mask_type" not in arcsinh.columns:
        raise ValueError("Expected a mask_type column in the ARK cell tables.")
    mask = arcsinh["mask_type"].eq("whole_cell")
    if not mask.any():
        raise ValueError("No rows found with mask_type == 'whole_cell'.")
    if not size.loc[mask, "mask_type"].eq("whole_cell").all():
        raise ValueError("Arcsinh and size-normalized mask_type rows do not align.")

    arcsinh = arcsinh.loc[mask].reset_index(drop=True)
    size = size.loc[mask].reset_index(drop=True)

    if "label" not in arcsinh.columns:
        raise ValueError("Expected label column to split expression and metadata.")
    split_idx = arcsinh.columns.get_loc("label")
    pre_label_columns = list(arcsinh.columns[:split_idx])
    missing_markers = [marker for marker in marker_order if marker not in pre_label_columns]
    if missing_markers:
        raise ValueError(f"Markers missing from ARK table: {missing_markers}")

    obs = arcsinh.iloc[:, split_idx:].copy()
    if "cell_size" in arcsinh.columns:
        obs["cell_size"] = pd.to_numeric(arcsinh["cell_size"], errors="coerce").to_numpy()

    obs = obs.rename(columns={"centroid-1": "X_centroid", "centroid-0": "Y_centroid"})
    for col in ["X_centroid", "Y_centroid", "fov", "area", "nc_ratio"]:
        if col not in obs.columns:
            raise ValueError(f"Missing required metadata column: {col}")

    obs["sample_id"] = SAMPLE_ID
    obs["slide_id"] = SAMPLE_ID
    obs["fov"] = obs["fov"].astype(str)
    obs["imageid"] = obs["fov"]
    obs["tissue_piece"] = obs["fov"]
    obs["fov_index"] = obs["fov"].str.extract(r"fov(\\d+)")[0].astype("Int64")
    if "source_ark_wdir" in obs.columns:
        obs["ark_wdir_piece"] = obs["source_ark_wdir"].astype(str)

    obs.index = [
        f"{SAMPLE_ID}_{fov}_{label}_{i}"
        for i, (fov, label) in enumerate(zip(obs["fov"].astype(str), obs["label"].astype(str)))
    ]
    obs.index.name = "cell_id"

    var = markers_df.copy()
    var.index = marker_order
    var.index.name = "marker"
    var["marker_order"] = np.arange(len(marker_order), dtype=int)

    adata = ad.AnnData(
        X=arcsinh[marker_order].to_numpy(dtype="float32"),
        obs=obs,
        var=var,
    )
    adata.layers["size_normalized"] = size[marker_order].to_numpy(dtype="float32")
    adata.raw = ad.AnnData(
        X=size[marker_order].to_numpy(dtype="float32"),
        obs=obs.copy(),
        var=var.copy(),
    )
    adata.obsm["spatial"] = obs[["X_centroid", "Y_centroid"]].to_numpy(dtype="float32")
    adata.uns["sample_id"] = SAMPLE_ID
    adata.uns["source_ark_wdirs"] = [str(path) for path in ark_wdirs]
    adata.uns["marker_order_source"] = str(marker_path)
    adata.uns["mask_type"] = "whole_cell"

    output_h5ad = paths["data"] / f"{SAMPLE_ID}_adata.h5ad"
    adata.write_h5ad(output_h5ad, compression="gzip")
    return output_h5ad


def adata_creation_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            f"""
            # {SAMPLE_ID} AnnData Creation

            This notebook creates a clean `AnnData` object from the two ARK working
            directories in `/Volumes/Shihong_3/{SAMPLE_ID}`. It preserves the TMA
            FOV labels as `imageid` so downstream image overlays can target one
            core at a time.
            """
        ),
        code(
            r"""
            # Keep matplotlib/numba caches in writable locations when running from locked-down envs.
            import os
            from pathlib import Path

            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
            os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

            import datetime as dt
            import warnings

            import anndata as ad
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            # Optional imports. Creating the h5ad only requires pandas + anndata.
            try:
                import spatioev as sv
            except Exception as exc:
                sv = None
                print(f"Optional import failed: spatioev ({type(exc).__name__}: {exc})")

            try:
                import scanpy as sc
            except Exception as exc:
                sc = None
                print(f"Optional import failed: scanpy ({type(exc).__name__}: {exc})")

            try:
                import scimap as sm
            except Exception as exc:
                sm = None
                print(f"Optional import failed: scimap ({type(exc).__name__}: {exc})")

            warnings.filterwarnings("ignore", category=FutureWarning)

            print("Current date and time:", dt.datetime.now())
            print("anndata version:", ad.__version__)
            if sv is not None:
                print("spatioev version:", getattr(sv, "__version__", "unknown"))
            """
        ),
        md(
            """
            ## 1. Paths And Analysis Settings

            The image/background folder for this TMA is `dearray`. A `background`
            compatibility link points to the same folder so this project follows
            the same layout pattern as `ST_Pat4_51` without duplicating OME-TIFFs.
            """
        ),
        code(
            f"""
            SAMPLE_ID = "{SAMPLE_ID}"
            PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_95")
            ARK_WDIRS = sorted(p for p in PROJECT_DIR.iterdir() if p.is_dir() and p.name.startswith("ark_wdir"))
            BACKGROUND_DIR = PROJECT_DIR / "background"
            DEARRAY_DIR = PROJECT_DIR / "dearray"
            MARKERS_CSV = BACKGROUND_DIR / f"{{SAMPLE_ID}}_markers.csv"
            DATA_DIR = PROJECT_DIR / "data"
            FIGURES_DIR = PROJECT_DIR / "figures"
            OUTPUT_H5AD = DATA_DIR / f"{{SAMPLE_ID}}_adata.h5ad"
            MASK_TYPE = "whole_cell"

            for folder in [DATA_DIR, FIGURES_DIR]:
                folder.mkdir(parents=True, exist_ok=True)

            if not ARK_WDIRS:
                raise FileNotFoundError(f"No ark_wdir folders found under {PROJECT_DIR}")
            for path in [DEARRAY_DIR, BACKGROUND_DIR, MARKERS_CSV]:
                if not path.exists():
                    raise FileNotFoundError(path)

            print("ARK working dirs:")
            for path in ARK_WDIRS:
                print(" -", path)
            print("Background/dearray folder:", BACKGROUND_DIR.resolve())
            print("Marker manifest:", MARKERS_CSV)
            print("Output AnnData:", OUTPUT_H5AD)
            """
        ),
        md(
            """
            ## 2. Load Marker Order

            The source folder did not contain a separate MCMICRO marker manifest,
            so the marker order is preserved from the original ST_TMAG notebook.
            """
        ),
        code(
            """
            markers_df = pd.read_csv(MARKERS_CSV)
            required_marker_columns = {"channel_number", "marker_name"}
            missing_marker_columns = required_marker_columns - set(markers_df.columns)
            if missing_marker_columns:
                raise ValueError(f"Markers CSV is missing columns: {sorted(missing_marker_columns)}")

            marker_order = markers_df["marker_name"].astype(str).tolist()
            print(f"Loaded {len(marker_order)} markers")
            marker_order
            """
        ),
        md(
            """
            ## 3. Load And Concatenate ARK Cell Tables

            `ark_wdir_1` contains FOV 1-9 and `ark_wdir_2` contains FOV 10-17.
            The notebook validates that arcsinh and size-normalized tables remain
            row-aligned before concatenating them.
            """
        ),
        code(
            """
            def read_ark_tables(ark_wdirs):
                arcsinh_tables = []
                size_tables = []
                for ark_dir in ark_wdirs:
                    cell_table_dir = ark_dir / "segmentation" / "cell_table"
                    arcsinh_path = cell_table_dir / "cell_table_arcsinh_transformed.csv"
                    size_path = cell_table_dir / "cell_table_size_normalized.csv"
                    if not arcsinh_path.exists() or not size_path.exists():
                        raise FileNotFoundError(f"Missing cell-table CSVs under {cell_table_dir}")

                    arcsinh = pd.read_csv(arcsinh_path)
                    size = pd.read_csv(size_path)
                    if len(arcsinh) != len(size):
                        raise ValueError(f"Row-count mismatch in {ark_dir.name}")
                    if list(arcsinh.columns) != list(size.columns):
                        raise ValueError(f"Column mismatch in {ark_dir.name}")

                    arcsinh["source_ark_wdir"] = ark_dir.name
                    size["source_ark_wdir"] = ark_dir.name
                    arcsinh_tables.append(arcsinh)
                    size_tables.append(size)

                    fovs = sorted(arcsinh["fov"].astype(str).unique(), key=lambda x: int(x.replace("fov", "")))
                    print(f"{ark_dir.name}: {arcsinh.shape[0]:,} rows, FOVs {fovs}")

                return pd.concat(arcsinh_tables, ignore_index=True), pd.concat(size_tables, ignore_index=True)


            ark_arcsinh, ark_size_normalized = read_ark_tables(ARK_WDIRS)
            print("Concatenated arcsinh:", ark_arcsinh.shape)
            print("Concatenated size-normalized:", ark_size_normalized.shape)
            """
        ),
        md(
            """
            ## 4. Filter To Whole-Cell Measurements

            The old notebook used `mask_type == "whole_cell"`. Nuclear marker
            summaries and nuclear morphology columns remain in `obs`.
            """
        ),
        code(
            """
            mask = ark_arcsinh["mask_type"].eq(MASK_TYPE)
            if not mask.any():
                raise ValueError(f"No rows found for mask_type == {MASK_TYPE!r}")
            if not ark_size_normalized.loc[mask, "mask_type"].eq(MASK_TYPE).all():
                raise ValueError("Arcsinh and size-normalized mask_type rows do not align.")

            ark_arcsinh = ark_arcsinh.loc[mask].reset_index(drop=True)
            ark_size_normalized = ark_size_normalized.loc[mask].reset_index(drop=True)
            print(f"Kept {len(ark_arcsinh):,} rows with mask_type == {MASK_TYPE!r}")
            print(ark_arcsinh["fov"].value_counts().sort_index())
            """
        ),
        md(
            """
            ## 5. Split Expression From Metadata

            Marker columns occur before `label`. The `cell_size` column is kept as
            metadata, not expression.
            """
        ),
        code(
            """
            if "label" not in ark_arcsinh.columns:
                raise ValueError("Expected a label column to split marker intensities from metadata.")

            split_idx = ark_arcsinh.columns.get_loc("label")
            pre_label_columns = list(ark_arcsinh.columns[:split_idx])
            missing_markers = [marker for marker in marker_order if marker not in pre_label_columns]
            if missing_markers:
                raise ValueError(f"Markers missing from ARK table: {missing_markers}")

            X_arcsinh = ark_arcsinh[marker_order].copy()
            X_size_normalized = ark_size_normalized[marker_order].copy()
            obs = ark_arcsinh.iloc[:, split_idx:].copy()
            if "cell_size" in ark_arcsinh.columns:
                obs["cell_size"] = pd.to_numeric(ark_arcsinh["cell_size"], errors="coerce").to_numpy()

            print("Expression matrix:", X_arcsinh.shape)
            print("Metadata table:", obs.shape)
            """
        ),
        md(
            """
            ## 6. Harmonize Spatial Metadata

            Coordinates remain in the per-FOV image pixel space. `imageid` is set
            to the ARK `fov` value so scimap and SpatioEv plotting can subset one
            TMA core/image at a time.
            """
        ),
        code(
            """
            obs = obs.rename(columns={"centroid-1": "X_centroid", "centroid-0": "Y_centroid"})
            for col in ["X_centroid", "Y_centroid", "fov", "area", "nc_ratio"]:
                if col not in obs.columns:
                    raise ValueError(f"Missing required metadata column: {col}")

            obs["sample_id"] = SAMPLE_ID
            obs["slide_id"] = SAMPLE_ID
            obs["fov"] = obs["fov"].astype(str)
            obs["imageid"] = obs["fov"]
            obs["tissue_piece"] = obs["fov"]
            obs["fov_index"] = obs["fov"].str.extract(r"fov(\\d+)")[0].astype("Int64")
            obs["ark_wdir_piece"] = obs["source_ark_wdir"].astype(str)

            obs.index = [
                f"{SAMPLE_ID}_{fov}_{label}_{i}"
                for i, (fov, label) in enumerate(zip(obs["fov"].astype(str), obs["label"].astype(str)))
            ]
            obs.index.name = "cell_id"
            obs[["imageid", "X_centroid", "Y_centroid", "source_ark_wdir"]].head()
            """
        ),
        md(
            """
            ## 7. Create AnnData

            `adata.X` stores arcsinh-transformed expression. The size-normalized
            values are stored in `adata.layers["size_normalized"]` and `adata.raw`.
            """
        ),
        code(
            """
            var = markers_df.copy()
            var.index = marker_order
            var.index.name = "marker"
            var["marker_order"] = np.arange(len(marker_order), dtype=int)

            adata = ad.AnnData(
                X=X_arcsinh.to_numpy(dtype="float32"),
                obs=obs,
                var=var,
            )
            adata.layers["size_normalized"] = X_size_normalized.to_numpy(dtype="float32")
            adata.raw = ad.AnnData(
                X=X_size_normalized.to_numpy(dtype="float32"),
                obs=obs.copy(),
                var=var.copy(),
            )
            adata.obsm["spatial"] = obs[["X_centroid", "Y_centroid"]].to_numpy(dtype="float32")
            adata.uns["sample_id"] = SAMPLE_ID
            adata.uns["source_ark_wdirs"] = [str(path) for path in ARK_WDIRS]
            adata.uns["marker_order_source"] = str(MARKERS_CSV)
            adata.uns["mask_type"] = MASK_TYPE

            print(adata)
            print("Layers:", list(adata.layers.keys()))
            print("obsm:", list(adata.obsm.keys()))
            """
        ),
        md(
            """
            ## 8. Basic QC Summaries

            These checks confirm marker values, coordinates, and FOV labels before
            saving the handoff `.h5ad`.
            """
        ),
        code(
            """
            print("Number of cells:", adata.n_obs)
            print("Number of markers:", adata.n_vars)
            print("NaNs in X:", int(np.isnan(adata.X).sum()))
            print("NaNs in spatial:", int(np.isnan(adata.obsm["spatial"]).sum()))
            print("Cells per imageid:")
            print(adata.obs["imageid"].value_counts().sort_index())

            marker_summary = pd.DataFrame(
                {
                    "arcsinh_mean": np.asarray(adata.X).mean(axis=0),
                    "arcsinh_std": np.asarray(adata.X).std(axis=0),
                    "size_normalized_mean": np.asarray(adata.layers["size_normalized"]).mean(axis=0),
                },
                index=adata.var_names,
            )
            marker_summary
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.bar(np.arange(adata.n_vars), np.asarray(adata.X).mean(axis=0))
            ax.set_xticks(np.arange(adata.n_vars))
            ax.set_xticklabels(adata.var_names, rotation=90)
            ax.set_ylabel("Mean arcsinh intensity")
            ax.set_title(f"{SAMPLE_ID}: marker means in preserved channel order")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_marker_mean_arcsinh.png", dpi=200)
            plt.show()
            plt.close(fig)
            """
        ),
        code(
            """
            def fov_sort_key(value):
                text = str(value)
                suffix = text.replace("fov", "")
                return int(suffix) if suffix.isdigit() else text


            imageids = sorted(adata.obs["imageid"].astype(str).unique(), key=fov_sort_key)
            ncols = 5
            nrows = int(np.ceil(len(imageids) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)

            for ax, imageid in zip(axes.ravel(), imageids):
                frame = adata.obs.loc[adata.obs["imageid"].astype(str).eq(imageid)]
                ax.scatter(frame["X_centroid"], frame["Y_centroid"], s=0.08, alpha=0.55, linewidths=0)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_title(f"{imageid} ({len(frame):,} cells)")
                ax.set_xticks([])
                ax.set_yticks([])

            for ax in axes.ravel()[len(imageids):]:
                ax.axis("off")

            fig.suptitle(f"{SAMPLE_ID}: FOV-local coordinate preview", y=1.01)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_fov_spatial_preview_grid.png", dpi=200, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## 9. Save And Reload

            The saved object is the primary handoff for QC, phenotyping, and
            downstream spatial analysis.
            """
        ),
        code(
            """
            adata.write_h5ad(OUTPUT_H5AD, compression="gzip")
            print("Saved:", OUTPUT_H5AD)
            print("Size:", f"{OUTPUT_H5AD.stat().st_size / 1024**2:.1f} MB")

            adata_check = ad.read_h5ad(OUTPUT_H5AD)
            print(adata_check)
            print("Reloaded markers match:", list(adata_check.var_names) == marker_order)
            print("Reloaded imageids:", sorted(adata_check.obs["imageid"].astype(str).unique()))
            """
        ),
    ]


def qc_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            f"""
            # {SAMPLE_ID} Segmentation QC

            This notebook runs segmentation QC on the `AnnData` object created by
            `00_{SAMPLE_ID}_adata_creation.ipynb`.
            """
        ),
        code(
            f"""
            import os
            import sys
            from pathlib import Path

            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
            os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

            import datetime as dt
            import warnings

            import anndata as ad
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            SPATIOEV_REPO = Path("{SPATIOEV_REPO}")
            if SPATIOEV_REPO.exists() and str(SPATIOEV_REPO) not in sys.path:
                sys.path.insert(0, str(SPATIOEV_REPO))

            import spatioev as sv
            from spatioev.config import QCConfig
            from spatioev.pp import run_segmentation_qc, filter_segmentation_errors, generate_qc_summary
            from spatioev.pl import plot_area_distribution, plot_nc_ratio_distribution

            warnings.filterwarnings("ignore", category=FutureWarning)
            print("Current date and time:", dt.datetime.now())
            print("spatioev version:", getattr(sv, "__version__", "unknown"))
            print("anndata version:", ad.__version__)
            """
        ),
        md(
            """
            ## 1. Paths And QC Settings

            The QC outputs are kept under `qc/segmentation_qc`, leaving the
            original MCMICRO `qc/coreograph`, `qc/mesmer`, and `qc/provenance`
            folders untouched.
            """
        ),
        code(
            f"""
            SAMPLE_ID = "{SAMPLE_ID}"
            PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_95")
            INPUT_H5AD = PROJECT_DIR / "data" / f"{{SAMPLE_ID}}_adata.h5ad"
            MARKERS_CSV = PROJECT_DIR / "background" / f"{{SAMPLE_ID}}_markers.csv"
            QC_DIR = PROJECT_DIR / "qc" / "segmentation_qc"
            FIGURES_DIR = QC_DIR / "figures"
            TABLES_DIR = QC_DIR / "tables"
            ADATA_DIR = QC_DIR / "adata"
            QC_ANNOTATED_H5AD = ADATA_DIR / f"{{SAMPLE_ID}}_adata_segmentation_qc.h5ad"
            QC_FILTERED_H5AD = ADATA_DIR / f"{{SAMPLE_ID}}_adata_segmentation_qc_filtered.h5ad"
            SAVE_OUTPUTS = True

            for folder in [FIGURES_DIR, TABLES_DIR, ADATA_DIR]:
                folder.mkdir(parents=True, exist_ok=True)

            config = QCConfig(
                pixel_size=0.325,
                min_area_um2=10,
                max_area_um2=1000,
                max_nc_ratio=1.0,
            )

            for path in [INPUT_H5AD, MARKERS_CSV]:
                if not path.exists():
                    raise FileNotFoundError(path)

            print("Input AnnData:", INPUT_H5AD)
            print("Marker order CSV:", MARKERS_CSV)
            print("QC output folder:", QC_DIR)
            print("QC config:", config)
            """
        ),
        md(
            """
            ## 2. Load AnnData

            The adata-creation notebook should have already:

            - Reordered markers according to `ST_TMAG_E1_95_markers.csv`
            - Stored arcsinh values in `adata.X`
            - Stored size-normalized values in `adata.layers["size_normalized"]`
            - Stored FOV-local pixel coordinates in `obs` and `obsm["spatial"]`
            """
        ),
        code(
            """
            adata = ad.read_h5ad(INPUT_H5AD)
            print(adata)
            print("Layers:", list(adata.layers.keys()))
            print("obsm:", list(adata.obsm.keys()))
            print("obs columns:", len(adata.obs.columns))
            print("var names:", list(adata.var_names))
            """
        ),
        md(
            """
            ## 3. Validate Marker Order And Required QC Fields

            This catches common handoff problems before any filtering is applied.
            """
        ),
        code(
            """
            required_obs_columns = [
                "area",
                "nc_ratio",
                "X_centroid",
                "Y_centroid",
                "imageid",
                "fov",
                "mask_type",
            ]
            missing_obs = [col for col in required_obs_columns if col not in adata.obs.columns]
            if missing_obs:
                raise ValueError(f"Missing required obs columns: {missing_obs}")

            marker_order = pd.read_csv(MARKERS_CSV)["marker_name"].astype(str).tolist()
            if list(adata.var_names) != marker_order:
                raise ValueError("adata.var_names do not match marker order from ST_TMAG_E1_95_markers.csv")
            print("Marker order matches marker manifest.")

            if "spatial" not in adata.obsm:
                adata.obsm["spatial"] = adata.obs[["X_centroid", "Y_centroid"]].to_numpy(dtype="float32")

            if not np.allclose(
                adata.obsm["spatial"],
                adata.obs[["X_centroid", "Y_centroid"]].to_numpy(dtype="float32"),
                equal_nan=True,
            ):
                warnings.warn("adata.obsm['spatial'] does not exactly match obs X/Y centroids.")

            print("Required QC columns present.")
            print("NaNs in X:", int(np.isnan(adata.X).sum()))
            print("NaNs in spatial:", int(np.isnan(adata.obsm["spatial"]).sum()))
            print("\\nimageid counts:")
            print(adata.obs["imageid"].value_counts().sort_index())
            print("\\nCoordinate ranges:")
            print(adata.obs[["X_centroid", "Y_centroid"]].agg(["min", "max"]))
            """
        ),
        md(
            """
            ## 4. Inspect Raw QC Metric Distributions
            """
        ),
        code(
            """
            metric_summary = adata.obs[["area", "nc_ratio"]].describe(
                percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
            ).T
            metric_summary
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(adata.obs["area"], bins=80, color="#4C78A8")
            axes[0].set_title("Cell area before QC")
            axes[0].set_xlabel("Area (pixels^2)")
            axes[0].set_ylabel("Cell count")

            nc_plot_max = adata.obs["nc_ratio"].quantile(0.995)
            axes[1].hist(adata.obs["nc_ratio"].clip(upper=nc_plot_max), bins=80, color="#F58518")
            axes[1].set_title("NC ratio before QC")
            axes[1].set_xlabel("NC ratio")
            axes[1].set_ylabel("Cell count")
            fig.tight_layout()
            if SAVE_OUTPUTS:
                fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_raw_qc_metric_distributions.png", dpi=200)
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## 5. Run SpatioEv Segmentation QC

            This adds:

            - `area_um2`
            - `area_category`
            - `nc_ratio_category`
            - `segmentation_qc_pass`
            - `segmentation_qc_reason`
            """
        ),
        code(
            """
            adata_qc = run_segmentation_qc(adata.copy(), config)
            pass_mask = (
                (adata_qc.obs["area_category"] == "normal_area")
                & (adata_qc.obs["nc_ratio_category"] == "normal_nc_ratio")
            )
            adata_qc.obs["segmentation_qc_pass"] = pass_mask

            reasons = []
            for _, row in adata_qc.obs[["area_category", "nc_ratio_category"]].iterrows():
                flags = []
                if row["area_category"] != "normal_area":
                    flags.append(row["area_category"])
                if row["nc_ratio_category"] != "normal_nc_ratio":
                    flags.append(row["nc_ratio_category"])
                reasons.append(";".join(flags) if flags else "pass")
            adata_qc.obs["segmentation_qc_reason"] = pd.Categorical(reasons)

            print(adata_qc)
            print("\\nArea categories:")
            print(adata_qc.obs["area_category"].value_counts())
            print("\\nNC ratio categories:")
            print(adata_qc.obs["nc_ratio_category"].value_counts())
            print("\\nUnion-aware pass/fail:")
            print(adata_qc.obs["segmentation_qc_pass"].value_counts())
            print("\\nReasons:")
            print(adata_qc.obs["segmentation_qc_reason"].value_counts())
            """
        ),
        md(
            """
            ## 6. Generate QC Summaries

            The SpatioEv template summary is included for continuity. This
            notebook also computes a union-aware summary from
            `segmentation_qc_pass`.
            """
        ),
        code(
            """
            def summarize_segmentation_qc(adata_qc, groupby=None):
                obs = adata_qc.obs
                groups = [(None, obs)] if groupby is None else list(obs.groupby(groupby, observed=True))
                rows = []
                for name, df in groups:
                    total = len(df)
                    failed = ~df["segmentation_qc_pass"].astype(bool)
                    debris = int((df["area_category"] == "debris_fragment").sum())
                    merged = int((df["area_category"] == "merged_cell").sum())
                    abnormal_nc = int((df["nc_ratio_category"] == "abnormal_nc_ratio").sum())
                    row = {
                        "total_cells": total,
                        "pass_cells": int((~failed).sum()),
                        "failed_cells": int(failed.sum()),
                        "percent_failed": round(float(failed.mean() * 100), 2) if total else 0,
                        "debris_fragment": debris,
                        "merged_cell": merged,
                        "abnormal_nc_ratio": abnormal_nc,
                    }
                    if groupby is not None:
                        row[groupby] = name
                    rows.append(row)
                columns = ([groupby] if groupby is not None else []) + [
                    "total_cells",
                    "pass_cells",
                    "failed_cells",
                    "percent_failed",
                    "debris_fragment",
                    "merged_cell",
                    "abnormal_nc_ratio",
                ]
                return pd.DataFrame(rows)[columns]


            spatioev_summary = generate_qc_summary(adata_qc)
            spatioev_summary_by_imageid = generate_qc_summary(adata_qc, groupby="imageid")

            qc_summary = summarize_segmentation_qc(adata_qc)
            qc_summary_by_imageid = summarize_segmentation_qc(adata_qc, groupby="imageid").sort_values(
                "imageid",
                key=lambda s: s.astype(str).str.replace("fov", "", regex=False).astype(int),
            )
            print("Union-aware overall summary:")
            display(qc_summary)
            print("Union-aware per-imageid summary:")
            display(qc_summary_by_imageid)

            if SAVE_OUTPUTS:
                qc_summary.to_csv(TABLES_DIR / f"{SAMPLE_ID}_segmentation_qc_summary.csv", index=False)
                qc_summary_by_imageid.to_csv(TABLES_DIR / f"{SAMPLE_ID}_segmentation_qc_summary_by_imageid.csv", index=False)
                spatioev_summary.to_csv(TABLES_DIR / f"{SAMPLE_ID}_spatioev_template_qc_summary.csv", index=False)
                spatioev_summary_by_imageid.to_csv(TABLES_DIR / f"{SAMPLE_ID}_spatioev_template_qc_summary_by_imageid.csv", index=False)
                print("Saved QC summary tables to:", TABLES_DIR)
            """
        ),
        md(
            """
            ## 7. Plot Template QC Distributions
            """
        ),
        code(
            """
            fig_area = plot_area_distribution(
                adata_qc,
                min_area=config.min_area_um2,
                max_area=config.max_area_um2,
            )
            if SAVE_OUTPUTS:
                fig_area.savefig(FIGURES_DIR / f"{SAMPLE_ID}_area_um2_distribution.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig_area)
            """
        ),
        code(
            """
            fig_nc = plot_nc_ratio_distribution(
                adata_qc,
                max_ratio=config.max_nc_ratio,
            )
            if SAVE_OUTPUTS:
                fig_nc.savefig(FIGURES_DIR / f"{SAMPLE_ID}_nc_ratio_distribution.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig_nc)
            """
        ),
        md(
            """
            ## 8. Per-FOV QC Overview
            """
        ),
        code(
            """
            plot_df = qc_summary_by_imageid.copy()
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].bar(plot_df["imageid"].astype(str), plot_df["total_cells"], color="#4C78A8")
            axes[0].set_title("Cells per imageid/fov")
            axes[0].set_xlabel("imageid")
            axes[0].set_ylabel("Cell count")
            axes[0].tick_params(axis="x", rotation=90)

            axes[1].bar(plot_df["imageid"].astype(str), plot_df["percent_failed"], color="#E45756")
            axes[1].set_title("QC failed cells per imageid/fov")
            axes[1].set_xlabel("imageid")
            axes[1].set_ylabel("Failed cells (%)")
            axes[1].tick_params(axis="x", rotation=90)
            fig.tight_layout()
            if SAVE_OUTPUTS:
                fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_qc_by_imageid.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## 9. Spatial View Of QC Flags

            ST_TMAG coordinates are FOV-local rather than one stitched full-slide
            canvas. The QC flag view is therefore faceted by `imageid` to avoid
            overlapping unrelated TMA cores.
            """
        ),
        code(
            """
            def fov_sort_key(value):
                text = str(value)
                suffix = text.replace("fov", "")
                return int(suffix) if suffix.isdigit() else text


            # Downsample points for fast plotting if needed; the full dataset is still retained.
            rng = np.random.default_rng(42)
            max_points = 60000
            if adata_qc.n_obs > max_points:
                plot_idx = rng.choice(adata_qc.n_obs, size=max_points, replace=False)
                obs_plot = adata_qc.obs.iloc[plot_idx].copy()
            else:
                obs_plot = adata_qc.obs.copy()

            imageids = sorted(obs_plot["imageid"].astype(str).unique(), key=fov_sort_key)
            ncols = 5
            nrows = int(np.ceil(len(imageids) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)

            for ax, imageid in zip(axes.ravel(), imageids):
                df = obs_plot.loc[obs_plot["imageid"].astype(str).eq(imageid)]
                pass_df = df[df["segmentation_qc_pass"].astype(bool)]
                fail_df = df[~df["segmentation_qc_pass"].astype(bool)]

                ax.scatter(
                    pass_df["X_centroid"],
                    pass_df["Y_centroid"],
                    s=0.08,
                    alpha=0.25,
                    color="#4C78A8",
                    linewidths=0,
                    label="pass",
                )
                ax.scatter(
                    fail_df["X_centroid"],
                    fail_df["Y_centroid"],
                    s=0.35,
                    alpha=0.8,
                    color="#E45756",
                    linewidths=0,
                    label="fail",
                )
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_title(f"{imageid} ({len(df):,} plotted)")
                ax.set_xticks([])
                ax.set_yticks([])

            for ax in axes.ravel()[len(imageids):]:
                ax.axis("off")

            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", frameon=False, markerscale=8)
            fig.suptitle(f"{SAMPLE_ID}: segmentation QC pass/fail by FOV", y=1.01)
            fig.tight_layout()
            if SAVE_OUTPUTS:
                fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_spatial_qc_flags_by_fov.png", dpi=300, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## 10. Create Filtered AnnData
            """
        ),
        code(
            """
            adata_filtered = adata_qc[adata_qc.obs["segmentation_qc_pass"].to_numpy()].copy()
            print("Original cells:", adata_qc.n_obs)
            print("Filtered cells:", adata_filtered.n_obs)
            print("Cells removed:", adata_qc.n_obs - adata_filtered.n_obs)
            print("Percent removed:", round((adata_qc.n_obs - adata_filtered.n_obs) / adata_qc.n_obs * 100, 2))

            print("\\nFiltered imageid counts:")
            print(adata_filtered.obs["imageid"].value_counts().sort_index())
            """
        ),
        md(
            """
            ## 11. Save QC Outputs

            Files written by this notebook:

            - QC-annotated AnnData: all cells with QC columns added
            - QC-filtered AnnData: only cells passing segmentation QC
            - Overall and per-FOV summary CSVs
            - QC distribution and per-FOV figures
            """
        ),
        code(
            """
            if SAVE_OUTPUTS:
                adata_qc.write_h5ad(QC_ANNOTATED_H5AD, compression="gzip")
                adata_filtered.write_h5ad(QC_FILTERED_H5AD, compression="gzip")
                print("Saved QC-annotated h5ad:", QC_ANNOTATED_H5AD)
                print("  size MB:", round(QC_ANNOTATED_H5AD.stat().st_size / 1024**2, 1))
                print("Saved QC-filtered h5ad:", QC_FILTERED_H5AD)
                print("  size MB:", round(QC_FILTERED_H5AD.stat().st_size / 1024**2, 1))
            else:
                print("SAVE_OUTPUTS is False; no h5ad files written.")
            """
        ),
        md(
            """
            ## 12. Reload Check

            A final quick check confirms the saved filtered object can be read
            and still contains the required spatial-analysis fields.
            """
        ),
        code(
            """
            if SAVE_OUTPUTS:
                check = ad.read_h5ad(QC_FILTERED_H5AD)
                print(check)
                for col in ["imageid", "fov", "X_centroid", "Y_centroid", "segmentation_qc_pass"]:
                    assert col in check.obs.columns, f"Missing column after reload: {col}"
                assert "spatial" in check.obsm, "Missing obsm['spatial'] after reload"
                print("Reload check passed.")
            else:
                print("SAVE_OUTPUTS is False; skipping reload check.")
            """
        ),
        md(
            f"""
            ## Notes For Downstream Analysis

            Use the filtered object for phenotyping and spatial analysis unless
            you have a reason to inspect failed cells:

            `/Volumes/Shihong_3/ST_TMAG_E1_95/qc/segmentation_qc/adata/{SAMPLE_ID}_adata_segmentation_qc_filtered.h5ad`

            For ST_TMAG, `imageid` is the FOV/core ID (`fov1` ... `fov17`).
            Keep using FOV-specific image IDs for neighborhood or image-overlay
            work because centroid coordinates are local to each TMA core.
            """
        ),
    ]


def manual_gating_notebook() -> list[nbf.NotebookNode]:
    return [
        md(
            f"""
            # {SAMPLE_ID}: Manual Gating And scimap Phenotyping

            This notebook rewrites the original ST_TMAG gating/rescaling notebook
            into the organized project layout. It uses the QC-filtered AnnData if
            available and falls back to the raw AnnData created by notebook 00.
            """
        ),
        code(
            f"""
            import os
            import sys
            from pathlib import Path

            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
            os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

            import datetime as dt
            import warnings

            import anndata as ad
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import scanpy as sc

            SPATIOEV_REPO = Path("{SPATIOEV_REPO}")
            if SPATIOEV_REPO.exists() and str(SPATIOEV_REPO) not in sys.path:
                sys.path.insert(0, str(SPATIOEV_REPO))

            try:
                import scimap as sm
            except ImportError:
                sm = None

            warnings.filterwarnings("ignore", category=FutureWarning)
            sc.settings.verbosity = 3
            sc.set_figure_params(dpi=150, fontsize=10, dpi_save=600)

            print("Current date and time:", dt.datetime.now())
            print("scimap available:", sm is not None)
            """
        ),
        md(
            """
            ## 1. Paths And Inputs

            The old root-level `manual_gates.csv` and `phenotype_workflow.csv`
            were copied into `phenotyping/manual_gating/tables`. The workflow copy
            uses the actual marker names present in the ARK cell table.
            """
        ),
        code(
            f"""
            SAMPLE_ID = "{SAMPLE_ID}"
            PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_95")
            BACKGROUND_DIR = PROJECT_DIR / "background"
            PHENOTYPING_DIR = PROJECT_DIR / "phenotyping" / "manual_gating"
            FIGURES_DIR = PHENOTYPING_DIR / "figures"
            TABLES_DIR = PHENOTYPING_DIR / "tables"
            ADATA_DIR = PHENOTYPING_DIR / "adata"
            QC_FILTERED_H5AD = PROJECT_DIR / "qc" / "segmentation_qc" / "adata" / f"{{SAMPLE_ID}}_adata_segmentation_qc_filtered.h5ad"
            FALLBACK_H5AD = PROJECT_DIR / "data" / f"{{SAMPLE_ID}}_adata.h5ad"
            INPUT_H5AD = QC_FILTERED_H5AD if QC_FILTERED_H5AD.exists() else FALLBACK_H5AD
            MARKERS_CSV = BACKGROUND_DIR / f"{{SAMPLE_ID}}_markers.csv"
            MANUAL_GATES_CSV = TABLES_DIR / f"{{SAMPLE_ID}}_manual_gates.csv"
            PHENOTYPE_WORKFLOW_CSV = TABLES_DIR / f"{{SAMPLE_ID}}_phenotype_workflow.csv"
            RESCALED_H5AD = ADATA_DIR / f"{{SAMPLE_ID}}_adata_scimap_rescaled.h5ad"
            PHENOTYPED_H5AD = ADATA_DIR / f"{{SAMPLE_ID}}_adata_scimap_phenotyped.h5ad"
            PHENOTYPE_ASSIGNMENTS_CSV = TABLES_DIR / f"{{SAMPLE_ID}}_phenotype_assignments.csv"

            for folder in [FIGURES_DIR, TABLES_DIR, ADATA_DIR]:
                folder.mkdir(parents=True, exist_ok=True)

            sc.settings.figdir = str(FIGURES_DIR)
            print("Input AnnData:", INPUT_H5AD)
            print("Manual gates:", MANUAL_GATES_CSV)
            print("Phenotype workflow:", PHENOTYPE_WORKFLOW_CSV)
            """
        ),
        md(
            """
            ## 2. Load AnnData And Validate Markers
            """
        ),
        code(
            """
            adata = ad.read_h5ad(INPUT_H5AD)
            markers_df = pd.read_csv(MARKERS_CSV)
            marker_order = markers_df["marker_name"].astype(str).tolist()
            adata = adata[:, marker_order].copy()

            required_obs = ["fov", "imageid", "X_centroid", "Y_centroid"]
            missing_obs = [col for col in required_obs if col not in adata.obs.columns]
            if missing_obs:
                raise ValueError(f"Missing required obs columns: {missing_obs}")

            print(adata)
            print("imageids:", sorted(adata.obs["imageid"].astype(str).unique(), key=lambda x: int(x.replace("fov", ""))))
            print("markers:", list(adata.var_names))
            """
        ),
        md(
            """
            ## 3. Manual Gate Review

            Fill or update the gate values in `ST_TMAG_E1_95_manual_gates.csv`.
            The table is FOV-specific, matching the original workflow.
            """
        ),
        code(
            """
            manual_gate = pd.read_csv(MANUAL_GATES_CSV)
            if "markers" not in manual_gate.columns:
                raise ValueError("Manual gate table must contain a 'markers' column.")

            gate_markers = manual_gate["markers"].dropna().astype(str).tolist()
            missing_gate_markers = [marker for marker in gate_markers if marker not in adata.var_names]
            if missing_gate_markers:
                raise ValueError(f"Gate markers missing from AnnData: {missing_gate_markers}")

            gate_value_columns = [col for col in manual_gate.columns if col != "markers"]
            blank_gate_count = int(manual_gate[gate_value_columns].isna().sum().sum())
            total_gate_slots = int(manual_gate[gate_value_columns].size)
            print(f"Blank gate values: {blank_gate_count:,} / {total_gate_slots:,}")
            manual_gate.head()
            """
        ),
        md(
            """
            ## 4. Optional Interactive Gate Finder

            Set `RUN_GATE_FINDER = True` for one marker and FOV at a time. For
            FOV `fov1`, the image is `background/1.ome.tif`; for `fov17`, it is
            `background/17.ome.tif`.
            """
        ),
        code("%gui qt"),
        code(
            """
            def require_scimap():
                if sm is None:
                    raise ImportError(
                        "scimap is not available in this Python kernel. "
                        "Activate/install a scimap-capable environment before running this cell."
                    )


            def image_path_for_fov(fov):
                fov_number = str(fov).replace("fov", "")
                path = BACKGROUND_DIR / f"{fov_number}.ome.tif"
                if not path.exists():
                    raise FileNotFoundError(path)
                return path


            RUN_GATE_FINDER = False
            GATE_FOV = "fov1"
            MARKER_OF_INTEREST = "MAGEA4"

            if RUN_GATE_FINDER:
                require_scimap()
                image_path = image_path_for_fov(GATE_FOV)
                sm.pl.gate_finder(
                    image_path=str(image_path),
                    adata=adata,
                    marker_of_interest=MARKER_OF_INTEREST,
                    imageid="fov",
                    subset=GATE_FOV,
                    from_gate=4,
                    to_gate=10,
                    increment=0.1,
                    point_size=6,
                )
            else:
                print("RUN_GATE_FINDER=False; set it to True when reviewing gates interactively.")
            """
        ),
        md(
            """
            ## 5. Rescale With Manual Gates

            This step is off by default because the current gate table is mostly
            blank. After filling the gate CSV, set `RUN_RESCALING = True`.
            """
        ),
        code(
            """
            RUN_RESCALING = False
            if RUN_RESCALING:
                require_scimap()
                manual_gate = pd.read_csv(MANUAL_GATES_CSV)
                gate_value_columns = [col for col in manual_gate.columns if col != "markers"]
                if manual_gate[gate_value_columns].isna().all(axis=None):
                    raise ValueError("Manual gate table has no thresholds yet.")

                adata_rescaled = sm.pp.rescale(
                    adata.copy(),
                    gate=manual_gate,
                    imageid="fov",
                )
                adata_rescaled.write_h5ad(RESCALED_H5AD, compression="gzip")
                print("Saved:", RESCALED_H5AD)
            else:
                adata_rescaled = ad.read_h5ad(RESCALED_H5AD) if RESCALED_H5AD.exists() else adata.copy()
                print("RUN_RESCALING=False; using existing rescaled h5ad if present, otherwise raw input.")
                print(adata_rescaled)
            """
        ),
        md(
            """
            ## 6. Load And Validate Phenotype Workflow
            """
        ),
        code(
            """
            phenotype = pd.read_csv(PHENOTYPE_WORKFLOW_CSV)
            marker_aliases = {"CLDN11_1": "CLDN11_2", "Vimentin_1": "Vimentin_2"}
            phenotype = phenotype.rename(columns=marker_aliases)

            metadata_cols = list(phenotype.columns[:2])
            workflow_marker_cols = list(phenotype.columns[2:])
            missing_workflow_markers = [marker for marker in workflow_marker_cols if marker not in adata.var_names]
            if missing_workflow_markers:
                raise ValueError(f"Phenotype workflow markers missing from AnnData: {missing_workflow_markers}")

            phenotype.to_csv(PHENOTYPE_WORKFLOW_CSV, index=False)
            print("Workflow marker columns:", workflow_marker_cols)
            phenotype.style.format(na_rep="")
            """
        ),
        md(
            """
            ## 7. Run scimap Phenotyping

            Set `RUN_PHENOTYPING = True` after the gate table has been filled and
            rescaling has been run or intentionally skipped.
            """
        ),
        code(
            """
            RUN_PHENOTYPING = False
            if RUN_PHENOTYPING:
                require_scimap()
                adata_pheno = sm.tl.phenotype_cells(
                    adata_rescaled.copy(),
                    phenotype=phenotype,
                    label="phenotype",
                )
                adata_pheno.write_h5ad(PHENOTYPED_H5AD, compression="gzip")
                adata_pheno.obs[["fov", "imageid", "phenotype"]].to_csv(PHENOTYPE_ASSIGNMENTS_CSV)
                print("Saved:", PHENOTYPED_H5AD)
                print("Saved:", PHENOTYPE_ASSIGNMENTS_CSV)
                print(adata_pheno.obs["phenotype"].value_counts())
            else:
                if PHENOTYPED_H5AD.exists():
                    adata_pheno = ad.read_h5ad(PHENOTYPED_H5AD)
                    print("Loaded existing phenotyped object:", PHENOTYPED_H5AD)
                else:
                    adata_pheno = None
                    print("RUN_PHENOTYPING=False and no phenotyped h5ad exists yet.")
            """
        ),
        md(
            """
            ## 8. Phenotype Heatmap
            """
        ),
        code(
            """
            HEATMAP_MARKERS = [
                "MAGEA4", "UTF1_2", "PIWIL4", "GFRA1", "DMRT1", "PCNA", "FMRP", "Acrosin",
                "CREM", "SOX9", "INSL3", "CYP17A1", "CYP11A1", "MYH11", "SMA", "CD31",
            ]
            HEATMAP_MARKERS = [marker for marker in HEATMAP_MARKERS if marker in adata.var_names]

            if adata_pheno is not None and "phenotype" in adata_pheno.obs.columns:
                matrixplot = sc.pl.matrixplot(
                    adata_pheno,
                    figsize=(8, 4),
                    var_names=HEATMAP_MARKERS,
                    groupby="phenotype",
                    dendrogram=True,
                    use_raw=False,
                    cmap="viridis",
                    standard_scale="var",
                    return_fig=True,
                )
                matrixplot.savefig(FIGURES_DIR / f"{SAMPLE_ID}_phenotype_matrixplot.pdf", dpi=300, bbox_inches="tight")
            else:
                print("No phenotype labels available yet.")
            """
        ),
        md(
            """
            ## 9. Image Overlay Review
            """
        ),
        code(
            """
            RUN_IMAGE_VIEWER = False
            VIEW_FOV = "fov1"

            if RUN_IMAGE_VIEWER:
                require_scimap()
                if adata_pheno is None:
                    raise ValueError("Run phenotyping before opening the phenotype overlay viewer.")
                sm.pl.image_viewer(
                    image_path=str(image_path_for_fov(VIEW_FOV)),
                    adata=adata_pheno,
                    imageid="fov",
                    subset=VIEW_FOV,
                    overlay="phenotype",
                    point_size=10,
                    point_color="white",
                )
            else:
                print("RUN_IMAGE_VIEWER=False; set it to True for Napari phenotype review.")
            """
        ),
        md(
            """
            ## 10. Static Spatial Scatter
            """
        ),
        code(
            """
            def spatial_scatter_plot(
                adata,
                color_by,
                x_coordinate="X_centroid",
                y_coordinate="Y_centroid",
                subset=None,
                imageid="fov",
                s=0.1,
                figsize=(10, 10),
                dpi=300,
                save_path=None,
            ):
                if subset is not None:
                    subset_values = [subset] if isinstance(subset, str) else list(subset)
                    bdata = adata[adata.obs[imageid].isin(subset_values)].copy()
                else:
                    bdata = adata.copy()

                color_values = bdata.obs[color_by].astype(str)
                categories = pd.Index(color_values.unique()).sort_values()
                cmap = plt.get_cmap("tab20")
                colors = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                for cat in categories:
                    mask = color_values.eq(cat)
                    ax.scatter(
                        bdata.obs.loc[mask, x_coordinate],
                        bdata.obs.loc[mask, y_coordinate],
                        s=s,
                        c=[colors[cat]],
                        linewidths=0,
                        label=cat,
                    )
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(color_by)
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=8)
                fig.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                return fig


            if adata_pheno is not None and "phenotype" in adata_pheno.obs.columns:
                fig = spatial_scatter_plot(
                    adata_pheno,
                    color_by="phenotype",
                    imageid="fov",
                    s=0.1,
                    save_path=FIGURES_DIR / f"{SAMPLE_ID}_phenotype_spatial_scatter.pdf",
                )
                fig
            else:
                print("No phenotype labels available yet.")
            """
        ),
        md(
            """
            ## Outputs

            Primary manual-gating outputs:

            - `phenotyping/manual_gating/adata/ST_TMAG_E1_95_adata_scimap_rescaled.h5ad`
            - `phenotyping/manual_gating/adata/ST_TMAG_E1_95_adata_scimap_phenotyped.h5ad`
            - `phenotyping/manual_gating/tables/ST_TMAG_E1_95_phenotype_assignments.csv`
            """
        ),
    ]


def write_rewritten_notebooks(paths: dict[str, Path]) -> list[Path]:
    notebook_dir = paths["notebooks"]
    notebook_paths = [
        notebook_dir / f"00_{SAMPLE_ID}_adata_creation.ipynb",
        notebook_dir / f"00_{SAMPLE_ID}_qc.ipynb",
        notebook_dir / f"01_{SAMPLE_ID}_manual_gating_phenotyping.ipynb",
    ]
    builders = [adata_creation_notebook, qc_notebook, manual_gating_notebook]
    for path, builder in zip(notebook_paths, builders):
        write_notebook(path, builder())
    return notebook_paths


def validate_notebooks(paths: list[Path]) -> None:
    for path in paths:
        nb = nbf.read(path, as_version=4)
        nbf.validate(nb)


def organize(project_dir: Path = PROJECT_DIR, skip_adata: bool = False) -> None:
    paths = ensure_layout(project_dir)
    marker_path = write_marker_manifest(paths)
    copy_legacy_inputs(paths, project_dir)
    output_h5ad = None if skip_adata else build_adata(paths, marker_path, project_dir)
    notebook_paths = write_rewritten_notebooks(paths)
    validate_notebooks(notebook_paths)

    print("Organized project:", project_dir)
    print("Marker manifest:", marker_path)
    if output_h5ad is not None:
        print("AnnData:", output_h5ad)
    print("Notebooks:")
    for path in notebook_paths:
        print(" -", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--skip-adata", action="store_true")
    args = parser.parse_args()
    organize(args.project_dir, skip_adata=args.skip_adata)


if __name__ == "__main__":
    main()
