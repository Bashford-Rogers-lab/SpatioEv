"""Write the ST_TMAG_E1_31 FOV-aware clustering phenotyping notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


SAMPLE_ID = "ST_TMAG_E1_31"
PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_31")
NOTEBOOK_PATH = PROJECT_DIR  / "paper" / "notebooks" / f"01_{SAMPLE_ID}_clustering_based_phenotyping.ipynb"
SPATIOEV_REPO = Path("/Users/shihongwu/SpatioEv")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            f"""
            # {SAMPLE_ID}: FOV-Aware Clustering-Based Phenotyping

            This notebook follows the same overall workflow as
            `01_ST_Pat4_51_clustering_based_phenotyping.ipynb`, adapted for the
            ST_TMAG microarray/TMA sample.

            Main adaptation: each `fov` is treated as an independent clustering
            unit, analogous to the manually drawn ROI units in ST_Pat4_51. This
            avoids mixing unrelated tissue cores whose pixel coordinates are
            local to each FOV.

            This copy also exposes `SELECTED_FOVS` near the top of the notebook
            so you can choose which microarray cores enter the analysis.

            Main steps:

            1. Load the QC-filtered AnnData and validate the marker order.
            2. Treat `imageid`/`fov` as the independent clustering unit.
            3. Z-score marker expression and run level-0 Leiden clustering
               independently inside each FOV.
            4. Write cluster review tables for manual phenotype naming.
            5. Apply reviewed names and save phenotyping assignments.
            6. Optionally run FOV-aware level-1 refinements after review.
            """
        ),
        md(
            """
            ## Runtime Setup

            Cache directories are set before importing Scanpy/Numba. This avoids
            cache-location errors when Jupyter is launched from a restricted
            environment.
            """
        ),
        code(
            f"""
            import os
            import sys
            from pathlib import Path
            from dataclasses import asdict
            import warnings

            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
            os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

            warnings.filterwarnings("ignore", category=FutureWarning)

            import anndata as ad
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import scanpy as sc
            from IPython.display import display

            SPATIOEV_REPO = Path("{SPATIOEV_REPO}")
            if SPATIOEV_REPO.exists() and str(SPATIOEV_REPO) not in sys.path:
                sys.path.insert(0, str(SPATIOEV_REPO))

            import spatioev as sv
            from spatioev.config import ClusteringConfig

            try:
                import scimap as sm
            except Exception:
                sm = None

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)

            sc.settings.verbosity = 2
            sc.settings.set_figure_params(dpi=120, facecolor="white", frameon=False)

            print(f"Scanpy: {{sc.__version__}}")
            print(f"SpatioEv: {{getattr(sv, '__version__', 'unknown')}}")
            print("scimap available:", sm is not None)
            """
        ),
        md(
            """
            ## Paths And Analysis Options

            By default this notebook uses the un-QCed/raw AnnData from the
            creation notebook. Set `USE_QC_FILTERED_INPUT = True` if you want to
            cluster the segmentation-QC-filtered object instead.
            """
        ),
        code(
            f"""
            SAMPLE_ID = "{SAMPLE_ID}"
            PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_31")

            QC_FILTERED_H5AD = PROJECT_DIR / "qc" / "segmentation_qc" / "adata" / f"{{SAMPLE_ID}}_adata_segmentation_qc_filtered.h5ad"
            RAW_H5AD = PROJECT_DIR / "data" / f"{{SAMPLE_ID}}_adata.h5ad"
            MARKERS_CSV = PROJECT_DIR / "background" / f"{{SAMPLE_ID}}_markers.csv"
            BACKGROUND_DIR = PROJECT_DIR / "background"

            PHENO_DIR = PROJECT_DIR / "phenotyping" / "clustering_based"
            FIGURES_DIR = PHENO_DIR / "figures"
            TABLES_DIR = PHENO_DIR / "tables"
            ADATA_DIR = PHENO_DIR / "adata"

            SAVE_OUTPUTS = True
            USE_QC_FILTERED_INPUT = False

            SELECTED_FOVS = ["fov1", "fov6", "fov10", "fov12", "fov15", "fov16", "fov17"]
            STRICT_SELECTED_FOVS = True

            FOV_KEY = "imageid"
            FOV_LABEL_KEY = "fov"

            for folder in [PHENO_DIR, FIGURES_DIR, TABLES_DIR, ADATA_DIR]:
                folder.mkdir(parents=True, exist_ok=True)

            INPUT_H5AD = QC_FILTERED_H5AD if USE_QC_FILTERED_INPUT else RAW_H5AD
            if not INPUT_H5AD.exists():
                raise FileNotFoundError(
                    "Could not find the selected AnnData file. "
                    f"USE_QC_FILTERED_INPUT={{USE_QC_FILTERED_INPUT}}; checked: {{INPUT_H5AD}}"
                )
            if not MARKERS_CSV.exists():
                raise FileNotFoundError(MARKERS_CSV)

            print(f"Input AnnData: {{INPUT_H5AD}}")
            print("Using QC-filtered input:", USE_QC_FILTERED_INPUT)
            print(f"Marker table: {{MARKERS_CSV}}")
            print(f"Background/dearray folder: {{BACKGROUND_DIR.resolve()}}")
            print(f"Output folder: {{PHENO_DIR}}")
            """
        ),
        md(
            """
            ## Load AnnData

            The AnnData input is controlled by `USE_QC_FILTERED_INPUT`.
            Coordinates remain FOV-local pixel coordinates, so all clustering
            and static spatial review is grouped by FOV. If `SELECTED_FOVS` is
            set, the notebook subsets to those FOVs before marker validation
            and clustering.
            """
        ),
        code(
            """
            adata = ad.read_h5ad(INPUT_H5AD)

            required_obs = ["X_centroid", "Y_centroid", FOV_KEY, FOV_LABEL_KEY]
            missing_obs = [col for col in required_obs if col not in adata.obs.columns]
            if missing_obs:
                raise ValueError(f"Missing required obs columns: {missing_obs}")

            adata.obs[FOV_KEY] = adata.obs[FOV_KEY].astype(str)
            adata.obs[FOV_LABEL_KEY] = adata.obs[FOV_LABEL_KEY].astype(str)
            adata.obs["clustering_unit"] = adata.obs[FOV_KEY].astype(str)

            def fov_sort_key(value):
                text = str(value)
                suffix = text.replace("fov", "")
                return int(suffix) if suffix.isdigit() else text

            def normalize_fov_label(value):
                text = str(value).strip()
                if not text:
                    return text
                text = text.lower().replace(" ", "")
                return text if text.startswith("fov") else f"fov{text}"

            available_fovs = sorted(adata.obs[FOV_KEY].astype(str).unique(), key=fov_sort_key)
            print("Available FOVs in input:", available_fovs)

            if SELECTED_FOVS:
                requested_fovs = list(dict.fromkeys(normalize_fov_label(fov) for fov in SELECTED_FOVS))
                missing_selected_fovs = [fov for fov in requested_fovs if fov not in available_fovs]
                selected_fovs_available = [fov for fov in requested_fovs if fov in available_fovs]

                if missing_selected_fovs:
                    message = f"Requested FOVs not present in {INPUT_H5AD.name}: {missing_selected_fovs}"
                    if STRICT_SELECTED_FOVS:
                        raise ValueError(message)
                    warnings.warn(message)

                if not selected_fovs_available:
                    raise ValueError("None of SELECTED_FOVS are present in the input AnnData.")

                adata = adata[adata.obs[FOV_KEY].astype(str).isin(selected_fovs_available)].copy()
                fov_order = selected_fovs_available
                print("Using selected FOVs:", fov_order)
            else:
                requested_fovs = []
                missing_selected_fovs = []
                fov_order = available_fovs
                print("SELECTED_FOVS is empty/None; using all available FOVs.")

            selection_table = adata.obs[FOV_KEY].value_counts().reindex(fov_order).rename("n_cells").to_frame()
            selection_table["included"] = True
            if SELECTED_FOVS and missing_selected_fovs:
                missing_table = pd.DataFrame({"n_cells": 0, "included": False}, index=missing_selected_fovs)
                selection_table = pd.concat([selection_table, missing_table], axis=0)

            summary = pd.Series(
                {
                    "n_cells": adata.n_obs,
                    "n_markers": adata.n_vars,
                    "n_fovs": len(fov_order),
                    "layers": ", ".join(adata.layers.keys()) if adata.layers else "none",
                    "x_min": float(adata.obs["X_centroid"].min()),
                    "x_max": float(adata.obs["X_centroid"].max()),
                    "y_min": float(adata.obs["Y_centroid"].min()),
                    "y_max": float(adata.obs["Y_centroid"].max()),
                },
                name="value",
            )

            display(summary.to_frame())
            display(selection_table)
            adata
            """
        ),
        md(
            """
            ## Validate Marker Order

            `ST_TMAG_E1_31_markers.csv` is the channel-order source of truth for
            this sample. The AnnData variables are reordered to match it before
            clustering.
            """
        ),
        code(
            """
            markers_df = pd.read_csv(MARKERS_CSV)
            marker_col = "marker_name" if "marker_name" in markers_df.columns else markers_df.columns[0]
            ordered_markers = markers_df[marker_col].astype(str).tolist()

            missing_markers = [marker for marker in ordered_markers if marker not in adata.var_names]
            if missing_markers:
                raise ValueError(f"Markers in the marker table are missing from AnnData: {missing_markers}")

            adata = adata[:, ordered_markers].copy()
            print("Marker order validated:")
            print(", ".join(adata.var_names))
            """
        ),
        md(
            """
            ## Markers Used For Clustering

            DNA channels are excluded from clustering by default. The remaining
            marker order follows `ST_TMAG_E1_31_markers.csv`.
            """
        ),
        code(
            """
            DNA_MARKERS = ["DNA_1"]

            def keep_present(markers):
                return [m for m in markers if m in adata.var_names]

            LEVEL0_MARKERS = [m for m in ordered_markers if m not in DNA_MARKERS]

            marker_check = pd.DataFrame(
                {
                    "order": np.arange(len(ordered_markers)),
                    "marker": ordered_markers,
                    "used_for_level0": [m in LEVEL0_MARKERS for m in ordered_markers],
                }
            )

            display(marker_check)
            print(f"Level-0 clustering markers ({len(LEVEL0_MARKERS)}):")
            print(", ".join(LEVEL0_MARKERS))
            """
        ),
        md(
            """
            ## FOV Units Before Clustering

            In ST_Pat4_51 the notebook draws ROIs before clustering. In ST_TMAG,
            each microarray core/FOV is already a separate analysis unit. The
            notebook therefore uses `imageid` as `clustering_unit`.
            """
        ),
        code(
            """
            UNIT_KEY = "clustering_unit"
            adata.obs[UNIT_KEY] = adata.obs[FOV_KEY].astype(str)

            unit_counts = adata.obs[UNIT_KEY].value_counts().reindex(fov_order).rename("n_cells").to_frame()
            display(unit_counts)

            UNIT_ASSIGNMENTS_CSV = TABLES_DIR / f"{SAMPLE_ID}_fov_clustering_units.csv"
            if SAVE_OUTPUTS:
                adata.obs[[UNIT_KEY, FOV_KEY, FOV_LABEL_KEY, "X_centroid", "Y_centroid"]].to_csv(UNIT_ASSIGNMENTS_CSV)
                print(f"FOV clustering-unit table available at: {UNIT_ASSIGNMENTS_CSV}")
            """
        ),
        md(
            """
            ## FOV Preview

            This static view checks each FOV independently. A single combined
            scatter would overlap the cores because ST_TMAG coordinates are
            FOV-local.
            """
        ),
        code(
            """
            def plot_fov_spatial_grid(
                adata_obj,
                color_key,
                fov_key=FOV_KEY,
                x_key="X_centroid",
                y_key="Y_centroid",
                fovs=None,
                ncols=5,
                point_size=0.12,
                alpha=0.75,
                figsize_per_panel=(3.0, 3.0),
                categorical=True,
                title=None,
                show_legend=False,
                legend_title=None,
                save_path=None,
            ):
                if fovs is None:
                    fovs = sorted(adata_obj.obs[fov_key].astype(str).unique(), key=fov_sort_key)
                else:
                    fovs = [str(fov) for fov in fovs]

                nrows = int(np.ceil(len(fovs) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                    squeeze=False,
                )

                if categorical:
                    color_series = adata_obj.obs[color_key]
                    observed_categories = set(color_series.astype(str).dropna().unique())
                    if isinstance(color_series.dtype, pd.CategoricalDtype):
                        categories = [
                            str(cat)
                            for cat in color_series.cat.categories
                            if str(cat) in observed_categories
                        ]
                    else:
                        categories = sorted(observed_categories)
                    cmap = plt.get_cmap("tab20", max(len(categories), 1))
                    color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

                for ax, fov in zip(axes.ravel(), fovs):
                    df = adata_obj.obs.loc[adata_obj.obs[fov_key].astype(str).eq(fov)].copy()
                    if df.empty:
                        ax.axis("off")
                        continue

                    if categorical:
                        values = df[color_key].astype(str)
                        for cat in sorted(values.unique()):
                            mask = values.eq(cat)
                            ax.scatter(
                                df.loc[mask, x_key],
                                df.loc[mask, y_key],
                                s=point_size,
                                alpha=alpha,
                                linewidths=0,
                                color=color_map.get(cat),
                                label=cat,
                            )
                    else:
                        sca = ax.scatter(
                            df[x_key],
                            df[y_key],
                            s=point_size,
                            alpha=alpha,
                            linewidths=0,
                            c=df[color_key],
                            cmap="viridis",
                        )
                        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)

                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.set_title(f"{fov} ({len(df):,})")
                    ax.set_xticks([])
                    ax.set_yticks([])

                for ax in axes.ravel()[len(fovs):]:
                    ax.axis("off")

                if title:
                    fig.suptitle(title, y=1.01)
                if categorical and show_legend and categories:
                    from matplotlib.lines import Line2D

                    legend_handles = [
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            linestyle="",
                            color="none",
                            markerfacecolor=color_map[cat],
                            markeredgewidth=0,
                            markersize=7,
                            label=cat,
                        )
                        for cat in categories
                    ]
                    fig.legend(
                        handles=legend_handles,
                        title=legend_title or color_key,
                        loc="center left",
                        bbox_to_anchor=(1.01, 0.5),
                        frameon=False,
                    )
                    fig.tight_layout(rect=[0, 0, 0.88, 1])
                else:
                    fig.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=220, bbox_inches="tight")
                return fig


            fig = plot_fov_spatial_grid(
                adata,
                color_key=UNIT_KEY,
                point_size=0.08,
                alpha=0.75,
                title=f"{SAMPLE_ID}: FOV clustering units",
                save_path=FIGURES_DIR / f"{SAMPLE_ID}_fov_units_preview.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Level-0 Clustering Configuration

            Resolution controls how many clusters are produced inside each FOV.
            If the review table shows clear over-splitting, lower `resolution`;
            if broad mixed clusters remain, increase it.
            """
        ),
        code(
            """
            def make_clustering_config(markers, resolution=0.4, n_neighbors=15, n_pcs=20):
                markers = keep_present(markers)
                if len(markers) < 2:
                    raise ValueError("At least two markers are required for clustering.")
                n_pcs = max(1, min(n_pcs, len(markers) - 1))
                return ClusteringConfig(
                    markers=markers,
                    resolution=resolution,
                    n_neighbors=n_neighbors,
                    n_pcs=n_pcs,
                    scale=True,
                )

            def adapt_config_for_subset(config, n_obs):
                if n_obs < 3:
                    raise ValueError("At least 3 cells are required for clustering.")
                max_pcs = max(1, min(len(config.markers) - 1, n_obs - 1))
                n_pcs = max(1, min(config.n_pcs, max_pcs))
                n_neighbors = max(2, min(config.n_neighbors, n_obs - 1))
                return ClusteringConfig(
                    markers=list(config.markers),
                    resolution=config.resolution,
                    n_neighbors=n_neighbors,
                    n_pcs=n_pcs,
                    scale=config.scale,
                )

            level0_config = make_clustering_config(
                LEVEL0_MARKERS,
                resolution=0.35,
                n_neighbors=15,
                n_pcs=20,
            )

            pd.Series(asdict(level0_config), name="level0_config").to_frame()
            """
        ),
        md(
            """
            ## Run Level-0 Clustering By FOV

            SpatioEv's clustering helper runs PCA, nearest neighbors, UMAP, and
            Leiden clustering on the marker subset. Here it is run separately for
            each FOV/core, so `cluster_level0` labels are prefixed by FOV.
            """
        ),
        code(
            """
            MIN_CELLS_PER_FOV_FOR_CLUSTERING = 50

            adata_work = adata.copy()
            print(f"Analyzing all included cells: {adata_work.n_obs:,}")

            def run_level0_clustering_by_fov(adata_in, config):
                results = {}
                summaries = []
                fovs = sorted(adata_in.obs[FOV_KEY].astype(str).dropna().unique(), key=fov_sort_key)

                for fov in fovs:
                    sub = adata_in[adata_in.obs[FOV_KEY].astype(str) == fov].copy()
                    n_cells = sub.n_obs
                    if n_cells < MIN_CELLS_PER_FOV_FOR_CLUSTERING:
                        summaries.append({"fov": fov, "n_cells": n_cells, "status": "skipped_too_few_cells"})
                        continue

                    sub = sv.zscore_normalize(sub)
                    fov_config = adapt_config_for_subset(config, sub.n_obs)
                    clustered = sv.cluster_cells(sub, fov_config)
                    clustered.obs[UNIT_KEY] = fov
                    clustered.obs["cluster_level0_local"] = clustered.obs["leiden"].astype(str)
                    clustered.obs["cluster_level0"] = (
                        clustered.obs[UNIT_KEY].astype(str) + "::" + clustered.obs["cluster_level0_local"].astype(str)
                    ).astype("category")

                    results[fov] = clustered
                    summaries.append(
                        {
                            "fov": fov,
                            "n_cells": n_cells,
                            "status": "clustered",
                            "n_clusters": clustered.obs["cluster_level0"].nunique(),
                            "n_neighbors": fov_config.n_neighbors,
                            "n_pcs": fov_config.n_pcs,
                            "resolution": fov_config.resolution,
                        }
                    )

                if not results:
                    raise RuntimeError(
                        "No FOV had enough cells to cluster. Lower MIN_CELLS_PER_FOV_FOR_CLUSTERING "
                        "or check the input AnnData."
                    )

                combined = ad.concat(results.values(), axis=0, join="outer", merge="same", index_unique=None)
                combined.obs["cluster_level0"] = combined.obs["cluster_level0"].astype("category")
                combined.obs["cluster_level0_local"] = combined.obs["cluster_level0_local"].astype("category")
                combined.obs[UNIT_KEY] = combined.obs[UNIT_KEY].astype("category")
                return combined, results, pd.DataFrame(summaries)

            adata_lvl0, level0_results, level0_fov_summary = run_level0_clustering_by_fov(adata_work, level0_config)

            cluster_counts = adata_lvl0.obs["cluster_level0"].value_counts().sort_index()
            display(level0_fov_summary)
            display(cluster_counts.rename("n_cells").to_frame())
            print(f"Level-0 clusters across clustered FOVs: {cluster_counts.shape[0]}")
            """
        ),
        md(
            """
            ## Save Level-0 Clustering Object

            This object contains FOV-prefixed Leiden labels, per-FOV local Leiden
            labels, and the UMAP coordinates from each FOV-specific clustering
            run.
            """
        ),
        code(
            """
            LEVEL0_H5AD = ADATA_DIR / f"{SAMPLE_ID}_level0_clusters_by_fov.h5ad"
            LEVEL0_SUMMARY_CSV = TABLES_DIR / f"{SAMPLE_ID}_level0_fov_clustering_summary.csv"

            if SAVE_OUTPUTS:
                adata_lvl0.write_h5ad(LEVEL0_H5AD, compression="gzip")
                level0_fov_summary.to_csv(LEVEL0_SUMMARY_CSV, index=False)
                print(f"Saved: {LEVEL0_H5AD}")
                print(f"Saved: {LEVEL0_SUMMARY_CSV}")
            else:
                print("SAVE_OUTPUTS=False; not writing level-0 outputs.")
            """
        ),
        md(
            """
            ## Level-0 UMAPs

            UMAP coordinates are generated separately per FOV, so they should be
            interpreted within each FOV rather than as one global embedding.
            """
        ),
        code(
            """
            for fov, sub in level0_results.items():
                fig = sc.pl.umap(
                    sub,
                    color=["cluster_level0_local"],
                    legend_loc="on data",
                    frameon=False,
                    title=f"{fov}: level-0 local clusters",
                    show=False,
                    return_fig=True,
                )
                if SAVE_OUTPUTS:
                    fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_{fov}_level0_umap_clusters.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)
            """
        ),
        md(
            """
            ## Inspect Level-0 Clusters Spatially

            This static grid is safe to run anywhere and helps catch FOV-specific
            artifacts, edge effects, or clusters driven by segmentation/image
            quality rather than biology.
            """
        ),
        code(
            """
            fig = plot_fov_spatial_grid(
                adata_lvl0,
                color_key="cluster_level0_local",
                point_size=0.12,
                alpha=0.8,
                title=f"{SAMPLE_ID}: level-0 clusters within each FOV",
                save_path=FIGURES_DIR / f"{SAMPLE_ID}_level0_spatial_clusters_by_fov.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Optional Image Overlay Review

            Set `RUN_CLUSTER_VIEWER = True` to inspect one FOV at a time on the
            corresponding `background/<fov_number>.ome.tif` image. This requires
            scimap/Napari in the active kernel.
            """
        ),
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

            RUN_CLUSTER_VIEWER = False
            VIEW_FOV = fov_order[0] if fov_order else "fov1"

            if RUN_CLUSTER_VIEWER:
                require_scimap()
                view_adata = adata_lvl0[adata_lvl0.obs[FOV_KEY].astype(str) == VIEW_FOV].copy()
                viewer_channel_names = list(adata.var_names)
                view_adata.uns["all_markers"] = viewer_channel_names
                sm.pl.image_viewer(
                    image_path=str(image_path_for_fov(VIEW_FOV)),
                    adata=view_adata,
                    channel_names=viewer_channel_names,
                    imageid=FOV_KEY,
                    subset=VIEW_FOV,
                    overlay="cluster_level0_local",
                    point_size=8,
                    point_color="white",
                )
            else:
                print("RUN_CLUSTER_VIEWER=False; set it to True for per-FOV Napari review.")
            """
        ),
        md(
            """
            ## Annotate Level-0 Clusters

            The draft review table has one row per FOV-prefixed cluster. Edit the
            `annotation` column in the reviewed CSV, or set
            `ANNOTATE_LEVEL0_INTERACTIVELY = True` to prompt inside the notebook.
            """
        ),
        code(
            """
            ANNOTATE_LEVEL0_INTERACTIVELY = False

            DRAFT_REVIEW_CSV = TABLES_DIR / f"{SAMPLE_ID}_level0_cluster_annotation_review.csv"
            REVIEWED_ANNOTATION_CSV = TABLES_DIR / f"{SAMPLE_ID}_level0_cluster_annotation_reviewed.csv"

            def _cluster_sort_key(value):
                text = str(value)
                if "::" in text:
                    fov, cluster = text.rsplit("::", 1)
                    cluster_key = int(cluster) if cluster.isdigit() else cluster
                    return (fov_sort_key(fov), cluster_key)
                return (9999, text)

            def make_annotation_table_from_clusters(adata_obj, cluster_key):
                table = (
                    adata_obj.obs[cluster_key]
                    .astype(str)
                    .value_counts()
                    .rename("n_cells")
                    .to_frame()
                )
                table = table.reindex(sorted(table.index, key=_cluster_sort_key))
                table["fov"] = [idx.rsplit("::", 1)[0] if "::" in idx else "" for idx in table.index]
                table["local_cluster"] = [idx.rsplit("::", 1)[1] if "::" in idx else idx for idx in table.index]
                table["annotation"] = table.index.astype(str)
                table["review_note"] = ""
                table.index.name = "cluster"
                return table

            def annotate_clusters_interactively(
                adata_obj,
                review_table=None,
                cluster_key="cluster_level0",
                new_key="annotation_level0",
            ):
                if review_table is None or review_table.empty:
                    review = make_annotation_table_from_clusters(adata_obj, cluster_key)
                else:
                    review = review_table.copy()
                    review.index = review.index.astype(str)

                clusters = sorted(adata_obj.obs[cluster_key].astype(str).unique(), key=_cluster_sort_key)
                mapping = {}

                print("Enter one annotation per cluster. Press Enter to keep the current value.\\n")
                for cluster in clusters:
                    row = review.loc[cluster] if cluster in review.index else pd.Series(dtype=object)
                    current = str(row.get("annotation", cluster))
                    n_cells = row.get("n_cells", int((adata_obj.obs[cluster_key].astype(str) == cluster).sum()))
                    label = input(f"Cluster {cluster} | n={n_cells} | current={current}: ").strip()
                    mapping[cluster] = label if label else current

                annotated = adata_obj.copy()
                annotated.obs[new_key] = (
                    annotated.obs[cluster_key]
                    .astype(str)
                    .map(mapping)
                    .fillna("unassigned")
                    .astype("category")
                )

                updated_table = review.copy()
                mapped_annotations = pd.Series(updated_table.index, index=updated_table.index).map(mapping)
                updated_table["annotation"] = mapped_annotations.fillna(updated_table["annotation"].astype(str))
                updated_table["review_note"] = updated_table.get("review_note", "")
                return annotated, mapping, updated_table

            cluster_annotation_table = make_annotation_table_from_clusters(adata_lvl0, "cluster_level0")

            if ANNOTATE_LEVEL0_INTERACTIVELY:
                adata_lvl0, interactive_mapping, cluster_annotation_table = annotate_clusters_interactively(
                    adata_lvl0,
                    cluster_annotation_table,
                    cluster_key="cluster_level0",
                    new_key="annotation_level0",
                )

                if SAVE_OUTPUTS:
                    cluster_annotation_table.to_csv(REVIEWED_ANNOTATION_CSV)
                    print(f"Saved interactive annotations: {REVIEWED_ANNOTATION_CSV}")
            else:
                print("Interactive annotation skipped.")
                if SAVE_OUTPUTS:
                    cluster_annotation_table.to_csv(DRAFT_REVIEW_CSV)
                    print(f"Saved draft cluster review table: {DRAFT_REVIEW_CSV}")

            display(cluster_annotation_table)
            """
        ),
        md(
            """
            ## Apply Reviewed Or Draft Level-0 Annotations

            If `ST_TMAG_E1_31_level0_cluster_annotation_reviewed.csv` exists, the
            notebook uses its `annotation` column. Otherwise it uses the draft
            table created above, where each annotation defaults to the cluster ID.
            """
        ),
        code(
            """
            annotation_source = None

            if REVIEWED_ANNOTATION_CSV.exists():
                annotation_table_path = REVIEWED_ANNOTATION_CSV
                annotation_table = pd.read_csv(annotation_table_path, dtype={"cluster": str})
                annotation_source = annotation_table_path
            elif DRAFT_REVIEW_CSV.exists():
                annotation_table_path = DRAFT_REVIEW_CSV
                annotation_table = pd.read_csv(annotation_table_path, dtype={"cluster": str})
                annotation_source = annotation_table_path
            else:
                annotation_table = cluster_annotation_table.reset_index().copy()
                annotation_source = "in-memory manual review table"

            if "cluster" not in annotation_table.columns:
                annotation_table = annotation_table.rename(columns={annotation_table.columns[0]: "cluster"})
            if "annotation" not in annotation_table.columns:
                raise ValueError("Expected an 'annotation' column in the cluster annotation table.")

            cluster_to_annotation = dict(
                zip(annotation_table["cluster"].astype(str), annotation_table["annotation"].astype(str))
            )

            adata_lvl0.obs["annotation_level0"] = (
                adata_lvl0.obs["cluster_level0"]
                .astype(str)
                .map(cluster_to_annotation)
                .fillna("unassigned")
                .astype("category")
            )

            print(f"Loaded annotations from: {annotation_source}")
            display(adata_lvl0.obs["annotation_level0"].value_counts().rename("n_cells").to_frame())
            """
        ),
        md(
            """
            ## View Annotated Level-0 Results
            """
        ),
        code(
            """
            fig = plot_fov_spatial_grid(
                adata_lvl0,
                color_key="annotation_level0",
                point_size=0.12,
                alpha=0.8,
                title=f"{SAMPLE_ID}: level-0 annotations within each FOV",
                save_path=FIGURES_DIR / f"{SAMPLE_ID}_level0_spatial_annotations_by_fov.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Save Phenotype Assignments

            This writes labels back onto a copy of the QC-filtered AnnData. FOVs
            skipped during clustering, if any, are marked as `not_clustered`.
            """
        ),
        code(
            """
            adata_pheno = adata.copy()
            adata_pheno.obs["cluster_level0"] = "not_clustered"
            adata_pheno.obs["annotation_level0"] = "not_clustered"

            adata_pheno.obs.loc[adata_lvl0.obs_names, "cluster_level0"] = adata_lvl0.obs["cluster_level0"].astype(str)
            adata_pheno.obs.loc[adata_lvl0.obs_names, "annotation_level0"] = adata_lvl0.obs["annotation_level0"].astype(str)

            adata_pheno.obs["cluster_level0"] = adata_pheno.obs["cluster_level0"].astype("category")
            adata_pheno.obs["annotation_level0"] = adata_pheno.obs["annotation_level0"].astype("category")

            PHENOTYPED_H5AD = ADATA_DIR / f"{SAMPLE_ID}_phenotyping_level0_by_fov.h5ad"
            ASSIGNMENTS_CSV = TABLES_DIR / f"{SAMPLE_ID}_level0_phenotype_assignments_by_fov.csv"

            assignment_cols = [
                FOV_KEY,
                FOV_LABEL_KEY,
                UNIT_KEY,
                "X_centroid",
                "Y_centroid",
                "cluster_level0",
                "annotation_level0",
            ]
            assignment_cols = [col for col in assignment_cols if col in adata_pheno.obs.columns]

            if SAVE_OUTPUTS:
                adata_pheno.write_h5ad(PHENOTYPED_H5AD, compression="gzip")
                adata_pheno.obs[assignment_cols].to_csv(ASSIGNMENTS_CSV)
                print(f"Saved: {PHENOTYPED_H5AD}")
                print(f"Saved: {ASSIGNMENTS_CSV}")

            display(adata_pheno.obs["annotation_level0"].value_counts().rename("n_cells").to_frame())
            """
        ),
        md(
            """
            ## Level-1 Clustering Configuration

            Level 1 refines selected level-0 annotations or cluster IDs. By
            default it is disabled until level-0 labels have been reviewed.
            When enabled, each selected target is reclustered independently inside
            each FOV.
            """
        ),
        code(
            """
            RUN_LEVEL1_CLUSTERING = False

            LEVEL1_SOURCE_KEY = "annotation_level0"
            LEVEL1_TARGETS = []  # Example after review: ["Spermatogonia", "spermatocytes"]

            MIN_CELLS_PER_LEVEL1_TARGET = 50
            LEVEL1_FOV_KEY = FOV_KEY

            LEVEL1_MARKERS_BY_TARGET = {
                "Spermatogonia": ["MAGEA4", "UTF1_2", "PIWIL4", "GFRA1", "DMRT1", "PCNA"],
                "spermatocytes": ["CREM", "SYCP3", "DMRT1", "PIWIL4"],
                "spermatids": ["Acrosin", "CREM", "SYCP3"],
                "sertoli cells": ["SOX9", "AMH", "CLDN11_2", "CX43_2"],
                "leydig cells": ["INSL3", "CYP17A1", "CYP11A1", "STAR"],
                "myoid cells": ["MYH11", "SMA", "Vimentin_2"],
                "immune": ["CD45", "CD68", "CD8"],
            }

            LEVEL1_DEFAULT_MARKERS = LEVEL0_MARKERS
            LEVEL1_RESOLUTION_BY_TARGET = {}

            if LEVEL1_SOURCE_KEY not in adata_pheno.obs.columns:
                raise ValueError(f"{LEVEL1_SOURCE_KEY!r} is not in adata_pheno.obs")

            available_level1_targets = (
                adata_pheno.obs[LEVEL1_SOURCE_KEY]
                .astype(str)
                .value_counts()
                .rename("n_cells")
                .to_frame()
            )
            display(available_level1_targets)

            if not LEVEL1_TARGETS:
                print("LEVEL1_TARGETS is empty. Fill it, then set RUN_LEVEL1_CLUSTERING=True.")
            """
        ),
        md(
            """
            ## Run Level-1 Clustering
            """
        ),
        code(
            """
            def markers_for_level1_target(target):
                if target in LEVEL1_MARKERS_BY_TARGET:
                    return keep_present(LEVEL1_MARKERS_BY_TARGET[target])
                return keep_present(LEVEL1_DEFAULT_MARKERS)

            def make_level1_group_label(target, fov):
                return f"{fov}::{target}"

            def run_level1_clustering(parent_adata, source_key, targets):
                results = {}
                summaries = []

                for target in targets:
                    target_data = parent_adata[parent_adata.obs[source_key].astype(str) == str(target)].copy()
                    n_target_cells = target_data.n_obs
                    if n_target_cells == 0:
                        summaries.append({"target": target, "fov": None, "n_cells": 0, "status": "skipped_missing_target"})
                        continue

                    fovs = sorted(target_data.obs[LEVEL1_FOV_KEY].astype(str).dropna().unique(), key=fov_sort_key)

                    for fov in fovs:
                        sub = target_data[target_data.obs[LEVEL1_FOV_KEY].astype(str) == str(fov)].copy()
                        n_cells = sub.n_obs
                        if n_cells < MIN_CELLS_PER_LEVEL1_TARGET:
                            summaries.append({"target": target, "fov": fov, "n_cells": n_cells, "status": "skipped_too_few_cells"})
                            continue

                        markers = markers_for_level1_target(str(target))
                        resolution = LEVEL1_RESOLUTION_BY_TARGET.get(str(target), 0.5)
                        base_config = make_clustering_config(markers, resolution=resolution, n_neighbors=20, n_pcs=20)
                        target_config = adapt_config_for_subset(base_config, sub.n_obs)

                        sub = sv.zscore_normalize(sub)
                        clustered = sv.cluster_cells(sub, target_config)
                        group_label = make_level1_group_label(str(target), str(fov))

                        clustered.obs["level1_parent_key"] = source_key
                        clustered.obs["level1_parent"] = str(target)
                        clustered.obs["level1_fov"] = str(fov)
                        clustered.obs["level1_group"] = group_label
                        clustered.obs["cluster_level1_local"] = clustered.obs["leiden"].astype(str)
                        clustered.obs["cluster_level1"] = (
                            clustered.obs["level1_group"].astype(str)
                            + "::"
                            + clustered.obs["cluster_level1_local"].astype(str)
                        ).astype("category")

                        results[group_label] = clustered
                        summaries.append(
                            {
                                "target": target,
                                "fov": fov,
                                "level1_group": group_label,
                                "n_cells": n_cells,
                                "status": "clustered",
                                "n_clusters": clustered.obs["cluster_level1"].nunique(),
                                "markers": ", ".join(markers),
                                "n_neighbors": target_config.n_neighbors,
                                "n_pcs": target_config.n_pcs,
                                "resolution": target_config.resolution,
                            }
                        )

                summary = pd.DataFrame(summaries)
                if not results:
                    return None, results, summary

                combined = ad.concat(results.values(), axis=0, join="outer", merge="same", index_unique=None)
                combined.obs["cluster_level1"] = combined.obs["cluster_level1"].astype("category")
                combined.obs["level1_group"] = combined.obs["level1_group"].astype("category")
                combined.obs["level1_fov"] = combined.obs["level1_fov"].astype("category")
                return combined, results, summary

            if RUN_LEVEL1_CLUSTERING and LEVEL1_TARGETS:
                adata_lvl1, level1_results, level1_summary = run_level1_clustering(
                    adata_pheno,
                    source_key=LEVEL1_SOURCE_KEY,
                    targets=LEVEL1_TARGETS,
                )
                display(level1_summary)

                if adata_lvl1 is not None and SAVE_OUTPUTS:
                    LEVEL1_H5AD = ADATA_DIR / f"{SAMPLE_ID}_level1_clusters_by_fov.h5ad"
                    LEVEL1_SUMMARY_CSV = TABLES_DIR / f"{SAMPLE_ID}_level1_fov_clustering_summary.csv"
                    adata_lvl1.write_h5ad(LEVEL1_H5AD, compression="gzip")
                    level1_summary.to_csv(LEVEL1_SUMMARY_CSV, index=False)
                    print(f"Saved: {LEVEL1_H5AD}")
                    print(f"Saved: {LEVEL1_SUMMARY_CSV}")
            else:
                adata_lvl1 = None
                level1_results = {}
                level1_summary = pd.DataFrame()
                print("Level-1 clustering skipped. Set RUN_LEVEL1_CLUSTERING=True and provide LEVEL1_TARGETS.")
            """
        ),
        md(
            """
            ## Level-1 UMAPs

            Review each FOV/target level-1 group before assigning phenotype
            names. UMAP coordinates are generated independently for each
            level-1 group.
            """
        ),
        code(
            """
            if adata_lvl1 is not None and level1_results:
                for group_label, sub in level1_results.items():
                    fig = sc.pl.umap(
                        sub,
                        color=["cluster_level1_local"],
                        legend_loc="on data",
                        frameon=False,
                        title=f"{group_label}: level-1 local clusters",
                        show=False,
                        return_fig=True,
                    )
                    safe_group = str(group_label).replace("::", "__").replace("/", "_")
                    if SAVE_OUTPUTS:
                        fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_{safe_group}_level1_umap_clusters.png", dpi=200, bbox_inches="tight")
                    plt.show()
                    plt.close(fig)
            else:
                print("No level-1 clustering results available yet.")
            """
        ),
        md(
            """
            ## Inspect Level-1 Clusters Spatially

            This shows all level-1-clustered cells in their FOV-local coordinate
            systems. Because level-1 runs only on selected parent labels, these
            plots show only those refined cells.
            """
        ),
        code(
            """
            if adata_lvl1 is not None:
                fig = plot_fov_spatial_grid(
                    adata_lvl1,
                    color_key="cluster_level1",
                    fov_key=LEVEL1_FOV_KEY,
                    point_size=0.18,
                    alpha=0.85,
                    title=f"{SAMPLE_ID}: level-1 clusters within each FOV",
                    save_path=FIGURES_DIR / f"{SAMPLE_ID}_level1_spatial_clusters_by_fov.png" if SAVE_OUTPUTS else None,
                )
                plt.show()
                plt.close(fig)
            else:
                print("No level-1 object available yet.")
            """
        ),
        md(
            """
            ## Optional Level-1 Image Overlay Review

            Set `RUN_LEVEL1_CLUSTER_VIEWER = True` to inspect one level-1
            FOV/target group on the original `background/<fov_number>.ome.tif`
            image before annotation.
            """
        ),
        code(
            """
            RUN_LEVEL1_CLUSTER_VIEWER = False
            LEVEL1_VIEW_GROUP = next(iter(level1_results), None) if level1_results else None

            if RUN_LEVEL1_CLUSTER_VIEWER:
                require_scimap()
                if adata_lvl1 is None or not level1_results:
                    raise ValueError("No level-1 clustering results are available for image overlay review.")
                if LEVEL1_VIEW_GROUP not in level1_results:
                    raise ValueError(f"LEVEL1_VIEW_GROUP must be one of: {list(level1_results)}")

                view_adata = level1_results[LEVEL1_VIEW_GROUP].copy()
                view_fov = str(view_adata.obs[LEVEL1_FOV_KEY].astype(str).iloc[0])
                viewer_channel_names = list(adata.var_names)
                view_adata.uns["all_markers"] = viewer_channel_names

                sm.pl.image_viewer(
                    image_path=str(image_path_for_fov(view_fov)),
                    adata=view_adata,
                    channel_names=viewer_channel_names,
                    imageid=LEVEL1_FOV_KEY,
                    subset=view_fov,
                    overlay="cluster_level1_local",
                    point_size=8,
                    point_color="white",
                )
            else:
                print("RUN_LEVEL1_CLUSTER_VIEWER=False; set it to True for one level-1 group overlay.")
                print("Available level-1 groups:", list(level1_results))
            """
        ),
        md(
            """
            ## Annotate And Merge Level-1 Labels
            """
        ),
        code(
            """
            ANNOTATE_LEVEL1_INTERACTIVELY = False

            DRAFT_LEVEL1_REVIEW_CSV = TABLES_DIR / f"{SAMPLE_ID}_level1_cluster_annotation_review.csv"
            REVIEWED_LEVEL1_ANNOTATION_CSV = TABLES_DIR / f"{SAMPLE_ID}_level1_cluster_annotation_reviewed.csv"

            if adata_lvl1 is not None:
                level1_annotation_table = make_annotation_table_from_clusters(adata_lvl1, "cluster_level1")

                if ANNOTATE_LEVEL1_INTERACTIVELY:
                    adata_lvl1, interactive_mapping_level1, level1_annotation_table = annotate_clusters_interactively(
                        adata_lvl1,
                        level1_annotation_table,
                        cluster_key="cluster_level1",
                        new_key="annotation_level1",
                    )

                    if SAVE_OUTPUTS:
                        level1_annotation_table.to_csv(REVIEWED_LEVEL1_ANNOTATION_CSV)
                        print(f"Saved interactive level-1 annotations: {REVIEWED_LEVEL1_ANNOTATION_CSV}")
                else:
                    print("Interactive level-1 annotation skipped.")
                    if SAVE_OUTPUTS:
                        level1_annotation_table.to_csv(DRAFT_LEVEL1_REVIEW_CSV)
                        print(f"Saved draft level-1 cluster table: {DRAFT_LEVEL1_REVIEW_CSV}")

                if REVIEWED_LEVEL1_ANNOTATION_CSV.exists():
                    level1_annotation_source = REVIEWED_LEVEL1_ANNOTATION_CSV
                    level1_table = pd.read_csv(level1_annotation_source, dtype={"cluster": str})
                elif DRAFT_LEVEL1_REVIEW_CSV.exists():
                    level1_annotation_source = DRAFT_LEVEL1_REVIEW_CSV
                    level1_table = pd.read_csv(level1_annotation_source, dtype={"cluster": str})
                else:
                    level1_annotation_source = "in-memory manual review table"
                    level1_table = level1_annotation_table.reset_index().copy()

                if "cluster" not in level1_table.columns:
                    level1_table = level1_table.rename(columns={level1_table.columns[0]: "cluster"})
                if "annotation" not in level1_table.columns:
                    raise ValueError("Expected an 'annotation' column in the level-1 annotation table.")

                cluster_to_annotation_level1 = dict(
                    zip(level1_table["cluster"].astype(str), level1_table["annotation"].astype(str))
                )
                adata_lvl1.obs["annotation_level1"] = (
                    adata_lvl1.obs["cluster_level1"]
                    .astype(str)
                    .map(cluster_to_annotation_level1)
                    .fillna("unassigned")
                    .astype("category")
                )

                adata_pheno.obs["cluster_level1"] = "not_refined"
                adata_pheno.obs["annotation_level1"] = adata_pheno.obs["annotation_level0"].astype(str)

                adata_pheno.obs.loc[adata_lvl1.obs_names, "cluster_level1"] = adata_lvl1.obs["cluster_level1"].astype(str)
                adata_pheno.obs.loc[adata_lvl1.obs_names, "annotation_level1"] = adata_lvl1.obs["annotation_level1"].astype(str)

                adata_pheno.obs["cluster_level1"] = adata_pheno.obs["cluster_level1"].astype("category")
                adata_pheno.obs["annotation_level1"] = adata_pheno.obs["annotation_level1"].astype("category")

                LEVEL1_PHENOTYPED_H5AD = ADATA_DIR / f"{SAMPLE_ID}_phenotyping_level1_by_fov.h5ad"
                LEVEL1_ASSIGNMENTS_CSV = TABLES_DIR / f"{SAMPLE_ID}_level1_phenotype_assignments_by_fov.csv"
                level1_assignment_cols = assignment_cols + ["cluster_level1", "annotation_level1"]
                level1_assignment_cols = [col for col in level1_assignment_cols if col in adata_pheno.obs.columns]

                if SAVE_OUTPUTS:
                    adata_pheno.write_h5ad(LEVEL1_PHENOTYPED_H5AD, compression="gzip")
                    adata_pheno.obs[level1_assignment_cols].to_csv(LEVEL1_ASSIGNMENTS_CSV)
                    print(f"Saved: {LEVEL1_PHENOTYPED_H5AD}")
                    print(f"Saved: {LEVEL1_ASSIGNMENTS_CSV}")

                display(adata_pheno.obs["annotation_level1"].value_counts().rename("n_cells").to_frame())
            else:
                level1_annotation_table = pd.DataFrame()
                print("No level-1 object available yet.")
            """
        ),
        md(
            """
            ## View Annotated Level-1 Results
            """
        ),
        code(
            """
            if adata_lvl1 is not None and "annotation_level1" in adata_lvl1.obs.columns:
                fig = sc.pl.umap(
                    adata_lvl1,
                    color="annotation_level1",
                    frameon=False,
                    title="Level-1 annotations",
                    show=False,
                    return_fig=True,
                )
                if SAVE_OUTPUTS:
                    fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_level1_umap_annotations.png", dpi=200, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                fig = plot_fov_spatial_grid(
                    adata_lvl1,
                    color_key="annotation_level1",
                    fov_key=LEVEL1_FOV_KEY,
                    point_size=0.18,
                    alpha=0.85,
                    title=f"{SAMPLE_ID}: level-1 annotations within each FOV",
                    save_path=FIGURES_DIR / f"{SAMPLE_ID}_level1_spatial_annotations_by_fov.png" if SAVE_OUTPUTS else None,
                )
                plt.show()
                plt.close(fig)
            else:
                print("No level-1 annotations available yet.")
            """
        ),
        md(
            """
            ## Consolidate Clean Annotations

            This section keeps the reviewed level-0 and level-1 annotations
            auditable, then collapses them into two clean labels for downstream
            spatial analysis:

            - `anno_broad`: `tubule`, `non_tubule`, or `artifact`
            - `anno_fine`: `germ_cell`, `sertoli_cell`, `leydig_cell`,
              `myoid_cell`, `vascular_cell`, `non_tubule`, or `artifact`

            Level-1 annotation is used first when it resolves to a clean fine
            class. Level-0 is used as a fallback. Generic or unresolved labels
            are assigned to `non_tubule`.
            """
        ),
        code(
            """
            CLEAN_FINE_ORDER = [
                "germ_cell",
                "sertoli_cell",
                "leydig_cell",
                "myoid_cell",
                "vascular_cell",
                "non_tubule",
                "artifact",
            ]
            CLEAN_BROAD_ORDER = ["tubule", "non_tubule", "artifact"]

            def normalize_annotation_label(value):
                text = str(value).strip().lower()
                text = text.replace("-", "_").replace(" ", "_")
                while "__" in text:
                    text = text.replace("__", "_")
                return text

            def clean_fine_from_annotation(value):
                text = normalize_annotation_label(value)
                if text in {"", "nan", "none", "na", "unassigned", "not_clustered", "not_refined"}:
                    return None
                if any(token in text for token in ["germ", "spermatogonia", "spermatocyte", "spermatid"]):
                    return "germ_cell"
                if "sertoli" in text:
                    return "sertoli_cell"
                if "leydig" in text:
                    return "leydig_cell"
                if "myoid" in text or "smooth_muscle" in text or text == "sma":
                    return "myoid_cell"
                if "vascular" in text or "endothelial" in text or text == "cd31":
                    return "vascular_cell"
                if "artifact" in text:
                    return "artifact"
                if "non_tubule" in text or "interstitial" in text or "immune" in text:
                    return "non_tubule"
                return None

            def consolidate_fine_annotation(row):
                for key in ["annotation_level1", "annotation_level0"]:
                    if key in row.index:
                        clean = clean_fine_from_annotation(row[key])
                        if clean is not None:
                            return clean
                return "non_tubule"

            def broad_from_fine(fine):
                if fine == "artifact":
                    return "artifact"
                return "tubule" if fine in {"germ_cell", "sertoli_cell"} else "non_tubule"

            def fov_annotation_counts(adata_obj, label_key, fov_key=FOV_KEY, columns=None):
                table = pd.crosstab(
                    adata_obj.obs[fov_key].astype(str),
                    adata_obj.obs[label_key].astype(str),
                )
                table = table.reindex(sorted(table.index, key=fov_sort_key))
                if columns is not None:
                    table = table.reindex(columns=columns, fill_value=0)
                return table

            adata_consolidated = adata_pheno.copy()
            adata_consolidated.obs["anno_fine"] = pd.Categorical(
                adata_consolidated.obs.apply(consolidate_fine_annotation, axis=1),
                categories=CLEAN_FINE_ORDER,
            )
            adata_consolidated.obs["anno_broad"] = pd.Categorical(
                adata_consolidated.obs["anno_fine"].astype(str).map(broad_from_fine),
                categories=CLEAN_BROAD_ORDER,
            )

            print("Level-0 annotation counts per FOV")
            level0_counts_by_fov = fov_annotation_counts(adata_consolidated, "annotation_level0")
            display(level0_counts_by_fov)

            if "annotation_level1" in adata_consolidated.obs.columns:
                print("Level-1 annotation counts per FOV")
                level1_counts_by_fov = fov_annotation_counts(adata_consolidated, "annotation_level1")
                display(level1_counts_by_fov)
            else:
                level1_counts_by_fov = pd.DataFrame()

            print("Clean anno_fine counts per FOV")
            anno_fine_counts_by_fov = fov_annotation_counts(
                adata_consolidated,
                "anno_fine",
                columns=CLEAN_FINE_ORDER,
            )
            display(anno_fine_counts_by_fov)

            print("Clean anno_broad counts per FOV")
            anno_broad_counts_by_fov = fov_annotation_counts(
                adata_consolidated,
                "anno_broad",
                columns=CLEAN_BROAD_ORDER,
            )
            display(anno_broad_counts_by_fov)

            print("Overall clean anno_fine counts")
            display(
                adata_consolidated.obs["anno_fine"]
                .value_counts()
                .reindex(CLEAN_FINE_ORDER, fill_value=0)
                .rename("n_cells")
                .to_frame()
            )

            print("Overall clean anno_broad counts")
            display(
                adata_consolidated.obs["anno_broad"]
                .value_counts()
                .reindex(CLEAN_BROAD_ORDER, fill_value=0)
                .rename("n_cells")
                .to_frame()
            )

            cleanup_group_cols = [
                col
                for col in ["annotation_level0", "annotation_level1", "anno_broad", "anno_fine"]
                if col in adata_consolidated.obs.columns
            ]
            annotation_cleanup_audit = (
                adata_consolidated.obs
                .groupby(cleanup_group_cols, observed=True)
                .size()
                .rename("n_cells")
                .reset_index()
                .sort_values("n_cells", ascending=False)
            )
            print("Annotation cleanup audit")
            display(annotation_cleanup_audit)

            CONSOLIDATED_H5AD = ADATA_DIR / f"{SAMPLE_ID}_phenotyping_consolidated_by_fov.h5ad"
            CONSOLIDATED_ASSIGNMENTS_CSV = TABLES_DIR / f"{SAMPLE_ID}_consolidated_phenotype_assignments_by_fov.csv"
            LEVEL0_COUNTS_BY_FOV_CSV = TABLES_DIR / f"{SAMPLE_ID}_level0_annotation_counts_by_fov.csv"
            LEVEL1_COUNTS_BY_FOV_CSV = TABLES_DIR / f"{SAMPLE_ID}_level1_annotation_counts_by_fov.csv"
            ANNO_FINE_COUNTS_BY_FOV_CSV = TABLES_DIR / f"{SAMPLE_ID}_anno_fine_counts_by_fov.csv"
            ANNO_BROAD_COUNTS_BY_FOV_CSV = TABLES_DIR / f"{SAMPLE_ID}_anno_broad_counts_by_fov.csv"
            CLEANUP_AUDIT_CSV = TABLES_DIR / f"{SAMPLE_ID}_annotation_cleanup_audit.csv"

            consolidated_assignment_cols = [
                FOV_KEY,
                FOV_LABEL_KEY,
                UNIT_KEY,
                "X_centroid",
                "Y_centroid",
                "cluster_level0",
                "annotation_level0",
                "cluster_level1",
                "annotation_level1",
                "anno_broad",
                "anno_fine",
            ]
            consolidated_assignment_cols = [
                col for col in consolidated_assignment_cols if col in adata_consolidated.obs.columns
            ]

            if SAVE_OUTPUTS:
                adata_consolidated.write_h5ad(CONSOLIDATED_H5AD, compression="gzip")
                adata_consolidated.obs[consolidated_assignment_cols].to_csv(CONSOLIDATED_ASSIGNMENTS_CSV)
                level0_counts_by_fov.to_csv(LEVEL0_COUNTS_BY_FOV_CSV)
                if not level1_counts_by_fov.empty:
                    level1_counts_by_fov.to_csv(LEVEL1_COUNTS_BY_FOV_CSV)
                anno_fine_counts_by_fov.to_csv(ANNO_FINE_COUNTS_BY_FOV_CSV)
                anno_broad_counts_by_fov.to_csv(ANNO_BROAD_COUNTS_BY_FOV_CSV)
                annotation_cleanup_audit.to_csv(CLEANUP_AUDIT_CSV, index=False)

                for path in [
                    CONSOLIDATED_H5AD,
                    CONSOLIDATED_ASSIGNMENTS_CSV,
                    LEVEL0_COUNTS_BY_FOV_CSV,
                    LEVEL1_COUNTS_BY_FOV_CSV,
                    ANNO_FINE_COUNTS_BY_FOV_CSV,
                    ANNO_BROAD_COUNTS_BY_FOV_CSV,
                    CLEANUP_AUDIT_CSV,
                ]:
                    if path.exists():
                        print(f"Saved: {path}")
            """
        ),
        md(
            """
            ## View Clean Consolidated Annotations
            """
        ),
        code(
            """
            def plot_fov_spatial_grid_with_side_legend(
                adata_obj,
                color_key,
                fov_key=FOV_KEY,
                x_key="X_centroid",
                y_key="Y_centroid",
                fovs=None,
                ncols=5,
                point_size=1,
                alpha=1,
                figsize_per_panel=(3.0, 3.0),
                title=None,
                legend_title=None,
                save_path=None,
            ):
                from matplotlib.lines import Line2D

                if fovs is None:
                    fovs = sorted(adata_obj.obs[fov_key].astype(str).unique(), key=fov_sort_key)
                else:
                    fovs = [str(fov) for fov in fovs]

                color_series = adata_obj.obs[color_key]
                observed_categories = set(color_series.astype(str).dropna().unique())
                if isinstance(color_series.dtype, pd.CategoricalDtype):
                    categories = [
                        str(cat)
                        for cat in color_series.cat.categories
                        if str(cat) in observed_categories
                    ]
                else:
                    categories = sorted(observed_categories)

                cmap = plt.get_cmap("tab20", max(len(categories), 1))
                color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}

                nrows = int(np.ceil(len(fovs) / ncols))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                    squeeze=False,
                )

                for ax, fov in zip(axes.ravel(), fovs):
                    df = adata_obj.obs.loc[adata_obj.obs[fov_key].astype(str).eq(fov)].copy()
                    if df.empty:
                        ax.axis("off")
                        continue

                    values = df[color_key].astype(str)
                    for cat in categories:
                        mask = values.eq(cat)
                        if not mask.any():
                            continue
                        ax.scatter(
                            df.loc[mask, x_key],
                            df.loc[mask, y_key],
                            s=point_size,
                            alpha=alpha,
                            linewidths=0,
                            color=color_map[cat],
                            label=cat,
                        )

                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.set_title(f"{fov} ({len(df):,})")
                    ax.set_xticks([])
                    ax.set_yticks([])

                for ax in axes.ravel()[len(fovs):]:
                    ax.axis("off")

                if title:
                    fig.suptitle(title, y=1.01)

                legend_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        color="none",
                        markerfacecolor=color_map[cat],
                        markeredgewidth=0,
                        markersize=7,
                        label=cat,
                    )
                    for cat in categories
                ]
                fig.legend(
                    handles=legend_handles,
                    title=legend_title or color_key,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=False,
                )
                fig.tight_layout(rect=[0, 0, 0.88, 1])
                if save_path is not None:
                    fig.savefig(save_path, dpi=220, bbox_inches="tight")
                return fig

            fig = plot_fov_spatial_grid_with_side_legend(
                adata_consolidated,
                color_key="anno_fine",
                point_size=1,
                alpha=1,
                title=f"{SAMPLE_ID}: clean fine annotations within each FOV",
                legend_title="anno_fine",
                save_path=FIGURES_DIR / f"{SAMPLE_ID}_anno_fine_spatial_by_fov.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)

            fig = plot_fov_spatial_grid_with_side_legend(
                adata_consolidated,
                color_key="anno_broad",
                point_size=1,
                alpha=1,
                title=f"{SAMPLE_ID}: clean broad annotations within each FOV",
                legend_title="anno_broad",
                save_path=FIGURES_DIR / f"{SAMPLE_ID}_anno_broad_spatial_by_fov.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            f"""
            ## Outputs For Spatial Analysis

            Primary outputs:

            - `phenotyping/clustering_based/adata/{SAMPLE_ID}_level0_clusters_by_fov.h5ad`
            - `phenotyping/clustering_based/adata/{SAMPLE_ID}_phenotyping_level0_by_fov.h5ad`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_level0_fov_clustering_summary.csv`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_level0_cluster_annotation_review.csv`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_level0_phenotype_assignments_by_fov.csv`
            - optional: `phenotyping/clustering_based/adata/{SAMPLE_ID}_phenotyping_level1_by_fov.h5ad`
            - `phenotyping/clustering_based/adata/{SAMPLE_ID}_phenotyping_consolidated_by_fov.h5ad`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_consolidated_phenotype_assignments_by_fov.csv`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_anno_fine_counts_by_fov.csv`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_anno_broad_counts_by_fov.csv`
            - `phenotyping/clustering_based/tables/{SAMPLE_ID}_annotation_cleanup_audit.csv`

            For ST_TMAG, keep FOV/core identity in downstream analysis. The
            cluster labels are intentionally FOV-prefixed because each FOV was
            clustered independently.
            """
        ),
    ]


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for folder in [
        PROJECT_DIR / "phenotyping" / "clustering_based" / "figures",
        PROJECT_DIR / "phenotyping" / "clustering_based" / "tables",
        PROJECT_DIR / "phenotyping" / "clustering_based" / "adata",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = notebook_cells()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    nbf.validate(nb)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    write_notebook()
