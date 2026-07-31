"""Write the ST_TMAG_E1_31 spatial tubule territory analysis notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


SAMPLE_ID = "ST_TMAG_E1_31"
PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_31")
NOTEBOOK_PATH = PROJECT_DIR  / "paper" / "notebooks" / f"02_{SAMPLE_ID}_spatial_tubule_territories_and_neighborhoods.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            f"""
            # {SAMPLE_ID}: Spatial Tubule Territories And Neighborhoods

            This notebook starts from the completed clustering-based phenotype
            object and runs the first spatial analysis layer for the testis
            microarray/FOV dataset.

            Analysis strategy:

            1. Use all non-artifact cells/FOVs for broad descriptive summaries.
            2. Define tubule-centered territories by expanding from tubule cores
               to the nearest neighboring tubule core.
            3. Select three neighboring tubule territories per FOV as a matched
               demo set because adult `fov1` has three tubules.
            4. Quantify compartment composition, tubule-level composition,
               distance-to-tubule measurements, local neighborhoods, and
               SCIMAP-style cell-type proximity enrichment.

            Biological note: the 1 to 6 year testis tissues are immature, so
            this notebook keeps germ cells as one conservative `germ_cell`
            category rather than forcing adult spermatogenic state calls.
            """
        ),
        md(
            """
            ## Runtime Setup
            """
        ),
        code(
            """
            import os
            import shutil
            import sys
            from pathlib import Path
            import warnings

            os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
            os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba-cache")
            Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

            warnings.filterwarnings("ignore", category=FutureWarning)

            import anndata as ad
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            import numpy as np
            import pandas as pd
            from IPython.display import display

            from scipy.spatial import ConvexHull, distance
            from scipy.sparse import csr_matrix
            from sklearn.cluster import KMeans
            from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

            SPATIOEV_REPO = Path("/Users/shihongwu/SpatioEv")
            if SPATIOEV_REPO.exists() and str(SPATIOEV_REPO) not in sys.path:
                sys.path.insert(0, str(SPATIOEV_REPO))

            import spatioev as sv

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)

            print(f"SpatioEv: {getattr(sv, '__version__', 'unknown')}")
            print("Ready.")
            """
        ),
        md(
            """
            ## Paths And Analysis Options

            The core parameters below are intentionally easy to edit after
            reviewing the first-pass figures.
            """
        ),
        code(
            f"""
            SAMPLE_ID = "{SAMPLE_ID}"
            PROJECT_DIR = Path("/Volumes/Shihong_3/ST_TMAG_E1_31")

            INPUT_H5AD = (
                PROJECT_DIR
                / "phenotyping"
                / "clustering_based"
                / "adata"
                / f"{{SAMPLE_ID}}_phenotyping_consolidated_by_fov.h5ad"
            )

            SPATIAL_DIR = PROJECT_DIR / "spatial_analysis" / "tubule_territories"
            FIGURES_DIR = SPATIAL_DIR / "figures"
            TABLES_DIR = SPATIAL_DIR / "tables"
            ADATA_DIR = SPATIAL_DIR / "adata"
            SEGMENTATION_LINK_DIR = SPATIAL_DIR / "segmentation_masks"

            for folder in [SPATIAL_DIR, FIGURES_DIR, TABLES_DIR, ADATA_DIR, SEGMENTATION_LINK_DIR]:
                folder.mkdir(parents=True, exist_ok=True)

            SAVE_OUTPUTS = True

            FOV_KEY = "imageid"
            X_KEY = "X_centroid"
            Y_KEY = "Y_centroid"

            AGE_BY_FOV = {{
                "fov1": "Adult",
                "fov6": "1 yr",
                "fov10": "2 yr",
                "fov12": "3 yr",
                "fov15": "4 yr",
                "fov16": "5 yr",
                "fov17": "6 yr",
            }}
            AGE_ORDER = ["1 yr", "2 yr", "3 yr", "4 yr", "5 yr", "6 yr", "Adult"]
            FOV_ORDER = ["fov6", "fov10", "fov12", "fov15", "fov16", "fov17", "fov1"]

            FINE_ORDER = [
                "germ_cell",
                "sertoli_cell",
                "leydig_cell",
                "myoid_cell",
                "vascular_cell",
                "non_tubule",
                "artifact",
            ]
            NON_ARTIFACT_FINE_ORDER = [x for x in FINE_ORDER if x != "artifact"]
            BROAD_ORDER = ["tubule", "non_tubule", "artifact"]
            ZONE_ORDER = ["tubule_core", "peritubular_ring", "outer_surrounding"]

            FINE_COLORS = {{
                "germ_cell": "#1f77b4",
                "sertoli_cell": "#ffb26b",
                "leydig_cell": "#d62728",
                "myoid_cell": "#8c564b",
                "vascular_cell": "#e377c2",
                "non_tubule": "#bcbd22",
                "artifact": "#9edae5",
            }}
            BROAD_COLORS = {{
                "tubule": "#1f77b4",
                "non_tubule": "#8c564b",
                "artifact": "#9edae5",
            }}
            ZONE_COLORS = {{
                "tubule_core": "#1f77b4",
                "peritubular_ring": "#ff7f0e",
                "outer_surrounding": "#9467bd",
            }}

            ARK_WDIR_1 = PROJECT_DIR / "ark_wdir_1"
            ARK_WDIR_2 = PROJECT_DIR / "ark_wdir_2"

            # Tubule-core components are built from direct adjacency of
            # tubule-cell labels in the ARK whole-cell segmentation masks.
            # Set this to 0 for strict direct touching. A small tolerance keeps
            # biologically continuous tubules from splitting across tiny
            # segmentation gaps, matching the approach used in the graph
            # pseudotime notebook.
            TUBULE_MASK_CONNECTION_MODE = "label_adjacency"
            TUBULE_MASK_GAP_TOLERANCE_PX = 5
            TUBULE_MASK_CONNECTIVITY = 2
            TUBULE_SEG_SUFFIX = "_whole_cell.tiff"
            TUBULE_MASK_COMPONENT_RAW_KEY = "tubule_mask_component_raw"
            TUBULE_MASK_TARGET_KEY = "is_tubule_mask_target"
            MIN_TUBULE_CORE_CELLS = 100
            MIN_TUBULE_CORE_CELLS_BY_FOV = {{
                # fov17 has many real but smaller immature tubule/cord regions.
                # Lowering only this FOV lets us choose three neighboring
                # candidates around the original T03 region.
                "fov17": 30,
            }}
            INITIAL_MASK_COMPONENT_MIN_CELLS = min(
                [MIN_TUBULE_CORE_CELLS] + list(MIN_TUBULE_CORE_CELLS_BY_FOV.values())
            )
            PERITUBULAR_RING_RADIUS_PX = 75

            # Automatic matched-demo selection chooses the largest tubule
            # territory and its two nearest neighboring tubules.
            N_DEMO_TUBULES_PER_FOV = 3
            SELECTED_TUBULE_IDS_BY_FOV = {{
                # Optional manual override after inspecting candidate labels:
                # "fov6": ["fov6_T03", "fov6_T04", "fov6_T08"],
                "fov17": ["fov17_T19", "fov17_T21", "fov17_T22"],
            }}

            # Local-neighborhood analysis.
            RUN_NEIGHBORHOODS = True
            NEIGHBOR_RADIUS_PX = 100
            MIN_NEIGHBORS_FOR_MOTIF = 5
            N_NEIGHBORHOOD_MOTIFS = 5

            # Cell-type proximity enrichment uses the same radius as the
            # neighborhood composition step.
            PROXIMITY_PAIR_TYPES = [
                ("germ_cell", "sertoli_cell"),
                ("sertoli_cell", "myoid_cell"),
                ("leydig_cell", "vascular_cell"),
                ("leydig_cell", "non_tubule"),
            ]
            """
        ),
        md(
            """
            ## Helper Functions
            """
        ),
        code(
            """
            def fov_sort_key(value):
                text = str(value)
                suffix = text.replace("fov", "")
                return int(suffix) if suffix.isdigit() else text

            def ordered_fovs(values):
                values = [str(v) for v in values]
                return [fov for fov in FOV_ORDER if fov in values] + [
                    fov for fov in sorted(values, key=fov_sort_key) if fov not in FOV_ORDER
                ]

            def fov_age_label(fov):
                age = AGE_BY_FOV.get(str(fov), "unknown")
                return f"{fov}\\n{age}"

            def min_tubule_core_cells_for_fov(fov):
                return int(MIN_TUBULE_CORE_CELLS_BY_FOV.get(str(fov), MIN_TUBULE_CORE_CELLS))

            def convex_hull_area(xy):
                xy = np.asarray(xy, dtype=float)
                if xy.shape[0] < 3:
                    return np.nan
                try:
                    return float(ConvexHull(xy).volume)
                except Exception:
                    return np.nan

            def add_figure_legend(fig, color_map, order, title, bbox_to_anchor=(1.01, 0.5)):
                handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        color="none",
                        markerfacecolor=color_map[label],
                        markeredgewidth=0,
                        markersize=7,
                        label=label,
                    )
                    for label in order
                    if label in color_map
                ]
                fig.legend(
                    handles=handles,
                    title=title,
                    loc="center left",
                    bbox_to_anchor=bbox_to_anchor,
                    frameon=False,
                )

            def plot_stacked_proportions(count_table, colors, title, ylabel, save_path=None):
                prop = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
                fig, ax = plt.subplots(figsize=(8, 4))
                bottom = np.zeros(len(prop))
                x = np.arange(len(prop))
                for col in prop.columns:
                    ax.bar(
                        x,
                        prop[col].to_numpy(),
                        bottom=bottom,
                        label=col,
                        color=colors.get(col),
                        width=0.72,
                    )
                    bottom += prop[col].to_numpy()
                ax.set_xticks(x)
                ax.set_xticklabels(prop.index, rotation=0)
                ax.set_ylim(0, 1)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                fig.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=220, bbox_inches="tight")
                return fig

            def plot_fov_grid(
                obs,
                color_key,
                color_map,
                order,
                title,
                fovs=None,
                point_size=1,
                alpha=1,
                save_path=None,
                extra_outline_obs=None,
            ):
                if fovs is None:
                    fovs = ordered_fovs(obs[FOV_KEY].astype(str).unique())
                ncols = 5
                nrows = int(np.ceil(len(fovs) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)

                for ax, fov in zip(axes.ravel(), fovs):
                    df = obs.loc[obs[FOV_KEY].astype(str).eq(fov)]
                    if df.empty:
                        ax.axis("off")
                        continue

                    values = df[color_key].astype(str)
                    for label in order:
                        mask = values.eq(label)
                        if not mask.any():
                            continue
                        ax.scatter(
                            df.loc[mask, X_KEY],
                            df.loc[mask, Y_KEY],
                            s=point_size,
                            alpha=alpha,
                            linewidths=0,
                            color=color_map.get(label, "#aaaaaa"),
                        )

                    if extra_outline_obs is not None:
                        ex = extra_outline_obs.loc[extra_outline_obs[FOV_KEY].astype(str).eq(fov)]
                        if not ex.empty:
                            ax.scatter(
                                ex[X_KEY],
                                ex[Y_KEY],
                                s=max(point_size * 3, 3),
                                facecolors="none",
                                edgecolors="black",
                                linewidths=0.25,
                            )

                    ax.set_title(fov_age_label(fov))
                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])

                for ax in axes.ravel()[len(fovs):]:
                    ax.axis("off")

                if title:
                    fig.suptitle(title, y=1.01)
                add_figure_legend(fig, color_map, order, title=color_key)
                fig.tight_layout(rect=[0, 0, 0.88, 1])
                if save_path is not None:
                    fig.savefig(save_path, dpi=220, bbox_inches="tight")
                return fig

            def save_table(df, path, index=True):
                if SAVE_OUTPUTS:
                    df.to_csv(path, index=index)
                    print(f"Saved: {path}")
            """
        ),
        md(
            """
            ## Load Consolidated Phenotypes
            """
        ),
        code(
            """
            adata = ad.read_h5ad(INPUT_H5AD)
            obs = adata.obs.copy()
            obs[FOV_KEY] = obs[FOV_KEY].astype(str)
            obs["age"] = obs[FOV_KEY].map(AGE_BY_FOV)
            obs["age"] = pd.Categorical(obs["age"], categories=AGE_ORDER, ordered=True)
            obs["fov_age"] = pd.Categorical(
                obs[FOV_KEY].map(fov_age_label),
                categories=[fov_age_label(fov) for fov in FOV_ORDER],
                ordered=True,
            )

            for col, order in [("anno_fine", FINE_ORDER), ("anno_broad", BROAD_ORDER)]:
                obs[col] = pd.Categorical(obs[col].astype(str), categories=order)

            adata.obs["age"] = obs["age"]
            adata.obs["fov_age"] = obs["fov_age"]

            print(adata)
            display(obs[[FOV_KEY, "age", "anno_broad", "anno_fine", X_KEY, Y_KEY]].head())
            display(obs.groupby([FOV_KEY, "age"], observed=True).size().rename("n_cells").to_frame())
            """
        ),
        md(
            """
            ## Whole-FOV Descriptive Summaries

            These summaries use all cells first, then non-artifact cells for
            tissue-composition measurements.
            """
        ),
        code(
            """
            fov_index = [fov_age_label(fov) for fov in FOV_ORDER]

            broad_counts = pd.crosstab(obs["fov_age"], obs["anno_broad"]).reindex(fov_index)
            broad_counts = broad_counts.reindex(columns=BROAD_ORDER, fill_value=0)
            fine_counts = pd.crosstab(obs["fov_age"], obs["anno_fine"]).reindex(fov_index)
            fine_counts = fine_counts.reindex(columns=FINE_ORDER, fill_value=0)

            broad_props = broad_counts.div(broad_counts.sum(axis=1), axis=0)
            fine_props = fine_counts.div(fine_counts.sum(axis=1), axis=0)

            display(broad_counts)
            display(fine_counts)

            save_table(broad_counts, TABLES_DIR / f"{SAMPLE_ID}_whole_fov_anno_broad_counts.csv")
            save_table(broad_props, TABLES_DIR / f"{SAMPLE_ID}_whole_fov_anno_broad_proportions.csv")
            save_table(fine_counts, TABLES_DIR / f"{SAMPLE_ID}_whole_fov_anno_fine_counts.csv")
            save_table(fine_props, TABLES_DIR / f"{SAMPLE_ID}_whole_fov_anno_fine_proportions.csv")

            fig = plot_stacked_proportions(
                broad_counts,
                BROAD_COLORS,
                f"{SAMPLE_ID}: whole-FOV compartment composition",
                "Fraction of all cells",
                FIGURES_DIR / f"{SAMPLE_ID}_whole_fov_anno_broad_proportions.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)

            fig = plot_stacked_proportions(
                fine_counts,
                FINE_COLORS,
                f"{SAMPLE_ID}: whole-FOV fine annotation composition",
                "Fraction of all cells",
                FIGURES_DIR / f"{SAMPLE_ID}_whole_fov_anno_fine_proportions.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        code(
            """
            non_artifact_obs = obs.loc[obs["anno_broad"].astype(str).ne("artifact")].copy()

            footprint_records = []
            for fov in FOV_ORDER:
                df_all = obs.loc[obs[FOV_KEY].eq(fov)]
                df = non_artifact_obs.loc[non_artifact_obs[FOV_KEY].eq(fov)]
                xy = df[[X_KEY, Y_KEY]].to_numpy(float)
                area_px2 = convex_hull_area(xy)
                footprint_records.append(
                    {
                        "fov": fov,
                        "age": AGE_BY_FOV[fov],
                        "n_all_cells": len(df_all),
                        "n_non_artifact_cells": len(df),
                        "n_artifact_cells": int((df_all["anno_broad"].astype(str) == "artifact").sum()),
                        "artifact_fraction": float((df_all["anno_broad"].astype(str) == "artifact").mean()),
                        "convex_hull_area_px2": area_px2,
                        "non_artifact_density_per_1e6_px2": len(df) / area_px2 * 1e6 if area_px2 > 0 else np.nan,
                    }
                )

            fov_footprint = pd.DataFrame(footprint_records)
            display(fov_footprint)
            save_table(fov_footprint, TABLES_DIR / f"{SAMPLE_ID}_whole_fov_density_and_artifact_summary.csv", index=False)

            fig, ax = plt.subplots(figsize=(7.5, 4))
            ax.bar(
                fov_footprint["age"],
                fov_footprint["non_artifact_density_per_1e6_px2"],
                color="#4c78a8",
                width=0.7,
            )
            ax.set_ylabel("Non-artifact cells per 1e6 px2")
            ax.set_xlabel("Age/FOV")
            ax.set_title(f"{SAMPLE_ID}: whole-FOV cell density")
            fig.tight_layout()
            if SAVE_OUTPUTS:
                fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_whole_fov_cell_density.png", dpi=220, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Define Tubule Components And Territories

            Tubule cores are connected components of `anno_broad == tubule`
            cells inside each FOV, identified from direct adjacency of their
            whole-cell segmentation labels. Every non-artifact cell is then
            assigned to its nearest retained tubule core. This creates a
            Voronoi-like territory for each tubule: surrounding cells expand
            away from a tubule until they are closer to another tubule.
            """
        ),
        code(
            """
            def ark_wdir_for_fov(fov):
                fov_number = int(str(fov).replace("fov", ""))
                return ARK_WDIR_1 if fov_number <= 9 else ARK_WDIR_2

            def segmentation_source_path_for_fov(fov):
                return (
                    ark_wdir_for_fov(fov)
                    / "segmentation"
                    / "deepcell_output"
                    / f"{fov}{TUBULE_SEG_SUFFIX}"
                )

            def prepare_segmentation_link_dir(fovs):
                SEGMENTATION_LINK_DIR.mkdir(parents=True, exist_ok=True)
                records = []
                for fov in fovs:
                    src = segmentation_source_path_for_fov(fov)
                    if not src.exists():
                        raise FileNotFoundError(src)
                    dst = SEGMENTATION_LINK_DIR / f"{fov}{TUBULE_SEG_SUFFIX}"
                    if dst.exists() or dst.is_symlink():
                        try:
                            if dst.resolve() == src.resolve():
                                records.append({"fov": fov, "source": src, "link": dst, "mode": "existing"})
                                continue
                        except FileNotFoundError:
                            pass
                        dst.unlink()
                    try:
                        dst.symlink_to(src)
                        mode = "symlink"
                    except OSError:
                        shutil.copy2(src, dst)
                        mode = "copy"
                    records.append({"fov": fov, "source": src, "link": dst, "mode": mode})
                return pd.DataFrame(records)

            segmentation_links = prepare_segmentation_link_dir(FOV_ORDER)
            display(segmentation_links)
            save_table(segmentation_links, TABLES_DIR / f"{SAMPLE_ID}_segmentation_mask_links.csv", index=False)

            adata_components = sv.cluster_spatial_components_from_mask(
                adata[obs["anno_broad"].astype(str).ne("artifact")].copy(),
                seg_dir=str(SEGMENTATION_LINK_DIR),
                label_key="anno_broad",
                label_value="tubule",
                fov_key=FOV_KEY,
                cell_label_key="label",
                component_key=TUBULE_MASK_COMPONENT_RAW_KEY,
                target_key=TUBULE_MASK_TARGET_KEY,
                seg_suffix=TUBULE_SEG_SUFFIX,
                connection_mode=TUBULE_MASK_CONNECTION_MODE,
                gap_tolerance=TUBULE_MASK_GAP_TOLERANCE_PX,
                connectivity=TUBULE_MASK_CONNECTIVITY,
                min_component_size=INITIAL_MASK_COMPONENT_MIN_CELLS,
                assign_singletons=False,
            )

            mask_component_params = adata_components.uns["spatial_niches"][
                f"{TUBULE_MASK_COMPONENT_RAW_KEY}_params"
            ].copy()
            display(mask_component_params)
            save_table(
                mask_component_params,
                TABLES_DIR / f"{SAMPLE_ID}_tubule_mask_component_params.csv",
                index=False,
            )

            non_artifact_obs = non_artifact_obs.copy()
            non_artifact_obs[TUBULE_MASK_COMPONENT_RAW_KEY] = adata_components.obs[
                TUBULE_MASK_COMPONENT_RAW_KEY
            ].reindex(non_artifact_obs.index)
            non_artifact_obs[TUBULE_MASK_TARGET_KEY] = adata_components.obs[
                TUBULE_MASK_TARGET_KEY
            ].reindex(non_artifact_obs.index).fillna(False).astype(bool)

            def tubule_components_for_fov(df, fov):
                df = df.copy()
                tubule_mask = df["anno_broad"].astype(str).eq("tubule")
                tubule_df = df.loc[tubule_mask].copy()

                if tubule_df.empty:
                    result = pd.DataFrame(index=df.index)
                    result["tubule_core_id"] = pd.NA
                    result["territory_tubule_id"] = pd.NA
                    result["distance_to_tubule_px"] = np.nan
                    result["territory_zone"] = pd.NA
                    return result, pd.DataFrame()

                raw_component = tubule_df[TUBULE_MASK_COMPONENT_RAW_KEY].astype(str)
                retained_component_mask = raw_component.str.contains("component", na=False)
                raw = tubule_df.loc[retained_component_mask, [FOV_KEY, X_KEY, Y_KEY, "anno_fine"]].copy()
                raw["raw_mask_component"] = raw_component.loc[retained_component_mask].to_numpy()

                if raw.empty:
                    result = pd.DataFrame(index=df.index)
                    result["tubule_core_id"] = pd.NA
                    result["territory_tubule_id"] = pd.NA
                    result["distance_to_tubule_px"] = np.nan
                    result["territory_zone"] = pd.NA
                    return result, pd.DataFrame()

                raw_summary = (
                    raw.groupby("raw_mask_component", observed=True)
                    .agg(
                        n_core_cells=("raw_mask_component", "size"),
                        x_center=(X_KEY, "mean"),
                        y_center=(Y_KEY, "mean"),
                        x_min=(X_KEY, "min"),
                        x_max=(X_KEY, "max"),
                        y_min=(Y_KEY, "min"),
                        y_max=(Y_KEY, "max"),
                    )
                    .reset_index()
                )
                min_core_cells = min_tubule_core_cells_for_fov(fov)
                retained = raw_summary.loc[raw_summary["n_core_cells"] >= min_core_cells].copy()
                retained = retained.sort_values(["y_center", "x_center"]).reset_index(drop=True)
                retained["tubule_core_id"] = [f"{fov}_T{i + 1:02d}" for i in range(len(retained))]
                component_map = dict(zip(retained["raw_mask_component"], retained["tubule_core_id"]))

                tubule_df["tubule_core_id"] = raw_component.map(component_map)
                retained_tubules = tubule_df.dropna(subset=["tubule_core_id"]).copy()

                result = pd.DataFrame(index=df.index)
                result["tubule_core_id"] = pd.NA
                result.loc[tubule_df.index, "tubule_core_id"] = tubule_df["tubule_core_id"]

                if retained_tubules.empty:
                    result["territory_tubule_id"] = pd.NA
                    result["distance_to_tubule_px"] = np.nan
                    result["territory_zone"] = pd.NA
                    return result, retained

                retained_xy = retained_tubules[[X_KEY, Y_KEY]].to_numpy(float)
                retained_labels = retained_tubules["tubule_core_id"].astype(str).to_numpy()
                query_xy = df[[X_KEY, Y_KEY]].to_numpy(float)

                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(retained_xy)
                distances, nearest_indices = nn.kneighbors(query_xy)
                nearest_labels = retained_labels[nearest_indices[:, 0]]

                result["territory_tubule_id"] = nearest_labels
                result["distance_to_tubule_px"] = distances[:, 0]
                core_mask = result["tubule_core_id"].notna()
                result.loc[core_mask, "distance_to_tubule_px"] = 0.0
                result["territory_zone"] = np.where(
                    core_mask,
                    "tubule_core",
                    np.where(
                        result["distance_to_tubule_px"].to_numpy(float) <= PERITUBULAR_RING_RADIUS_PX,
                        "peritubular_ring",
                        "outer_surrounding",
                    ),
                )

                retained["fov"] = fov
                retained["age"] = AGE_BY_FOV.get(fov)
                retained["component_method"] = TUBULE_MASK_CONNECTION_MODE
                retained["mask_gap_tolerance_px"] = TUBULE_MASK_GAP_TOLERANCE_PX
                retained["mask_connectivity"] = TUBULE_MASK_CONNECTIVITY
                retained["initial_mask_component_min_cells"] = INITIAL_MASK_COMPONENT_MIN_CELLS
                retained["min_tubule_core_cells"] = min_core_cells
                retained["segmentation_path"] = str(segmentation_source_path_for_fov(fov))
                return result, retained

            territory_frames = []
            component_summaries = []

            for fov in FOV_ORDER:
                df = non_artifact_obs.loc[non_artifact_obs[FOV_KEY].eq(fov)].copy()
                territory_df, summary = tubule_components_for_fov(df, fov)
                territory_frames.append(territory_df)
                component_summaries.append(summary)

            territory_obs = pd.concat(territory_frames, axis=0)
            tubule_component_summary = pd.concat(component_summaries, axis=0, ignore_index=True)

            adata_spatial = adata[territory_obs.index].copy()
            for col in ["tubule_core_id", "territory_tubule_id", "distance_to_tubule_px", "territory_zone"]:
                adata_spatial.obs[col] = territory_obs[col]
            for col in [TUBULE_MASK_COMPONENT_RAW_KEY, TUBULE_MASK_TARGET_KEY]:
                adata_spatial.obs[col] = non_artifact_obs.loc[adata_spatial.obs_names, col]
            adata_spatial.obs["age"] = obs.loc[adata_spatial.obs_names, "age"]
            adata_spatial.obs["fov_age"] = obs.loc[adata_spatial.obs_names, "fov_age"]
            adata_spatial.obs["territory_zone"] = pd.Categorical(
                adata_spatial.obs["territory_zone"].astype(str),
                categories=ZONE_ORDER,
            )

            print(adata_spatial)
            display(tubule_component_summary.groupby(["fov", "age"], observed=True).size().rename("n_retained_tubules").to_frame())
            display(tubule_component_summary.head())

            save_table(tubule_component_summary, TABLES_DIR / f"{SAMPLE_ID}_tubule_component_summary.csv", index=False)
            if SAVE_OUTPUTS:
                TERRITORY_H5AD = ADATA_DIR / f"{SAMPLE_ID}_spatial_tubule_territories.h5ad"
                adata_spatial.write_h5ad(TERRITORY_H5AD, compression="gzip")
                print(f"Saved: {TERRITORY_H5AD}")
            """
        ),
        md(
            """
            ## Preview Tubule Component Candidates

            Use this figure to decide whether the automatic candidate detection
            is good enough or whether `SELECTED_TUBULE_IDS_BY_FOV` should be
            edited near the top of the notebook.
            """
        ),
        code(
            """
            def plot_tubule_candidate_preview(adata_obj, component_summary, save_path=None):
                obs_plot = adata_obj.obs.copy()
                fovs = ordered_fovs(obs_plot[FOV_KEY].astype(str).unique())
                ncols = 5
                nrows = int(np.ceil(len(fovs) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), squeeze=False)
                cmap = plt.get_cmap("tab20")

                for ax, fov in zip(axes.ravel(), fovs):
                    df = obs_plot.loc[obs_plot[FOV_KEY].astype(str).eq(fov)]
                    ax.scatter(df[X_KEY], df[Y_KEY], s=0.25, color="#dddddd", linewidths=0)

                    sub = df.loc[df["anno_broad"].astype(str).eq("tubule")]
                    ids = sorted(sub["territory_tubule_id"].dropna().astype(str).unique())
                    for i, tid in enumerate(ids):
                        core = sub.loc[sub["territory_tubule_id"].astype(str).eq(tid)]
                        ax.scatter(core[X_KEY], core[Y_KEY], s=1.0, color=cmap(i % cmap.N), linewidths=0)

                    centers = component_summary.loc[component_summary["fov"].eq(fov)]
                    for _, row in centers.iterrows():
                        ax.text(
                            row["x_center"],
                            row["y_center"],
                            row["tubule_core_id"].replace(f"{fov}_", ""),
                            fontsize=7,
                            ha="center",
                            va="center",
                            bbox=dict(facecolor="white", edgecolor="black", linewidth=0.3, alpha=0.75),
                        )

                    ax.set_title(f"{fov_age_label(fov)}: {len(centers)} candidates")
                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])

                for ax in axes.ravel()[len(fovs):]:
                    ax.axis("off")

                fig.suptitle(f"{SAMPLE_ID}: retained tubule-core candidates", y=1.01)
                fig.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=220, bbox_inches="tight")
                return fig

            fig = plot_tubule_candidate_preview(
                adata_spatial,
                tubule_component_summary,
                FIGURES_DIR / f"{SAMPLE_ID}_tubule_candidate_preview.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Select Three Neighboring Tubules Per FOV

            By default, the notebook selects the largest retained tubule and
            its two nearest neighboring retained tubules. Manual choices in
            `SELECTED_TUBULE_IDS_BY_FOV` override the automatic selection.
            """
        ),
        code(
            """
            def auto_select_neighboring_tubules(component_summary, n=3):
                selections = {}
                for fov in FOV_ORDER:
                    sub = component_summary.loc[component_summary["fov"].eq(fov)].copy()
                    if sub.empty:
                        selections[fov] = []
                        continue
                    if len(sub) <= n:
                        selections[fov] = sub.sort_values("n_core_cells", ascending=False)["tubule_core_id"].tolist()
                        continue

                    seed = sub.sort_values("n_core_cells", ascending=False).iloc[0]
                    centers = sub[["x_center", "y_center"]].to_numpy(float)
                    seed_xy = np.array([[seed["x_center"], seed["y_center"]]], dtype=float)
                    sub["distance_to_seed"] = distance.cdist(centers, seed_xy).ravel()
                    chosen = (
                        sub.sort_values(["distance_to_seed", "n_core_cells"], ascending=[True, False])
                        .head(n)
                        .sort_values(["y_center", "x_center"])
                    )
                    selections[fov] = chosen["tubule_core_id"].tolist()
                return selections

            AUTO_SELECTED_TUBULE_IDS_BY_FOV = auto_select_neighboring_tubules(
                tubule_component_summary,
                n=N_DEMO_TUBULES_PER_FOV,
            )

            SELECTED_TUBULE_IDS_FINAL = {
                fov: SELECTED_TUBULE_IDS_BY_FOV.get(fov, AUTO_SELECTED_TUBULE_IDS_BY_FOV.get(fov, []))
                for fov in FOV_ORDER
            }

            selected_records = []
            for fov, ids in SELECTED_TUBULE_IDS_FINAL.items():
                for order, tubule_id in enumerate(ids, start=1):
                    selected_records.append(
                        {
                            "fov": fov,
                            "age": AGE_BY_FOV[fov],
                            "selection_order": order,
                            "tubule_core_id": tubule_id,
                            "selection_source": "manual" if fov in SELECTED_TUBULE_IDS_BY_FOV else "auto_largest_plus_nearest",
                        }
                    )

            selected_tubules = pd.DataFrame(selected_records)
            display(selected_tubules)
            save_table(selected_tubules, TABLES_DIR / f"{SAMPLE_ID}_selected_demo_tubules.csv", index=False)

            selected_ids = set(selected_tubules["tubule_core_id"].astype(str))
            adata_spatial.obs["is_demo_tubule_territory"] = (
                adata_spatial.obs["territory_tubule_id"].astype(str).isin(selected_ids)
            )
            adata_spatial.obs["is_demo_tubule_core"] = (
                adata_spatial.obs["tubule_core_id"].astype(str).isin(selected_ids)
            )

            if SAVE_OUTPUTS:
                TERRITORY_H5AD = ADATA_DIR / f"{SAMPLE_ID}_spatial_tubule_territories.h5ad"
                adata_spatial.write_h5ad(TERRITORY_H5AD, compression="gzip")
                print(f"Updated: {TERRITORY_H5AD}")
            """
        ),
        code(
            """
            def plot_selected_tubule_territories(adata_obj, selected_tubules, save_path=None):
                obs_plot = adata_obj.obs.copy()
                selected_ids = set(selected_tubules["tubule_core_id"].astype(str))
                fovs = ordered_fovs(obs_plot[FOV_KEY].astype(str).unique())
                ncols = 5
                nrows = int(np.ceil(len(fovs) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), squeeze=False)

                for ax, fov in zip(axes.ravel(), fovs):
                    df = obs_plot.loc[obs_plot[FOV_KEY].astype(str).eq(fov)]
                    ax.scatter(df[X_KEY], df[Y_KEY], s=0.2, color="#dddddd", linewidths=0, alpha=0.45)
                    selected = df.loc[df["territory_tubule_id"].astype(str).isin(selected_ids)]
                    for zone in ZONE_ORDER:
                        sub = selected.loc[selected["territory_zone"].astype(str).eq(zone)]
                        if sub.empty:
                            continue
                        ax.scatter(
                            sub[X_KEY],
                            sub[Y_KEY],
                            s=1.0,
                            color=ZONE_COLORS[zone],
                            linewidths=0,
                            alpha=1,
                        )
                    centers = tubule_component_summary.loc[
                        tubule_component_summary["tubule_core_id"].astype(str).isin(selected_ids)
                        & tubule_component_summary["fov"].eq(fov)
                    ]
                    for _, row in centers.iterrows():
                        ax.text(
                            row["x_center"],
                            row["y_center"],
                            row["tubule_core_id"].replace(f"{fov}_", ""),
                            fontsize=8,
                            ha="center",
                            va="center",
                            bbox=dict(facecolor="white", edgecolor="black", linewidth=0.3, alpha=0.8),
                        )
                    ax.set_title(fov_age_label(fov))
                    ax.set_aspect("equal")
                    ax.invert_yaxis()
                    ax.set_xticks([])
                    ax.set_yticks([])

                for ax in axes.ravel()[len(fovs):]:
                    ax.axis("off")

                fig.suptitle(f"{SAMPLE_ID}: selected three-tubule territories", y=1.01)
                add_figure_legend(fig, ZONE_COLORS, ZONE_ORDER, "territory_zone")
                fig.tight_layout(rect=[0, 0, 0.88, 1])
                if save_path is not None:
                    fig.savefig(save_path, dpi=220, bbox_inches="tight")
                return fig

            fig = plot_selected_tubule_territories(
                adata_spatial,
                selected_tubules,
                FIGURES_DIR / f"{SAMPLE_ID}_selected_three_tubule_territories.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Tubule-Level And Matched-Demo Metrics
            """
        ),
        code(
            """
            spatial_obs = adata_spatial.obs.copy()

            territory_counts = (
                spatial_obs.groupby(["fov", "age", "territory_tubule_id", "territory_zone", "anno_fine"], observed=True)
                .size()
                .rename("n_cells")
                .reset_index()
            )
            save_table(territory_counts, TABLES_DIR / f"{SAMPLE_ID}_territory_zone_fine_counts_long.csv", index=False)

            territory_wide = (
                spatial_obs.groupby(["fov", "age", "territory_tubule_id", "anno_fine"], observed=True)
                .size()
                .rename("n_cells")
                .reset_index()
                .pivot_table(
                    index=["fov", "age", "territory_tubule_id"],
                    columns="anno_fine",
                    values="n_cells",
                    fill_value=0,
                    observed=True,
                )
                .reindex(columns=NON_ARTIFACT_FINE_ORDER, fill_value=0)
                .reset_index()
            )
            territory_wide["total_cells"] = territory_wide[NON_ARTIFACT_FINE_ORDER].sum(axis=1)
            territory_wide["germ_to_sertoli_ratio"] = territory_wide["germ_cell"] / territory_wide["sertoli_cell"].replace(0, np.nan)
            territory_wide["leydig_fraction"] = territory_wide["leydig_cell"] / territory_wide["total_cells"].replace(0, np.nan)
            territory_wide["myoid_fraction"] = territory_wide["myoid_cell"] / territory_wide["total_cells"].replace(0, np.nan)
            territory_wide["vascular_fraction"] = territory_wide["vascular_cell"] / territory_wide["total_cells"].replace(0, np.nan)
            territory_wide["is_demo_tubule_territory"] = territory_wide["territory_tubule_id"].astype(str).isin(selected_ids)

            display(territory_wide.head())
            save_table(territory_wide, TABLES_DIR / f"{SAMPLE_ID}_territory_fine_summary.csv", index=False)

            demo_obs = spatial_obs.loc[spatial_obs["is_demo_tubule_territory"].astype(bool)].copy()
            demo_fine_counts = (
                pd.crosstab(demo_obs["fov_age"], demo_obs["anno_fine"])
                .reindex(fov_index)
                .reindex(columns=NON_ARTIFACT_FINE_ORDER, fill_value=0)
            )
            demo_zone_counts = (
                pd.crosstab(demo_obs["fov_age"], demo_obs["territory_zone"])
                .reindex(fov_index)
                .reindex(columns=ZONE_ORDER, fill_value=0)
            )

            display(demo_fine_counts)
            display(demo_zone_counts)
            save_table(demo_fine_counts, TABLES_DIR / f"{SAMPLE_ID}_selected_demo_tubules_anno_fine_counts.csv")
            save_table(demo_zone_counts, TABLES_DIR / f"{SAMPLE_ID}_selected_demo_tubules_zone_counts.csv")

            fig = plot_stacked_proportions(
                demo_fine_counts,
                FINE_COLORS,
                f"{SAMPLE_ID}: matched three-tubule territories, fine composition",
                "Fraction of selected-territory cells",
                FIGURES_DIR / f"{SAMPLE_ID}_selected_demo_tubules_anno_fine_proportions.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)

            fig = plot_stacked_proportions(
                demo_zone_counts,
                ZONE_COLORS,
                f"{SAMPLE_ID}: matched three-tubule territory zones",
                "Fraction of selected-territory cells",
                FIGURES_DIR / f"{SAMPLE_ID}_selected_demo_tubules_zone_proportions.png" if SAVE_OUTPUTS else None,
            )
            plt.show()
            plt.close(fig)
            """
        ),
        code(
            """
            demo_territory_summary = territory_wide.loc[territory_wide["is_demo_tubule_territory"]].copy()
            demo_fov_summary = (
                demo_territory_summary.groupby(["fov", "age"], observed=True)
                .agg(
                    n_selected_tubules=("territory_tubule_id", "nunique"),
                    mean_total_cells=("total_cells", "mean"),
                    mean_germ_to_sertoli_ratio=("germ_to_sertoli_ratio", "mean"),
                    mean_leydig_fraction=("leydig_fraction", "mean"),
                    mean_myoid_fraction=("myoid_fraction", "mean"),
                    mean_vascular_fraction=("vascular_fraction", "mean"),
                )
                .reset_index()
            )
            demo_fov_summary["age"] = pd.Categorical(demo_fov_summary["age"], categories=AGE_ORDER, ordered=True)
            demo_fov_summary = demo_fov_summary.sort_values("age")
            display(demo_fov_summary)
            save_table(demo_fov_summary, TABLES_DIR / f"{SAMPLE_ID}_selected_demo_tubules_fov_summary.csv", index=False)

            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
            metrics = [
                ("mean_germ_to_sertoli_ratio", "Mean germ/Sertoli ratio"),
                ("mean_myoid_fraction", "Mean myoid fraction"),
                ("mean_leydig_fraction", "Mean Leydig fraction"),
            ]
            for ax, (metric, label) in zip(axes, metrics):
                ax.plot(demo_fov_summary["age"].astype(str), demo_fov_summary[metric], marker="o", linewidth=1.5)
                ax.set_title(label)
                ax.set_xlabel("Age")
                ax.tick_params(axis="x", rotation=45)
            axes[0].set_ylabel("Value")
            fig.suptitle(f"{SAMPLE_ID}: matched three-tubule summary metrics", y=1.05)
            fig.tight_layout()
            if SAVE_OUTPUTS:
                fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_selected_demo_tubules_summary_metrics.png", dpi=220, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Distance To Nearest Tubule Core

            Distances are measured from each non-artifact cell to the nearest
            retained tubule-core cell in the same FOV. Tubule-core cells are
            assigned distance 0.
            """
        ),
        code(
            """
            distance_summary = (
                spatial_obs.groupby(["fov", "age", "anno_fine"], observed=True)
                .agg(
                    n_cells=("distance_to_tubule_px", "size"),
                    mean_distance_px=("distance_to_tubule_px", "mean"),
                    median_distance_px=("distance_to_tubule_px", "median"),
                    p25_distance_px=("distance_to_tubule_px", lambda x: np.nanpercentile(x, 25)),
                    p75_distance_px=("distance_to_tubule_px", lambda x: np.nanpercentile(x, 75)),
                )
                .reset_index()
            )
            distance_summary["age"] = pd.Categorical(distance_summary["age"], categories=AGE_ORDER, ordered=True)
            distance_summary = distance_summary.sort_values(["age", "anno_fine"])
            display(distance_summary)
            save_table(distance_summary, TABLES_DIR / f"{SAMPLE_ID}_distance_to_tubule_by_fov_celltype.csv", index=False)

            distance_plot_types = ["leydig_cell", "myoid_cell", "vascular_cell", "non_tubule"]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x_positions = np.arange(len(AGE_ORDER))
            for cell_type in distance_plot_types:
                sub = (
                    distance_summary.loc[distance_summary["anno_fine"].astype(str).eq(cell_type)]
                    .set_index("age")
                    .reindex(AGE_ORDER)
                )
                ax.plot(
                    x_positions,
                    sub["median_distance_px"],
                    marker="o",
                    linewidth=1.5,
                    label=cell_type,
                    color=FINE_COLORS.get(cell_type),
                )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(AGE_ORDER)
            ax.set_ylabel("Median distance to tubule core (px)")
            ax.set_xlabel("Age/FOV")
            ax.set_title(f"{SAMPLE_ID}: distance of non-tubule cell types to tubule core")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
            fig.tight_layout()
            if SAVE_OUTPUTS:
                fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_distance_to_tubule_by_celltype.png", dpi=220, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            """
        ),
        md(
            """
            ## Local Neighborhood Composition And Motifs

            This is a lightweight TissueSchematics-inspired layer. Each cell is
            represented by the composition of neighboring `anno_fine` cell types
            within a fixed radius. These neighborhood vectors are clustered into
            recurring tissue motifs.
            """
        ),
        code(
            """
            def neighborhood_composition_for_fov(df, categories, radius_px):
                xy = df[[X_KEY, Y_KEY]].to_numpy(float)
                labels = df["anno_fine"].astype(str).to_numpy()
                if len(df) == 0:
                    return pd.DataFrame(index=df.index), csr_matrix((0, 0))

                graph = radius_neighbors_graph(
                    xy,
                    radius=radius_px,
                    mode="connectivity",
                    include_self=False,
                    n_jobs=-1,
                ).tocsr()
                one_hot = np.zeros((len(df), len(categories)), dtype=float)
                cat_to_i = {cat: i for i, cat in enumerate(categories)}
                for row_i, label in enumerate(labels):
                    if label in cat_to_i:
                        one_hot[row_i, cat_to_i[label]] = 1.0
                counts = graph @ one_hot
                n_neighbors = counts.sum(axis=1)
                proportions = np.divide(
                    counts,
                    n_neighbors[:, None],
                    out=np.zeros_like(counts, dtype=float),
                    where=n_neighbors[:, None] > 0,
                )
                out = pd.DataFrame(
                    proportions,
                    index=df.index,
                    columns=[f"nh_{cat}_fraction" for cat in categories],
                )
                out["n_neighbors"] = n_neighbors
                return out, graph

            if RUN_NEIGHBORHOODS:
                neighborhood_frames = []
                adjacency_by_fov = {}
                for fov in FOV_ORDER:
                    df = spatial_obs.loc[spatial_obs[FOV_KEY].astype(str).eq(fov)].copy()
                    nh, graph = neighborhood_composition_for_fov(df, NON_ARTIFACT_FINE_ORDER, NEIGHBOR_RADIUS_PX)
                    neighborhood_frames.append(nh)
                    adjacency_by_fov[fov] = (df.index.to_numpy(), graph)

                neighborhood_table = pd.concat(neighborhood_frames, axis=0)
                for col in neighborhood_table.columns:
                    adata_spatial.obs[col] = neighborhood_table[col]

                motif_feature_cols = [f"nh_{cat}_fraction" for cat in NON_ARTIFACT_FINE_ORDER]
                motif_mask = adata_spatial.obs["n_neighbors"].fillna(0).to_numpy() >= MIN_NEIGHBORS_FOR_MOTIF
                motif_features = adata_spatial.obs.loc[motif_mask, motif_feature_cols].to_numpy(float)

                kmeans = KMeans(
                    n_clusters=N_NEIGHBORHOOD_MOTIFS,
                    random_state=RANDOM_STATE,
                    n_init=25,
                )
                motif_codes = kmeans.fit_predict(motif_features)
                motif_ids = [f"motif_{i + 1}" for i in range(N_NEIGHBORHOOD_MOTIFS)]
                motif_labels = pd.Series(pd.NA, index=adata_spatial.obs_names, dtype="object")
                motif_labels.loc[adata_spatial.obs_names[motif_mask]] = [motif_ids[i] for i in motif_codes]
                adata_spatial.obs["neighborhood_motif"] = pd.Categorical(motif_labels, categories=motif_ids)

                motif_centers = pd.DataFrame(kmeans.cluster_centers_, index=motif_ids, columns=NON_ARTIFACT_FINE_ORDER)
                motif_names = {}
                for motif_id, row in motif_centers.iterrows():
                    top = row.sort_values(ascending=False).head(2)
                    motif_names[motif_id] = f"{motif_id}: {top.index[0]}+{top.index[1]}"
                adata_spatial.obs["neighborhood_motif_label"] = (
                    adata_spatial.obs["neighborhood_motif"].astype(str).map(motif_names)
                )

                display(motif_centers)
                save_table(neighborhood_table, TABLES_DIR / f"{SAMPLE_ID}_neighborhood_composition_by_cell.csv")
                save_table(motif_centers, TABLES_DIR / f"{SAMPLE_ID}_neighborhood_motif_centers.csv")

                motif_counts = (
                    pd.crosstab(adata_spatial.obs["fov_age"], adata_spatial.obs["neighborhood_motif"])
                    .reindex(fov_index)
                    .reindex(columns=motif_ids, fill_value=0)
                )
                display(motif_counts)
                save_table(motif_counts, TABLES_DIR / f"{SAMPLE_ID}_neighborhood_motif_counts_by_fov.csv")

                if SAVE_OUTPUTS:
                    TERRITORY_H5AD = ADATA_DIR / f"{SAMPLE_ID}_spatial_tubule_territories.h5ad"
                    adata_spatial.write_h5ad(TERRITORY_H5AD, compression="gzip")
                    print(f"Updated: {TERRITORY_H5AD}")
            else:
                neighborhood_table = pd.DataFrame()
                motif_centers = pd.DataFrame()
                motif_counts = pd.DataFrame()
                adjacency_by_fov = {}
                print("Neighborhood motif analysis skipped.")
            """
        ),
        code(
            """
            if RUN_NEIGHBORHOODS and not motif_centers.empty:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                im = ax.imshow(motif_centers.to_numpy(), aspect="auto", cmap="viridis")
                ax.set_xticks(np.arange(len(motif_centers.columns)))
                ax.set_xticklabels(motif_centers.columns, rotation=45, ha="right")
                ax.set_yticks(np.arange(len(motif_centers.index)))
                ax.set_yticklabels([motif_names[motif_id] for motif_id in motif_centers.index])
                ax.set_title(f"{SAMPLE_ID}: local-neighborhood motif composition")
                fig.colorbar(im, ax=ax, label="Neighbor fraction")
                fig.tight_layout()
                if SAVE_OUTPUTS:
                    fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_neighborhood_motif_composition_heatmap.png", dpi=220, bbox_inches="tight")
                plt.show()
                plt.close(fig)

                motif_color_map = {
                    motif_id: plt.get_cmap("tab10")(i)
                    for i, motif_id in enumerate(motif_centers.index)
                }
                fig = plot_fov_grid(
                    adata_spatial.obs.dropna(subset=["neighborhood_motif"]).copy(),
                    color_key="neighborhood_motif",
                    color_map=motif_color_map,
                    order=list(motif_centers.index),
                    title=f"{SAMPLE_ID}: spatial map of local-neighborhood motifs",
                    point_size=1,
                    alpha=1,
                    save_path=FIGURES_DIR / f"{SAMPLE_ID}_neighborhood_motifs_spatial_by_fov.png" if SAVE_OUTPUTS else None,
                )
                plt.show()
                plt.close(fig)

                fig = plot_stacked_proportions(
                    motif_counts,
                    motif_color_map,
                    f"{SAMPLE_ID}: neighborhood motif abundance by FOV",
                    "Fraction of motif-assigned cells",
                    FIGURES_DIR / f"{SAMPLE_ID}_neighborhood_motif_proportions_by_fov.png" if SAVE_OUTPUTS else None,
                )
                plt.show()
                plt.close(fig)
            """
        ),
        md(
            """
            ## SCIMAP-Style Cell-Type Proximity Enrichment

            This section computes a simple observed/expected enrichment of
            neighboring cell-type pairs within `NEIGHBOR_RADIUS_PX`. Positive
            log2 enrichment means the two cell types are neighbors more often
            than expected from their FOV-level frequencies.
            """
        ),
        code(
            """
            def pair_key(a, b):
                return tuple(sorted((str(a), str(b))))

            def proximity_enrichment_for_fov(df, graph, categories):
                labels = df["anno_fine"].astype(str).to_numpy()
                valid = np.isin(labels, categories)
                category_counts = pd.Series(labels[valid]).value_counts().reindex(categories, fill_value=0)
                p = category_counts / category_counts.sum()

                coo = graph.tocoo()
                edge_records = []
                for i, j in zip(coo.row, coo.col):
                    if i >= j:
                        continue
                    if labels[i] not in categories or labels[j] not in categories:
                        continue
                    edge_records.append(pair_key(labels[i], labels[j]))

                pair_counts = pd.Series(edge_records).value_counts()
                total_edges = pair_counts.sum()
                records = []
                for a in categories:
                    for b in categories:
                        if categories.index(a) > categories.index(b):
                            continue
                        observed_edges = int(pair_counts.get(pair_key(a, b), 0))
                        observed_fraction = observed_edges / total_edges if total_edges else np.nan
                        expected_fraction = (p[a] ** 2) if a == b else (2 * p[a] * p[b])
                        enrichment = observed_fraction / expected_fraction if expected_fraction > 0 else np.nan
                        records.append(
                            {
                                "cell_type_a": a,
                                "cell_type_b": b,
                                "pair": f"{a}__{b}",
                                "observed_edges": observed_edges,
                                "observed_fraction": observed_fraction,
                                "expected_fraction": expected_fraction,
                                "obs_over_exp": enrichment,
                                "log2_obs_over_exp": np.log2(enrichment) if enrichment > 0 else np.nan,
                            }
                        )
                return pd.DataFrame(records)

            if RUN_NEIGHBORHOODS:
                enrichment_tables = []
                for fov in FOV_ORDER:
                    cell_index, graph = adjacency_by_fov[fov]
                    df = spatial_obs.loc[cell_index].copy()
                    enrich = proximity_enrichment_for_fov(df, graph, NON_ARTIFACT_FINE_ORDER)
                    enrich["fov"] = fov
                    enrich["age"] = AGE_BY_FOV[fov]
                    enrichment_tables.append(enrich)

                proximity_enrichment = pd.concat(enrichment_tables, axis=0, ignore_index=True)
                proximity_enrichment["age"] = pd.Categorical(
                    proximity_enrichment["age"],
                    categories=AGE_ORDER,
                    ordered=True,
                )
                display(proximity_enrichment.head())
                save_table(
                    proximity_enrichment,
                    TABLES_DIR / f"{SAMPLE_ID}_celltype_proximity_enrichment_by_fov.csv",
                    index=False,
                )

                selected_pair_keys = []
                for a, b in PROXIMITY_PAIR_TYPES:
                    if NON_ARTIFACT_FINE_ORDER.index(a) <= NON_ARTIFACT_FINE_ORDER.index(b):
                        selected_pair_keys.append(f"{a}__{b}")
                    else:
                        selected_pair_keys.append(f"{b}__{a}")
                selected_enrichment = proximity_enrichment.loc[
                    proximity_enrichment["pair"].isin(selected_pair_keys)
                ].copy()
                selected_enrichment = selected_enrichment.sort_values(["pair", "age"])
                display(selected_enrichment)
                save_table(
                    selected_enrichment,
                    TABLES_DIR / f"{SAMPLE_ID}_selected_celltype_proximity_enrichment.csv",
                    index=False,
                )

                fig, ax = plt.subplots(figsize=(8, 4.5))
                x_positions = np.arange(len(AGE_ORDER))
                for pair, sub in selected_enrichment.groupby("pair", observed=True):
                    sub = sub.set_index("age").reindex(AGE_ORDER)
                    ax.plot(
                        x_positions,
                        sub["log2_obs_over_exp"],
                        marker="o",
                        linewidth=1.5,
                        label=pair.replace("__", " - "),
                    )
                ax.set_xticks(x_positions)
                ax.set_xticklabels(AGE_ORDER)
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylabel("log2 observed/expected proximity")
                ax.set_xlabel("Age/FOV")
                ax.set_title(f"{SAMPLE_ID}: selected cell-type proximity enrichment")
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                fig.tight_layout()
                if SAVE_OUTPUTS:
                    fig.savefig(FIGURES_DIR / f"{SAMPLE_ID}_selected_celltype_proximity_enrichment.png", dpi=220, bbox_inches="tight")
                plt.show()
                plt.close(fig)
            else:
                proximity_enrichment = pd.DataFrame()
                print("Proximity enrichment skipped because RUN_NEIGHBORHOODS=False.")
            """
        ),
        md(
            """
            ## Outputs

            Main outputs are written under:

            `spatial_analysis/tubule_territories/`

            Key files:

            - `adata/ST_TMAG_E1_31_spatial_tubule_territories.h5ad`
            - `tables/ST_TMAG_E1_31_whole_fov_anno_fine_proportions.csv`
            - `tables/ST_TMAG_E1_31_tubule_component_summary.csv`
            - `tables/ST_TMAG_E1_31_selected_demo_tubules.csv`
            - `tables/ST_TMAG_E1_31_selected_demo_tubules_fov_summary.csv`
            - `tables/ST_TMAG_E1_31_distance_to_tubule_by_fov_celltype.csv`
            - `tables/ST_TMAG_E1_31_neighborhood_motif_centers.csv`
            - `tables/ST_TMAG_E1_31_celltype_proximity_enrichment_by_fov.csv`

            First refinement pass:

            1. Inspect `ST_TMAG_E1_31_tubule_candidate_preview.png`.
            2. Inspect `ST_TMAG_E1_31_selected_three_tubule_territories.png`.
            3. If a selected tubule is merged or visually poor, edit
               `SELECTED_TUBULE_IDS_BY_FOV` near the top and rerun from
               "Select Three Neighboring Tubules Per FOV" onward.
            """
        ),
    ]


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
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
