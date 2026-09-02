"""Rewrite the core QC and phenotyping notebooks as clean public-API tutorials."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT  / "paper" / "notebooks"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


COMMON_SETUP = r"""
from pathlib import Path
import importlib.util

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

import spatioev as sv

ROOT = Path.cwd()
if not (ROOT / "spatioev").exists():
    for candidate in ROOT.parents:
        if (candidate / "spatioev").exists():
            ROOT = candidate
            break

DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

EXP1_H5AD = DATA_DIR / "exp_1" / "exp_1.h5ad"
ANNOTATION_PATH = RESULT_DIR / "phenotyping_annotations.csv"

MARKER_CANDIDATES = [
    "DNA_1", "PCK", "CK19", "NaKATPase",
    "CD45", "CD3", "CD4", "CD68", "CD11c", "CD14", "CD19", "HLADR",
    "Vimentin", "SMA", "FN", "COL6A1", "COL4A1", "CD90", "PDPN", "Tenascin",
    "CD31", "CD146",
]

MORPH_FEATURES = [
    "area",
    "convex_area",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "feret_diameter_max",
    "equivalent_diameter",
    "num_concavities",
    "centroid_dif",
    "eccentricity",
    "solidity",
    "major_minor_axis_ratio",
    "perim_square_over_area",
    "major_axis_equiv_diam_ratio",
    "convex_hull_resid",
    "circularity",
    "fractal_dimension",
    "boundary_irregularity",
    "nc_ratio",
]

LINEAGE_MARKER_SETS = {
    "epithelial": ["PCK", "CK19", "NaKATPase"],
    "immune": ["CD45", "CD3", "CD4", "CD68", "CD11c", "CD14", "CD19", "HLADR"],
    "fibroblast/stroma": ["Vimentin", "SMA", "FN", "COL6A1", "COL4A1", "CD90", "PDPN", "Tenascin"],
    "endothelial": ["CD31", "CD146"],
}

IMMUNE_SUBTYPE_MARKERS = {
    "T cells": ["CD3"],
    "CD4 T cells": ["CD3", "CD4"],
    "macrophage/monocyte": ["CD68", "CD14", "HLADR"],
    "dendritic/myeloid": ["CD11c", "HLADR"],
    "B lineage": ["CD19"],
}

FIBROBLAST_SUBTYPE_MARKERS = {
    "myofibroblast-like": ["SMA", "COL1A1", "COL6A1"],
    "matrix fibroblast": ["FN", "COL4A1", "COL6A1", "Tenascin"],
    "PDPN+ fibroblast": ["PDPN", "Vimentin"],
    "CD90+ fibroblast": ["CD90", "Vimentin"],
}


def present(values, candidates):
    return [item for item in candidates if item in values]


def dense_matrix(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def expression_frame(adata, markers):
    markers = present(adata.var_names, markers)
    if not markers:
        return pd.DataFrame(index=adata.obs_names)
    x = dense_matrix(adata[:, markers].X)
    return pd.DataFrame(x, index=adata.obs_names, columns=markers)


def zscore_frame(df):
    if df.empty:
        return df.copy()
    out = df.astype(float).copy()
    std = out.std(axis=0).replace(0, np.nan)
    out = (out - out.mean(axis=0)) / std
    return out.fillna(0.0)


def make_synthetic_adata(n_cells=1200, seed=7):
    rng = np.random.default_rng(seed)
    phenotypes = rng.choice(
        ["epithelial", "immune", "fibroblast/stroma", "endothelial"],
        size=n_cells,
        p=[0.35, 0.28, 0.30, 0.07],
    )
    obs = pd.DataFrame(
        {
            "label": np.arange(n_cells),
            "imageid": rng.choice(["synthetic_fov_0", "synthetic_fov_1"], size=n_cells),
            "X_centroid": rng.normal(500, 180, size=n_cells),
            "Y_centroid": rng.normal(500, 160, size=n_cells),
            "area": rng.gamma(8, 22, size=n_cells),
            "eccentricity": rng.uniform(0.2, 0.9, size=n_cells),
            "major_axis_length": rng.uniform(15, 55, size=n_cells),
            "minor_axis_length": rng.uniform(6, 25, size=n_cells),
            "perimeter": rng.uniform(50, 170, size=n_cells),
            "convex_area": rng.gamma(9, 24, size=n_cells),
            "equivalent_diameter": rng.uniform(10, 35, size=n_cells),
            "solidity": rng.uniform(0.75, 0.99, size=n_cells),
            "feret_diameter_max": rng.uniform(16, 65, size=n_cells),
            "major_minor_axis_ratio": rng.uniform(1.1, 3.2, size=n_cells),
            "perim_square_over_area": rng.uniform(12, 22, size=n_cells),
            "major_axis_equiv_diam_ratio": rng.uniform(1.0, 2.2, size=n_cells),
            "convex_hull_resid": rng.uniform(0.0, 0.35, size=n_cells),
            "centroid_dif": rng.uniform(0.0, 4.0, size=n_cells),
            "num_concavities": rng.integers(0, 8, size=n_cells),
            "circularity": rng.uniform(0.3, 0.95, size=n_cells),
            "fractal_dimension": rng.uniform(1.0, 1.45, size=n_cells),
            "boundary_irregularity": rng.uniform(0.0, 0.8, size=n_cells),
            "nc_ratio": rng.beta(5, 2, size=n_cells),
            "manual_lineage": phenotypes,
        },
        index=[f"synthetic_cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=MARKER_CANDIDATES)
    x = rng.normal(size=(n_cells, len(var)))
    for marker in ["PCK", "CK19", "NaKATPase"]:
        x[:, var.index.get_loc(marker)] += (phenotypes == "epithelial") * 2.2
    for marker in ["CD45", "CD3", "CD4", "CD68", "CD11c", "CD14", "CD19", "HLADR"]:
        x[:, var.index.get_loc(marker)] += (phenotypes == "immune") * 1.7
    for marker in ["Vimentin", "SMA", "FN", "COL6A1", "COL4A1", "CD90", "PDPN", "Tenascin"]:
        x[:, var.index.get_loc(marker)] += (phenotypes == "fibroblast/stroma") * 1.7
    for marker in ["CD31", "CD146"]:
        x[:, var.index.get_loc(marker)] += (phenotypes == "endothelial") * 2.2
    return ad.AnnData(x, obs=obs, var=var)


def ensure_demo_columns(adata):
    rng = np.random.default_rng(11)
    obs = adata.obs
    n = adata.n_obs
    if "imageid" not in obs.columns:
        obs["imageid"] = obs["fov"].astype(str) if "fov" in obs.columns else "image_0"
    if "X_centroid" not in obs.columns:
        obs["X_centroid"] = rng.normal(500, 120, size=n)
    if "Y_centroid" not in obs.columns:
        obs["Y_centroid"] = rng.normal(500, 120, size=n)
    if "fractal_dimension" not in obs.columns and "fractual_dimension" in obs.columns:
        obs["fractal_dimension"] = obs["fractual_dimension"]
    for col in MORPH_FEATURES:
        if col not in obs.columns:
            if col == "nc_ratio":
                obs[col] = rng.beta(5, 2, size=n)
            elif col in {"eccentricity", "solidity", "circularity", "fractal_dimension", "boundary_irregularity"}:
                obs[col] = rng.uniform(0.2, 0.95, size=n)
            else:
                obs[col] = rng.gamma(8, 20, size=n)
    return adata


def load_example_adata(max_cells=5000, seed=13):
    if EXP1_H5AD.exists():
        backed = ad.read_h5ad(EXP1_H5AD, backed="r")
        if backed.n_obs > max_cells:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(backed.n_obs, max_cells, replace=False))
            adata = backed[idx].to_memory()
        else:
            adata = backed.to_memory()
        source = f"local example data: {EXP1_H5AD.relative_to(ROOT)}"
    else:
        adata = make_synthetic_adata(n_cells=max_cells, seed=seed)
        source = "synthetic fallback data"
    adata = ensure_demo_columns(adata)
    return adata, source


def load_saved_annotations(adata, path=ANNOTATION_PATH):
    if not path.exists():
        return adata
    annotations = pd.read_csv(path, index_col=0)
    overlap = adata.obs_names.intersection(annotations.index)
    for col in annotations.columns:
        adata.obs.loc[overlap, col] = annotations.loc[overlap, col].astype(str)
    return adata


def add_marker_obs(adata, markers):
    markers = present(adata.var_names, markers)
    if markers:
        adata = sv.pp.add_obs_from_var(adata, markers, overwrite=True)
    return adata


def score_marker_sets(adata, marker_sets):
    expr_z = zscore_frame(expression_frame(adata, sorted({m for markers in marker_sets.values() for m in markers})))
    scores = pd.DataFrame(index=adata.obs_names)
    for label, marker_list in marker_sets.items():
        cols = present(expr_z.columns, marker_list)
        if cols:
            scores[label] = expr_z[cols].mean(axis=1)
    return scores


def weak_lineage_labels(adata):
    scores = score_marker_sets(adata, LINEAGE_MARKER_SETS)
    if scores.empty:
        return pd.Series("Unknown", index=adata.obs_names)
    labels = scores.idxmax(axis=1)
    confidence = scores.max(axis=1) - scores.apply(lambda row: row.nlargest(min(2, len(row))).iloc[-1], axis=1)
    labels = labels.where(confidence >= -0.25, "Unknown")
    return labels.astype(str)


def assign_cluster_lineages(clustered, markers, cluster_key="leiden"):
    expr_z = zscore_frame(expression_frame(clustered, markers))
    if expr_z.empty or cluster_key not in clustered.obs.columns:
        clustered.obs["annotation"] = weak_lineage_labels(clustered)
        return pd.DataFrame()
    cluster_means = expr_z.groupby(clustered.obs[cluster_key].astype(str)).mean()
    cluster_scores = pd.DataFrame(index=cluster_means.index)
    for label, marker_list in LINEAGE_MARKER_SETS.items():
        cols = present(cluster_means.columns, marker_list)
        if cols:
            cluster_scores[label] = cluster_means[cols].mean(axis=1)
    mapping = cluster_scores.idxmax(axis=1).to_dict() if not cluster_scores.empty else {}
    clustered.obs["annotation"] = clustered.obs[cluster_key].astype(str).map(mapping).fillna("Unknown")
    return cluster_scores


def subtype_by_marker_scores(adata, marker_sets, default_label):
    scores = score_marker_sets(adata, marker_sets)
    if scores.empty:
        return pd.Series(default_label, index=adata.obs_names), scores
    labels = scores.idxmax(axis=1).astype(str)
    confidence = scores.max(axis=1) - scores.apply(lambda row: row.nlargest(min(2, len(row))).iloc[-1], axis=1)
    labels = labels.where(confidence >= -0.1, default_label)
    return labels, scores


def safe_spatial_scatter(adata, color, title=None, size=2):
    if color not in adata.obs.columns and color not in adata.var_names:
        print(f"Skipping spatial plot: {color!r} is not available.")
        return None
    fig = sv.pl.spatial_scatter_plot(
        adata,
        colorBy=color,
        x_coordinate="X_centroid",
        y_coordinate="Y_centroid",
        imageid="imageid",
        s=size,
        figsize=(5, 5),
    )
    if title:
        plt.suptitle(title)
    plt.show()
    plt.close("all")
    return fig


def show_counts(series, label):
    counts = series.astype(str).value_counts().rename_axis(label).to_frame("n_cells")
    display(counts.head(20))
    return counts


adata, data_source = load_example_adata()
markers = present(adata.var_names, MARKER_CANDIDATES)
adata = add_marker_obs(adata, markers)
print(f"SpatioEv {sv.__version__}")
print(f"Using {data_source}: {adata.n_obs:,} cells x {adata.n_vars:,} markers")
print("Detected marker panel:", markers)
"""


def notebook(title: str, intro: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(f"# {title}\n\n{intro}"),
        code(COMMON_SETUP),
        *cells,
    ]
    return nb


def build_notebooks() -> dict[str, nbf.NotebookNode]:
    return {
        "00_dev_seg_qc_testing.ipynb": notebook(
            "SpatioEv Tutorial 00: Segmentation QC",
            """
            This refined notebook turns the original segmentation-QC scratchpad into a reproducible first
            tutorial. The goal is to flag cells whose segmentation-derived morphology is biologically
            implausible before downstream phenotyping or spatial statistics. Very small cells often
            represent debris or fragmented masks; very large cells can represent merged objects; extreme
            nuclear-to-cell ratios often indicate nucleus/cytoplasm mask mismatch.
            """,
            [
                md("## Configure QC thresholds"),
                code("""
                qc_config = sv.pp.QCConfig(
                    pixel_size=0.325,
                    min_area_um2=10,
                    max_area_um2=1000,
                    max_nc_ratio=1.0,
                )
                print(qc_config)
                """),
                md("## Run segmentation QC"),
                code("""
                required = ["area", "nc_ratio"]
                missing = [col for col in required if col not in adata.obs.columns]
                if missing:
                    raise ValueError(f"Missing required QC columns: {missing}")

                adata_qc = sv.pp.run_segmentation_qc(adata.copy(), qc_config)
                display(adata_qc.obs[["area", "area_um2", "area_category", "nc_ratio", "nc_ratio_category"]].head())
                """),
                md("## Interpret flagged cells"),
                code("""
                area_counts = show_counts(adata_qc.obs["area_category"], "area_category")
                nc_counts = show_counts(adata_qc.obs["nc_ratio_category"], "nc_ratio_category")

                n_flagged = (
                    (adata_qc.obs["area_category"] != "normal_area")
                    | (adata_qc.obs["nc_ratio_category"] != "normal_nc_ratio")
                ).sum()
                print(f"Flagged cells: {n_flagged:,} / {adata_qc.n_obs:,} ({n_flagged / adata_qc.n_obs:.1%})")
                """),
                md("## Visualize QC distributions"),
                code("""
                sv.pl.plot_area_distribution(
                    adata_qc,
                    min_area=qc_config.min_area_um2,
                    max_area=qc_config.max_area_um2,
                )
                plt.show()
                plt.close("all")

                sv.pl.plot_nc_ratio_distribution(
                    adata_qc,
                    max_ratio=qc_config.max_nc_ratio,
                )
                plt.show()
                plt.close("all")
                """),
                md("## Filter cells and summarize by image"),
                code("""
                adata_filtered = sv.pp.filter_segmentation_errors(adata_qc)
                summary = sv.pp.generate_qc_summary(adata_qc, groupby="imageid")
                display(summary.head())

                print(f"Original cells: {adata_qc.n_obs:,}")
                print(f"Retained cells:  {adata_filtered.n_obs:,}")
                print(f"Removed cells:   {adata_qc.n_obs - adata_filtered.n_obs:,}")
                """),
                md("## Save lightweight QC outputs"),
                code("""
                qc_summary_path = RESULT_DIR / "segmentation_qc_summary.csv"
                summary.to_csv(qc_summary_path, index=False)
                print(f"Wrote {qc_summary_path.relative_to(ROOT)}")
                """),
            ],
        ),
        "01_dev_clustering_based_phenotyping_test.ipynb": notebook(
            "SpatioEv Tutorial 01: Clustering-Based Phenotyping",
            """
            This notebook keeps the original human-in-the-loop phenotyping idea but makes the logic explicit:
            build a marker feature space, cluster cells, assign broad biological lineages from marker-enrichment
            scores, then refine heterogeneous lineages into more specific annotations. Interactive image review
            remains encouraged, but the notebook now has a deterministic non-interactive path.
            """,
            [
                md("## Select a level-0 marker panel"),
                code("""
                level0_markers = present(markers, [
                    "PCK", "CK19", "NaKATPase",
                    "CD45", "CD3", "CD68", "CD19",
                    "SMA", "FN", "COL6A1", "COL4A1", "CD90", "PDPN", "Vimentin",
                    "CD31", "CD146",
                ])
                level0_config = sv.pp.ClusteringConfig(
                    markers=level0_markers,
                    resolution=0.8,
                    n_neighbors=12,
                    n_pcs=min(10, max(2, len(level0_markers) - 1)),
                )
                print(level0_config)
                """),
                md("## Cluster cells"),
                code("""
                adata_norm = sv.pp.zscore_normalize(adata.copy())
                try:
                    adata_lvl0 = sv.tl.cluster_cells(adata_norm, level0_config)
                    print("Scanpy clustering completed.")
                except Exception as exc:
                    print(f"Scanpy clustering skipped; using marker-score fallback. Reason: {exc}")
                    adata_lvl0 = adata.copy()
                    weak = weak_lineage_labels(adata_lvl0)
                    adata_lvl0.obs["leiden"] = pd.Categorical(pd.factorize(weak)[0].astype(str))

                cluster_scores = assign_cluster_lineages(adata_lvl0, level0_markers)
                display(cluster_scores)
                show_counts(adata_lvl0.obs["annotation"], "annotation")
                """),
                md("## Plot marker enrichment and spatial distribution"),
                code("""
                if "X_umap" in adata_lvl0.obsm and importlib.util.find_spec("scanpy") is not None:
                    import scanpy as sc
                    sc.pl.umap(adata_lvl0, color=["leiden", "annotation"], wspace=0.4)
                    plt.close("all")

                try:
                    sv.pl.plot_cluster_heatmap(adata_lvl0, level0_markers, cluster_key="leiden")
                    plt.show()
                    plt.close("all")
                except Exception as exc:
                    print(f"Cluster heatmap skipped: {exc}")

                safe_spatial_scatter(adata_lvl0, "annotation", "Broad phenotype annotations")
                """),
                md("## Refine heterogeneous lineages"),
                code("""
                refinement_configs = {
                    "immune": sv.pp.ClusteringConfig(
                        markers=present(markers, ["CD45", "CD3", "CD4", "CD68", "CD11c", "CD14", "CD19", "HLADR"]),
                        resolution=0.7,
                        n_neighbors=10,
                        n_pcs=6,
                    ),
                    "fibroblast/stroma": sv.pp.ClusteringConfig(
                        markers=present(markers, ["Vimentin", "SMA", "FN", "COL6A1", "COL4A1", "CD90", "PDPN", "Tenascin"]),
                        resolution=0.7,
                        n_neighbors=10,
                        n_pcs=6,
                    ),
                }
                clusters_to_refine = [
                    label for label in refinement_configs
                    if label in set(adata_lvl0.obs["annotation"].astype(str))
                    and len(refinement_configs[label].markers) >= 2
                ]

                refined = {}
                try:
                    refined = sv.tl.refine_clusters(
                        adata_lvl0,
                        clusters_to_refine,
                        {key: refinement_configs[key] for key in clusters_to_refine},
                        annotation_key="annotation",
                    )
                except Exception as exc:
                    print(f"Automated sub-clustering skipped: {exc}")

                for lineage, subset in refined.items():
                    subtype_sets = IMMUNE_SUBTYPE_MARKERS if lineage == "immune" else FIBROBLAST_SUBTYPE_MARKERS
                    labels, subtype_scores = subtype_by_marker_scores(subset, subtype_sets, default_label=lineage)
                    subset.obs["annotation"] = labels.astype(str)
                    print(f"{lineage}: {subset.n_obs:,} cells")
                    display(subtype_scores.head())

                adata_final = sv.tl.merge_refinements(adata_lvl0.copy(), refined, new_key="annotation_level2")
                if "annotation_level2" not in adata_final.obs:
                    adata_final.obs["annotation_level2"] = adata_final.obs["annotation"].astype(str)
                show_counts(adata_final.obs["annotation_level2"], "annotation_level2")
                """),
                md("## Save phenotyping annotations"),
                code("""
                output = adata_final.obs[["annotation", "annotation_level2"]].copy()
                output.to_csv(ANNOTATION_PATH)
                print(f"Wrote {ANNOTATION_PATH.relative_to(ROOT)}")
                safe_spatial_scatter(adata_final, "annotation_level2", "Refined phenotypes")
                """),
            ],
        ),
        "01_dev_scimap_phenotype_workflow.ipynb": notebook(
            "SpatioEv Tutorial 02: Rule-Based Phenotyping With Optional Scimap",
            """
            The original notebook used Scimap gate finding and phenotype workflow tables. This refined version
            keeps that intent but separates optional interactive Scimap steps from a reproducible marker-rule
            path. The rule-based outputs can be reviewed spatially and then fed directly into downstream SpatioEv
            analyses or the SVM probability notebook.
            """,
            [
                md("## Load broad annotations from the clustering tutorial"),
                code("""
                adata = load_saved_annotations(adata)
                if "annotation_level2" not in adata.obs.columns:
                    adata.obs["annotation_level2"] = weak_lineage_labels(adata)
                    print("No saved clustering annotations found; using weak marker-score lineages.")

                show_counts(adata.obs["annotation_level2"], "annotation_level2")
                """),
                md("## Optional Scimap resources"),
                code("""
                scimap_available = importlib.util.find_spec("scimap") is not None
                print("scimap available:", scimap_available)

                workflow_paths = {
                    "immune": DATA_DIR / "exp_1" / "immune_phenotype_workflow.csv",
                    "fibroblast": DATA_DIR / "exp_1" / "fibroblast_phenotype_workflow.csv",
                    "manual_gates": DATA_DIR / "exp_1" / "manual_gates.csv",
                }
                for name, path in workflow_paths.items():
                    print(f"{name}: {path.exists()} ({path.relative_to(ROOT) if path.exists() else path})")

                if workflow_paths["immune"].exists():
                    display(pd.read_csv(workflow_paths["immune"]).head())
                if workflow_paths["fibroblast"].exists():
                    display(pd.read_csv(workflow_paths["fibroblast"]).head())
                """),
                md("## Subset immune and fibroblast compartments"),
                code("""
                labels = adata.obs["annotation_level2"].astype(str)
                immune_labels = [x for x in labels.unique() if any(token in x.lower() for token in ["immune", "t cell", "macrophage", "b lineage", "myeloid"])]
                fibroblast_labels = [x for x in labels.unique() if any(token in x.lower() for token in ["fibro", "stroma", "myofibro", "matrix", "pdpn", "cd90"])]

                immune_cells = sv.tl.subset_cells(adata, "annotation_level2", immune_labels) if immune_labels else adata[weak_lineage_labels(adata) == "immune"].copy()
                fibroblasts = sv.tl.subset_cells(adata, "annotation_level2", fibroblast_labels) if fibroblast_labels else adata[weak_lineage_labels(adata) == "fibroblast/stroma"].copy()

                print(f"Immune subset: {immune_cells.n_obs:,} cells")
                print(f"Fibroblast/stroma subset: {fibroblasts.n_obs:,} cells")
                """),
                md("## Apply reproducible marker-rule phenotyping"),
                code("""
                adata.obs["annotated_clusters_update3"] = adata.obs["annotation_level2"].astype(str)

                if immune_cells.n_obs:
                    immune_labels, immune_scores = subtype_by_marker_scores(immune_cells, IMMUNE_SUBTYPE_MARKERS, default_label="immune")
                    adata.obs.loc[immune_cells.obs_names, "annotated_clusters_update3"] = immune_labels.astype(str)
                    display(immune_scores.head())

                if fibroblasts.n_obs:
                    fibro_labels, fibro_scores = subtype_by_marker_scores(fibroblasts, FIBROBLAST_SUBTYPE_MARKERS, default_label="fibroblast/stroma")
                    adata.obs.loc[fibroblasts.obs_names, "annotated_clusters_update3"] = fibro_labels.astype(str)
                    display(fibro_scores.head())

                show_counts(adata.obs["annotated_clusters_update3"], "annotated_clusters_update3")
                """),
                md("## Review and save the updated phenotype labels"),
                code("""
                safe_spatial_scatter(adata, "annotated_clusters_update3", "Rule-based refined phenotypes")

                annotation_cols = present(
                    adata.obs.columns,
                    ["annotation", "annotation_level2", "annotated_clusters_update3"],
                )
                adata.obs[annotation_cols].to_csv(ANNOTATION_PATH)
                print(f"Wrote {ANNOTATION_PATH.relative_to(ROOT)}")
                """),
            ],
        ),
        "02_dev_SVM_phenotype_probility.ipynb": notebook(
            "SpatioEv Tutorial 03: SVM Phenotype Probabilities",
            """
            This notebook refines the original SVM probability workflow. The aim is not to replace expert
            annotation, but to propagate reviewed labels to ambiguous or unlabelled cells while keeping
            probability scores for uncertainty-aware downstream analysis.
            """,
            [
                md("## Load manual or rule-based annotations"),
                code("""
                adata = load_saved_annotations(adata)
                label_source = None
                for candidate in ["annotated_clusters_update3", "annotation_level2", "annotation", "manual_lineage"]:
                    if candidate in adata.obs.columns:
                        label_source = candidate
                        break

                if label_source is None:
                    adata.obs["manual_label"] = weak_lineage_labels(adata)
                    label_source = "manual_label"
                else:
                    adata.obs["manual_label"] = adata.obs[label_source].astype(str)

                adata.obs["manual_label"] = adata.obs["manual_label"].replace({"Unknown": "unknown", "nan": "unknown", "None": "unknown"})
                show_counts(adata.obs["manual_label"], "manual_label")
                print("Label source:", label_source)
                """),
                md("## Choose SVM features"),
                code("""
                svm_markers = present(markers, [
                    "PCK", "CK19", "NaKATPase",
                    "CD45", "CD3", "CD4", "CD68", "CD11c", "CD14", "CD19", "HLADR",
                    "Vimentin", "SMA", "FN", "COL6A1", "COL4A1", "CD90", "PDPN",
                    "CD31", "CD146",
                ])

                counts = adata.obs["manual_label"].astype(str).value_counts()
                trainable_labels = counts[(counts >= 10) & (~counts.index.isin(["unknown", "noise"]))].index
                adata.obs["svm_training_label"] = np.where(
                    adata.obs["manual_label"].isin(trainable_labels),
                    adata.obs["manual_label"].astype(str),
                    "unknown",
                )

                if len(trainable_labels) < 2:
                    print("Not enough reviewed classes; falling back to weak marker-score labels.")
                    weak = weak_lineage_labels(adata)
                    counts = weak.value_counts()
                    trainable_labels = counts[counts >= 10].index
                    adata.obs["svm_training_label"] = np.where(weak.isin(trainable_labels), weak, "unknown")

                print("SVM markers:", svm_markers)
                show_counts(adata.obs["svm_training_label"], "svm_training_label")
                """),
                md("## Train the SVM and predict all cells"),
                code("""
                if len(set(adata.obs["svm_training_label"]) - {"unknown"}) < 2:
                    raise ValueError("SVM training requires at least two labelled classes.")

                svm_adata, model, report = sv.tl.run_svm_phenotyping(
                    adata.copy(),
                    markers=svm_markers,
                    label_key="svm_training_label",
                    morph_weight=0.4,
                )

                report_df = pd.DataFrame(report).transpose()
                display(report_df)
                show_counts(svm_adata.obs["svm_prediction"], "svm_prediction")
                """),
                md("## Inspect uncertainty and disagreements"),
                code("""
                prob_cols = [col for col in svm_adata.obs.columns if col.startswith("svm_prob_")]
                svm_adata.obs["svm_max_probability"] = svm_adata.obs[prob_cols].max(axis=1)
                svm_adata.obs["svm_prediction_confident"] = np.where(
                    svm_adata.obs["svm_max_probability"] >= 0.60,
                    svm_adata.obs["svm_prediction"].astype(str),
                    "uncertain",
                )

                display(
                    pd.crosstab(
                        svm_adata.obs["svm_training_label"],
                        svm_adata.obs["svm_prediction"],
                        normalize="index",
                    ).round(3)
                )
                display(svm_adata.obs[["svm_prediction", "svm_max_probability", "svm_prediction_confident"]].head())
                """),
                md("## Spatially review predictions and save results"),
                code("""
                safe_spatial_scatter(svm_adata, "svm_prediction_confident", "SVM predictions with uncertainty threshold")

                svm_result_cols = [
                    "manual_label",
                    "svm_training_label",
                    "svm_prediction",
                    "svm_max_probability",
                    "svm_prediction_confident",
                ]
                svm_adata.obs[svm_result_cols].to_csv(RESULT_DIR / "svm_phenotyping_results.csv")
                svm_adata.obs[prob_cols].to_csv(RESULT_DIR / "svm_phenotyping_probabilities.csv")
                print("Wrote SVM phenotype results and probability table to results/.")
                """),
            ],
        ),
    }


def main() -> None:
    for filename, nb in build_notebooks().items():
        out = NOTEBOOK_DIR / filename
        nbf.write(nb, out)
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
