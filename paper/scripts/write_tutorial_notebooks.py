"""Generate the expanded SpatioEv tutorial notebook series.

The notebooks are narrative tutorials, not just smoke tests. They explain the
purpose of the public API, run lightweight examples, and use the local example
dataset when it is present.
"""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = ROOT / "tutorials"
DOCS_DIR = ROOT / "docs"


CURATED_PURPOSE = {
    "QCConfig": "Dataclass storing segmentation QC thresholds such as pixel size, area limits, and N:C-ratio limits.",
    "ClusteringConfig": "Dataclass storing Scanpy/Leiden clustering settings.",
    "load_h5ad": "Load an AnnData h5ad file from disk.",
    "compute_area_um2": "Convert segmented cell area from pixels to square microns.",
    "categorize_area": "Flag debris-like and merged-cell-like objects from physical area thresholds.",
    "categorize_nc_ratio": "Flag cells with implausible nuclear-to-cell area ratio.",
    "filter_segmentation_errors": "Return a filtered AnnData containing cells that pass area and N:C-ratio QC.",
    "run_segmentation_qc": "Run the complete area and N:C-ratio segmentation QC workflow.",
    "zscore_normalize": "Z-score normalize the expression matrix in AnnData.X.",
    "train_svm_classifier": "Train a balanced radial-basis SVM and return a classification report.",
    "cluster_cells": "Run Scanpy preprocessing, neighbor graph construction, UMAP, and Leiden clustering.",
    "annotate_interactive": "Launch an interactive cluster-annotation workflow.",
    "annotate_from_csv": "Attach saved cluster annotations from a CSV file.",
    "annotate_refinements": "Review and annotate subsets of cells selected for refinement.",
    "subset_cells": "Create an AnnData subset from an observation-column query.",
    "merge_annotations": "Merge multiple annotation columns into one prioritized annotation.",
    "merge_refinements": "Merge refined annotations back into a parent annotation column.",
    "refine_clusters": "Split one annotated population for subclustering/refinement.",
    "plot_area_distribution": "Plot cell-area distributions with optional QC thresholds.",
    "plot_nc_ratio_distribution": "Plot nuclear-to-cell ratio distributions with optional QC threshold.",
    "compute_kde_density": "Estimate a smoothed 2D kernel-density surface for all cells or a phenotype.",
    "plot_kde_density": "Plot a smoothed KDE density map.",
    "cross_ripleys_k_all_pairs": "Compute cross-Ripley statistics for all phenotype pairs.",
    "calculate_polarity_score": "Measure displacement between geometric and intensity centroids normalized by cell size.",
    "calculate_moment_of_inertia": "Quantify how dispersed intensity is around the intensity centroid.",
    "calculate_haralick_features": "Compute gray-level texture descriptors from an intensity patch.",
    "calculate_haralick_features_rescaled": "Compute Haralick texture after rescaling intensity to uint8.",
    "calculate_entropy": "Compute Shannon entropy of an intensity patch.",
    "calculate_lacunarity": "Measure texture gap heterogeneity in a local image patch.",
    "calculate_channel_correlation": "Compute correlation between two channels inside a cell mask.",
    "spatial_scatter_plot": "Plot spatial coordinates colored by phenotype or continuous feature.",
    "plot_cluster_heatmap": "Plot a cluster-marker heatmap with Scanpy.",
    "plot_refinement_umaps": "Plot UMAPs from refinement runs.",
    "inspect_clusters": "Open interactive image/cluster inspection with optional viewer dependencies.",
}


STAGE_BY_PATH = {
    "spatioev/config.py": "configuration",
    "spatioev/io": "I/O",
    "spatioev/qc": "segmentation QC",
    "spatioev/preprocessing": "preprocessing",
    "spatioev/ml": "phenotyping/SVM",
    "spatioev/phenotype": "annotation refinement",
    "spatioev/plot": "QC plotting",
    "spatioev/visualization": "visualization",
    "spatioev/spatial/general_density.py": "density",
    "spatioev/spatial/local_density": "density",
    "spatioev/spatial/interaction.py": "cell-cell interaction",
    "spatioev/spatial/spatial_stats.py": "spatial statistics",
    "spatioev/spatial/preprocessing.py": "spatial preprocessing",
    "spatioev/spatial/spatial_niche": "niche/graph",
    "spatioev/spatial/spatial_cell_graph.py": "niche/graph",
    "spatioev/spatial/spatial_ecm": "ECM-cell analysis",
    "spatioev/spatial/cell_pixel_features.py": "pixel morphology/Xenium",
    "spatioev/spatial/pseudotime.py": "trajectory pseudotime",
    "spatioev/spatial/pseudotime_dynamics.py": "trajectory dynamics",
    "spatioev/spatial/pseudotime_trends.py": "trajectory dynamics",
    "spatioev/spatial/visualization.py": "spatial visualization",
    "spatioev/xenium": "Xenium spatial transcriptomics",
}


NOTEBOOK_BY_STAGE = {
    "configuration": "00",
    "I/O": "00",
    "segmentation QC": "01",
    "preprocessing": "01",
    "QC plotting": "01",
    "pixel morphology/Xenium": "01/06",
    "phenotyping/SVM": "02",
    "annotation refinement": "02",
    "visualization": "02/03",
    "density": "03",
    "cell-cell interaction": "03",
    "spatial statistics": "03",
    "spatial preprocessing": "03",
    "niche/graph": "04",
    "ECM-cell analysis": "05",
    "trajectory pseudotime": "06",
    "trajectory dynamics": "06",
    "Xenium spatial transcriptomics": "06",
    "spatial visualization": "03/04",
}


STAGE_DISPLAY = {
    "ECM-cell analysis": "ECM-Cell Analysis",
    "I/O": "I/O",
    "QC plotting": "QC Plotting",
    "segmentation QC": "Segmentation QC",
    "phenotyping/SVM": "Phenotyping/SVM",
    "trajectory pseudotime": "Trajectory Pseudotime",
    "trajectory dynamics": "Trajectory Dynamics",
    "Xenium spatial transcriptomics": "Xenium Spatial Transcriptomics",
}


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def stage_for_path(path: str) -> str:
    for key, stage in STAGE_BY_PATH.items():
        if path == key or key in path:
            return stage
    return "other"


def build_function_catalog() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "spatioev").glob("**/*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if "/archive/" in rel or rel.endswith("__init__.py"):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            if node.name.startswith("_"):
                continue

            doc = ast.get_docstring(node)
            purpose = CURATED_PURPOSE.get(node.name)
            if purpose is None and doc:
                purpose = doc.strip().splitlines()[0].strip()
            if purpose is None:
                purpose = f"{node.name.replace('_', ' ').capitalize()} helper used by the {stage_for_path(rel)} workflow."

            stage = stage_for_path(rel)
            rows.append(
                {
                    "stage": stage,
                    "module": rel.removesuffix(".py").replace("/", "."),
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "purpose": purpose,
                    "tutorial": NOTEBOOK_BY_STAGE.get(stage, ""),
                }
            )
    df = pd.DataFrame(rows).sort_values(["stage", "module", "name"]).reset_index(drop=True)
    return df


def write_function_catalog() -> pd.DataFrame:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_function_catalog()
    df.to_csv(DOCS_DIR / "function_catalog.csv", index=False)

    lines = [
        "# SpatioEv Function Catalog",
        "",
        "This catalog is generated from the current package source by `scripts/write_tutorial_notebooks.py`. It excludes private helpers and generated artifacts to keep the user-facing API readable.",
        "",
    ]
    for stage, sub in df.groupby("stage", sort=True):
        lines.append(f"## {STAGE_DISPLAY.get(stage, stage.title())}")
        lines.append("")
        lines.append("| Module | API | Purpose | Tutorial |")
        lines.append("|---|---|---|---|")
        for row in sub.itertuples(index=False):
            lines.append(
                f"| `{row.module}` | `{row.name}` | {row.purpose} | `{row.tutorial}` |"
            )
        lines.append("")

    (DOCS_DIR / "function_catalog.md").write_text("\n".join(lines), encoding="utf-8")
    return df


COMMON_SETUP = r"""
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spatioev

ROOT = Path.cwd()
if not (ROOT / "spatioev").exists():
    for candidate in [ROOT.parent, *ROOT.parents]:
        if (candidate / "spatioev").exists():
            ROOT = candidate
            break
CATALOG_PATH = ROOT / "docs" / "function_catalog.csv"
EXAMPLE_H5AD = ROOT / "data" / "exp_2" / "34434_1_adata.h5ad"
EXAMPLE_ANNOTATION = ROOT / "data" / "exp_2" / "34434_1_annotation.csv"


def make_synthetic_adata(n=240, seed=3):
    rng = np.random.default_rng(seed)
    branch = np.where(np.arange(n) < n * 0.45, "niche_a", np.where(np.arange(n) < n * 0.75, "niche_b", "niche_c"))
    obs = pd.DataFrame(
        {
            "label": np.arange(n),
            "imageid": np.where(np.arange(n) < n // 2, "synthetic_1", "synthetic_2"),
            "phenotype": rng.choice(["ductal", "fibroblast", "T cell", "B cell", "endothelial"], size=n, p=[0.36, 0.26, 0.18, 0.10, 0.10]),
            "niche_id": branch,
            "niche_region": rng.choice(["core", "inner_border", "surround"], size=n, p=[0.62, 0.18, 0.20]),
            "X_centroid": rng.normal(np.where(branch == "niche_a", 260, np.where(branch == "niche_b", 620, 960)), 80),
            "Y_centroid": rng.normal(np.where(branch == "niche_a", 360, np.where(branch == "niche_b", 620, 300)), 90),
            "area": rng.gamma(8, 22, size=n),
            "eccentricity": rng.uniform(0.2, 0.9, size=n),
            "major_axis_length": rng.uniform(15, 55, size=n),
            "minor_axis_length": rng.uniform(6, 25, size=n),
            "perimeter": rng.uniform(50, 170, size=n),
            "convex_area": rng.gamma(9, 24, size=n),
            "equivalent_diameter": rng.uniform(10, 35, size=n),
            "orientation": rng.uniform(-1.57, 1.57, size=n),
            "solidity": rng.uniform(0.75, 0.99, size=n),
            "feret_diameter_max": rng.uniform(16, 65, size=n),
            "major_minor_axis_ratio": rng.uniform(1.1, 3.2, size=n),
            "perim_square_over_area": rng.uniform(12, 22, size=n),
            "major_axis_equiv_diam_ratio": rng.uniform(1.0, 2.2, size=n),
            "convex_hull_resid": rng.uniform(0.0, 0.35, size=n),
            "centroid_dif": rng.uniform(0.0, 4.0, size=n),
            "num_concavities": rng.integers(0, 8, size=n),
            "circularity": rng.uniform(0.3, 0.95, size=n),
            "fractal_dimension": rng.uniform(1.0, 1.45, size=n),
            "boundary_irregularity": rng.uniform(0.0, 0.8, size=n),
            "nc_ratio": rng.beta(5, 2, size=n),
            "pseudotime": np.linspace(0, 1, n),
            "Ki67_expr_z": rng.normal(size=n),
            "FAP_expr_z": rng.normal(size=n),
        },
        index=[f"synthetic_cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=["CD8", "Ki67", "PCK", "Vimentin", "CD20", "CD31"])
    X = rng.normal(size=(n, len(var)))
    X[:, var.index.get_loc("PCK")] += (obs["phenotype"].eq("ductal").to_numpy() * 2.5)
    X[:, var.index.get_loc("CD8")] += (obs["phenotype"].eq("T cell").to_numpy() * 2.2)
    X[:, var.index.get_loc("CD20")] += (obs["phenotype"].eq("B cell").to_numpy() * 2.0)
    X[:, var.index.get_loc("Vimentin")] += (obs["phenotype"].eq("fibroblast").to_numpy() * 1.8)
    return ad.AnnData(X=X, obs=obs, var=var)


def load_example_or_synthetic(n_obs=1600):
    if EXAMPLE_H5AD.exists():
        print(f"Loading local example: {EXAMPLE_H5AD}")
        adata = ad.read_h5ad(EXAMPLE_H5AD, backed="r")[:n_obs].to_memory()
        if EXAMPLE_ANNOTATION.exists():
            annotations = pd.read_csv(EXAMPLE_ANNOTATION, index_col=0)
            adata.obs = adata.obs.join(annotations, how="left")
            source = "Tier_A" if "Tier_A" in adata.obs else "annotation_level1"
            adata.obs["phenotype"] = adata.obs[source].fillna("unannotated").astype(str)
        else:
            adata.obs["phenotype"] = "unannotated"
        adata.obs["niche_id"] = pd.qcut(
            pd.to_numeric(adata.obs["X_centroid"], errors="coerce").rank(method="average"),
            q=4,
            labels=[f"example_niche_{i}" for i in range(4)],
            duplicates="drop",
        ).astype(str)
        adata.obs["niche_region"] = np.where(
            pd.to_numeric(adata.obs["nc_ratio"], errors="coerce") > adata.obs["nc_ratio"].median(),
            "inner_border",
            "core",
        )
        adata.obs["pseudotime"] = pd.to_numeric(adata.obs["X_centroid"], errors="coerce").rank(pct=True).fillna(0.5)
        return adata

    print("Local example data not found; using synthetic fallback.")
    return make_synthetic_adata(n=n_obs)


def make_synthetic_fibers(adata, n=160, seed=11):
    rng = np.random.default_rng(seed)
    obs = adata.obs
    fibers = pd.DataFrame(
        {
            "imageid": rng.choice(obs["imageid"].unique(), size=n),
            "X_centroid": rng.uniform(obs["X_centroid"].min(), obs["X_centroid"].max(), size=n),
            "Y_centroid": rng.uniform(obs["Y_centroid"].min(), obs["Y_centroid"].max(), size=n),
            "orientation": rng.uniform(-1.57, 1.57, size=n),
            "major_axis_length": rng.uniform(20, 110, size=n),
            "minor_axis_length": rng.uniform(3, 18, size=n),
            "area": rng.gamma(6, 18, size=n),
            "eccentricity": rng.uniform(0.45, 0.98, size=n),
            "alignment_score": rng.normal(0, 1, size=n),
            "fiber_type": rng.choice(["COL6A1", "COL4A1", "FN", "CHP"], size=n, p=[0.35, 0.25, 0.25, 0.15]),
            "pathology": rng.choice(["RA", "OA"], size=n),
        },
        index=[f"fiber_{i}" for i in range(n)],
    )
    return fibers


adata = load_example_or_synthetic()
catalog = pd.read_csv(CATALOG_PATH) if CATALOG_PATH.exists() else pd.DataFrame()
print("SpatioEv", spatioev.__version__, "| AnnData shape:", adata.shape)
"""


def notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def write(path: Path, cells: list[nbf.NotebookNode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook(cells), path)
    print(f"Wrote {path.relative_to(ROOT)}")


def notebook_00() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 00. SpatioEv data model and function catalog

            SpatioEv is organized around one practical idea: each spatial analysis
            should produce an auditable table that can be inspected, plotted,
            reused in downstream models, and regenerated later. The package uses
            AnnData for cell-level imaging or spatial transcriptomics data, and
            plain pandas DataFrames for derived tables such as tile densities,
            cell-fiber links, niche feature matrices, and branch-time summaries.

            This notebook introduces the package layout, shows how the local
            example dataset is loaded, and displays the generated public-function
            catalog. Use it as the map before entering the method-specific
            tutorials.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            # The catalog is generated from the current source tree.
            catalog.groupby("stage").size().sort_values(ascending=False)
            """
        ),
        code(
            """
            catalog[["stage", "module", "name", "purpose", "tutorial"]].head(25)
            """
        ),
        md(
            """
            ## How to choose a workflow

            - Start with `qc` and `preprocessing` when validating segmentation,
              marker columns, and morphology features.
            - Use `ml` and `phenotype` when categorical annotations need
              probability-aware refinement or semi-supervised correction.
            - Use `spatial.general_density`, `local_density_*`, `interaction`,
              and `spatial_stats` when the biological question is about
              clustering, exclusion, co-enrichment, or feature autocorrelation.
            - Use `spatial_niche_*` and `spatial_cell_graph` when the object of
              interest is a recurring tissue structure rather than an individual
              cell.
            - Use `spatial_ecm_*` when ECM fibers are segmented as a second
              spatial object layer.
            - Use `pseudotime_dynamics` when niches or epithelial cells have a
              trajectory coordinate and the question is how the surrounding
              microenvironment changes along that coordinate.
            """
        ),
        code(
            """
            adata.obs[["imageid", "phenotype", "X_centroid", "Y_centroid", "area", "nc_ratio"]].head()
            """
        ),
        code(
            """
            # A compact audit of the tutorial input.
            pd.DataFrame(
                {
                    "n_cells": [adata.n_obs],
                    "n_markers": [adata.n_vars],
                    "n_images": [adata.obs["imageid"].nunique()],
                    "n_phenotypes": [adata.obs["phenotype"].nunique()],
                    "coordinate_columns_present": [
                        {"X_centroid", "Y_centroid"}.issubset(adata.obs.columns)
                    ],
                }
            )
            """
        ),
    ]


def notebook_01() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 01. Segmentation QC, preprocessing, and pixel-level features

            Segmentation QC is the first reproducibility checkpoint. SpatioEv
            does not hide QC decisions inside plotting code: area thresholds,
            N:C-ratio thresholds, and pass/fail labels are written into
            `adata.obs`. That makes the removal criteria reviewable and lets the
            same flags be reused by density, interaction, or trajectory analyses.

            Functions covered here include `QCConfig`, `compute_area_um2`,
            `categorize_area`, `categorize_nc_ratio`, `run_segmentation_qc`,
            `filter_segmentation_errors`, `generate_qc_summary`,
            `zscore_normalize`, `add_obs_from_var`, `add_zscore_obs_features`,
            `plot_area_distribution`, `plot_nc_ratio_distribution`, and the
            pixel-feature primitives used for polarity, entropy, lacunarity,
            Haralick texture, and channel correlation.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            catalog[catalog["stage"].isin(["segmentation QC", "preprocessing", "QC plotting", "pixel morphology/Xenium"])][
                ["module", "name", "purpose"]
            ].reset_index(drop=True)
            """
        ),
        code(
            """
            from spatioev.config import QCConfig
            from spatioev.qc import (
                compute_area_um2,
                categorize_area,
                categorize_nc_ratio,
                run_segmentation_qc,
                filter_segmentation_errors,
                generate_qc_summary,
            )

            qc_config = QCConfig(pixel_size=0.325, min_area_um2=5, max_area_um2=650, max_nc_ratio=1.0)
            adata_qc = run_segmentation_qc(adata.copy(), qc_config)
            generate_qc_summary(adata_qc)
            """
        ),
        code(
            """
            passing = filter_segmentation_errors(adata_qc)
            print("Before QC:", adata_qc.n_obs, "| after QC:", passing.n_obs)
            adata_qc.obs[["area_um2", "area_category", "nc_ratio_category"]].head()
            """
        ),
        code(
            """
            from spatioev.preprocessing.normalize import add_obs_from_var, add_zscore_obs_features

            markers = [m for m in ["CD8", "Ki67", "PCK", "Vimentin", "CD20"] if m in adata_qc.var_names]
            if markers:
                adata_qc = add_obs_from_var(adata_qc, markers, overwrite=True)
            adata_qc = add_zscore_obs_features(adata_qc, ["area", "nc_ratio"], overwrite=True)
            adata_qc.obs.filter(regex="(_expr|_z)$").head()
            """
        ),
        code(
            """
            from spatioev.plot.plot import plot_area_distribution, plot_nc_ratio_distribution

            plot_area_distribution(adata_qc, min_area=qc_config.min_area_um2, max_area=qc_config.max_area_um2)
            plot_nc_ratio_distribution(adata_qc, max_ratio=qc_config.max_nc_ratio)
            """
        ),
        md(
            """
            Pixel-level functions are lower-level building blocks. They are used
            when raw masks and channel images are available and you want to
            summarize subcellular organization: membrane polarity, nuclear
            texture, channel co-localization, or Xenium DAPI morphology.
            """
        ),
        code(
            """
            from spatioev.spatial.cell_pixel_features import (
                calculate_channel_correlation,
                calculate_entropy,
                calculate_lacunarity,
                calculate_moment_of_inertia,
                calculate_polarity_score,
            )

            rng = np.random.default_rng(4)
            patch_a = rng.normal(size=(20, 20))
            patch_b = patch_a * 0.4 + rng.normal(size=(20, 20)) * 0.6
            mask = np.zeros((20, 20), dtype=bool)
            mask[5:16, 6:17] = True

            {
                "polarity_score": calculate_polarity_score((10, 10), (11.2, 8.8), cell_size=12),
                "moment_of_inertia": calculate_moment_of_inertia(patch_a * mask, cx=10, cy=10),
                "entropy": calculate_entropy(patch_a * mask),
                "lacunarity": calculate_lacunarity(patch_a * mask, box_size=4),
                "channel_correlation": calculate_channel_correlation(mask, patch_a, patch_b),
            }
            """
        ),
    ]


def notebook_02() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 02. Phenotyping, SVM probability profiles, and annotation refinement

            SpatioEv treats phenotype labels as useful but imperfect evidence.
            Marker intensity, cell shape, and neighborhood context can reveal
            misassigned cells or transition-like cells that are compressed by
            hard categorical labels. The `ml` module builds marker/morphology
            matrices and trains SVM classifiers; the `phenotype` module stores
            cluster annotation and refinement utilities around that workflow.

            Functions covered here include `build_marker_features`,
            `build_morphology_features`, `build_feature_matrix`,
            `train_svm_classifier`, `predict_svm`, `run_svm_phenotyping`,
            `cluster_cells`, `annotate_from_csv`, `annotate_interactive`,
            `subset_cells`, `merge_annotations`, `merge_refinements`,
            `refine_clusters`, `annotate_refinements`,
            `inspect_reassigned_cells`, and `inspect_disagreements`.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            catalog[catalog["stage"].isin(["phenotyping/SVM", "annotation refinement", "visualization"])][
                ["module", "name", "purpose"]
            ].reset_index(drop=True).head(40)
            """
        ),
        code(
            """
            from spatioev.ml.features import build_feature_matrix, build_marker_features, build_morphology_features
            from spatioev.ml.prediction import predict_svm
            from spatioev.ml.training import train_svm_classifier

            markers = [m for m in ["CD8", "Ki67", "PCK", "Vimentin", "CD20", "CD31"] if m in adata.var_names]
            model_adata = adata[adata.obs["phenotype"].isin(adata.obs["phenotype"].value_counts().head(5).index)].copy()
            X = build_feature_matrix(model_adata, markers=markers, morph_weight=0.25)
            y = model_adata.obs["phenotype"].astype(str)

            model, report = train_svm_classifier(X, y, test_size=0.25, random_state=1)
            pred, prob_df = predict_svm(model, X, model.classes_)
            model_adata.obs["svm_prediction"] = pred
            model_adata.obs = model_adata.obs.join(prob_df.set_index(model_adata.obs.index))

            pd.DataFrame(report).T.head(10)
            """
        ),
        code(
            """
            # Probability columns are useful for confidence filtering and transition-state review.
            confidence_cols = [c for c in model_adata.obs.columns if c.startswith("svm_prob_")]
            model_adata.obs[["phenotype", "svm_prediction", *confidence_cols[:4]]].head()
            """
        ),
        code(
            """
            from spatioev.phenotype.merge import merge_annotations
            from spatioev.phenotype.subset import subset_cells

            # Simulate a refinement: take one phenotype subset, write a reviewed
            # annotation column, then merge that subset back into the parent object.
            focus_label = model_adata.obs["phenotype"].value_counts().index[0]
            reviewed_subset = subset_cells(model_adata, annotation_key="phenotype", labels=focus_label)
            reviewed_subset.obs["annotation"] = reviewed_subset.obs["svm_prediction"].astype(str)
            merged_adata = merge_annotations(model_adata.copy(), reviewed_subset, new_key="reviewed_annotation")
            merged_adata.obs[["phenotype", "svm_prediction", "reviewed_annotation"]].head()
            """
        ),
        md(
            """
            Optional viewer and Scanpy functions are deliberately imported lazily.
            `cluster_cells`, `plot_cluster_heatmap`, and `plot_refinement_umaps`
            need Scanpy; `inspect_reassigned_cells` and `inspect_disagreements`
            need scimap/Napari-style viewer dependencies. The core package can be
            installed and tested without those extras.
            """
        ),
        code(
            """
            optional_functions = catalog[catalog["name"].isin([
                "cluster_cells", "plot_cluster_heatmap", "plot_refinement_umaps",
                "inspect_reassigned_cells", "inspect_disagreements"
            ])]
            optional_functions[["name", "purpose", "module"]]
            """
        ),
    ]


def notebook_03() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 03. Density, interaction, Ripley statistics, and Moran statistics

            This notebook converts spatial impressions into quantitative
            evidence. Tile density asks where cells or phenotypes accumulate.
            Radius/kNN density asks how crowded each cell's local environment is.
            Cross-phenotype interaction asks whether a source phenotype sees
            more target cells than expected locally. Ripley functions quantify
            clustering over distance scales, while Moran functions quantify
            spatial autocorrelation of continuous features.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            catalog[catalog["stage"].isin(["density", "cell-cell interaction", "spatial statistics", "spatial preprocessing", "spatial visualization"])][
                ["module", "name", "purpose"]
            ].reset_index(drop=True).head(80)
            """
        ),
        code(
            """
            from spatioev.spatial.preprocessing import compute_tissue_areas, detect_edge_cells, validate_spatial_coordinates

            validate_spatial_coordinates(adata)
            tissue_area = compute_tissue_areas(adata)
            edge_adata = detect_edge_cells(adata.copy(), radius=80)
            tissue_area.head(), edge_adata.obs[["edge_cell", "distance_to_boundary"]].head()
            """
        ),
        code(
            """
            from spatioev.spatial.general_density import assign_tiles, compute_general_density, compute_phenotype_density, phenotype_density_correlation

            tiled = assign_tiles(adata, tile_size=256)
            general_density = compute_general_density(tiled, tile_size=256)
            phenotype_density = compute_phenotype_density(tiled, phenotype_key="phenotype", tile_size=256)
            density_corr = phenotype_density_correlation(phenotype_density, phenotype_key="phenotype")
            general_density.head(), density_corr.round(2)
            """
        ),
        code(
            """
            from spatioev.spatial.local_density_KNN import compute_local_density_all_cells, compute_local_density_by_phenotype
            from spatioev.spatial.local_density_radius import compute_radius_density
            from spatioev.spatial.interaction import phenotype_interaction_density

            adata_density = compute_local_density_all_cells(adata.copy(), k_neighbors=8)
            adata_density = compute_local_density_by_phenotype(adata_density, phenotype_key="phenotype", k_neighbors=5)
            adata_density = compute_radius_density(adata_density, radius=120)

            phenotypes = adata_density.obs["phenotype"].value_counts().index.tolist()
            source, target = phenotypes[:2] if len(phenotypes) >= 2 else (phenotypes[0], phenotypes[0])
            adata_density = phenotype_interaction_density(
                adata_density,
                phenotype_key="phenotype",
                source_pheno=source,
                target_pheno=target,
                radius=160,
            )
            adata_density.obs.filter(regex="density|interaction").head()
            """
        ),
        code(
            """
            from spatioev.spatial.spatial_stats import (
                add_local_morans_i,
                cross_morans_i,
                cross_ripleys_curve_by_phenotype,
                morans_i,
                ripleys_curve,
                ripley_interaction_scale,
                ripley_spatial_scales,
            )

            coords = adata_density.obs[["X_centroid", "Y_centroid"]].to_numpy()
            area_values = adata_density.obs["area"].to_numpy()
            global_moran = morans_i(coords, area_values, k=8)
            adata_density = add_local_morans_i(adata_density, value_key="area", k=8)

            radii = np.array([50, 100, 150, 250, 400])
            ripley_curve = ripleys_curve(coords, radii)
            cross_curve = cross_ripleys_curve_by_phenotype(
                adata_density,
                phenotype_key="phenotype",
                source_phenotype=source,
                target_phenotype=target,
                radii=radii,
            )
            global_moran, ripley_curve.head(), cross_curve.head()
            """
        ),
        code(
            """
            import seaborn as sns

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(density_corr, center=0, cmap="vlag", ax=axes[0])
            axes[0].set_title("Phenotype density correlation")
            axes[1].scatter(
                adata_density.obs["X_centroid"],
                adata_density.obs["Y_centroid"],
                c=adata_density.obs["local_morans_i__area"],
                s=4,
                cmap="coolwarm",
            )
            axes[1].invert_yaxis()
            axes[1].set_title("Local Moran's I for area")
            plt.tight_layout()
            """
        ),
    ]


def notebook_04() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 04. Niche boundaries, cell graphs, and niche feature tables

            Cell-level spatial statistics are powerful, but many biological
            hypotheses are about tissue structures: ducts, tumor nests, immune
            aggregates, tertiary lymphoid structures, or stromal regions. This
            notebook shows how SpatioEv moves from cells to graph-defined niches.

            Boundary functions use geometric dependencies such as Shapely and
            HDBSCAN when available. The graph workflow below is dependency-light
            and works in the base tutorial environment.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            catalog[catalog["stage"].eq("niche/graph")][["module", "name", "purpose"]].reset_index(drop=True)
            """
        ),
        code(
            """
            from spatioev.spatial.spatial_cell_graph import build_cell_graph, extract_all_niche_subgraphs, extract_niche_subgraph
            from spatioev.spatial.spatial_niche_graph_features import (
                build_niche_feature_table,
                score_pdac_niche_pathology_modules,
                summarize_niche_surrounding_context,
            )

            feature_cols = [c for c in ["area", "nc_ratio", "eccentricity", "major_axis_length"] if c in adata.obs.columns]
            graph_adata = build_cell_graph(
                adata.copy(),
                feature_cols=feature_cols,
                phenotype_key="phenotype",
                radius=180,
                compute_weights=True,
            )
            graph_adata.uns["cell_graph"]
            """
        ),
        code(
            """
            one_niche = graph_adata.obs["niche_id"].dropna().iloc[0]
            sub = extract_niche_subgraph(graph_adata, niche_key="niche_id", niche_value=one_niche)
            all_subgraphs = extract_all_niche_subgraphs(graph_adata, niche_key="niche_id")
            print("Selected niche:", one_niche)
            print("Subgraph cells:", len(sub["cell_ids"]), "| all subgraphs:", len(all_subgraphs))
            """
        ),
        code(
            """
            niche_features = build_niche_feature_table(
                graph_adata,
                niche_key="niche_id",
                feature_cols=feature_cols,
                state_feature_cols=feature_cols,
                phenotype_key="phenotype",
                region_key="niche_region",
                min_cells=5,
                lightweight=True,
            )
            niche_scores = score_pdac_niche_pathology_modules(niche_features)
            niche_features.head(), niche_scores.head()
            """
        ),
        code(
            """
            surround = summarize_niche_surrounding_context(
                graph_adata,
                niche_key="niche_id",
                phenotype_key="phenotype",
                feature_cols=feature_cols[:2],
                min_cells=5,
            )
            surround.head()
            """
        ),
        md(
            """
            Boundary functions are used when a niche needs a geometric core,
            shrunken inner boundary, expanded outer boundary, or composition
            summary for cells inside those regions. In a full installation, use
            `cluster_spatial_components`, `build_niche_boundaries`,
            `buffer_niche_boundaries`, `assign_cells_to_niche_regions`,
            `summarize_niche_composition`, and `add_niche_regions_to_obs`.
            """
        ),
        code(
            """
            boundary_api = catalog[catalog["name"].isin([
                "estimate_density_adaptive_dbscan_params",
                "estimate_spatial_component_params",
                "cluster_spatial_niches",
                "cluster_spatial_components",
                "cluster_spatial_components_hdbscan",
                "cluster_spatial_components_from_mask",
                "build_niche_boundaries",
                "buffer_niche_boundaries",
                "assign_cells_to_niche_regions",
                "summarize_niche_composition",
                "add_niche_regions_to_obs",
            ])]
            boundary_api[["name", "purpose"]]
            """
        ),
    ]


def notebook_05() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 05. ECM-cell interactions and fiber-aware neighborhoods

            SpatioEv treats ECM fibers as spatial objects, not just background
            intensity. The workflow is: build cell-fiber links, compute distance
            and density summaries, quantify spatial coupling with Ripley/Moran
            style statistics, then optionally cluster radius-neighborhood
            profiles that combine cell phenotypes and ECM features.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            catalog[catalog["stage"].eq("ECM-cell analysis")][["module", "name", "purpose"]].reset_index(drop=True)
            """
        ),
        code(
            """
            from spatioev.spatial.spatial_ecm_links import build_cell_fiber_links, build_nearest_cell_fiber_map

            adata_ecm = adata.copy()
            adata_ecm.obs["pathology"] = np.where(np.arange(adata_ecm.n_obs) % 2 == 0, "RA_like", "OA_like")
            fiber_df = make_synthetic_fibers(adata, n=180)
            links = build_cell_fiber_links(adata_ecm, fiber_df, radius=180)
            nearest = build_nearest_cell_fiber_map(adata_ecm, fiber_df)
            links.head(), nearest.head()
            """
        ),
        code(
            """
            from spatioev.spatial.spatial_ecm_stats import (
                cell_fiber_alignment,
                cell_to_fiber_distance,
                cross_morans_i_ecm_cells,
                fiber_density_near_cells,
                fiber_vectors,
                map_cells_to_fibers,
                morans_i_fibers,
                spatial_enrichment_score,
            )

            dist_adata = cell_to_fiber_distance(adata_ecm.copy(), fiber_df, links, phenotype_key="phenotype", phenotype=None)
            density_adata = fiber_density_near_cells(
                dist_adata,
                fiber_df,
                links,
                phenotype_key="phenotype",
                fiber_type_key="fiber_type",
                normalize=True,
                density_radius=180,
            )
            moran = morans_i_fibers(fiber_df, feature="alignment_score", k=6)
            map_phenotype = adata_ecm.obs["phenotype"].value_counts().index[0]
            mapped = map_cells_to_fibers(adata_ecm, fiber_df, links, phenotype_key="phenotype", phenotype=map_phenotype)
            vectors = fiber_vectors(fiber_df)

            density_adata.obs.filter(regex="dist_to_|_density").head(), moran, mapped.head(), vectors.head()
            """
        ),
        code(
            """
            from spatioev.spatial.spatial_ecm_neighborhoods import (
                build_ecm_cell_neighborhood_features,
                cluster_ecm_cell_neighborhoods,
                default_ecm_cell_neighborhood_feature_columns,
                summarize_ecm_cell_neighborhoods,
            )

            nbhd = build_ecm_cell_neighborhood_features(
                adata_ecm,
                fiber_df,
                links,
                radius=180,
                phenotype_key="phenotype",
                fiber_types=["COL6A1", "COL4A1", "FN", "CHP"],
                pixel_size_um=0.325,
            )
            feature_cols = default_ecm_cell_neighborhood_feature_columns(nbhd)
            clustered, kmeans_model, scaler, used_features = cluster_ecm_cell_neighborhoods(
                nbhd,
                feature_columns=feature_cols,
                n_clusters=4,
                random_state=2,
            )
            summary, group_counts, phenotype_counts = summarize_ecm_cell_neighborhoods(clustered, phenotype_key="phenotype")
            clustered.head(), summary.head()
            """
        ),
        code(
            """
            from spatioev.spatial.spatial_ecm_graph import (
                assign_niches_to_fibers,
                build_ecm_bipartite_graph_per_image,
                compute_invasion_score,
                detect_ecm_niches_per_image,
                project_fiber_graph_per_image,
            )

            graphs = build_ecm_bipartite_graph_per_image(adata_ecm, fiber_df, links)
            fiber_graphs = project_fiber_graph_per_image(graphs)
            niche_maps = detect_ecm_niches_per_image(fiber_graphs, resolution=1.0, random_state=1)
            fibers_with_niches = assign_niches_to_fibers(fiber_df, niche_maps)
            fibers_with_niches["tumor_density"] = mapped[f"{map_phenotype}_density"].fillna(0).to_numpy()
            invasion = compute_invasion_score(fibers_with_niches)
            invasion.head()
            """
        ),
    ]


def notebook_06() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 06. Pseudotime dynamics, Xenium extension, and manuscript figures

            SpatioEv's trajectory layer asks how a spatial ecosystem changes
            along a progression coordinate. The coordinate can come from an
            epithelial niche trajectory, an external trajectory algorithm, a
            histology score, or a spatial-transcriptomics branch-time model.
            The key design is source-centered: for each source cell or niche,
            summarize the target phenotypes nearby, bin by pseudotime, and
            interpret changes as dynamic tissue remodeling.
            """
        ),
        code(COMMON_SETUP),
        code(
            """
            catalog[catalog["stage"].isin([
                "trajectory pseudotime",
                "trajectory dynamics",
                "Xenium spatial transcriptomics",
                "pixel morphology/Xenium",
            ])][
                ["module", "name", "purpose"]
            ].reset_index(drop=True)
            """
        ),
        code(
            """
            from spatioev.spatial import (
                block_balance_feature_matrix,
                compute_feature_trend_table,
                prepare_pseudotime_feature_matrix,
                score_signed_feature_module,
            )

            candidate_features = [
                "area",
                "eccentricity",
                "major_minor_axis_ratio",
                "boundary_irregularity",
                "nc_ratio",
                "Ki67_expr_z",
                "FAP_expr_z",
            ]
            feature_result = prepare_pseudotime_feature_matrix(
                adata.obs,
                feature_cols=candidate_features,
                priority_features=["nc_ratio", "boundary_irregularity"],
                correlation_threshold=0.95,
            )
            balanced_features, feature_blocks = block_balance_feature_matrix(
                feature_result.matrix,
                return_blocks=True,
            )
            adata.obs["example_pathology_like_score"] = score_signed_feature_module(
                adata.obs,
                [
                    (1, ["nc_ratio"]),
                    (1, ["boundary_irregularity"]),
                    (1, ["Ki67_expr_z"]),
                    (-1, ["solidity"]),
                ],
                min_features=2,
            )
            trend_table = compute_feature_trend_table(
                adata.obs,
                ["example_pathology_like_score", *feature_result.selected_features[:4]],
                pseudotime_col="pseudotime",
                min_n=30,
            )
            feature_result.diagnostics.head(), feature_blocks.head(), trend_table.head()
            """
        ),
        code(
            """
            from spatioev.xenium import (
                assign_labels_from_marker_rules,
                compute_marker_set_scores,
                summarize_cluster_marker_scores,
            )

            marker_sets = {
                "epithelial_score": ["PCK", "KRT19", "EPCAM"],
                "immune_score": ["CD8", "PTPRC", "CD3D"],
                "proliferation_score": ["Ki67", "MKI67"],
            }
            marker_scores = compute_marker_set_scores(adata, marker_sets, min_markers=1)
            cluster_summary = summarize_cluster_marker_scores(
                marker_scores,
                adata.obs["phenotype"],
            )
            suggested = assign_labels_from_marker_rules(
                cluster_summary,
                {
                    "epithelial-like": {"epithelial_score": 0.2},
                    "immune-like": {"immune_score": 0.2},
                    "cycling-like": {"proliferation_score": 0.2},
                },
                cluster_col="cluster",
            )
            suggested.head()
            """
        ),
        code(
            """
            from spatioev.spatial.pseudotime_dynamics import (
                assign_pseudotime_bins,
                compute_epithelial_centered_interaction_dynamics,
                summarize_epithelial_interaction_dynamics,
            )

            pseudotime_bins, bin_summary = assign_pseudotime_bins(adata.obs["pseudotime"], n_bins=5)
            binned = adata.copy()
            binned.obs["pseudotime_bin"] = pseudotime_bins
            phenotypes = binned.obs["phenotype"].value_counts().index.tolist()
            source = phenotypes[0]
            targets = phenotypes[1:4]

            dynamics = compute_epithelial_centered_interaction_dynamics(
                binned,
                source_phenotype=source,
                target_phenotypes=targets,
                phenotype_key="phenotype",
                pseudotime_key="pseudotime",
                radius=180,
            )
            dyn_summary = summarize_epithelial_interaction_dynamics(dynamics, pseudotime_key="pseudotime")
            bin_summary, dyn_summary.head()
            """
        ),
        md(
            """
            Existing local result tables extend this idea to multiplexed imaging
            and Xenium outputs. If the local data are present, the next cells
            load the branch-level summaries used to generate manuscript figures.
            """
        ),
        code(
            """
            paths = {
                "multiplexed_changes": ROOT  / "paper" / "notebooks" / "results" / "trajectory_microenvironment_interactions" / "tables" / "top_trajectory_microenvironment_changes.csv",
                "xenium_branch_biology": ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_branch_biology_summary.csv",
                "analysis_summary": ROOT / "manuscript" / "analysis_summary.json",
            }
            available = {name: path.exists() for name, path in paths.items()}
            available
            """
        ),
        code(
            """
            if paths["multiplexed_changes"].exists():
                changes = pd.read_csv(paths["multiplexed_changes"])
                display(changes.sort_values("abs_spearman_r", ascending=False).head(12)[
                    ["analysis", "label", "spearman_r", "late_minus_early_median"]
                ])

            if paths["xenium_branch_biology"].exists():
                branch_biology = pd.read_csv(paths["xenium_branch_biology"])
                display(branch_biology[["branch", "n_niches", "suggested_biology", "top_enriched_scores"]].head())
            """
        ),
        code(
            """
            # Regenerate manuscript figures after changing package code or local analyses.
            # This line is intentionally displayed rather than executed in notebooks.
            print("python scripts/generate_manuscript_figures.py")
            """
        ),
    ]


def write_readme() -> None:
    text = """# SpatioEv Tutorials

These notebooks are regenerated by `python scripts/write_tutorial_notebooks.py`.
They use the local example data when present and otherwise fall back to a small
synthetic AnnData object, so they remain runnable in lightweight GitHub clones.

1. `00_data_model_and_function_catalog.ipynb` - package map, AnnData/table model, and full public API catalog.
2. `01_qc_preprocessing_pixel_features.ipynb` - segmentation QC, marker copying, z-scoring, and pixel-feature primitives.
3. `02_phenotyping_svm_annotation_refinement.ipynb` - feature matrices, SVM probability profiles, and annotation refinement tools.
4. `03_density_interaction_spatial_statistics.ipynb` - tile/radius/kNN density, cell-cell interaction, Ripley, and Moran statistics.
5. `04_niche_boundaries_cell_graphs.ipynb` - graph construction, niche feature tables, surrounding context, and boundary API guide.
6. `05_ecm_cell_interactions.ipynb` - cell-fiber links, ECM statistics, ECM neighborhoods, and ECM graph niches.
7. `06_pseudotime_xenium_manuscript_figures.ipynb` - trajectory dynamics, Xenium outputs, and manuscript figure regeneration.

The companion API table is generated at `docs/function_catalog.md` and
`docs/function_catalog.csv`.
"""
    (TUTORIAL_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    TUTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    for old in TUTORIAL_DIR.glob("*.ipynb"):
        old.unlink()

    catalog = write_function_catalog()
    print(f"Wrote docs/function_catalog.md ({len(catalog)} public functions/classes)")

    write(TUTORIAL_DIR / "00_data_model_and_function_catalog.ipynb", notebook_00())
    write(TUTORIAL_DIR / "01_qc_preprocessing_pixel_features.ipynb", notebook_01())
    write(TUTORIAL_DIR / "02_phenotyping_svm_annotation_refinement.ipynb", notebook_02())
    write(TUTORIAL_DIR / "03_density_interaction_spatial_statistics.ipynb", notebook_03())
    write(TUTORIAL_DIR / "04_niche_boundaries_cell_graphs.ipynb", notebook_04())
    write(TUTORIAL_DIR / "05_ecm_cell_interactions.ipynb", notebook_05())
    write(TUTORIAL_DIR / "06_pseudotime_xenium_manuscript_figures.ipynb", notebook_06())
    write_readme()
    print("Wrote tutorials/README.md")


if __name__ == "__main__":
    main()
