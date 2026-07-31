"""Generate public-API validation notebooks for the reorganized SpatioEv tree.

These notebooks replace the older development notebooks as runnable examples
for the strict public API layout:

    spatioev.io, spatioev.pp, spatioev.tl, spatioev.pl, spatioev.hl, spatioev.xe

The historical implementation code remains under ``spatioev.archive`` but the
notebooks intentionally use only the public namespaces.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "public_api_validation"


HISTORICAL_NOTEBOOKS = {
    "00_public_api_data_model_and_qc.ipynb": [
        "00_dev_seg_qc_testing.ipynb",
    ],
    "01_public_api_phenotyping_and_svm.ipynb": [
        "01_dev_clustering_based_phenotyping_test.ipynb",
        "01_dev_scimap_phenotype_workflow.ipynb",
        "02_dev_SVM_phenotype_probility.ipynb",
    ],
    "02_public_api_density_interaction_spatial_stats.ipynb": [
        "03_dev_general_density.ipynb",
        "03_dev_local_density_KNN.ipynb",
        "03_dev_local_density_radius.ipynb",
        "04_dev_spatial_stats_exp3.ipynb",
        "04_dev_spatial_stats.ipynb",
        "04_global_organization_PDAC_IgG4AIP.ipynb",
    ],
    "03_public_api_niche_boundaries_and_graphs.ipynb": [
        "05_dev_spatial_niche_boundaries.ipynb",
    ],
    "04_public_api_pseudotime_single_and_combined.ipynb": [
        "06_dev_graph_pseudotime_v2_exp_2.ipynb",
        "06_dev_graph_pseudotime_v2_exp_3.ipynb",
        "06_dev_graph_pseudotime_v2_exp_4.ipynb",
        "06_dev_graph_pseudotime_v2_exp_5.ipynb",
        "06_dev_graph_pseudotime_v2_combined_exp_2_3_4_5.ipynb",
    ],
    "05_public_api_xenium_workflows.ipynb": [
        "07_xenium_00_data_audit_and_spatialdata.ipynb",
        "07_xenium_01_cell_annotation.ipynb",
        "07_xenium_02_epithelial_niche_features.ipynb",
        "07_xenium_03_pooled_pseudotime.ipynb",
    ],
    "06_public_api_ecm_cell_and_microenvironment.ipynb": [
        "08_RA_OA_ECM_cell_00_prepare_links.ipynb",
        "08_RA_OA_ECM_cell_05_chp_density_micro_holes_col6_dark_zone_segmentation.ipynb",
        "08_trajectory_microenvironment_interactions.ipynb",
        "09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb",
    ],
    "07_public_api_external_integrations.ipynb": [
        "09_xenium_banksy_pseudotime_integration.ipynb",
        "10_xenium_spatialcellchat_pseudotime_integration.ipynb",
    ],
}


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


COMMON_SETUP = r"""
from pathlib import Path
import importlib.util

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import spatioev as sv

ROOT = Path.cwd()
if not (ROOT / "spatioev").exists():
    for candidate in ROOT.parents:
        if (candidate / "spatioev").exists():
            ROOT = candidate
            break

DATA_DIR = ROOT / "data"
EXP2_H5AD = DATA_DIR / "exp_2" / "34434_1_adata.h5ad"
EXP2_ANNOTATION = DATA_DIR / "exp_2" / "34434_1_annotation.csv"

MARKER_CANDIDATES = ["CD8", "Ki67", "CD3", "PCK", "Vimentin", "CD20", "CD31", "FAP"]
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


def present(values, candidates):
    return [item for item in candidates if item in values]


def make_synthetic_adata(n=360, seed=7):
    rng = np.random.default_rng(seed)
    imageid = np.where(np.arange(n) < n // 2, "synthetic_A", "synthetic_B")
    branch = np.select(
        [np.arange(n) < n * 0.40, np.arange(n) < n * 0.72],
        ["ductal_niche_early", "ductal_niche_transition"],
        default="ductal_niche_late",
    )
    phenotype = rng.choice(
        ["ductal", "fibroblast", "T cell", "B cell", "endothelial"],
        size=n,
        p=[0.38, 0.26, 0.18, 0.08, 0.10],
    )
    obs = pd.DataFrame(
        {
            "label": np.arange(n),
            "imageid": imageid,
            "phenotype": phenotype,
            "Tier_A": phenotype,
            "Tier_B": phenotype,
            "niche_id": branch,
            "niche_region": rng.choice(["core", "inner_border", "surround"], size=n, p=[0.55, 0.25, 0.20]),
            "X_centroid": rng.normal(np.where(branch == "ductal_niche_early", 260, np.where(branch == "ductal_niche_transition", 620, 950)), 85),
            "Y_centroid": rng.normal(np.where(branch == "ductal_niche_early", 340, np.where(branch == "ductal_niche_transition", 620, 360)), 95),
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
            "pathology": rng.choice(["normal_like", "PDAC_like"], size=n),
        },
        index=[f"synthetic_cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=["CD8", "Ki67", "CD3", "PCK", "Vimentin", "CD20", "CD31", "FAP"])
    X = rng.normal(size=(n, len(var)))
    X[:, var.index.get_loc("PCK")] += (phenotype == "ductal") * 2.5
    X[:, var.index.get_loc("FAP")] += (phenotype == "fibroblast") * 2.0
    X[:, var.index.get_loc("CD3")] += (phenotype == "T cell") * 2.0
    X[:, var.index.get_loc("CD20")] += (phenotype == "B cell") * 2.0
    X[:, var.index.get_loc("CD31")] += (phenotype == "endothelial") * 2.0
    return ad.AnnData(X=X, obs=obs, var=var)


def ensure_demo_columns(adata, seed=11):
    rng = np.random.default_rng(seed)
    obs = adata.obs
    n = adata.n_obs
    if "imageid" not in obs.columns:
        obs["imageid"] = obs["image_id"].astype(str) if "image_id" in obs.columns else "image_0"
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
    if "phenotype" not in obs.columns:
        if "Tier_A" in obs.columns:
            obs["phenotype"] = obs["Tier_A"].astype(str)
        else:
            obs["phenotype"] = rng.choice(["ductal", "fibroblast", "T cell", "B cell", "endothelial"], size=n)
    if "Tier_A" not in obs.columns:
        obs["Tier_A"] = obs["phenotype"].astype(str)
    if "Tier_B" not in obs.columns:
        obs["Tier_B"] = obs["phenotype"].astype(str)
    if "niche_id" not in obs.columns:
        obs["niche_id"] = np.where(
            np.arange(n) < n * 0.40,
            "ductal_niche_early",
            np.where(np.arange(n) < n * 0.72, "ductal_niche_transition", "ductal_niche_late"),
        )
    if "niche_region" not in obs.columns:
        obs["niche_region"] = rng.choice(["core", "inner_border", "surround"], size=n, p=[0.55, 0.25, 0.20])
    if "pseudotime" not in obs.columns:
        obs["pseudotime"] = np.linspace(0, 1, n)
    if "pathology" not in obs.columns:
        obs["pathology"] = rng.choice(["normal_like", "PDAC_like"], size=n)
    for marker in ["Ki67", "FAP"]:
        z_col = f"{marker}_expr_z"
        if z_col in obs.columns:
            continue
        if marker in adata.var_names:
            adata = sv.pp.add_obs_from_var(adata, [marker], overwrite=True)
        else:
            obs[z_col] = rng.normal(size=n)
    return adata


def attach_exp2_annotations(adata):
    if not EXP2_ANNOTATION.exists():
        return adata
    annot = pd.read_csv(EXP2_ANNOTATION, index_col=0)
    overlap = adata.obs_names.intersection(annot.index)
    for col in ["Tier_A", "Tier_B", "annotation_level1", "annotation_level2", "annotation_level3"]:
        if col in annot.columns:
            adata.obs.loc[overlap, col] = annot.loc[overlap, col].astype(str)
    if "Tier_A" in adata.obs.columns:
        adata.obs["phenotype"] = adata.obs["Tier_A"].astype(str)
    return adata


def load_demo_adata(max_cells=800):
    if EXP2_H5AD.exists():
        adata = ad.read_h5ad(EXP2_H5AD, backed="r")[:max_cells].to_memory()
        adata = attach_exp2_annotations(adata)
        source = f"local example subset: {EXP2_H5AD.relative_to(ROOT)}"
    else:
        adata = make_synthetic_adata(n=max_cells)
        source = "synthetic fallback dataset"
    adata = ensure_demo_columns(adata)
    return adata, source


def usable_markers(adata, minimum=2):
    markers = present(adata.var_names, MARKER_CANDIDATES)
    if len(markers) < minimum:
        adata = make_synthetic_adata(n=adata.n_obs)
        markers = present(adata.var_names, MARKER_CANDIDATES)
    return markers, adata


adata, data_source = load_demo_adata()
markers, adata = usable_markers(adata)
print(f"SpatioEv {sv.__version__}")
print(f"Using {data_source}: {adata.n_obs:,} cells x {adata.n_vars:,} markers")
print("Public namespaces:", [name for name in ["io", "pp", "tl", "pl", "hl", "xe"] if hasattr(sv, name)])
print("Markers:", markers)
"""


def mapping_md(new_name: str) -> str:
    bullets = "\n".join(f"- `{name}`" for name in HISTORICAL_NOTEBOOKS[new_name])
    return f"""
    **Historical notebooks recapitulated here**

    {bullets}
    """


def notebook(title: str, new_name: str, body_cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(f"# {title}\n\nThis notebook uses the strict public SpatioEv API after package cleanup."),
        md(mapping_md(new_name)),
        code(COMMON_SETUP),
        *body_cells,
    ]
    return nb


def write_notebook(filename: str, nb: nbf.NotebookNode) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT_DIR / filename)


def write_readme() -> None:
    lines = [
        "# Public API Validation Notebooks",
        "",
        "These notebooks recapitulate the historical development/testing notebooks using the strict public API:",
        "",
        "```python",
        "import spatioev as sv",
        "sv.pp  # preprocessing and QC",
        "sv.tl  # analysis tools",
        "sv.pl  # plotting",
        "sv.hl  # helper functions",
        "sv.xe  # Xenium/spatial-transcriptomics helpers",
        "sv.io  # input/output",
        "```",
        "",
        "The old implementation namespaces were moved under `spatioev.archive`; these notebooks intentionally do not import them directly.",
        "",
        "| New notebook | Historical notebooks recapitulated |",
        "| --- | --- |",
    ]
    for new_name, old_names in HISTORICAL_NOTEBOOKS.items():
        old = "<br>".join(f"`{name}`" for name in old_names)
        lines.append(f"| `{new_name}` | {old} |")
    lines.append("")
    lines.append("Most notebooks use the local `data/exp_2/34434_1_adata.h5ad` subset when present, with synthetic fallback data for portability.")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_notebooks() -> dict[str, nbf.NotebookNode]:
    notebooks: dict[str, nbf.NotebookNode] = {}

    notebooks["00_public_api_data_model_and_qc.ipynb"] = notebook(
        "00. Public API, Data Model, QC, and Pixel Features",
        "00_public_api_data_model_and_qc.ipynb",
        [
            md("""
            ## Package layout check

            The cleaned package exposes only the public namespaces plus `archive` internally.
            This cell verifies that the old direct namespaces are no longer importable.
            """),
            code("""
            public = ["io", "pp", "tl", "pl", "hl", "xe"]
            retired = ["ml", "phenotype", "plot", "preprocessing", "qc", "spatial", "visualization", "xenium"]
            print({name: hasattr(sv, name) for name in public})
            print({name: hasattr(sv, name) for name in retired})
            assert all(hasattr(sv, name) for name in public)
            assert not any(hasattr(sv, name) for name in retired)
            """),
            md("## Segmentation QC"),
            code("""
            qc_adata = sv.pp.run_segmentation_qc(adata.copy(), sv.pp.QCConfig(pixel_size=0.325))
            qc_summary = sv.pp.generate_qc_summary(qc_adata, groupby="imageid")
            display(qc_summary.head())
            qc_adata.obs[["area_um2", "area_category", "nc_ratio_category"]].head()
            """),
            md("## Marker preprocessing"),
            code("""
            marker_subset = markers[:3]
            expr_adata = sv.pp.add_obs_from_var(qc_adata.copy(), marker_subset, overwrite=True)
            expr_adata = sv.pp.add_zscore_obs_features(expr_adata, ["area"], overwrite=True)
            display(expr_adata.obs.filter(regex="_expr|_z$").head())
            """),
            md("## QC plotting and pixel morphology helpers"),
            code("""
            sv.pl.plot_area_distribution(qc_adata)
            plt.close("all")

            patch = np.arange(100, dtype=float).reshape(10, 10)
            mask = patch > 25
            yy, xx = np.indices(mask.shape)
            geometric_centroid = [xx[mask].mean(), yy[mask].mean()]
            intensity_centroid = [
                np.average(xx[mask], weights=patch[mask]),
                np.average(yy[mask], weights=patch[mask]),
            ]
            print("entropy:", sv.hl.calculate_entropy(patch))
            print("lacunarity:", sv.hl.calculate_lacunarity(mask.astype(float)))
            print("polarity:", sv.hl.calculate_polarity_score(geometric_centroid, intensity_centroid, cell_size=float(mask.sum())))
            """),
        ],
    )

    notebooks["01_public_api_phenotyping_and_svm.ipynb"] = notebook(
        "01. Phenotyping, SVM Annotation, and Refinement",
        "01_public_api_phenotyping_and_svm.ipynb",
        [
            md("""
            ## Feature matrix construction

            This recapitulates the clustering/phenotyping and SVM development notebooks with
            public tools. Scanpy/scimap interactive workflows are optional; the reusable
            package layer starts from marker and morphology feature matrices.
            """),
            code("""
            X = sv.tl.build_feature_matrix(adata, markers=markers[:4], morph_weight=0.25)
            print("Feature matrix:", X.shape)
            """),
            md("## SVM phenotype model"),
            code("""
            svm_adata = adata.copy()
            labels = svm_adata.obs["phenotype"].astype(str)
            keep = labels.groupby(labels).transform("size") >= 8
            svm_adata.obs["manual_label"] = np.where(keep, labels, "unknown")

            result_adata, model, report = sv.tl.run_svm_phenotyping(
                svm_adata,
                markers=markers[:4],
                label_key="manual_label",
                morph_weight=0.25,
            )
            print("Classes:", list(model.classes_))
            display(pd.DataFrame(report).T.head())
            display(result_adata.obs.filter(regex="svm_").head())
            """),
            md("## Refinement helpers"),
            code("""
            subset = sv.tl.subset_cells(result_adata, "svm_prediction", model.classes_[0])
            subset.obs["annotation"] = str(model.classes_[0]) + "_reviewed"
            merged = sv.tl.merge_annotations(result_adata.copy(), subset, new_key="annotation_level2_public_api")
            display(merged.obs[["svm_prediction", "annotation_level2_public_api"]].head())
            """),
            md("## Optional Scanpy/scimap notes"),
            code("""
            print("scanpy available:", importlib.util.find_spec("scanpy") is not None)
            print("scimap available:", importlib.util.find_spec("scimap") is not None)
            print("Use sv.tl.cluster_cells when scanpy dependencies are installed.")
            """),
        ],
    )

    notebooks["02_public_api_density_interaction_spatial_stats.ipynb"] = notebook(
        "02. Density, Interaction, and Spatial Statistics",
        "02_public_api_density_interaction_spatial_stats.ipynb",
        [
            md("## Tile density and phenotype density"),
            code("""
            tiled = sv.tl.assign_tiles(adata, tile_size=256)
            density = sv.tl.compute_general_density(tiled, tile_size=256)
            phenotype_density = sv.tl.compute_phenotype_density(tiled, phenotype_key="phenotype", tile_size=256)
            corr = sv.tl.phenotype_density_correlation(phenotype_density, phenotype_key="phenotype")
            display(density.head())
            display(phenotype_density.head())
            display(corr.head())
            """),
            md("## Local kNN/radius density and cell-cell interactions"),
            code("""
            local = sv.tl.compute_local_density_all_cells(adata.copy(), k_neighbors=8)
            local = sv.tl.compute_radius_density(local, radius=120)
            source = local.obs["phenotype"].astype(str).value_counts().index[0]
            target = local.obs["phenotype"].astype(str).value_counts().index[min(1, local.obs["phenotype"].nunique() - 1)]
            local = sv.tl.phenotype_interaction_density(
                local,
                phenotype_key="phenotype",
                source_pheno=source,
                target_pheno=target,
                radius=160,
            )
            display(local.obs.filter(regex="density").head())
            """),
            md("## Ripley and Moran statistics"),
            code("""
            tissue = sv.pp.compute_tissue_areas(adata)
            edge_adata = sv.pp.detect_edge_cells(adata.copy(), radius=20)
            ripley = sv.tl.ripleys_k_by_phenotype(adata, phenotype_key="phenotype", radius=160)
            cross = sv.tl.cross_ripleys_k_by_phenotype(
                adata,
                phenotype_key="phenotype",
                source_phenotype=source,
                target_phenotype=target,
                radius=160,
            )
            coords = adata.obs[["X_centroid", "Y_centroid"]].to_numpy()
            feature_a = pd.to_numeric(adata.obs["area"], errors="coerce")
            feature_b = pd.to_numeric(adata.obs["eccentricity"], errors="coerce")
            print("Moran I area:", sv.tl.morans_i(coords, feature_a, k=8))
            print("Cross Moran I area/eccentricity:", sv.tl.cross_morans_i(coords, feature_a, feature_b, k=8))
            display(tissue.head())
            display(ripley.head())
            display(cross.head())
            """),
            md("## Source-centered feature summaries"),
            code("""
            summary = sv.tl.summarize_target_features_around_source_cells(
                adata,
                phenotype_key="phenotype",
                source_phenotype=source,
                target_phenotype=target,
                target_feature_keys=["area", "eccentricity"],
                radius=180,
            )
            display(summary.head())
            """),
        ],
    )

    notebooks["03_public_api_niche_boundaries_and_graphs.ipynb"] = notebook(
        "03. Spatial Niche Boundaries and Cell Graphs",
        "03_public_api_niche_boundaries_and_graphs.ipynb",
        [
            md("## Niche component labels"),
            code("""
            if importlib.util.find_spec("shapely") is None:
                print("Shapely is not installed; using existing niche labels as component placeholders.")
                components = adata.copy()
                components.obs["public_api_component"] = components.obs["niche_id"].astype(str)
            else:
                label_value = adata.obs["phenotype"].astype(str).value_counts().index[0]
                components = sv.tl.cluster_spatial_components(
                    adata,
                    label_key="phenotype",
                    label_value=label_value,
                    component_key="public_api_component",
                    radius=180,
                    min_component_size=3,
                )
            display(components.obs["public_api_component"].value_counts().head())
            """),
            md("## Optional boundary geometries"),
            code("""
            if importlib.util.find_spec("shapely") is None:
                print("Shapely is not installed; skipping geometry boundary construction.")
                boundary_df = pd.DataFrame()
            else:
                boundary_df = sv.tl.build_niche_boundaries(
                    components,
                    component_key="public_api_component",
                    min_cluster_size=5,
                    method="concave_hull",
                    point_buffer=20,
                    line_buffer=20,
                )
                if not boundary_df.empty:
                    boundary_df = sv.tl.buffer_niche_boundaries(boundary_df, "public_api_component", expand_by=40, shrink_by=20)
                    assignments = sv.tl.assign_cells_to_niche_regions(
                        components,
                        boundary_df,
                        component_key="public_api_component",
                        region_key="public_api_region",
                    )
                    composition = sv.tl.summarize_niche_composition(
                        components,
                        assignments,
                        component_key="public_api_component",
                        phenotype_key="phenotype",
                        region_key="public_api_region",
                    )
                    display(boundary_df.head())
                    display(composition.head())
            """),
            md("## Cell graph and niche feature table"),
            code("""
            graph_adata = sv.tl.build_cell_graph(
                adata,
                feature_cols=["area", "eccentricity", "boundary_irregularity", "nc_ratio"],
                phenotype_key="phenotype",
                radius=180,
                compute_weights=True,
            )
            subgraphs = sv.tl.extract_all_niche_subgraphs(graph_adata, "niche_id", min_cells=5)
            print("Subgraphs:", list(subgraphs)[:5])

            niche_features = sv.tl.build_niche_feature_table(
                graph_adata,
                niche_key="niche_id",
                feature_cols=["area", "eccentricity", "boundary_irregularity", "nc_ratio"],
                state_feature_cols=["Ki67_expr_z", "FAP_expr_z"],
                phenotype_key="phenotype",
                region_key="niche_region",
                min_cells=5,
                lightweight=True,
            )
            modules = sv.tl.score_pdac_niche_pathology_modules(niche_features, niche_key="niche_id")
            display(niche_features.head())
            display(modules.head())
            """),
            md("## Surrounding context"),
            code("""
            context = sv.tl.summarize_niche_surrounding_context(
                graph_adata,
                niche_key="niche_id",
                phenotype_key="phenotype",
                feature_cols=["area", "eccentricity"],
                surround_hops=2,
                min_cells=5,
            )
            display(context.head())
            """),
        ],
    )

    notebooks["04_public_api_pseudotime_single_and_combined.ipynb"] = notebook(
        "04. Pseudotime: Single-Sample and Combined Analysis",
        "04_public_api_pseudotime_single_and_combined.ipynb",
        [
            md("## Build a trajectory feature table"),
            code("""
            feature_table = pd.DataFrame(
                {
                    "niche_id": adata.obs["niche_id"].astype(str).to_numpy(),
                    "imageid": adata.obs["imageid"].astype(str).to_numpy(),
                    "n_cells": 1,
                    "pdac_early_duct_anchor_score": 1 - adata.obs["pseudotime"].to_numpy(dtype=float),
                    "pdac_dysplasia_score": adata.obs["pseudotime"].to_numpy(dtype=float),
                    "state__Ki67_expr_z__mean": pd.to_numeric(adata.obs["Ki67_expr_z"], errors="coerce").to_numpy(),
                    "surround_prop__fibroblast": adata.obs["phenotype"].astype(str).eq("fibroblast").astype(float).to_numpy(),
                    "graph_density": pd.to_numeric(adata.obs["area"], errors="coerce").rank(pct=True).to_numpy(),
                },
                index=adata.obs_names,
            )
            result = sv.tl.prepare_pseudotime_feature_matrix(
                feature_table,
                priority_features=["pdac_early_duct_anchor_score", "pdac_dysplasia_score"],
                correlation_threshold=0.98,
            )
            balanced, blocks = sv.tl.block_balance_feature_matrix(result.matrix, return_blocks=True)
            print("Selected features:", result.selected_features)
            display(blocks.head())
            """),
            md("## Principal-tree branch labels without ElPiGraph"),
            code("""
            graph = nx.Graph([(0, 1), (1, 2), (2, 3), (2, 4)])
            observations = pd.DataFrame(
                {
                    "node_id": np.repeat([0, 1, 2, 3, 4], repeats=max(1, len(feature_table) // 5) + 1)[: len(feature_table)],
                    "pseudotime": feature_table["pdac_dysplasia_score"].to_numpy(),
                    "pdac_dysplasia_score": feature_table["pdac_dysplasia_score"].to_numpy(),
                },
                index=feature_table.index,
            )
            labels, metadata = sv.tl.infer_branch_labels(graph, observations, source_node=0, node_col="node_id")
            observations["branch"] = labels.to_numpy()
            branch_summary = sv.tl.summarize_branches(
                observations,
                branch_col="branch",
                pseudotime_col="pseudotime",
                score_cols=["pdac_dysplasia_score"],
            )
            display(metadata)
            display(branch_summary)
            """),
            md("## Feature trends and branch-time matrices"),
            code("""
            trend_table = sv.tl.compute_feature_trend_table(
                feature_table,
                ["pdac_early_duct_anchor_score", "pdac_dysplasia_score", "state__Ki67_expr_z__mean"],
                pseudotime_col="pdac_dysplasia_score",
                min_n=20,
            )
            branch_bins = sv.tl.add_branch_time_bins(observations, "branch", "pseudotime", n_bins=3)
            branch_matrix = sv.tl.branch_time_feature_matrix(
                branch_bins.join(feature_table[["pdac_early_duct_anchor_score"]]),
                ["pdac_early_duct_anchor_score", "pdac_dysplasia_score"],
                branch_col="branch",
                time_bin_col="branch_time_bin",
            )
            display(trend_table)
            display(branch_matrix.head())
            """),
            md("## Cell interaction dynamics over pseudotime"),
            code("""
            source = adata.obs["phenotype"].astype(str).value_counts().index[0]
            targets = adata.obs["phenotype"].astype(str).value_counts().index[1:3].tolist()
            dynamics = sv.tl.compute_epithelial_centered_interaction_dynamics(
                adata,
                pseudotime_key="pseudotime",
                phenotype_key="phenotype",
                source_phenotype=source,
                target_phenotypes=targets,
                radius=180,
                pseudotime_bin_count=5,
            )
            dynamics_summary = sv.tl.summarize_epithelial_interaction_dynamics(dynamics, pseudotime_key="pseudotime")
            display(dynamics.head())
            display(dynamics_summary.head())
            """),
            md("## Combined-sample output audit"),
            code("""
            combined_dir = DATA_DIR / "combined_exp_2_3_4_5"
            combined_files = [
                "pooled_pathology_feature_df.pkl",
                "pooled_niche_result_df.pkl",
                "pooled_embedding_df.pkl",
                "pooled_pathology_with_panin_validation_scores.pkl",
            ]
            for name in combined_files:
                path = combined_dir / name
                print(name, "present" if path.exists() else "missing")
                if path.exists():
                    obj = pd.read_pickle(path)
                    print("  shape:", getattr(obj, "shape", None))
            """),
        ],
    )

    notebooks["05_public_api_xenium_workflows.ipynb"] = notebook(
        "05. Xenium and Spatial-Transcriptomics Helpers",
        "05_public_api_xenium_workflows.ipynb",
        [
            md("## Xenium data audit"),
            code("""
            xenium_dir = DATA_DIR / "xenium_pancreas_10x" / "annotated_h5ad"
            xenium_files = sorted(xenium_dir.glob("*.h5ad")) if xenium_dir.exists() else []
            print("Xenium annotated files:", [p.name for p in xenium_files])
            """),
            md("## Marker-set scores and rule-based annotation"),
            code("""
            rng = np.random.default_rng(12)
            expression = pd.DataFrame(
                {
                    "EPCAM": [5, 4, 0, 0, 1, 0],
                    "KRT19": [3, 4, 0, 0, 1, 0],
                    "PTPRC": [0, 0, 5, 4, 1, 2],
                    "COL1A1": [0, 1, 0, 1, 5, 4],
                    "LYZ": [0, 0, 2, 1, 0, 4],
                },
                index=[f"xenium_cell_{i}" for i in range(6)],
            )
            marker_sets = {
                "epithelial_score": ["EPCAM", "KRT19"],
                "immune_score": ["PTPRC", "LYZ"],
                "fibroblast_score": ["COL1A1"],
            }
            scores = sv.xe.compute_marker_set_scores(expression, marker_sets)
            clusters = pd.Series(["c0", "c0", "c1", "c1", "c2", "c2"], index=expression.index)
            cluster_summary = sv.xe.summarize_cluster_marker_scores(scores, clusters)
            labels = sv.xe.assign_labels_from_marker_rules(
                cluster_summary,
                {
                    "epithelial": {"epithelial_score": 0.5},
                    "immune": {"immune_score": 0.5},
                    "fibroblast": {"fibroblast_score": 0.5},
                },
            )
            display(scores)
            display(labels)
            """),
            md("## Xenium epithelial-niche histology modules"),
            code("""
            feature_table = pd.DataFrame(
                {
                    "duct_lumen__fraction": [1.0, 0.4, 0.1],
                    "duct_continuity__score": [1.0, 0.5, 0.2],
                    "state__EPCAM_mean": [3.0, 2.0, 1.0],
                    "interface__stromal_fraction": [0.0, 0.4, 0.9],
                    "state__dapi_area_mean": [1.0, 2.0, 3.0],
                    "state__nucleus_boundary_irregularity_mean": [0.1, 0.4, 0.8],
                    "surround__LYZ_mean": [0.1, 0.5, 1.2],
                }
            )
            module_scores = sv.xe.score_xenium_histology_modules(feature_table, min_features=2)
            with_modules = sv.xe.add_module_scores(feature_table, module_scores)
            display(module_scores)
            display(with_modules)
            """),
            md("## Xenium DAPI feature extraction entry point"),
            code("""
            print("Public function available:", callable(sv.xe.extract_xenium_dapi_features))
            print("Full image extraction is intentionally not run in this lightweight validation notebook.")
            """),
        ],
    )

    notebooks["06_public_api_ecm_cell_and_microenvironment.ipynb"] = notebook(
        "06. ECM-Cell and Microenvironment Interaction Workflows",
        "06_public_api_ecm_cell_and_microenvironment.ipynb",
        [
            md("## Synthetic ECM fiber table"),
            code("""
            rng = np.random.default_rng(22)
            fiber_df = pd.DataFrame(
                {
                    "imageid": rng.choice(adata.obs["imageid"].unique(), size=180),
                    "X_centroid": rng.normal(adata.obs["X_centroid"].mean(), adata.obs["X_centroid"].std(), size=180),
                    "Y_centroid": rng.normal(adata.obs["Y_centroid"].mean(), adata.obs["Y_centroid"].std(), size=180),
                    "fiber_type": rng.choice(["COL6A1", "CHP", "COL1A1"], size=180, p=[0.4, 0.35, 0.25]),
                    "orientation": rng.uniform(-np.pi / 2, np.pi / 2, size=180),
                    "length": rng.gamma(4, 15, size=180),
                    "tumor_density": rng.uniform(0, 1, size=180),
                    "alignment_score": rng.uniform(0, 1, size=180),
                },
                index=[f"fiber_{i}" for i in range(180)],
            )
            display(fiber_df.head())
            """),
            md("## Cell-fiber links and proximity statistics"),
            code("""
            links = sv.tl.build_cell_fiber_links(adata, fiber_df, radius=180)
            nearest = sv.tl.build_nearest_cell_fiber_map(adata, fiber_df)
            ecm_adata = sv.tl.cell_to_fiber_distance(adata.copy(), fiber_df, links, fiber_type="COL6A1")
            ecm_adata = sv.tl.fiber_density_near_cells(ecm_adata, fiber_df, links, fiber_type="COL6A1", density_radius=180)
            display(links.head())
            display(nearest.head())
            display(ecm_adata.obs.filter(regex="dist_to|fiber_density").head())
            """),
            md("## ECM-cell neighborhood features"),
            code("""
            neighborhoods = sv.tl.build_ecm_cell_neighborhood_features(
                ecm_adata,
                fiber_df,
                links,
                radius=180,
                phenotype_key="phenotype",
                fiber_types=("COL6A1", "CHP"),
            )
            feature_cols = sv.tl.default_ecm_cell_neighborhood_feature_columns(neighborhoods)
            clustered_result = sv.tl.cluster_ecm_cell_neighborhoods(
                neighborhoods,
                feature_columns=feature_cols,
                n_clusters=3,
                random_state=0,
            )
            clustered, ecm_cluster_model, ecm_cluster_scaler, ecm_cluster_features = clustered_result
            neighborhood_summary, group_counts, phenotype_counts = sv.tl.summarize_ecm_cell_neighborhoods(clustered, phenotype_key="phenotype")
            scored, dark_cluster_scores = sv.tl.score_col6_dark_neighborhoods(clustered)
            display(clustered.head())
            display(neighborhood_summary.head())
            display(phenotype_counts.head())
            display(scored.head())
            display(dark_cluster_scores.head())
            """),
            md("## ECM graph and invasion score"),
            code("""
            graphs = sv.tl.build_ecm_bipartite_graph_per_image(ecm_adata, fiber_df, links, distance_scale=90)
            fiber_graphs = sv.tl.project_fiber_graph_per_image(graphs)
            niche_maps = sv.tl.detect_ecm_niches_per_image(fiber_graphs, random_state=0)
            fibers_with_niches = sv.tl.assign_niches_to_fibers(fiber_df, niche_maps)
            invasion = sv.tl.compute_invasion_score(fibers_with_niches)
            print({key: graph.number_of_nodes() for key, graph in graphs.items()})
            display(fibers_with_niches.head())
            display(invasion.head())
            """),
        ],
    )

    notebooks["07_public_api_external_integrations.ipynb"] = notebook(
        "07. External Integration Audit: BANKSY and SpatialCellChat",
        "07_public_api_external_integrations.ipynb",
        [
            md("""
            ## Why this notebook is lighter

            BANKSY and SpatialCellChat are external workflows with their own dependencies and
            often large intermediate files. This notebook recapitulates the validation logic:
            check expected artifacts, load compact result tables when present, and connect
            those outputs back to SpatioEv pseudotime/dynamics tables.
            """),
            code("""
            expected_patterns = {
                "banksy": list((ROOT / "results").glob("**/*banksy*")) + list(DATA_DIR.glob("**/*banksy*")),
                "spatialcellchat": list((ROOT / "results").glob("**/*spatialcellchat*")) + list(DATA_DIR.glob("**/*spatialcellchat*")),
                "pseudotime": list(DATA_DIR.glob("**/*pseudotime*")) + list((ROOT / "results").glob("**/*pseudotime*")),
            }
            for key, paths in expected_patterns.items():
                print(key, len(paths))
                for path in paths[:5]:
                    print("  ", path.relative_to(ROOT))
            """),
            md("## Minimal pseudotime-linked interaction table"),
            code("""
            bins, bin_summary = sv.tl.assign_pseudotime_bins(adata.obs["pseudotime"], n_bins=5)
            interaction = pd.DataFrame(
                {
                    "cell_id": adata.obs_names,
                    "pseudotime": adata.obs["pseudotime"].to_numpy(),
                    "pseudotime_bin": bins.to_numpy(),
                    "phenotype": adata.obs["phenotype"].astype(str).to_numpy(),
                    "banksy_domain": adata.obs["niche_id"].astype(str).to_numpy(),
                }
            )
            display(bin_summary)
            display(interaction.head())
            """),
            md("## Public API hooks used by integration notebooks"),
            code("""
            hooks = [
                sv.tl.assign_pseudotime_bins,
                sv.tl.compute_epithelial_centered_interaction_dynamics,
                sv.tl.summarize_epithelial_interaction_dynamics,
                sv.tl.compute_feature_trend_table,
                sv.xe.score_xenium_histology_modules,
            ]
            for fn in hooks:
                print(fn.__name__, "ready")
            """),
        ],
    )

    return notebooks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, nb in build_notebooks().items():
        write_notebook(filename, nb)
    write_readme()
    print(f"Wrote notebooks to {OUT_DIR}")


if __name__ == "__main__":
    main()
