#!/usr/bin/env python3
"""Write Xenium pancreas pseudotime workflow notebooks.

The notebooks are intentionally split into small stages:

1. audit raw 10x Xenium outputs and optionally convert to SpatialData
2. annotate cells with transparent marker-score rules
3. build epithelial niche/context feature tables
4. fit a pooled Xenium niche trajectory

The notebooks include both SpatialData conversion and a Scanpy + 10x H5/CSV
fallback, which is still useful for fast tabular annotation/modeling.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path("/Users/shihongwu/SpatioEv")
NOTEBOOK_DIR = ROOT / "notebooks"
DOCS_DIR = ROOT / "docs"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.rstrip() + "\n")


COMMON_SETUP = r'''
%matplotlib inline

import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba")

from pathlib import Path
import gc
import json
import importlib.util
import tarfile
import warnings
from io import StringIO

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse

plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 8
sns.set_style("white")

ROOT = Path("/Users/shihongwu/SpatioEv")
DATA_ROOT = Path("/Volumes/Shihong_5/for_spatioev/pancreas_Xenium_example_data_from_10X")
OUTPUT_DIR = ROOT / "data" / "xenium_pancreas_10x"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_CONFIGS = [
    {
        "sample_id": "pdac_pancreas_v1",
        "display_name": "Human Pancreas FFPE",
        "disease_group": "PDAC",
        "panel_hint": "Human Multi-Tissue and Cancer",
        "outs_path": DATA_ROOT / "Xenium_V1_human_Pancreas_FFPE_outs",
    },
    {
        "sample_id": "pdac_io_v1",
        "display_name": "Human Ductal Adenocarcinoma FFPE",
        "disease_group": "PDAC",
        "panel_hint": "Human Immuno-Oncology",
        "outs_path": DATA_ROOT / "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs",
    },
    {
        "sample_id": "pdac_addon_v1",
        "display_name": "hPancreas Cancer Add-on FFPE",
        "disease_group": "PDAC",
        "panel_hint": "Human Multi-Tissue + Add-on",
        "outs_path": DATA_ROOT / "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs",
    },
    {
        "sample_id": "normal_nondiseased_v1",
        "display_name": "hPancreas nondiseased section",
        "disease_group": "NormalPancreas",
        "panel_hint": "Human Multi-Tissue and Cancer",
        "outs_path": DATA_ROOT / "Xenium_V1_hPancreas_nondiseased_section_outs",
    },
]

CLINICAL_SAMPLE_METADATA = {
    "normal_nondiseased_v1": {
        "clinical_diagnosis": "Nondiseased pancreas",
        "clinical_stage": "Normal",
        "clinical_grade": "Normal",
        "clinical_grade_order": 0.0,
        "tumor_content_percent": 0.0,
        "clinical_progression_label": "Normal pancreas",
        "clinical_progression_order": 0,
        "clinical_note": "10x nondiseased pancreas reference",
    },
    "pdac_addon_v1": {
        "clinical_diagnosis": "Adenocarcinoma",
        "clinical_stage": "Not provided",
        "clinical_grade": "Grade I-II",
        "clinical_grade_order": 1.5,
        "tumor_content_percent": 50.0,
        "clinical_progression_label": "Grade I-II, 50% tumor",
        "clinical_progression_order": 1,
        "clinical_note": "10x pancreas cancer add-on sample; adenocarcinoma, Grade I-II, 50% tumor",
    },
    "pdac_pancreas_v1": {
        "clinical_diagnosis": "Adenocarcinoma",
        "clinical_stage": "Stage III",
        "clinical_grade": "Not provided",
        "clinical_grade_order": np.nan,
        "tumor_content_percent": np.nan,
        "clinical_progression_label": "Stage III adenocarcinoma",
        "clinical_progression_order": 2,
        "clinical_note": "10x human pancreas FFPE sample; Stage III adenocarcinoma",
    },
    "pdac_io_v1": {
        "clinical_diagnosis": "Pancreatic ductal adenocarcinoma",
        "clinical_stage": "Stage IIB",
        "clinical_grade": "Grade 3",
        "clinical_grade_order": 3.0,
        "tumor_content_percent": np.nan,
        "clinical_progression_label": "Stage IIB, Grade 3 PDAC",
        "clinical_progression_order": 3,
        "clinical_note": "10x human ductal adenocarcinoma FFPE sample; Stage IIB, Grade 3",
    },
}

CLINICAL_SAMPLE_ORDER = [
    "normal_nondiseased_v1",
    "pdac_addon_v1",
    "pdac_pancreas_v1",
    "pdac_io_v1",
]
CLINICAL_LABEL_ORDER = [
    CLINICAL_SAMPLE_METADATA[sample_id]["clinical_progression_label"]
    for sample_id in CLINICAL_SAMPLE_ORDER
]
CLINICAL_PALETTE = {
    "Normal pancreas": "#4daf4a",
    "Grade I-II, 50% tumor": "#ffb000",
    "Stage III adenocarcinoma": "#e41a1c",
    "Stage IIB, Grade 3 PDAC": "#6a3d9a",
}

def clinical_metadata_frame():
    return (
        pd.DataFrame.from_dict(CLINICAL_SAMPLE_METADATA, orient="index")
        .rename_axis("sample_id")
        .reset_index()
        .sort_values("clinical_progression_order")
    )

def attach_clinical_metadata(df):
    meta = clinical_metadata_frame()
    existing_meta_cols = [col for col in meta.columns if col != "sample_id" and col in df.columns]
    if existing_meta_cols:
        df = df.drop(columns=existing_meta_cols)
    return df.merge(meta, on="sample_id", how="left", validate="many_to_one")

def package_available(name):
    return importlib.util.find_spec(name) is not None

def save_df(df, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pkl":
        df.to_pickle(path)
    else:
        df.to_csv(path, index=index)

def load_df(path):
    path = Path(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path)

def present_columns(df, cols):
    return [c for c in cols if c in df.columns]

def make_sparse_safe_copy(X):
    return X.copy() if sparse.issparse(X) else np.asarray(X).copy()
'''


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "spatioev_env",
        "language": "python",
        "name": "spatioev_env",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"Wrote {path}")


audit_cells = [
    md(
        """
# Xenium Pancreas 10x Data Audit and Optional SpatialData Conversion

This notebook checks the four downloaded 10x Xenium pancreas datasets before we start annotation and pseudotime.

Why this stage matters:

- the public 10x examples use different panels, so we need to know which genes are truly shared
- `spatialdata-io` can manage Xenium outputs nicely
- the downstream notebooks also keep a Scanpy fallback from `cell_feature_matrix.h5` and `cells.csv.gz`, which is faster for tabular annotation/modeling
"""
    ),
    code(COMMON_SETUP),
    md(
        """
## Package Status

SpatialData support is available when `spatialdata-io` is installed. The conversion cell below can write `.zarr` stores; if a future environment lacks SpatialData, the notebook records a clear install message and continues with Scanpy-compatible files.
"""
    ),
    code(
        r'''
package_status = pd.Series(
    {
        "spatialdata": package_available("spatialdata"),
        "spatialdata_io": package_available("spatialdata_io"),
        "spatialdata_plot": package_available("spatialdata_plot"),
        "pyarrow": package_available("pyarrow"),
        "celltypist": package_available("celltypist"),
        "scvi": package_available("scvi"),
        "scanpy": package_available("scanpy"),
        "anndata": package_available("anndata"),
        "elpigraph": package_available("elpigraph"),
    },
    name="available",
).to_frame()
package_status
'''
    ),
    md(
        """
## Audit Raw Outputs

This reads only lightweight metadata plus the 10x feature matrix headers. It does not load transcript coordinates or full images.
"""
    ),
    code(
        r'''
FOCUS_GENES = [
    "EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "SOX9", "MUC1", "MUC5AC",
    "TFF1", "TFF2", "TFF3", "CEACAM5", "CEACAM6", "AGR2", "AGR3", "S100P",
    "MKI67", "UBE2C", "TOP2A", "CENPF", "CDK1",
    "ACTA2", "PDGFRA", "FAP", "THY1", "PDPN", "DCN", "LUM", "VIM",
    "PECAM1", "VWF", "KDR", "CDH5",
    "PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "FOXP3", "GZMB", "NKG7",
    "CD19", "MS4A1", "CD79A", "MZB1", "JCHAIN", "SDC1",
    "LST1", "LYZ", "CD68", "C1QA", "C1QB", "C1QC",
    "AMY2A", "PRSS1", "PRSS2", "CPA1", "CTRB1", "REG1A",
    "INS", "GCG", "SST", "PPY", "CHGA",
]

audit_rows = []
gene_sets = {}
focus_rows = []

for cfg in SAMPLE_CONFIGS:
    outs = Path(cfg["outs_path"])
    matrix_path = outs / "cell_feature_matrix.h5"
    cells_path = outs / "cells.csv.gz"
    metrics_path = outs / "metrics_summary.csv"
    gene_panel_path = outs / "gene_panel.json"
    cell_groups_path = outs / "cell_groups.csv"

    missing = [
        name
        for name, path in {
            "cell_feature_matrix.h5": matrix_path,
            "cells.csv.gz": cells_path,
            "metrics_summary.csv": metrics_path,
            "gene_panel.json": gene_panel_path,
        }.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"{cfg['sample_id']} missing required files: {missing}")

    adata_head = sc.read_10x_h5(matrix_path)
    gene_sets[cfg["sample_id"]] = set(adata_head.var_names)
    cells_head = pd.read_csv(cells_path, nrows=5)
    metrics = pd.read_csv(metrics_path)
    metrics_row = metrics.iloc[0].to_dict() if len(metrics) else {}

    audit_rows.append(
        {
            "sample_id": cfg["sample_id"],
            "display_name": cfg["display_name"],
            "disease_group": cfg["disease_group"],
            "panel_hint": cfg["panel_hint"],
            "outs_path": str(outs),
            "n_cells_matrix": int(adata_head.n_obs),
            "n_genes": int(adata_head.n_vars),
            "cells_csv_columns": ", ".join(cells_head.columns),
            "has_cell_groups_csv": cell_groups_path.exists(),
            "panel_name_metrics": metrics_row.get("panel_name", np.nan),
            "num_cells_metrics": metrics_row.get("num_cells_detected", np.nan),
            "median_genes_per_cell": metrics_row.get("median_genes_per_cell", np.nan),
            "median_transcripts_per_cell": metrics_row.get("median_transcripts_per_cell", np.nan),
        }
    )

    for gene in FOCUS_GENES:
        focus_rows.append(
            {
                "sample_id": cfg["sample_id"],
                "disease_group": cfg["disease_group"],
                "gene": gene,
                "present": gene in adata_head.var_names,
            }
        )

audit_df = pd.DataFrame(audit_rows)
focus_gene_df = pd.DataFrame(focus_rows)

common_genes = sorted(set.intersection(*gene_sets.values()))
union_genes = sorted(set.union(*gene_sets.values()))

save_df(audit_df, OUTPUT_DIR / "xenium_dataset_audit.csv")
save_df(focus_gene_df, OUTPUT_DIR / "xenium_focus_gene_availability.csv")
pd.Series(common_genes, name="gene").to_csv(OUTPUT_DIR / "xenium_common_genes.csv", index=False)

print(f"Common genes across all four datasets: {len(common_genes)}")
print(f"Union genes across all four datasets: {len(union_genes)}")
audit_df
'''
    ),
    code(
        r'''
focus_matrix = (
    focus_gene_df
    .pivot_table(index="gene", columns="sample_id", values="present", aggfunc="first")
    .reindex(FOCUS_GENES)
)

plt.figure(figsize=(8, 10))
sns.heatmap(
    focus_matrix.astype(float),
    cmap=sns.color_palette(["#f2f2f2", "#2b8cbe"], as_cmap=True),
    cbar=False,
    linewidths=0.2,
    linecolor="white",
)
plt.title("Focus gene availability across Xenium panels")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
'''
    ),
    md(
        """
## Optional: Convert Raw Xenium Outputs to SpatialData Zarr

The SpatialData docs describe Xenium loading through `spatialdata_io.xenium(path, ...)`, which reads the Xenium `outs` folder into a `SpatialData` object containing images, labels, points, shapes, and an AnnData table. This conversion is skipped automatically unless `spatialdata_io` is installed.
"""
    ),
    code(
        r'''
SPATIALDATA_DIR = OUTPUT_DIR / "spatialdata_zarr"
SPATIALDATA_DIR.mkdir(exist_ok=True)
FORCE_SPATIALDATA_CONVERT = False

if not package_available("spatialdata_io"):
    print(
        "spatialdata_io is not installed in this environment. "
        "If you want SpatialData conversion, install spatialdata, spatialdata-io, spatialdata-plot, and pyarrow in spatioev_env, then rerun this cell."
    )
else:
    from spatialdata_io import xenium

    for cfg in SAMPLE_CONFIGS:
        zarr_path = SPATIALDATA_DIR / f"{cfg['sample_id']}.zarr"
        if zarr_path.exists() and not FORCE_SPATIALDATA_CONVERT:
            print(f"Already exists: {zarr_path}")
            continue

        print(f"Converting {cfg['sample_id']} to SpatialData zarr...")
        sdata = xenium(
            cfg["outs_path"],
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_as_circles=True,
            transcripts=False,
            cells_labels=False,
            nucleus_labels=False,
            morphology_mip=False,
            morphology_focus=False,
            aligned_images=False,
        )
        sdata.write(zarr_path)
        print(f"Wrote {zarr_path}")
'''
    ),
]


annotation_cells = [
    md(
        """
# Xenium Pancreas Cell Annotation

This notebook creates transparent, reproducible Tier_A/Tier_B annotations for the four 10x Xenium pancreas examples.

Important design choice:

- Each sample is annotated independently using all genes/probes available in that sample after minimal QC filtering.
- We first run PCA/UMAP/Leiden, then annotate Leiden clusters using marker programs and marker expression.
- The output includes QC plots and a cluster-review table so labels can be curated before downstream pseudotime.

SingleR/reference annotation can be added later, but this notebook does not assume an R/SingleR installation. The current default is the standard spatial transcriptomics/scRNA-seq pattern: unsupervised clustering plus marker validation.
"""
    ),
    code(COMMON_SETUP),
    code(
        r'''
ANNOTATED_DIR = OUTPUT_DIR / "annotated_h5ad"
ANNOTATED_DIR.mkdir(exist_ok=True)
ANNOTATION_QC_DIR = OUTPUT_DIR / "annotation_qc"
ANNOTATION_QC_DIR.mkdir(exist_ok=True)

GENE_SETS = {
    "ductal_epithelial": ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "SOX9", "MUC1", "MUC5AC", "TFF1", "TFF2", "TFF3", "CEACAM6", "AGR2", "AGR3", "EGFR", "ERBB2", "CFTR", "FXYD2", "TM4SF4", "PROX1", "EHF"],
    # Keep duodenum strict. REG4/DMBT1/TFF3/TMPRSS2 can be PDAC/PanIN or
    # intestinalized ductal programs, so they are scored separately below.
    "duodenum_epithelial": ["CDX2", "MUC2", "KRT20", "VIL1", "FABP1", "FABP2", "ALPI", "APOA1", "APOA4", "SI"],
    "acinar_epithelial": ["AMY2A", "PRSS1", "PRSS2", "CPA1", "CPA2", "CTRB1", "CTRB2", "REG1A", "REG1B", "AQP8", "GATM", "ANPEP", "KLK1", "KLK11", "PNLIP", "CLPS"],
    "islet_endocrine": ["INS", "GCG", "SST", "PPY", "IAPP", "CHGA", "CHGB"],
    "fibroblast_stellate": ["ACTA2", "PDGFRA", "FAP", "THY1", "PDPN", "DCN", "LUM", "COL1A1", "COL1A2", "VIM", "TAGLN"],
    "endothelial": ["PECAM1", "VWF", "KDR", "CDH5", "SOX17", "ACKR1", "ADGRL4", "PLVAP", "FLT1", "CLDN5", "ESAM", "ENG", "EMCN", "CD34", "ECSCR", "CLEC14A", "AQP1", "SPARCL1", "IGFBP7"],
    "t_cell": ["PTPRC", "CD3D", "CD3E", "CD2", "CD247", "TRAC", "CD4", "CD8A", "CD8B"],
    "cytotoxic_t_nk": ["NKG7", "GNLY", "GZMA", "GZMB", "GZMK", "PRF1", "KLRD1", "KLRC1"],
    "treg_checkpoint": ["FOXP3", "IL2RA", "CTLA4", "PDCD1", "LAG3", "HAVCR2"],
    "b_cell": ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "TCL1A", "IGHM", "IGKC"],
    "plasma_cell": ["MZB1", "JCHAIN", "SDC1", "XBP1", "DERL3", "TNFRSF17", "PRDM1", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2", "IGKC"],
    "myeloid": ["LST1", "LYZ", "CD14", "CD68", "AIF1", "TYROBP", "FCGR3A", "FCGR1A", "FCGR2A", "C1QA", "C1QB", "C1QC", "CD163", "MPEG1", "CSF1R", "APOE", "CTSD", "PLA2G7", "S100A8", "S100A9", "CXCR1", "CXCR2", "ITGAX", "FGR", "LILRA5", "MCEMP1", "IL1B", "IL1A"],
    "mast_cell": ["KIT", "CPA3", "MS4A2", "GATA2", "HPGDS"],
    "proliferation": ["MKI67", "UBE2C", "TOP2A", "CENPF", "CDK1"],
    "panin_mucin_remodeling": ["MUC5AC", "TFF1", "TFF2", "TFF3", "CEACAM6", "AGR2", "AGR3", "DMBT1", "REG4", "GPX2"],
    "intestinal_like_ductal_remodeling": ["CDX2", "REG4", "DMBT1", "TMPRSS2", "GPX2", "TFF3"],
}

BROAD_SCORE_TO_TIER_A = {
    "ductal_epithelial_score": "pancreatic ductal epithelium",
    "duodenum_epithelial_score": "Duodenum epithelial",
    "acinar_epithelial_score": "pancreatic acinar epithelium",
    "islet_endocrine_score": "Islets",
    "fibroblast_stellate_score": "Fibroblasts",
    "endothelial_score": "Endothelial cells",
    "t_cell_score": "T cells",
    "b_cell_score": "B lineage",
    "plasma_cell_score": "B lineage",
    "myeloid_score": "Myeloid cells",
    "mast_cell_score": "Mast cells",
}

MARKER_EXPORT_GENES = sorted(set().union(*GENE_SETS.values()))
VALIDATION_MARKERS = [
    "EPCAM", "KRT7", "KRT19", "SOX9", "MUC1", "MUC5AC", "TFF1", "TFF2", "TFF3", "CEACAM6", "AGR2", "AGR3", "MKI67", "TOP2A",
    "CFTR", "FXYD2", "TM4SF4", "PROX1", "EHF", "CDX2", "REG4", "DMBT1", "TMPRSS2", "GPX2", "MUC2", "KRT20", "VIL1", "FABP1", "FABP2", "ALPI",
    "AMY2A", "PRSS1", "CPA1", "REG1A", "AQP8", "GATM", "ANPEP", "KLK1", "KLK11", "INS", "GCG", "SST", "PPY",
    "ACTA2", "PDGFRA", "FAP", "THY1", "PDPN", "DCN", "LUM", "COL1A1", "VIM",
    "PECAM1", "VWF", "KDR", "PLVAP", "FLT1", "SPARCL1", "IGFBP7", "RGS5", "SOX17", "CD34",
    "PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "FOXP3", "GZMB", "NKG7",
    "CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "MZB1", "JCHAIN", "SDC1", "TNFRSF17", "IGHM", "IGKC", "IGHG1", "IGHA1",
    "LST1", "LYZ", "CD68", "AIF1", "CD14", "FCGR3A", "FCGR2A", "CD163", "MPEG1", "CSF1R", "S100A9", "CXCR2", "ITGAX", "KIT", "CPA3",
]
DOTPLOT_MARKERS = [
    "EPCAM", "KRT7", "SOX9", "CFTR", "FXYD2", "TM4SF4", "MUC5AC", "TFF2", "TFF3", "CEACAM6", "AGR3",
    "CDX2", "REG4", "DMBT1", "TMPRSS2", "GPX2", "MUC2", "KRT20", "VIL1", "FABP1", "FABP2", "ALPI",
    "AMY2A", "AQP8", "GATM", "ANPEP", "KLK11", "INS", "GCG", "SST", "CHGA",
    "ACTA2", "PDGFRA", "FAP", "THY1", "PDPN", "DCN", "LUM", "PECAM1", "VWF", "KDR", "PLVAP", "FLT1", "SPARCL1", "IGFBP7", "RGS5", "SOX17", "CD34",
    "PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "FOXP3", "GZMB", "NKG7",
    "CD19", "MS4A1", "CD79A", "BANK1", "MZB1", "JCHAIN", "SDC1", "TNFRSF17", "IGKC", "IGHM",
    "LST1", "LYZ", "CD68", "AIF1", "CD14", "FCGR2A", "CD163", "MPEG1", "CSF1R", "S100A9", "CXCR2", "ITGAX", "KIT", "CPA3", "MKI67", "TOP2A",
]
TIER_A_PALETTE = {
    "pancreatic ductal epithelium": "#1f78b4",
    "Duodenum epithelial": "#8dd3c7",
    "Mucosa gland": "#80cdc1",
    "Submucosa": "#8c6d31",
    "pancreatic acinar epithelium": "#33a02c",
    "Islets": "#ff7f00",
    "Fibroblasts": "#e31a1c",
    "Endothelial cells": "#6a3d9a",
    "T cells": "#b15928",
    "B lineage": "#a6cee3",
    "Myeloid cells": "#cab2d6",
    "Mast cells": "#fb9a99",
    "Unknown": "#bdbdbd",
}
TIER_B_BASE_PALETTE = {
    "ductal/tumor epithelial": "#1f78b4",
    "proliferative ductal/tumor epithelial": "#08519c",
    "mucin/PanIN-like ductal epithelial": "#4292c6",
    "duodenum/intestinal epithelial": "#8dd3c7",
    "mucosa gland": "#80cdc1",
    "submucosa": "#8c6d31",
    "intestinal-like/PanIN-like ductal epithelial": "#2171b5",
    "acinar epithelial": "#33a02c",
    "endocrine/islet": "#ff7f00",
    "activated fibroblast/stellate": "#e31a1c",
    "fibroblast/stellate": "#fb6a4a",
    "endothelial": "#6a3d9a",
    "endothelial/perivascular": "#9e9ac8",
    "Tregs": "#b15928",
    "CD8 T cells": "#d95f02",
    "CD4/other T cells": "#fdbf6f",
    "B cells": "#a6cee3",
    "plasma-like B lineage": "#1f78b4",
    "myeloid/macrophage": "#cab2d6",
    "inflammatory myeloid": "#756bb1",
    "mast cell": "#fb9a99",
    "Unknown": "#bdbdbd",
}
LEIDEN_KEY = "leiden_annotation"
XENIUM_GRAPHCLUST_KEY = "xenium_graphclust"
XENIUM_KMEANS10_KEY = "xenium_kmeans_10"
ANNOTATION_CLUSTER_PREFERENCE = [XENIUM_GRAPHCLUST_KEY, LEIDEN_KEY]
CLUSTER_REVIEW_PATH = ANNOTATION_QC_DIR / "xenium_cluster_annotation_review.csv"
LEIDEN_RESOLUTION = 0.8
ANNOTATION_VERSION = "cluster_full_panel_v9_xenium_graphclust_io_mucosa_submucosa_k24"
CURATED_CLUSTER_LABEL_OVERRIDES = {
    ("pdac_io_v1", XENIUM_GRAPHCLUST_KEY, "2"): {
        "final_Tier_A": "Mucosa gland",
        "final_Tier_B": "mucosa gland",
        "notes": "Manual Xenium Explorer review: 10x graphclust 2 is mucosa gland, not pancreatic ductal epithelium.",
    },
    ("pdac_io_v1", XENIUM_GRAPHCLUST_KEY, "17"): {
        "final_Tier_A": "Submucosa",
        "final_Tier_B": "submucosa",
        "notes": "Manual Xenium Explorer review: 10x graphclust 17 is submucosa, not pancreatic ductal epithelium.",
    },
}

STRICT_DUODENUM_MARKERS = ["CDX2", "MUC2", "KRT20", "VIL1", "FABP1", "FABP2", "ALPI", "APOA1", "APOA4", "SI"]
NON_CDX2_DUODENUM_MARKERS = ["MUC2", "KRT20", "VIL1", "FABP1", "FABP2", "ALPI", "APOA1", "APOA4", "SI"]
INTESTINAL_LIKE_DUCTAL_MARKERS = ["CDX2", "REG4", "DMBT1", "TMPRSS2", "GPX2", "TFF3"]
EPITHELIAL_REFINEMENT_REQUIRED_GENES = ["EPCAM"]
EPITHELIAL_REFINEMENT_CANDIDATE_TIER_A = [
    "pancreatic ductal epithelium",
    "pancreatic acinar epithelium",
    "Islets",
    "Duodenum epithelial",
]
EPITHELIAL_REFINEMENT_K = 24
EPITHELIAL_REFINEMENT_MIN_CELLS = 1000

TOP_MARKER_RULES_TO_TIER_A = {
    "Duodenum epithelial": STRICT_DUODENUM_MARKERS,
    "Endothelial cells": ["PECAM1", "VWF", "KDR", "CDH5", "PLVAP", "FLT1", "CLDN5", "ESAM", "EMCN", "CD34", "ECSCR", "CLEC14A", "AQP1"],
    "Myeloid cells": ["LST1", "LYZ", "CD68", "AIF1", "CD14", "FCGR3A", "FCGR2A", "CD163", "MPEG1", "CSF1R", "APOE", "CTSD", "PLA2G7", "S100A8", "S100A9", "CXCR1", "CXCR2", "ITGAX", "FGR", "LILRA5", "MCEMP1", "IL1B", "IL1A"],
    "pancreatic acinar epithelium": ["AMY2A", "PRSS1", "PRSS2", "CPA1", "CPA2", "AQP8", "GATM", "KLK11", "REG1A", "REG1B"],
    "Fibroblasts": ["ACTA2", "PDGFRA", "FAP", "THY1", "PDPN", "DCN", "LUM", "COL1A1", "COL1A2", "VIM", "SPARC", "FN1", "VCAN", "FBN1"],
    "T cells": ["CD3D", "CD3E", "CD2", "TRAC", "CD4", "CD8A", "IL7R", "TCF7", "CCL5"],
    "B lineage": ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "MZB1", "JCHAIN", "SDC1", "IGHM", "IGKC", "IGHG1", "IGHG2", "IGHG3", "IGHG4"],
    "Mast cells": ["KIT", "CPA3", "MS4A2", "GATA2", "HPGDS", "CTSG"],
    "Islets": ["INS", "GCG", "SST", "PPY", "CHGA", "CHGB", "PCSK2"],
}
TOP_MARKER_RULE_WEIGHTS = {
    "Duodenum epithelial": 2.6,
    "Endothelial cells": 2.5,
    "Myeloid cells": 2.4,
    "T cells": 2.3,
    "B lineage": 2.3,
    "Mast cells": 2.3,
    "Islets": 2.3,
    "pancreatic acinar epithelium": 2.2,
    "Fibroblasts": 2.0,
}
TOP_MARKER_RULE_MIN_HITS = {
    "Duodenum epithelial": 2,
    "Endothelial cells": 2,
    "Myeloid cells": 2,
    "T cells": 2,
    "B lineage": 2,
    "Mast cells": 2,
    "Islets": 2,
    "pancreatic acinar epithelium": 2,
    "Fibroblasts": 2,
}

def palette_for_values(values, base_palette):
    values = pd.Index(pd.Series(values).dropna().astype(str).unique())
    fallback = sns.color_palette("tab20", n_colors=max(len(values), 1)).as_hex()
    out = {}
    for i, value in enumerate(values):
        out[value] = base_palette.get(value, fallback[i % len(fallback)])
    return out

def read_xenium_adata(cfg):
    adata = sc.read_10x_h5(Path(cfg["outs_path"]) / "cell_feature_matrix.h5")
    adata.var_names_make_unique()
    cells = pd.read_csv(Path(cfg["outs_path"]) / "cells.csv.gz").set_index("cell_id")
    obs = adata.obs.join(cells, how="left")
    obs["sample_id"] = cfg["sample_id"]
    obs["display_name"] = cfg["display_name"]
    obs["disease_group"] = cfg["disease_group"]
    obs["panel_hint"] = cfg["panel_hint"]

    adata.obs = obs
    adata = add_xenium_precomputed_clusters(adata, cfg)
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    adata.layers["counts"] = make_sparse_safe_copy(adata.X)
    return adata

def read_analysis_tar_csv(cfg, member_path):
    archive_path = Path(cfg["outs_path"]) / "analysis.tar.gz"
    if not archive_path.exists():
        return None
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            handle = tar.extractfile(member_path)
            if handle is None:
                return None
            return pd.read_csv(StringIO(handle.read().decode("utf-8")))
    except KeyError:
        return None

def load_xenium_precomputed_clusters(cfg):
    frames = []
    graphclust = read_analysis_tar_csv(
        cfg,
        "analysis/clustering/gene_expression_graphclust/clusters.csv",
    )
    if graphclust is not None:
        graphclust = graphclust.rename(columns={"Barcode": "cell_id", "Cluster": XENIUM_GRAPHCLUST_KEY})
        graphclust["cell_id"] = graphclust["cell_id"].astype(str)
        graphclust[XENIUM_GRAPHCLUST_KEY] = graphclust[XENIUM_GRAPHCLUST_KEY].astype(str)
        frames.append(graphclust[["cell_id", XENIUM_GRAPHCLUST_KEY]])

    kmeans10 = read_analysis_tar_csv(
        cfg,
        "analysis/clustering/gene_expression_kmeans_10_clusters/clusters.csv",
    )
    if kmeans10 is not None:
        kmeans10 = kmeans10.rename(columns={"Barcode": "cell_id", "Cluster": XENIUM_KMEANS10_KEY})
        kmeans10["cell_id"] = kmeans10["cell_id"].astype(str)
        kmeans10[XENIUM_KMEANS10_KEY] = kmeans10[XENIUM_KMEANS10_KEY].astype(str)
        frames.append(kmeans10[["cell_id", XENIUM_KMEANS10_KEY]])

    groups_path = Path(cfg["outs_path"]) / "cell_groups.csv"
    if groups_path.exists():
        groups = pd.read_csv(groups_path).rename(columns={"group": "xenium_10x_cell_group"})
        groups["cell_id"] = groups["cell_id"].astype(str)
        groups["xenium_10x_cell_group"] = groups["xenium_10x_cell_group"].astype(str)
        frames.append(groups[["cell_id", "xenium_10x_cell_group"]])

    if len(frames) == 0:
        return pd.DataFrame(columns=["cell_id"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="cell_id", how="outer")
    return out

def add_xenium_precomputed_clusters(adata, cfg):
    cluster_df = load_xenium_precomputed_clusters(cfg)
    if cluster_df.empty:
        adata.uns["xenium_precomputed_cluster_status"] = "not_found"
        return adata
    cluster_df = cluster_df.drop_duplicates("cell_id").set_index("cell_id")
    for col in cluster_df.columns:
        adata.obs[col] = cluster_df[col].reindex(adata.obs_names).astype("string").fillna("unassigned").astype(str)
    adata.uns["xenium_precomputed_cluster_status"] = "loaded"
    adata.uns["xenium_precomputed_cluster_columns"] = cluster_df.columns.tolist()
    return adata

def choose_annotation_cluster_key(adata):
    for key in ANNOTATION_CLUSTER_PREFERENCE:
        if key in adata.obs.columns and adata.obs[key].astype(str).ne("unassigned").any():
            return key
    if LEIDEN_KEY in adata.obs.columns:
        return LEIDEN_KEY
    raise KeyError("No usable annotation clustering key found.")

def get_matrix(adata, layer=None):
    if layer is None:
        return adata.X
    return adata.layers[layer]

def matrix_mean(adata, genes, layer=None):
    genes = [g for g in genes if g in adata.var_names]
    if len(genes) == 0:
        return np.full(adata.n_obs, np.nan, dtype=float), genes
    X = get_matrix(adata[:, genes], layer=layer)
    if sparse.issparse(X):
        vals = np.asarray(X.mean(axis=1)).ravel()
    else:
        vals = np.asarray(X).mean(axis=1)
    return vals.astype(float), genes

def zscore(values):
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or np.isclose(sd, 0):
        return np.full(values.shape, np.nan)
    return (values - mu) / sd

def vector_from_gene(adata, gene, layer=None):
    X = get_matrix(adata[:, [gene]], layer=layer)
    vals = X.toarray().ravel() if sparse.issparse(X) else np.asarray(X).ravel()
    return vals.astype(float)

def add_gene_expression_obs(adata, genes, layer="log1p"):
    for gene in genes:
        if gene not in adata.var_names:
            continue
        vals = vector_from_gene(adata, gene, layer=layer)
        adata.obs[f"{gene}_expr"] = vals.astype(float)
        adata.obs[f"{gene}_expr_z"] = zscore(vals)
    return adata

def preprocess_for_independent_annotation(adata):
    sc.pp.filter_cells(adata, min_counts=5)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.layers["counts"] = make_sparse_safe_copy(adata.X)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = make_sparse_safe_copy(adata.X)

    # Use the full retained Xenium panel for PCA, not a cross-sample HVG subset.
    n_comps = int(min(40, max(2, adata.n_vars - 1), max(2, adata.n_obs - 1)))
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", random_state=42)
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=min(30, n_comps), random_state=42)
    sc.tl.umap(adata, min_dist=0.35, random_state=42)
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY, random_state=42)
    return adata

def add_marker_scores(adata):
    resolved_sets = {}
    for set_name, genes in GENE_SETS.items():
        vals, present = matrix_mean(adata, genes, layer="log1p")
        adata.obs[f"{set_name}_score"] = vals
        adata.obs[f"{set_name}_score_z"] = zscore(vals)
        resolved_sets[set_name] = present
    adata = add_gene_expression_obs(adata, MARKER_EXPORT_GENES, layer="log1p")
    adata.uns["xenium_marker_sets_resolved"] = resolved_sets
    return adata

def top_markers_by_cluster(adata, cluster_key=LEIDEN_KEY, n_top=8):
    X = adata.layers["log1p"]
    if sparse.issparse(X):
        global_mean = np.asarray(X.mean(axis=0)).ravel()
    else:
        global_mean = np.asarray(X).mean(axis=0)

    out = {}
    clusters = adata.obs[cluster_key].astype(str)
    for cluster in sorted(clusters.unique(), key=lambda x: (len(x), x)):
        mask = clusters.to_numpy() == cluster
        if sparse.issparse(X):
            mean = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            mean = np.asarray(X[mask]).mean(axis=0)
        diff = mean - global_mean
        order = np.argsort(diff)[::-1][:n_top]
        out[cluster] = ", ".join(adata.var_names[order].tolist())
    return out

def summarize_annotation_clusters(adata, cfg, cluster_key=None):
    cluster_key = cluster_key or choose_annotation_cluster_key(adata)
    clusters = adata.obs[cluster_key].astype(str)
    top_markers = top_markers_by_cluster(adata, cluster_key=cluster_key)
    score_cols = [f"{name}_score_z" for name in GENE_SETS if f"{name}_score_z" in adata.obs.columns]
    marker_genes = [g for g in VALIDATION_MARKERS if g in adata.var_names and f"{g}_expr" in adata.obs.columns]

    rows = []
    for cluster in sorted(clusters.unique(), key=lambda x: (len(x), x)):
        mask = clusters == cluster
        row = {
            "sample_id": cfg["sample_id"],
            "display_name": cfg["display_name"],
            "disease_group": cfg["disease_group"],
            "cluster_key": cluster_key,
            "leiden": cluster,
            "n_cells": int(mask.sum()),
            "fraction": float(mask.mean()),
            "top_marker_genes": top_markers.get(cluster, ""),
        }
        for col in score_cols:
            row[col] = float(pd.to_numeric(adata.obs.loc[mask, col], errors="coerce").mean())
        for gene in marker_genes:
            expr = pd.to_numeric(adata.obs.loc[mask, f"{gene}_expr"], errors="coerce")
            row[f"{gene}_mean"] = float(expr.mean())
            row[f"{gene}_frac_pos"] = float((expr > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)

def suggest_tier_a_from_cluster(row):
    candidates = []
    for score_col, label in BROAD_SCORE_TO_TIER_A.items():
        z_col = score_col.replace("_score", "_score_z")
        if z_col in row and pd.notna(row[z_col]):
            candidates.append((float(row[z_col]), label, z_col))

    # Unsupervised marker genes are treated as additional evidence. This is
    # especially important for panels where canonical markers are missing from
    # a curated score, e.g. pdac_io_v1 endothelial cells with PLVAP/FLT1.
    for label, markers in TOP_MARKER_RULES_TO_TIER_A.items():
        marker_hits = top_marker_count(row, markers)
        if marker_hits >= TOP_MARKER_RULE_MIN_HITS.get(label, 1):
            score_cols = [
                score_col.replace("_score", "_score_z")
                for score_col, score_label in BROAD_SCORE_TO_TIER_A.items()
                if score_label == label
            ]
            score_vals = [row_value(row, col, np.nan) for col in score_cols]
            score_vals = [v for v in score_vals if pd.notna(v)]
            base_score = max(score_vals) if len(score_vals) > 0 else 0.0
            rule_score = TOP_MARKER_RULE_WEIGHTS.get(label, 1.75)
            candidates.append((max(float(base_score), rule_score), label, f"top_marker_rule_{marker_hits}_hits"))

    if not candidates:
        return "Unknown", np.nan, "no marker program available"
    candidates = sorted(candidates, reverse=True)
    supported = []
    unsupported = []
    for score, label, col in candidates:
        if label_supported_by_anchors(row, label):
            supported.append((score, label, col))
        else:
            unsupported.append((score, label, col))

    if len(supported) == 0:
        return "Unknown", candidates[0][0], f"no candidate label passed anchor checks; best was {candidates[0][2]}"

    best_score, best_label, best_col = supported[0]
    second_score = supported[1][0] if len(supported) > 1 else np.nan
    if not np.isfinite(best_score) or best_score < -0.15:
        return "Unknown", best_score, f"weak best score: {best_col}"
    if np.isfinite(second_score) and (best_score - second_score) < 0.08:
        note = f"ambiguous: {best_col} close to second-best"
    else:
        note = f"best marker program: {best_col}"
    if unsupported and unsupported[0][0] > best_score:
        note += f"; vetoed unsupported {unsupported[0][2]}"
    return best_label, best_score, note

def row_value(row, key, default=np.nan):
    return row[key] if key in row and pd.notna(row[key]) else default

def top_marker_hit(row, markers):
    return top_marker_count(row, markers) > 0

def top_marker_count(row, markers):
    top = str(row_value(row, "top_marker_genes", "")).replace(" ", "")
    top_markers = set(top.split(",")) if top else set()
    return sum(marker in top_markers for marker in markers)

def max_marker_mean(row, markers):
    vals = [row_value(row, f"{marker}_mean", np.nan) for marker in markers]
    vals = [v for v in vals if pd.notna(v)]
    return np.nanmax(vals) if vals else np.nan

def max_marker_frac(row, markers):
    vals = [row_value(row, f"{marker}_frac_pos", np.nan) for marker in markers]
    vals = [v for v in vals if pd.notna(v)]
    return np.nanmax(vals) if vals else np.nan

def label_supported_by_anchors(row, label):
    # Ductal and endocrine classes can use marker programs directly. Acinar and
    # duodenum calls are checked with top-marker anchors because some panels have
    # only weak overlapping epithelial genes.
    if label in {"pancreatic ductal epithelium", "Islets"}:
        return True

    anchor_markers = {
        "Duodenum epithelial": STRICT_DUODENUM_MARKERS,
        "pancreatic acinar epithelium": ["AMY2A", "PRSS1", "PRSS2", "CPA1", "CPA2", "AQP8", "GATM", "KLK11", "REG1A", "REG1B"],
        "Fibroblasts": ["PDGFRA", "ACTA2", "THY1", "PDPN", "DCN", "LUM", "COL1A1", "VIM", "SPARC", "FN1", "VCAN", "FBN1"],
        "Endothelial cells": ["PECAM1", "VWF", "KDR", "CDH5", "SOX17", "ACKR1", "ADGRL4", "PLVAP", "FLT1", "CLDN5", "ESAM", "EMCN", "CD34", "ECSCR", "CLEC14A", "AQP1"],
        "T cells": ["CD3D", "CD3E", "CD2", "TRAC", "CD4", "CD8A", "IL7R", "TCF7", "CCL5"],
        "B lineage": ["CD19", "MS4A1", "CD79A", "CD79B", "BANK1", "MZB1", "JCHAIN", "SDC1", "TNFRSF17", "IGHM", "IGKC", "IGHG1", "IGHG2", "IGHG3", "IGHG4", "IGHA1", "IGHA2"],
        "Myeloid cells": ["LST1", "LYZ", "CD14", "CD68", "AIF1", "FCGR3A", "FCGR1A", "FCGR2A", "C1QA", "C1QB", "C1QC", "CD163", "MPEG1", "CSF1R", "APOE", "CTSD", "PLA2G7", "S100A8", "S100A9", "CXCR1", "CXCR2", "ITGAX", "FGR", "LILRA5", "MCEMP1", "IL1B", "IL1A"],
        "Mast cells": ["KIT", "CPA3", "MS4A2", "GATA2", "HPGDS", "CTSG"],
    }.get(label, [])

    if len(anchor_markers) == 0:
        return True
    max_mean = max_marker_mean(row, anchor_markers)
    max_frac = max_marker_frac(row, anchor_markers)
    ptprc_mean = row_value(row, "PTPRC_mean", 0)

    if label in {"T cells", "B lineage", "Myeloid cells"}:
        if top_marker_count(row, anchor_markers) >= 3:
            return True
        if top_marker_count(row, anchor_markers) >= 2 and ptprc_mean >= 1.2:
            return True
        return (ptprc_mean >= 2.0) and (
            (pd.notna(max_mean) and max_mean >= 1.2)
            or (pd.notna(max_frac) and max_frac >= 0.25)
        )
    if label == "Endothelial cells":
        return (
            (pd.notna(max_mean) and max_mean >= 1.0)
            or (pd.notna(max_frac) and max_frac >= 0.18)
            or top_marker_hit(row, anchor_markers)
        )
    if label == "Duodenum epithelial":
        # REG4/DMBT1/TFF3/TMPRSS2 are deliberately excluded here because they
        # can be PDAC/PanIN-like ductal remodeling. A duodenum call needs either
        # two strict intestinal markers, or CDX2 plus another strict intestinal
        # marker such as KRT20/VIL1/FABP/ALPI/MUC2.
        strict_hits = top_marker_count(row, STRICT_DUODENUM_MARKERS)
        cdx2_frac = row_value(row, "CDX2_frac_pos", np.nan)
        cdx2_mean = row_value(row, "CDX2_mean", np.nan)
        non_cdx2_frac = max_marker_frac(row, NON_CDX2_DUODENUM_MARKERS)
        non_cdx2_mean = max_marker_mean(row, NON_CDX2_DUODENUM_MARKERS)
        has_cdx2 = (
            top_marker_hit(row, ["CDX2"])
            or (pd.notna(cdx2_frac) and cdx2_frac >= 0.25)
            or (pd.notna(cdx2_mean) and cdx2_mean >= 1.0)
        )
        has_second_intestinal_anchor = (
            top_marker_hit(row, NON_CDX2_DUODENUM_MARKERS)
            or (pd.notna(non_cdx2_frac) and non_cdx2_frac >= 0.20)
            or (pd.notna(non_cdx2_mean) and non_cdx2_mean >= 1.0)
        )
        return strict_hits >= 2 or (has_cdx2 and has_second_intestinal_anchor)
    if top_marker_hit(row, anchor_markers):
        return True
    if label == "Mast cells":
        return (
            pd.notna(max_mean)
            and pd.notna(max_frac)
            and max_mean >= 1.5
            and max_frac >= 0.30
        )
    return (
        (pd.notna(max_mean) and max_mean >= 1.0)
        or (pd.notna(max_frac) and max_frac >= 0.20)
    )

def suggest_tier_b_from_cluster(row, tier_a):
    if tier_a == "pancreatic ductal epithelium":
        intestinal_like_score = row_value(row, "intestinal_like_ductal_remodeling_score_z", np.nan)
        intestinal_like_hits = top_marker_count(row, INTESTINAL_LIKE_DUCTAL_MARKERS)
        cdx2_frac = row_value(row, "CDX2_frac_pos", np.nan)
        support_frac = max_marker_frac(row, ["REG4", "DMBT1", "TMPRSS2", "GPX2", "TFF3"])
        panin = np.nanmax([
            row_value(row, "panin_mucin_remodeling_score_z"),
            row_value(row, "MUC5AC_mean"),
            row_value(row, "TFF1_mean"),
            row_value(row, "TFF2_mean"),
            row_value(row, "TFF3_mean"),
            row_value(row, "CEACAM6_mean"),
        ])
        prolif = np.nanmax([row_value(row, "proliferation_score_z"), row_value(row, "MKI67_mean"), row_value(row, "TOP2A_mean")])
        if (
            intestinal_like_hits >= 2
            or (
                pd.notna(cdx2_frac)
                and cdx2_frac >= 0.15
                and (
                    (pd.notna(support_frac) and support_frac >= 0.25)
                    or (np.isfinite(intestinal_like_score) and intestinal_like_score > 0.75)
                )
            )
            or (
                np.isfinite(intestinal_like_score)
                and intestinal_like_score > 1.0
                and top_marker_hit(row, INTESTINAL_LIKE_DUCTAL_MARKERS)
            )
        ):
            return "intestinal-like/PanIN-like ductal epithelial"
        if np.isfinite(prolif) and prolif > 0.35:
            return "proliferative ductal/tumor epithelial"
        if np.isfinite(panin) and panin > 0.35:
            return "mucin/PanIN-like ductal epithelial"
        return "ductal/tumor epithelial"
    if tier_a == "Duodenum epithelial":
        return "duodenum/intestinal epithelial"
    if tier_a == "pancreatic acinar epithelium":
        return "acinar epithelial"
    if tier_a == "Islets":
        return "endocrine/islet"
    if tier_a == "Fibroblasts":
        if row_value(row, "ACTA2_mean", 0) > 0 or row_value(row, "fibroblast_stellate_score_z", 0) > 0.6:
            return "activated fibroblast/stellate"
        return "fibroblast/stellate"
    if tier_a == "Endothelial cells":
        if top_marker_hit(row, ["PLVAP", "FLT1", "RGS5", "SPARCL1", "IGFBP7"]):
            return "endothelial/perivascular"
        return "endothelial"
    if tier_a == "T cells":
        if row_value(row, "FOXP3_mean", 0) > 0:
            return "Tregs"
        if row_value(row, "CD8A_mean", 0) > row_value(row, "CD4_mean", 0):
            return "CD8 T cells"
        if row_value(row, "cytotoxic_t_nk_score_z", 0) > 0.4:
            return "cytotoxic T/NK-like"
        return "CD4/other T cells"
    if tier_a == "B lineage":
        if row_value(row, "plasma_cell_score_z", 0) > row_value(row, "b_cell_score_z", 0):
            return "plasma-like B lineage"
        return "B cells"
    if tier_a == "Myeloid cells":
        if top_marker_hit(row, ["S100A8", "S100A9", "CXCR1", "CXCR2", "IL1B", "IL1A", "MCEMP1"]):
            return "inflammatory myeloid"
        return "myeloid/macrophage"
    if tier_a == "Mast cells":
        return "mast cell"
    return "Unknown"

def add_suggested_cluster_labels(cluster_df):
    cluster_df = cluster_df.copy()
    suggested = cluster_df.apply(lambda row: suggest_tier_a_from_cluster(row), axis=1)
    cluster_df["suggested_Tier_A"] = [x[0] for x in suggested]
    cluster_df["annotation_score"] = [x[1] for x in suggested]
    cluster_df["annotation_note"] = [x[2] for x in suggested]
    cluster_df["suggested_Tier_B"] = [
        suggest_tier_b_from_cluster(row, row["suggested_Tier_A"])
        for _, row in cluster_df.iterrows()
    ]
    return cluster_df

def apply_curated_cluster_label_overrides(review_df):
    review_df = review_df.copy()
    for (sample_id, cluster_key, cluster), override in CURATED_CLUSTER_LABEL_OVERRIDES.items():
        mask = (
            review_df["sample_id"].astype(str).eq(sample_id)
            & review_df["cluster_key"].astype(str).eq(cluster_key)
            & review_df["leiden"].astype(str).eq(str(cluster))
        )
        if not mask.any():
            continue
        for col, value in override.items():
            review_df.loc[mask, col] = value
        review_df.loc[mask, "annotation_note"] = (
            review_df.loc[mask, "annotation_note"].astype(str)
            + "; curated graphclust override"
        )
    return review_df

def update_cluster_review_table(cluster_df):
    review_cols = [
        "annotation_version", "sample_id", "cluster_key", "leiden", "n_cells", "fraction", "top_marker_genes",
        "suggested_Tier_A", "suggested_Tier_B", "annotation_score", "annotation_note",
        "final_Tier_A", "final_Tier_B", "notes",
    ]
    new_review = cluster_df.copy()
    new_review["annotation_version"] = ANNOTATION_VERSION
    new_review["final_Tier_A"] = new_review["suggested_Tier_A"]
    new_review["final_Tier_B"] = new_review["suggested_Tier_B"]
    new_review["notes"] = ""
    new_review = new_review[[c for c in review_cols if c in new_review.columns]]

    if CLUSTER_REVIEW_PATH.exists():
        old = pd.read_csv(CLUSTER_REVIEW_PATH, dtype={"leiden": str})
        if "annotation_version" in old.columns:
            old = old.loc[old["annotation_version"] == ANNOTATION_VERSION].copy()
        else:
            old = old.iloc[0:0].copy()
        if "cluster_key" not in old.columns:
            old["cluster_key"] = ""
        keep_cols = ["sample_id", "cluster_key", "leiden", "final_Tier_A", "final_Tier_B", "notes"]
        old_keep = old.reindex(columns=keep_cols).dropna(subset=["sample_id", "leiden"], how="any").copy()
        if "cluster_key" not in old_keep.columns or old_keep["cluster_key"].isna().all():
            old_keep["cluster_key"] = ""
            new_review["cluster_key"] = new_review.get("cluster_key", "")
        merge_cols = ["sample_id", "cluster_key", "leiden"]
        new_review = new_review.drop(columns=["final_Tier_A", "final_Tier_B", "notes"], errors="ignore").merge(
            old_keep,
            on=merge_cols,
            how="left",
        )
        new_review["final_Tier_A"] = new_review["final_Tier_A"].fillna(new_review["suggested_Tier_A"])
        new_review["final_Tier_B"] = new_review["final_Tier_B"].fillna(new_review["suggested_Tier_B"])
        new_review["notes"] = new_review["notes"].fillna("")

        old_other = old.loc[~old.set_index(merge_cols).index.isin(
            new_review.set_index(merge_cols).index
        )]
        if len(old_other) > 0:
            new_review = pd.concat([old_other.reindex(columns=new_review.columns), new_review], ignore_index=True)

    new_review = apply_curated_cluster_label_overrides(new_review)
    new_review = new_review.sort_values(["sample_id", "cluster_key", "leiden"], key=lambda s: s.astype(str))
    new_review.to_csv(CLUSTER_REVIEW_PATH, index=False)
    return new_review

def apply_review_labels(adata, review_df, cfg):
    sample_review = review_df.loc[review_df["sample_id"] == cfg["sample_id"]].copy()
    sample_review["leiden"] = sample_review["leiden"].astype(str)
    if "cluster_key" in sample_review.columns and sample_review["cluster_key"].notna().any():
        cluster_key = sample_review["cluster_key"].dropna().astype(str).mode().iat[0]
    else:
        cluster_key = choose_annotation_cluster_key(adata)
    tier_a_map = sample_review.set_index("leiden")["final_Tier_A"].to_dict()
    tier_b_map = sample_review.set_index("leiden")["final_Tier_B"].to_dict()
    adata.obs["spatioev_annotation_cluster_key"] = cluster_key
    adata.obs["spatioev_annotation_cluster"] = adata.obs[cluster_key].astype(str)
    adata.uns["spatioev_annotation_cluster_key"] = cluster_key
    adata.obs["Tier_A"] = adata.obs[cluster_key].astype(str).map(tier_a_map).fillna("Unknown").astype(str)
    adata.obs["Tier_B"] = adata.obs[cluster_key].astype(str).map(tier_b_map).fillna("Unknown").astype(str)
    adata.obs["Tier_A_auto"] = adata.obs["Tier_A"]
    adata.obs["Tier_B_auto"] = adata.obs["Tier_B"]
    return adata

def refine_epithelial_labels_by_subclustering(adata, cfg):
    """Refine epithelial labels after broad all-cell Leiden annotation.

    The first pass assigns one label per whole-sample Leiden cluster. That can
    hide normal ductal cells inside large exocrine/acinar clusters. Here we run a
    focused epithelial-only PCA + MiniBatchKMeans pass across ductal, acinar,
    islet, and duodenum candidates, then interpret each subcluster using the
    marker-program z-scores already computed per cell. Non-epithelial labels are
    deliberately left untouched.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    adata.obs["Tier_A_cluster_level"] = adata.obs["Tier_A"].astype(str)
    adata.obs["Tier_B_cluster_level"] = adata.obs["Tier_B"].astype(str)
    adata.obs["epithelial_refinement_cluster"] = "not_run"
    adata.obs["epithelial_refinement_label"] = "not_run"

    available_genes = set(adata.var_names)
    if not set(EPITHELIAL_REFINEMENT_REQUIRED_GENES).issubset(available_genes):
        adata.uns["epithelial_refinement_status"] = "skipped_required_genes_missing"
        return adata, pd.DataFrame()
    if "log1p" not in adata.layers.keys():
        adata.uns["epithelial_refinement_status"] = "skipped_log1p_layer_missing"
        return adata, pd.DataFrame()

    candidate_mask = adata.obs["Tier_A"].astype(str).isin(EPITHELIAL_REFINEMENT_CANDIDATE_TIER_A).to_numpy()
    if int(candidate_mask.sum()) < EPITHELIAL_REFINEMENT_MIN_CELLS:
        adata.uns["epithelial_refinement_status"] = "skipped_too_few_epithelial_candidates"
        return adata, pd.DataFrame()

    X = adata.layers["log1p"][candidate_mask, :]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    gene_keep = (X > 0).sum(axis=0) >= 20
    if int(gene_keep.sum()) < 5:
        adata.uns["epithelial_refinement_status"] = "skipped_too_few_expressed_genes"
        return adata, pd.DataFrame()

    X = X[:, gene_keep]
    genes = np.asarray(adata.var_names)[gene_keep]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    Xz = (X - mean) / std

    n_components = int(min(30, Xz.shape[1] - 1, Xz.shape[0] - 1))
    if n_components < 2:
        adata.uns["epithelial_refinement_status"] = "skipped_too_few_components"
        return adata, pd.DataFrame()

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    embedding = pca.fit_transform(Xz)
    n_clusters = int(min(EPITHELIAL_REFINEMENT_K, max(4, candidate_mask.sum() // 2500)))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4096,
        n_init=10,
    )
    subcluster_labels = kmeans.fit_predict(embedding).astype(str)

    candidate_obs_names = adata.obs_names[candidate_mask]
    candidate_obs = adata.obs.loc[candidate_obs_names].copy()
    annotation_cluster_key = choose_annotation_cluster_key(adata)
    adata.obs.loc[candidate_obs_names, "epithelial_refinement_cluster"] = subcluster_labels

    marker_genes = [
        "EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "SOX9", "MUC1",
        "CFTR", "FXYD2", "TM4SF4", "PROX1", "EHF", "MUC5AC", "TFF1",
        "TFF2", "TFF3", "CEACAM6", "AGR2", "AGR3", "CDX2", "REG4",
        "DMBT1", "TMPRSS2", "GPX2", "KRT20", "MUC2", "VIL1", "FABP1",
        "FABP2", "ALPI", "AMY2A", "PRSS1", "PRSS2", "CPA1", "CPA2",
        "AQP8", "GATM", "ANPEP", "KLK11", "INS", "GCG", "SST", "PPY",
        "CHGA", "MKI67", "UBE2C", "TOP2A",
    ]
    marker_genes = [gene for gene in marker_genes if gene in genes]
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    marker_expr = pd.DataFrame(
        X[:, [gene_to_idx[gene] for gene in marker_genes]],
        index=candidate_obs_names,
        columns=marker_genes,
    )

    present_counts = {
        set_name: len([gene for gene in geneset if gene in available_genes])
        for set_name, geneset in GENE_SETS.items()
    }
    acinar_ready = present_counts.get("acinar_epithelial", 0) >= 3
    islet_ready = present_counts.get("islet_endocrine", 0) >= 2
    duodenum_ready = all(gene in available_genes for gene in ["CDX2", "REG4", "DMBT1", "TMPRSS2"])

    score_cols = [
        "ductal_epithelial_score_z",
        "duodenum_epithelial_score_z",
        "acinar_epithelial_score_z",
        "islet_endocrine_score_z",
        "panin_mucin_remodeling_score_z",
        "intestinal_like_ductal_remodeling_score_z",
        "proliferation_score_z",
    ]
    score_cols = [col for col in score_cols if col in candidate_obs.columns]

    def finite_mean(values):
        values = [value for value in values if pd.notna(value)]
        return float(np.mean(values)) if len(values) > 0 else np.nan

    def classify_epithelial_subcluster(row):
        parent_a = row.get("parent_Tier_A_major", "Unknown")
        parent_b = row.get("parent_Tier_B_major", "Unknown")
        ductal = row_value(row, "ductal_epithelial_score_z", np.nan)
        acinar = row_value(row, "acinar_epithelial_score_z", np.nan)
        islet = row_value(row, "islet_endocrine_score_z", np.nan)
        duodenum = row_value(row, "duodenum_epithelial_score_z", np.nan)

        is_duodenum_like = (
            duodenum_ready
            and row.get("CDX2_frac_pos", 0) >= 0.25
            and row.get("REG4_frac_pos", 0) >= 0.60
            and row.get("DMBT1_frac_pos", 0) >= 0.65
            and row.get("TMPRSS2_frac_pos", 0) >= 0.25
            and row["duodenum_minus_ductal_score"] >= 1.50
        )
        if is_duodenum_like:
            return "Duodenum epithelial", "duodenum/intestinal epithelial"

        # Preserve existing duodenum calls unless the subcluster is clearly a
        # ductal/PanIN-like population. This avoids undoing the pdac_io_v1
        # duodenum refinement unless the marker programs strongly disagree.
        if parent_a == "Duodenum epithelial":
            if np.isfinite(ductal) and np.isfinite(duodenum) and ductal > duodenum + 0.75 and ductal > 0.75:
                return "pancreatic ductal epithelium", suggest_tier_b_from_cluster(pd.Series(row), "pancreatic ductal epithelium")
            return parent_a, parent_b

        # Existing ductal calls are trusted unless a subcluster has very strong
        # endocrine support. The goal here is mainly to rescue missed ductal
        # cells, not aggressively remove ductal cells from PDAC regions.
        if parent_a == "pancreatic ductal epithelium":
            if islet_ready and np.isfinite(islet) and islet > max(ductal, acinar) + 1.00 and islet > 1.00:
                return "Islets", "endocrine/islet"
            return parent_a, parent_b

        if parent_a == "Islets":
            if np.isfinite(ductal) and ductal > islet + 1.00 and ductal > 1.00:
                return "pancreatic ductal epithelium", suggest_tier_b_from_cluster(pd.Series(row), "pancreatic ductal epithelium")
            return parent_a, parent_b

        if (
            parent_a == "pancreatic acinar epithelium"
            and np.isfinite(ductal)
            and ductal > 0
            and (
                (np.isfinite(acinar) and ductal > acinar + 0.25)
                or not acinar_ready
            )
        ):
            return "pancreatic ductal epithelium", suggest_tier_b_from_cluster(pd.Series(row), "pancreatic ductal epithelium")

        if (
            parent_a == "pancreatic acinar epithelium"
            and acinar_ready
            and np.isfinite(acinar)
            and acinar > 0
        ):
            return "pancreatic acinar epithelium", "acinar epithelial"

        return parent_a, parent_b

    rows = []
    refined_tier_a_by_subcluster = {}
    refined_tier_b_by_subcluster = {}
    for subcluster in sorted(pd.unique(subcluster_labels), key=lambda x: (len(str(x)), str(x))):
        idx = subcluster_labels == subcluster
        obs_idx = candidate_obs_names[idx]
        row = {
            "sample_id": cfg["sample_id"],
            "method": f"epithelial_pca_kmeans_{n_clusters}",
            "subcluster": str(subcluster),
            "n_cells": int(idx.sum()),
            "fraction_of_epithelial_candidates": float(idx.mean()),
        }
        parent_leiden_mix = candidate_obs.loc[obs_idx, annotation_cluster_key].astype(str).value_counts(normalize=True)
        parent_tier_a_mix = candidate_obs.loc[obs_idx, "Tier_A"].astype(str).value_counts(normalize=True)
        parent_tier_b_mix = candidate_obs.loc[obs_idx, "Tier_B"].astype(str).value_counts(normalize=True)
        row["parent_annotation_cluster_key"] = annotation_cluster_key
        row["parent_annotation_cluster_mix"] = "; ".join(f"{label}:{frac:.2f}" for label, frac in parent_leiden_mix.head(4).items())
        row["parent_leiden_mix"] = row["parent_annotation_cluster_mix"]
        row["parent_Tier_A_mix"] = "; ".join(f"{label}:{frac:.2f}" for label, frac in parent_tier_a_mix.head(4).items())
        row["parent_Tier_B_mix"] = "; ".join(f"{label}:{frac:.2f}" for label, frac in parent_tier_b_mix.head(4).items())
        row["parent_Tier_A_major"] = parent_tier_a_mix.index[0]
        row["parent_Tier_B_major"] = parent_tier_b_mix.index[0]

        for col in score_cols:
            row[col] = float(pd.to_numeric(candidate_obs.loc[obs_idx, col], errors="coerce").mean())
        for gene in marker_genes:
            vals = marker_expr.loc[obs_idx, gene]
            row[f"{gene}_mean"] = float(vals.mean())
            row[f"{gene}_frac_pos"] = float((vals > 0).mean())

        duodenum_panel_score = finite_mean([
            row.get("CDX2_mean", np.nan),
            row.get("REG4_mean", np.nan),
            row.get("DMBT1_mean", np.nan),
            row.get("TMPRSS2_mean", np.nan),
        ])
        ductal_tumor_score = finite_mean([
            row.get("SOX9_mean", np.nan),
            row.get("CEACAM6_mean", np.nan),
            row.get("MUC5AC_mean", np.nan),
            row.get("TFF3_mean", np.nan),
        ])
        row["duodenum_panel_score"] = float(duodenum_panel_score)
        row["ductal_tumor_score"] = float(ductal_tumor_score)
        row["duodenum_minus_ductal_score"] = float(duodenum_panel_score - ductal_tumor_score)

        refined_tier_a, refined_tier_b = classify_epithelial_subcluster(row)
        row["refined_Tier_A"] = refined_tier_a
        row["refined_Tier_B"] = refined_tier_b
        row["refinement_action"] = (
            "kept"
            if refined_tier_a == row["parent_Tier_A_major"] and refined_tier_b == row["parent_Tier_B_major"]
            else f"{row['parent_Tier_A_major']} -> {refined_tier_a}"
        )
        refined_tier_a_by_subcluster[str(subcluster)] = refined_tier_a
        refined_tier_b_by_subcluster[str(subcluster)] = refined_tier_b
        rows.append(row)

    refinement_df = pd.DataFrame(rows)
    refined_tier_a = pd.Series(subcluster_labels).map(refined_tier_a_by_subcluster).to_numpy()
    refined_tier_b = pd.Series(subcluster_labels).map(refined_tier_b_by_subcluster).to_numpy()

    adata.obs.loc[candidate_obs_names, "Tier_A"] = refined_tier_a
    adata.obs.loc[candidate_obs_names, "Tier_B"] = refined_tier_b
    adata.obs.loc[candidate_obs_names, "epithelial_refinement_label"] = refined_tier_a

    n_changed = int((candidate_obs["Tier_A"].astype(str).to_numpy() != refined_tier_a).sum())
    n_rescued_ductal = int(
        (
            (candidate_obs["Tier_A"].astype(str).to_numpy() != "pancreatic ductal epithelium")
            & (refined_tier_a == "pancreatic ductal epithelium")
        ).sum()
    )
    adata.uns["epithelial_refinement_status"] = (
        f"completed_{n_clusters}_subclusters_{n_changed}_changed_{n_rescued_ductal}_ductal_rescued"
    )
    refinement_path = ANNOTATION_QC_DIR / f"{cfg['sample_id']}_epithelial_refinement_qc.csv"
    refinement_df.to_csv(refinement_path, index=False)

    assignment_df = pd.DataFrame(
        {
            "cell_id": candidate_obs_names,
            "epithelial_refinement_cluster": subcluster_labels,
            "original_Tier_A": candidate_obs["Tier_A"].astype(str).to_numpy(),
            "original_Tier_B": candidate_obs["Tier_B"].astype(str).to_numpy(),
            "refined_Tier_A": refined_tier_a,
            "refined_Tier_B": refined_tier_b,
        }
    )
    assignment_df.to_csv(ANNOTATION_QC_DIR / f"{cfg['sample_id']}_epithelial_refinement_assignments.csv", index=False)
    return adata, refinement_df
'''
    ),
    code(
        r'''
FORCE_REANNOTATE = False
summary_frames = []
cluster_summary_frames = []
marker_availability_rows = []

for cfg in SAMPLE_CONFIGS:
    out_path = ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad"
    if out_path.exists() and not FORCE_REANNOTATE:
        adata = sc.read_h5ad(out_path, backed="r")
        cache_ok = (
            LEIDEN_KEY in adata.obs.columns
            and "Tier_A" in adata.obs.columns
            and (
                XENIUM_GRAPHCLUST_KEY in adata.obs.columns
                or adata.uns.get("xenium_precomputed_cluster_status") == "not_found"
            )
            and adata.uns.get("spatioev_xenium_annotation_version") == ANNOTATION_VERSION
        )
        can_fast_relabel = (
            LEIDEN_KEY in adata.obs.columns
            and "log1p" in adata.layers.keys()
        )
        if not cache_ok:
            adata.file.close()
            if can_fast_relabel:
                print(f"Cached annotation is older, but Leiden/log1p are available; fast relabeling {cfg['sample_id']}.")
                adata = sc.read_h5ad(out_path)
                adata = add_xenium_precomputed_clusters(adata, cfg)
                available_genes = set(adata.var_names)
                for set_name, genes in GENE_SETS.items():
                    present = [g for g in genes if g in available_genes]
                    marker_availability_rows.append(
                        {
                            "sample_id": cfg["sample_id"],
                            "gene_set": set_name,
                            "n_requested": len(genes),
                            "n_present": len(present),
                            "present_genes": ", ".join(present),
                        }
                    )
                adata = add_marker_scores(adata)
                cluster_summary_df = summarize_annotation_clusters(adata, cfg)
                cluster_summary_df = add_suggested_cluster_labels(cluster_summary_df)
                review_df = update_cluster_review_table(cluster_summary_df)
                adata = apply_review_labels(adata, review_df, cfg)
                adata, refinement_df = refine_epithelial_labels_by_subclustering(adata, cfg)
                adata.uns["spatioev_xenium_annotation_version"] = ANNOTATION_VERSION
                cluster_summary_df.to_csv(ANNOTATION_QC_DIR / f"{cfg['sample_id']}_cluster_annotation_summary.csv", index=False)
                adata.write_h5ad(out_path)
                cluster_summary_frames.append(cluster_summary_df)

                counts = adata.obs["Tier_A"].value_counts().rename("n").reset_index()
                counts.columns = ["Tier_A", "n"]
                counts["sample_id"] = cfg["sample_id"]
                counts["disease_group"] = cfg["disease_group"]
                summary_frames.append(counts)
                del adata
                gc.collect()
                continue
            print(f"Cached annotation is from an older workflow and cannot be fast-relabelled; rebuilding {cfg['sample_id']}.")
        else:
            print(f"Using cached annotation: {out_path}")
            counts = adata.obs["Tier_A"].value_counts().rename("n").reset_index()
            counts.columns = ["Tier_A", "n"]
            counts["sample_id"] = cfg["sample_id"]
            counts["disease_group"] = cfg["disease_group"]
            summary_frames.append(counts)
            cluster_path = ANNOTATION_QC_DIR / f"{cfg['sample_id']}_cluster_annotation_summary.csv"
            if cluster_path.exists():
                cluster_summary_frames.append(pd.read_csv(cluster_path, dtype={"leiden": str}))
            adata.file.close()
            continue

    if out_path.exists() and FORCE_REANNOTATE:
        print(f"FORCE_REANNOTATE=True; rebuilding {cfg['sample_id']}.")
    elif not out_path.exists():
        print(f"No cached annotation found for {cfg['sample_id']}.")

    print(f"Clustering and annotating {cfg['sample_id']} with its own full panel...")
    adata = read_xenium_adata(cfg)
    available_genes = set(adata.var_names)
    for set_name, genes in GENE_SETS.items():
        present = [g for g in genes if g in available_genes]
        marker_availability_rows.append(
            {
                "sample_id": cfg["sample_id"],
                "gene_set": set_name,
                "n_requested": len(genes),
                "n_present": len(present),
                "present_genes": ", ".join(present),
            }
        )

    adata = preprocess_for_independent_annotation(adata)
    adata = add_marker_scores(adata)
    cluster_summary_df = summarize_annotation_clusters(adata, cfg)
    cluster_summary_df = add_suggested_cluster_labels(cluster_summary_df)
    review_df = update_cluster_review_table(cluster_summary_df)
    adata = apply_review_labels(adata, review_df, cfg)
    adata, refinement_df = refine_epithelial_labels_by_subclustering(adata, cfg)
    adata.uns["spatioev_xenium_annotation_version"] = ANNOTATION_VERSION

    cluster_summary_df.to_csv(ANNOTATION_QC_DIR / f"{cfg['sample_id']}_cluster_annotation_summary.csv", index=False)
    adata.write_h5ad(out_path)
    cluster_summary_frames.append(cluster_summary_df)

    counts = adata.obs["Tier_A"].value_counts().rename("n").reset_index()
    counts.columns = ["Tier_A", "n"]
    counts["sample_id"] = cfg["sample_id"]
    counts["disease_group"] = cfg["disease_group"]
    summary_frames.append(counts)
    del adata
    gc.collect()

tier_a_counts_df = pd.concat(summary_frames, ignore_index=True)
tier_a_counts_df["fraction"] = tier_a_counts_df["n"] / tier_a_counts_df.groupby("sample_id")["n"].transform("sum")
save_df(tier_a_counts_df, OUTPUT_DIR / "xenium_tier_a_counts.csv")

cluster_annotation_summary_df = (
    pd.concat(cluster_summary_frames, ignore_index=True)
    if len(cluster_summary_frames) > 0
    else pd.DataFrame()
)
if len(marker_availability_rows) > 0:
    marker_availability_df = pd.DataFrame(marker_availability_rows)
    save_df(marker_availability_df, ANNOTATION_QC_DIR / "marker_gene_availability.csv")
else:
    marker_availability_path = ANNOTATION_QC_DIR / "marker_gene_availability.csv"
    marker_availability_df = pd.read_csv(marker_availability_path) if marker_availability_path.exists() else pd.DataFrame()

tier_a_counts_df.head()
'''
    ),
    md(
        """
## Export Xenium Explorer Cell Groups

These CSVs can be imported into Xenium Explorer as custom cell groups. `Tier_A` is best for broad QC; `Tier_B` is best when you want the refined subtype labels.
"""
    ),
    code(
        r'''
XENIUM_EXPLORER_GROUP_DIR = OUTPUT_DIR / "xenium_explorer_cell_groups"
XENIUM_EXPLORER_GROUP_DIR.mkdir(exist_ok=True)

export_rows = []
for cfg in SAMPLE_CONFIGS:
    adata = sc.read_h5ad(ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad", backed="r")
    for obs_col, suffix in {"Tier_A": "tier_a", "Tier_B": "tier_b"}.items():
        if obs_col not in adata.obs.columns:
            continue
        export_df = adata.obs[[obs_col]].copy()
        export_df.insert(0, "cell_id", export_df.index.astype(str))
        export_df = export_df.rename(columns={obs_col: "group"})
        export_df["group"] = export_df["group"].astype(str).fillna("Unknown")
        out_path = XENIUM_EXPLORER_GROUP_DIR / f"{cfg['sample_id']}_{suffix}_cell_groups.csv"
        export_df.to_csv(out_path, index=False)
        export_rows.append(
            {
                "sample_id": cfg["sample_id"],
                "annotation": obs_col,
                "n_cells": len(export_df),
                "path": str(out_path),
            }
        )
    adata.file.close()

xenium_explorer_export_df = pd.DataFrame(export_rows)
xenium_explorer_export_df
'''
    ),
    md(
        """
## Annotation Review Table

The table below is the key checkpoint. If any automated cluster label looks wrong, edit `final_Tier_A` and `final_Tier_B` in `xenium_cluster_annotation_review.csv`, set `FORCE_REANNOTATE = True`, and rerun the annotation cell. The notebook will preserve curated labels where `sample_id + leiden` still match.
"""
    ),
    code(
        r'''
review_df = pd.read_csv(CLUSTER_REVIEW_PATH, dtype={"leiden": str})
review_df.head(20)
'''
    ),
    code(
        r'''
if not marker_availability_df.empty:
    display(marker_availability_df.sort_values(["sample_id", "gene_set"]).head(30))
else:
    print("Marker availability table is empty because cached annotations were used before this table existed.")
'''
    ),
    md(
        """
### Epithelial Ambiguity Check

This table is specifically for the duodenum-vs-ductal issue. True duodenum should have strict intestinal anchors (`CDX2` plus `KRT20/VIL1/FABP/ALPI/MUC2` when available). In PDAC, `REG4`, `DMBT1`, `TFF3`, `TMPRSS2`, and `GPX2` are treated as intestinal-like/PanIN-like ductal remodeling unless those stricter intestinal anchors are also present.
"""
    ),
    code(
        r'''
epithelial_qc_labels = ["pancreatic ductal epithelium", "Duodenum epithelial"]
epithelial_qc_cols = [
    "sample_id", "leiden", "n_cells", "fraction", "top_marker_genes",
    "suggested_Tier_A", "suggested_Tier_B", "annotation_note",
    "ductal_epithelial_score_z", "duodenum_epithelial_score_z",
    "intestinal_like_ductal_remodeling_score_z", "panin_mucin_remodeling_score_z",
    "EPCAM_frac_pos", "KRT7_frac_pos", "SOX9_frac_pos", "CFTR_frac_pos", "FXYD2_frac_pos",
    "CDX2_frac_pos", "REG4_frac_pos", "DMBT1_frac_pos", "TMPRSS2_frac_pos", "GPX2_frac_pos",
    "KRT20_frac_pos", "VIL1_frac_pos", "FABP1_frac_pos", "FABP2_frac_pos", "ALPI_frac_pos", "MUC2_frac_pos",
]
epithelial_qc_df = cluster_annotation_summary_df[
    cluster_annotation_summary_df["suggested_Tier_A"].isin(epithelial_qc_labels)
    | cluster_annotation_summary_df["suggested_Tier_B"].astype(str).str.contains("ductal|duodenum|intestinal", case=False, na=False)
].copy()
epithelial_qc_df = epithelial_qc_df[[c for c in epithelial_qc_cols if c in epithelial_qc_df.columns]]
display(epithelial_qc_df.sort_values(["sample_id", "suggested_Tier_A", "fraction"], ascending=[True, True, False]))
save_df(epithelial_qc_df, ANNOTATION_QC_DIR / "epithelial_ductal_duodenum_qc.csv")

refinement_paths = sorted(ANNOTATION_QC_DIR.glob("*_epithelial_refinement_qc.csv"))
if len(refinement_paths) > 0:
    epithelial_refinement_qc_df = pd.concat(
        [pd.read_csv(path) for path in refinement_paths],
        ignore_index=True,
    )
    display(
        epithelial_refinement_qc_df.sort_values(
            ["sample_id", "duodenum_minus_ductal_score"],
            ascending=[True, False],
        )
    )
else:
    print("No epithelial refinement QC files found. This is expected only if too few epithelial candidates are available.")
'''
    ),
    code(
        r'''
plt.figure(figsize=(9, 4.5))
plot_df = tier_a_counts_df.sort_values(["sample_id", "fraction"], ascending=[True, False])
sns.barplot(
    data=plot_df,
    x="sample_id",
    y="fraction",
    hue="Tier_A",
    palette=TIER_A_PALETTE,
    linewidth=0,
)
plt.title("Marker-rule Tier_A composition")
plt.xlabel("")
plt.ylabel("Fraction of cells")
plt.xticks(rotation=25, ha="right")
plt.legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(False)
plt.tight_layout()
plt.show()
'''
    ),
    md(
        """
## Cluster-Level Marker QC

These are the checks that should make wrong labels obvious: cluster score heatmaps, UMAPs by 10x graphclust / Leiden / Tier_A / Tier_B, marker dotplots by the annotation cluster key, and spatial maps.
"""
    ),
    code(
        r'''
if cluster_annotation_summary_df.empty and CLUSTER_REVIEW_PATH.exists():
    cluster_annotation_summary_df = pd.read_csv(CLUSTER_REVIEW_PATH, dtype={"leiden": str})

score_cols = [c for c in cluster_annotation_summary_df.columns if c.endswith("_score_z")]
if len(score_cols) > 0:
    heatmap_df = cluster_annotation_summary_df.copy()
    if "cluster_key" not in heatmap_df.columns:
        heatmap_df["cluster_key"] = "cluster"
    heatmap_df["cluster_key"] = heatmap_df["cluster_key"].astype(str)
    heatmap_df["cluster"] = (
        heatmap_df["sample_id"].astype(str)
        + ":"
        + heatmap_df["cluster_key"].str.replace("xenium_", "", regex=False)
        + ":"
        + heatmap_df["leiden"].astype(str)
    )
    heatmap_df = heatmap_df.set_index("cluster")[score_cols]
    plt.figure(figsize=(max(7, 0.35 * len(score_cols)), max(4, 0.16 * len(heatmap_df))))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0,
        linewidths=0,
        cbar_kws={"label": "Mean cluster marker-program z-score"},
    )
    plt.title("Cluster marker-program validation")
    plt.tight_layout()
    plt.show()
else:
    print("No score columns available for cluster-level heatmap.")
'''
    ),
    code(
        r'''
def plot_downsampled_embedding(ax, obs, x, y, color, title, max_cells=50000, palette=None, cmap="viridis"):
    rng = np.random.default_rng(42)
    plot_obs = obs[[x, y, color]].dropna().copy()
    if len(plot_obs) > max_cells:
        plot_obs = plot_obs.iloc[rng.choice(len(plot_obs), size=max_cells, replace=False)]
    if pd.api.types.is_numeric_dtype(plot_obs[color]):
        sca = ax.scatter(plot_obs[x], plot_obs[y], c=plot_obs[color], cmap=cmap, s=1, alpha=0.65, linewidths=0)
        plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    else:
        sns.scatterplot(
            data=plot_obs,
            x=x,
            y=y,
            hue=color,
            palette=palette,
            s=1,
            linewidth=0,
            alpha=0.65,
            ax=ax,
            legend=False,
        )
    ax.set_title(title)
    ax.grid(False)

for cfg in SAMPLE_CONFIGS:
    adata = sc.read_h5ad(ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad")
    annotation_cluster_key = adata.uns.get("spatioev_annotation_cluster_key", None)
    if annotation_cluster_key is None or annotation_cluster_key not in adata.obs.columns:
        annotation_cluster_key = choose_annotation_cluster_key(adata)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6))
    obs = adata.obs.join(pd.DataFrame(adata.obsm["X_umap"], index=adata.obs_names, columns=["UMAP1", "UMAP2"]))
    plot_downsampled_embedding(
        axes[0],
        obs,
        "UMAP1",
        "UMAP2",
        annotation_cluster_key,
        f"{cfg['sample_id']}: annotation clusters ({annotation_cluster_key})",
    )
    plot_downsampled_embedding(axes[1], obs, "UMAP1", "UMAP2", LEIDEN_KEY, f"{cfg['sample_id']}: Scanpy Leiden QC")
    plot_downsampled_embedding(
        axes[2],
        obs,
        "UMAP1",
        "UMAP2",
        "Tier_A",
        f"{cfg['sample_id']}: Tier_A",
        palette=TIER_A_PALETTE,
    )
    plot_downsampled_embedding(
        axes[3],
        obs,
        "UMAP1",
        "UMAP2",
        "Tier_B",
        f"{cfg['sample_id']}: Tier_B",
        palette=palette_for_values(obs["Tier_B"], TIER_B_BASE_PALETTE),
    )
    plt.tight_layout()
    plt.show()

    dotplot_markers = [g for g in DOTPLOT_MARKERS if g in adata.var_names]
    if len(dotplot_markers) > 0:
        sc.pl.dotplot(
            adata,
            var_names=dotplot_markers,
            groupby=annotation_cluster_key,
            layer="log1p",
            standard_scale="var",
            dendrogram=False,
            show=True,
            title=f"{cfg['sample_id']}: marker expression by annotation cluster ({annotation_cluster_key})",
        )
        sc.pl.dotplot(
            adata,
            var_names=dotplot_markers,
            groupby="Tier_A",
            layer="log1p",
            standard_scale="var",
            dendrogram=False,
            show=True,
            title=f"{cfg['sample_id']}: marker expression by final Tier_A annotation",
        )
        tier_b_counts = adata.obs["Tier_B"].value_counts()
        tier_b_keep = tier_b_counts.loc[tier_b_counts >= 100].index.tolist()
        adata_tier_b = adata[adata.obs["Tier_B"].isin(tier_b_keep)].copy()
        sc.pl.dotplot(
            adata_tier_b,
            var_names=dotplot_markers,
            groupby="Tier_B",
            layer="log1p",
            standard_scale="var",
            dendrogram=False,
            show=True,
            title=f"{cfg['sample_id']}: marker expression by final Tier_B annotation",
        )
        del adata_tier_b
    del adata
    gc.collect()
'''
    ),
    md(
        """
## Quick Spatial QC

These are lightweight downsampled plots to catch obvious annotation problems before building niches.
"""
    ),
    code(
        r'''
MAX_PLOT_CELLS_PER_SAMPLE = 30000
rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.ravel()

for ax, cfg in zip(axes, SAMPLE_CONFIGS):
    adata = sc.read_h5ad(ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad", backed="r")
    obs = adata.obs[["x_centroid", "y_centroid", "Tier_A", "disease_group"]].copy()
    if len(obs) > MAX_PLOT_CELLS_PER_SAMPLE:
        obs = obs.iloc[rng.choice(len(obs), size=MAX_PLOT_CELLS_PER_SAMPLE, replace=False)]
    sns.scatterplot(
        data=obs,
        x="x_centroid",
        y="y_centroid",
        hue="Tier_A",
        palette=TIER_A_PALETTE,
        s=1,
        linewidth=0,
        alpha=0.65,
        ax=ax,
        legend=False,
    )
    ax.set_title(cfg["sample_id"])
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    adata.file.close()

plt.tight_layout()
plt.show()
'''
    ),
]


niche_cells = [
    md(
        """
# Xenium Epithelial Niche Feature Construction

This notebook turns annotated Xenium cells into epithelial-centered niches.

The key design choice is to define a niche as a connected spatial component of cells annotated as `pancreatic ductal epithelium`.

For Xenium, the preferred component definition uses the 10x cell-boundary polygons: two epithelial cells are connected when their segmentation boundaries touch or are within a small gap. This is closer to the multiplexed-imaging mask-adjacency logic than a centroid-radius rule. A centroid fallback is still available when boundary files are missing or too slow.

Then we summarize:

- epithelial niche geometry/topology
- epithelial expression/state features
- surrounding Tier_A/Tier_B composition
- compartment-specific marker changes in the local surround

Feature guardrail:

The current Xenium trajectory can capture epithelial transcript state, nuclear/cell shape from segmentation boundaries, optional DAPI texture/intensity, tissue architecture, and microenvironment context. It still cannot reproduce CK19/NaKATPase membrane polarity unless aligned protein imaging is available.
"""
    ),
    code(COMMON_SETUP),
    code(
        r'''
import spatioev as se
from scipy.sparse.csgraph import connected_components
from spatioev.spatial.cell_pixel_features import extract_xenium_dapi_features

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    SHAPELY_AVAILABLE = True
except Exception as exc:
    SHAPELY_AVAILABLE = False
    SHAPELY_IMPORT_ERROR = repr(exc)

ANNOTATED_DIR = OUTPUT_DIR / "annotated_h5ad"
NICHE_DIR = OUTPUT_DIR / "niche_features"
NICHE_DIR.mkdir(exist_ok=True)
DAPI_FEATURE_DIR = OUTPUT_DIR / "dapi_features"
DAPI_FEATURE_DIR.mkdir(exist_ok=True)

NICHE_KEY = "xenium_ductal_epithelium_component"
NICHE_FEATURE_VERSION = "boundary_components_v3_lumen_continuity_interface"
EXPECTED_ANNOTATION_VERSION = "cluster_full_panel_v9_xenium_graphclust_io_mucosa_submucosa_k24"
EPITHELIAL_COMPONENT_METHOD = "boundary_proximity"
EPITHELIAL_BOUNDARY_GAP_UM = 3.0
EPITHELIAL_COMPONENT_RADIUS_UM = 35.0  # fallback only
CELL_GRAPH_RADIUS_UM = 35.0
SURROUND_HOPS = 5
MIN_NICHE_CELLS = 5
LUMEN_CLOSING_BUFFER_UM = 2.0
DUCT_CONTINUITY_RADIUS_UM = 150.0
BUDDING_DISTANCE_UM = 50.0
DAPI_TARGET_TIER_A = "pancreatic ductal epithelium"
DAPI_COMPUTE_TEXTURE = True
DAPI_COMPUTE_HARALICK = False
DAPI_FEATURE_VERSION = "user_verified_focus_dapi_sources_v2"

DAPI_IMAGE_SOURCE_OVERRIDES = {
    "pdac_pancreas_v1": {
        "relative_path": "morphology_focus/morphology_focus_0000.ome.tif",
        "image_kind": "verified_focus_dir0_channel0_dapi",
        "z_projection": "max",
        "channel_index": 0,
        "notes": "Use the first channel of morphology_focus/morphology_focus_0000.ome.tif as DAPI.",
    },
    "pdac_io_v1": {
        "relative_path": "morphology_focus/morphology_focus_0000.ome.tif",
        "image_kind": "verified_focus_dir0_dapi",
        "z_projection": "max",
        "channel_index": 0,
        "notes": "Use morphology_focus/morphology_focus_0000.ome.tif as DAPI.",
    },
    "pdac_addon_v1": {
        "relative_path": "morphology_focus.ome.tif",
        "image_kind": "verified_focus_dapi",
        "z_projection": "max",
        "channel_index": 0,
        "notes": "User-verified morphology_focus.ome.tif is the DAPI image.",
    },
    "normal_nondiseased_v1": {
        "relative_path": "morphology_focus.ome.tif",
        "image_kind": "verified_focus_dapi",
        "z_projection": "max",
        "channel_index": 0,
        "notes": "Use morphology_focus.ome.tif as DAPI.",
    },
}

PHENOTYPE_FEATURE_MAP_CANDIDATES = {
    "pancreatic ductal epithelium": [
        "EPCAM_expr_z", "KRT7_expr_z", "SOX9_expr_z", "MUC5AC_expr_z",
        "TFF2_expr_z", "TFF3_expr_z", "CEACAM6_expr_z", "AGR3_expr_z",
        "CFTR_expr_z", "FXYD2_expr_z", "TM4SF4_expr_z", "PROX1_expr_z",
        "MKI67_expr_z", "UBE2C_expr_z", "TOP2A_expr_z",
    ],
    "Duodenum epithelial": [
        "CDX2_expr_z", "REG4_expr_z", "DMBT1_expr_z", "TMPRSS2_expr_z",
        "MUC2_expr_z", "KRT20_expr_z", "VIL1_expr_z",
    ],
    "Fibroblasts": [
        "ACTA2_expr_z", "PDGFRA_expr_z", "FAP_expr_z", "THY1_expr_z",
        "PDPN_expr_z", "DCN_expr_z", "LUM_expr_z",
    ],
    "Endothelial cells": [
        "PECAM1_expr_z", "VWF_expr_z", "KDR_expr_z", "CDH5_expr_z",
        "PLVAP_expr_z", "FLT1_expr_z", "SPARCL1_expr_z", "IGFBP7_expr_z",
        "SOX17_expr_z", "CD34_expr_z",
    ],
    "T cells": ["CD3D_expr_z", "CD3E_expr_z", "CD4_expr_z", "CD8A_expr_z", "FOXP3_expr_z", "GZMB_expr_z", "NKG7_expr_z"],
    "B lineage": ["CD19_expr_z", "MS4A1_expr_z", "CD79A_expr_z", "MZB1_expr_z", "JCHAIN_expr_z", "SDC1_expr_z"],
    "Myeloid cells": [
        "LST1_expr_z", "LYZ_expr_z", "CD68_expr_z", "AIF1_expr_z", "C1QA_expr_z", "C1QB_expr_z",
        "CD163_expr_z", "MPEG1_expr_z", "CSF1R_expr_z", "S100A9_expr_z", "CXCR2_expr_z", "ITGAX_expr_z",
    ],
    "pancreatic acinar epithelium": [
        "AMY2A_expr_z", "PRSS1_expr_z", "CPA1_expr_z", "REG1A_expr_z",
        "AQP8_expr_z", "GATM_expr_z", "ANPEP_expr_z", "KLK11_expr_z",
    ],
    "Islets": ["INS_expr_z", "GCG_expr_z", "SST_expr_z", "PPY_expr_z", "CHGA_expr_z"],
}

STATE_FEATURE_CANDIDATES = sorted(
    set(
        [
            "cell_area",
            "nucleus_area",
            "nucleus_to_cell_area",
            "nucleus_to_cell_area_z",
            "cell_boundary_area",
            "cell_boundary_perimeter",
            "cell_boundary_circularity",
            "cell_boundary_solidity",
            "cell_boundary_major_minor_axis_ratio",
            "cell_boundary_feret_diameter_max",
            "cell_boundary_irregularity",
            "nucleus_boundary_area",
            "nucleus_boundary_perimeter",
            "nucleus_boundary_circularity",
            "nucleus_boundary_solidity",
            "nucleus_boundary_major_minor_axis_ratio",
            "nucleus_boundary_feret_diameter_max",
            "nucleus_boundary_irregularity",
            "dapi_n_pixels",
            "dapi_area_um2",
            "dapi_total_intensity_z",
            "dapi_mean_z",
            "dapi_std_z",
            "dapi_iqr_z",
            "dapi_entropy_z",
            "dapi_lacunarity_z",
            "dapi_polarity_score_z",
            "dapi_inertia_z",
            "transcript_counts",
            "total_counts",
            "ductal_epithelial_score_z",
            "acinar_epithelial_score_z",
            "islet_endocrine_score_z",
            "fibroblast_stellate_score_z",
            "proliferation_score_z",
        ]
        + [g for genes in PHENOTYPE_FEATURE_MAP_CANDIDATES.values() for g in genes]
    )
)

def safe_zscore(values):
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or np.isclose(sd, 0):
        return np.full(values.shape, np.nan)
    return (values - mu) / sd

def available_feature_map(adata):
    out = {}
    for label, cols in PHENOTYPE_FEATURE_MAP_CANDIDATES.items():
        cols_present = [c for c in cols if c in adata.obs.columns]
        if cols_present:
            out[label] = cols_present
    return out

def add_shape_qc_features(adata):
    obs = adata.obs
    if {"nucleus_area", "cell_area"}.issubset(obs.columns):
        cell_area = pd.to_numeric(obs["cell_area"], errors="coerce")
        nucleus_area = pd.to_numeric(obs["nucleus_area"], errors="coerce")
        adata.obs["nucleus_to_cell_area"] = nucleus_area / cell_area.replace(0, np.nan)
        adata.obs["nucleus_to_cell_area_z"] = safe_zscore(adata.obs["nucleus_to_cell_area"].to_numpy(dtype=float))
    return adata

def boundary_path(cfg, boundary_kind):
    return Path(cfg["outs_path"]) / f"{boundary_kind}_boundaries.parquet"

def load_boundary_points(cfg, boundary_kind, cell_ids=None):
    path = boundary_path(cfg, boundary_kind)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, columns=["cell_id", "vertex_x", "vertex_y"])
    if cell_ids is not None:
        cell_ids = set(pd.Index(cell_ids).astype(str))
        df = df.loc[df["cell_id"].astype(str).isin(cell_ids)].copy()
    return df

def polygon_from_xy(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 3:
        return None
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly

def max_pairwise_distance(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 2:
        return np.nan
    if coords.shape[0] > 256:
        idx = np.linspace(0, coords.shape[0] - 1, 256).astype(int)
        coords = coords[idx]
    diff = coords[:, None, :] - coords[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).max())

def polygon_shape_record(cell_id, group, prefix):
    xy = group[["vertex_x", "vertex_y"]].to_numpy(dtype=float)
    poly = polygon_from_xy(xy)
    if poly is None:
        return None
    perimeter = float(poly.length)
    area = float(poly.area)
    hull = poly.convex_hull
    hull_area = float(hull.area) if hull is not None else np.nan
    hull_perimeter = float(hull.length) if hull is not None else np.nan
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else np.nan
    solidity = (area / hull_area) if hull_area and hull_area > 0 else np.nan
    irregularity = (perimeter / hull_perimeter - 1.0) if hull_perimeter and hull_perimeter > 0 else np.nan

    centered = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T) if xy.shape[0] >= 3 else np.full((2, 2), np.nan)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.sort(np.maximum(eigvals, 0))
        major_minor = np.sqrt(eigvals[-1]) / np.sqrt(eigvals[0]) if eigvals[0] > 0 else np.nan
        major_vec = eigvecs[:, np.argmax(eigvals)]
        orientation = float(np.arctan2(major_vec[1], major_vec[0]))
    except Exception:
        major_minor = np.nan
        orientation = np.nan

    hull_coords = np.asarray(hull.exterior.coords) if hasattr(hull, "exterior") else xy
    feret = max_pairwise_distance(hull_coords)
    return {
        "cell_id": str(cell_id),
        f"{prefix}_area": area,
        f"{prefix}_perimeter": perimeter,
        f"{prefix}_circularity": circularity,
        f"{prefix}_solidity": solidity,
        f"{prefix}_major_minor_axis_ratio": major_minor,
        f"{prefix}_orientation": orientation,
        f"{prefix}_feret_diameter_max": feret,
        f"{prefix}_irregularity": irregularity,
    }

def summarize_boundary_shape_features(cfg, boundary_kind, cell_ids=None, force=False):
    prefix = "cell_boundary" if boundary_kind == "cell" else "nucleus_boundary"
    cache_path = NICHE_DIR / f"{cfg['sample_id']}_{prefix}_shape_features.pkl"
    if cache_path.exists() and not force:
        return load_df(cache_path)

    points = load_boundary_points(cfg, boundary_kind, cell_ids=cell_ids)
    rows = []
    for cell_id, group in points.groupby("cell_id", sort=False):
        rec = polygon_shape_record(cell_id, group, prefix=prefix)
        if rec is not None:
            rows.append(rec)
    out = pd.DataFrame(rows)
    save_df(out, cache_path)
    return out

def add_boundary_shape_features(adata, cfg, force=False):
    frames = []
    for boundary_kind in ["cell", "nucleus"]:
        try:
            frames.append(summarize_boundary_shape_features(cfg, boundary_kind, cell_ids=adata.obs_names, force=force))
        except Exception as exc:
            print(f"Boundary shape extraction skipped for {cfg['sample_id']} {boundary_kind}: {exc}")
    if len(frames) == 0:
        return adata

    shape_df = frames[0]
    for frame in frames[1:]:
        shape_df = shape_df.merge(frame, on="cell_id", how="outer")
    shape_df = shape_df.set_index("cell_id")
    new_cols = [c for c in shape_df.columns if c not in adata.obs.columns]
    adata.obs = adata.obs.join(shape_df[new_cols], how="left")

    for col in new_cols:
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
            z_col = f"{col}_z"
            if z_col not in adata.obs:
                adata.obs[z_col] = safe_zscore(pd.to_numeric(adata.obs[col], errors="coerce").to_numpy(dtype=float))
                if z_col not in STATE_FEATURE_CANDIDATES:
                    STATE_FEATURE_CANDIDATES.append(z_col)
    return adata

def get_xenium_pixel_size_um(cfg):
    experiment_path = Path(cfg["outs_path"]) / "experiment.xenium"
    if not experiment_path.exists():
        return np.nan
    meta = json.loads(experiment_path.read_text())
    return float(meta.get("pixel_size", np.nan))

def resolve_dapi_image_source(cfg):
    source = DAPI_IMAGE_SOURCE_OVERRIDES.get(cfg["sample_id"], {}).copy()
    if len(source) == 0:
        source = {
            "relative_path": "morphology.ome.tif",
            "image_kind": "default_morphology_zstack",
            "z_projection": "max",
            "channel_index": 0,
            "notes": "Default fallback; please verify before full extraction.",
        }
    image_path = Path(cfg["outs_path"]) / source["relative_path"]
    out = {
        "sample_id": cfg["sample_id"],
        "dapi_image_path": image_path,
        "dapi_image_exists": image_path.exists(),
        "dapi_image_kind": source.get("image_kind", "custom"),
        "dapi_z_projection": source.get("z_projection", "max"),
        "dapi_channel_index": int(source.get("channel_index", 0)),
        "dapi_source_notes": source.get("notes", ""),
        "has_morphology_mip": (Path(cfg["outs_path"]) / "morphology_mip.ome.tif").exists(),
        "has_morphology_focus": (Path(cfg["outs_path"]) / "morphology_focus.ome.tif").exists(),
        "has_morphology_focus_0000": (Path(cfg["outs_path"]) / "morphology_focus" / "morphology_focus_0000.ome.tif").exists(),
        "has_zstack_morphology": (Path(cfg["outs_path"]) / "morphology.ome.tif").exists(),
    }
    return out

def dapi_feature_path(cfg, suffix="epithelial"):
    return DAPI_FEATURE_DIR / f"{cfg['sample_id']}_dapi_features_{DAPI_FEATURE_VERSION}_{suffix}.csv"

def add_dapi_features_to_obs(adata, dapi_df):
    if dapi_df is None or dapi_df.empty:
        return adata
    dapi_df = dapi_df.copy()
    if "cell_id" not in dapi_df.columns:
        raise ValueError("DAPI feature table must contain a cell_id column.")
    dapi_df["cell_id"] = dapi_df["cell_id"].astype(str)
    dapi_df = dapi_df.drop_duplicates("cell_id").set_index("cell_id")
    new_cols = [c for c in dapi_df.columns if c not in adata.obs.columns]
    adata.obs = adata.obs.join(dapi_df[new_cols], how="left")

    for col in new_cols:
        if not col.startswith("dapi_"):
            continue
        vals = pd.to_numeric(adata.obs[col], errors="coerce")
        if vals.notna().sum() < 5:
            continue
        z_col = f"{col}_z"
        if z_col in adata.obs.columns:
            continue
        adata.obs[z_col] = safe_zscore(vals.to_numpy(dtype=float))
        if z_col not in STATE_FEATURE_CANDIDATES:
            STATE_FEATURE_CANDIDATES.append(z_col)
    return adata

def extract_dapi_features_for_cfg(
    cfg,
    annotated_path,
    output_path,
    max_cells=None,
    force=False,
    target_tier_a=DAPI_TARGET_TIER_A,
):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"Using cached DAPI features: {output_path}")
        return pd.read_csv(output_path)

    adata = sc.read_h5ad(annotated_path, backed="r")
    try:
        if target_tier_a is None:
            cell_ids = adata.obs_names.astype(str).tolist()
        else:
            cell_ids = adata.obs_names[adata.obs["Tier_A"].astype(str) == target_tier_a].astype(str).tolist()
    finally:
        adata.file.close()

    dapi_source = resolve_dapi_image_source(cfg)
    if not dapi_source["dapi_image_exists"]:
        raise FileNotFoundError(dapi_source["dapi_image_path"])

    print(f"Extracting DAPI features for {cfg['sample_id']} ({len(cell_ids):,} requested cells; max_cells={max_cells})")
    print(f"Using DAPI source: {dapi_source['dapi_image_path']}")
    return extract_xenium_dapi_features(
        outs_path=cfg["outs_path"],
        cell_ids=cell_ids,
        output_path=output_path,
        image_kind=dapi_source["dapi_image_kind"],
        image_path=dapi_source["dapi_image_path"],
        channel_index=dapi_source["dapi_channel_index"],
        z_projection=dapi_source["dapi_z_projection"],
        max_cells=max_cells,
        random_state=42,
        compute_texture=DAPI_COMPUTE_TEXTURE,
        compute_haralick=DAPI_COMPUTE_HARALICK,
        progress_every=10000,
    )

class DisjointSet:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

def strtree_query_indices(tree, query_geom, geom_to_idx):
    hits = tree.query(query_geom)
    out = []
    for hit in hits:
        if isinstance(hit, (int, np.integer)):
            out.append(int(hit))
        else:
            idx = geom_to_idx.get(id(hit))
            if idx is not None:
                out.append(idx)
    return out

def assign_boundary_epithelial_components(adata, cfg, force=False):
    if not SHAPELY_AVAILABLE:
        raise ImportError(f"shapely import failed: {SHAPELY_IMPORT_ERROR}")

    epithelial_mask = adata.obs["Tier_A"].astype(str) == "pancreatic ductal epithelium"
    epithelial_ids = adata.obs_names[epithelial_mask].astype(str)
    if len(epithelial_ids) < MIN_NICHE_CELLS:
        adata.obs[NICHE_KEY] = pd.NA
        adata.obs["xenium_epithelial_component_method"] = "not_enough_epithelial_cells"
        return adata

    points = load_boundary_points(cfg, "cell", cell_ids=epithelial_ids)
    if points.empty:
        raise ValueError("No epithelial cell-boundary points found.")

    cell_ids = []
    polygons = []
    for cell_id, group in points.groupby("cell_id", sort=False):
        poly = polygon_from_xy(group[["vertex_x", "vertex_y"]].to_numpy(dtype=float))
        if poly is not None:
            cell_ids.append(str(cell_id))
            polygons.append(poly)

    if len(polygons) < MIN_NICHE_CELLS:
        raise ValueError("Too few valid epithelial polygons.")

    tree = STRtree(polygons)
    geom_to_idx = {id(geom): i for i, geom in enumerate(polygons)}
    dsu = DisjointSet(len(polygons))

    for i, poly in enumerate(polygons):
        query_geom = poly.buffer(EPITHELIAL_BOUNDARY_GAP_UM)
        for j in strtree_query_indices(tree, query_geom, geom_to_idx):
            if j <= i:
                continue
            if poly.distance(polygons[j]) <= EPITHELIAL_BOUNDARY_GAP_UM:
                dsu.union(i, j)

    comp_roots = np.array([dsu.find(i) for i in range(len(polygons))])
    comp_df = pd.DataFrame({"cell_id": cell_ids, "_root": comp_roots})
    comp_sizes = comp_df["_root"].value_counts()
    keep_roots = comp_sizes.loc[comp_sizes >= MIN_NICHE_CELLS].index.tolist()

    adata.obs[NICHE_KEY] = pd.NA
    adata.obs["xenium_epithelial_component_method"] = EPITHELIAL_COMPONENT_METHOD
    label_map = {
        root: f"{cfg['sample_id']}__ductal_boundary_{idx:05d}"
        for idx, root in enumerate(sorted(keep_roots), start=1)
    }
    comp_df = comp_df.loc[comp_df["_root"].isin(keep_roots)].copy()
    comp_df[NICHE_KEY] = comp_df["_root"].map(label_map)
    adata.obs.loc[comp_df["cell_id"].to_numpy(), NICHE_KEY] = comp_df[NICHE_KEY].to_numpy()
    return adata

def assign_centroid_epithelial_components(adata):
    adata = se.cluster_spatial_components(
        adata,
        label_key="Tier_A",
        label_value="pancreatic ductal epithelium",
        image_key="sample_id",
        x_key="x_centroid",
        y_key="y_centroid",
        component_key=NICHE_KEY,
        radius=EPITHELIAL_COMPONENT_RADIUS_UM,
        min_component_size=MIN_NICHE_CELLS,
        assign_singletons=False,
    )
    adata.obs["xenium_epithelial_component_method"] = "centroid_radius_fallback"
    return adata

def assign_epithelial_components(adata, cfg):
    if EPITHELIAL_COMPONENT_METHOD == "boundary_proximity":
        try:
            return assign_boundary_epithelial_components(adata, cfg)
        except Exception as exc:
            print(f"Boundary epithelial components failed for {cfg['sample_id']}; using centroid fallback. Reason: {exc}")
            return assign_centroid_epithelial_components(adata)
    return assign_centroid_epithelial_components(adata)

def polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [part for part in geom.geoms if part.geom_type == "Polygon"]
    return []

def polygon_hole_metrics(geom):
    outer_area = 0.0
    hole_area = 0.0
    max_hole_area = 0.0
    n_holes = 0
    for poly in polygon_parts(geom):
        try:
            outer_area += float(Polygon(poly.exterior).area)
        except Exception:
            continue
        for interior in poly.interiors:
            area = float(Polygon(interior).area)
            hole_area += area
            max_hole_area = max(max_hole_area, area)
            n_holes += 1
    return outer_area, hole_area, max_hole_area, n_holes

def rotated_rect_axis_metrics(geom):
    if geom is None or geom.is_empty:
        return np.nan, np.nan, np.nan, np.nan
    try:
        rect = geom.minimum_rotated_rectangle
        coords = np.asarray(rect.exterior.coords, dtype=float)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    if coords.shape[0] < 4:
        return np.nan, np.nan, np.nan, np.nan
    edges = coords[1:] - coords[:-1]
    lengths = np.sqrt((edges * edges).sum(axis=1))
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if len(lengths) == 0:
        return np.nan, np.nan, np.nan, np.nan
    major = float(lengths.max())
    minor = float(lengths.min())
    major_edge = edges[np.argmax(np.sqrt((edges * edges).sum(axis=1)))]
    orientation = float(np.arctan2(major_edge[1], major_edge[0]))
    axis_ratio = major / minor if minor > 0 else np.nan
    return major, minor, axis_ratio, orientation

def summarize_epithelial_interface_features(adata, niche_key=NICHE_KEY):
    if "cell_graph_connectivities" not in adata.obsp:
        return pd.DataFrame(columns=[niche_key, "sample_id"])

    A = adata.obsp["cell_graph_connectivities"].tocsr()
    obs = adata.obs
    tier_a = obs["Tier_A"].astype(str).to_numpy()
    niche_values = (
        obs[niche_key]
        .astype("string")
        .fillna("__not_in_epithelial_niche__")
        .astype(str)
        .to_numpy()
    )
    sample_values = obs["sample_id"].astype(str).to_numpy() if "sample_id" in obs.columns else np.repeat("sample", obs.shape[0])

    rows = []
    niche_series = obs[niche_key].dropna()
    for niche_value, niche_obs_names in niche_series.groupby(niche_series, sort=False).groups.items():
        niche_idx = obs.index.get_indexer(pd.Index(niche_obs_names))
        niche_idx = niche_idx[niche_idx >= 0]
        if len(niche_idx) == 0:
            continue

        neighbor_idx = A[niche_idx].nonzero()[1]
        if len(neighbor_idx) > 0:
            in_niche = np.asarray(niche_values[neighbor_idx] == str(niche_value), dtype=bool)
            neighbor_idx = np.unique(neighbor_idx[~in_niche])
        else:
            neighbor_idx = np.array([], dtype=int)

        neighbor_labels = tier_a[neighbor_idx] if len(neighbor_idx) > 0 else np.array([], dtype=str)
        fibro_count = int(np.sum(neighbor_labels == "Fibroblasts"))
        non_epi_count = int(np.sum(neighbor_labels != "pancreatic ductal epithelium"))
        all_count = int(len(neighbor_idx))

        rows.append(
            {
                niche_key: niche_value,
                "sample_id": sample_values[niche_idx[0]],
                "interface__hop1_neighbor_cells": all_count,
                "interface__fibroblast_contact_cells_hop1": fibro_count,
                "interface__fibroblast_contact_fraction_hop1": fibro_count / all_count if all_count > 0 else np.nan,
                "interface__fibroblast_contacts_per_epithelial_cell": fibro_count / len(niche_idx),
                "interface__non_epithelial_contact_fraction_hop1": non_epi_count / all_count if all_count > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)

def summarize_epithelial_architecture_extensions(adata, cfg, force=False):
    """Add literature-motivated duct/lumen, duct-continuity, and interface features."""
    if not SHAPELY_AVAILABLE:
        print(f"Skipping architecture extensions; shapely import failed: {SHAPELY_IMPORT_ERROR}")
        return pd.DataFrame(columns=[NICHE_KEY, "sample_id"])

    cache_path = NICHE_DIR / f"{cfg['sample_id']}_architecture_extensions_{NICHE_FEATURE_VERSION}.pkl"
    if cache_path.exists() and not force:
        return load_df(cache_path)

    obs = adata.obs.copy()
    epithelial_mask = obs["Tier_A"].astype(str) == "pancreatic ductal epithelium"
    epithelial_ids = obs.index[epithelial_mask].astype(str)
    if len(epithelial_ids) == 0:
        return pd.DataFrame(columns=[NICHE_KEY, "sample_id"])

    points = load_boundary_points(cfg, "cell", cell_ids=epithelial_ids)
    polygon_by_cell = {}
    for cell_id, group in points.groupby("cell_id", sort=False):
        poly = polygon_from_xy(group[["vertex_x", "vertex_y"]].to_numpy(dtype=float))
        if poly is not None:
            polygon_by_cell[str(cell_id)] = poly

    assigned = obs.loc[epithelial_mask & obs[NICHE_KEY].notna(), [NICHE_KEY]].copy()
    if assigned.empty:
        return pd.DataFrame(columns=[NICHE_KEY, "sample_id"])

    all_assigned_ids = set(assigned.index.astype(str))
    unassigned_ids = [
        str(cell_id)
        for cell_id in epithelial_ids
        if str(cell_id) in polygon_by_cell and str(cell_id) not in all_assigned_ids
    ]
    unassigned_polygons = [polygon_by_cell[cell_id] for cell_id in unassigned_ids]
    unassigned_tree = STRtree(unassigned_polygons) if len(unassigned_polygons) > 0 else None
    unassigned_lookup = {id(geom): i for i, geom in enumerate(unassigned_polygons)}

    rows = []
    component_geoms = []
    component_labels = []
    component_orientations = []

    for niche_value, cell_index in assigned.groupby(NICHE_KEY, sort=False).groups.items():
        cell_ids = [str(cell_id) for cell_id in cell_index if str(cell_id) in polygon_by_cell]
        polys = [polygon_by_cell[cell_id] for cell_id in cell_ids]
        if len(polys) == 0:
            continue

        geom = unary_union(polys)
        if LUMEN_CLOSING_BUFFER_UM > 0:
            try:
                geom_closed = geom.buffer(LUMEN_CLOSING_BUFFER_UM).buffer(-LUMEN_CLOSING_BUFFER_UM)
                if geom_closed is not None and not geom_closed.is_empty:
                    geom_for_lumen = geom_closed
                else:
                    geom_for_lumen = geom
            except Exception:
                geom_for_lumen = geom
        else:
            geom_for_lumen = geom

        epi_area = float(geom.area)
        boundary_length = float(geom.length)
        hull = geom.convex_hull
        hull_area = float(hull.area) if hull is not None and not hull.is_empty else np.nan
        hull_perimeter = float(hull.length) if hull is not None and not hull.is_empty else np.nan
        outer_area, lumen_area, max_lumen_area, n_lumens = polygon_hole_metrics(geom_for_lumen)
        outer_area = outer_area if outer_area > 0 else epi_area + lumen_area
        outer_equiv_diameter = np.sqrt(4.0 * outer_area / np.pi) if outer_area > 0 else np.nan
        lumen_equiv_diameter = np.sqrt(4.0 * lumen_area / np.pi) if lumen_area > 0 else 0.0
        ring_thickness = (outer_equiv_diameter - lumen_equiv_diameter) / 2.0 if lumen_area > 0 else np.nan
        major, minor, axis_ratio, orientation = rotated_rect_axis_metrics(geom)
        boundary_roughness = boundary_length / hull_perimeter - 1.0 if hull_perimeter and hull_perimeter > 0 else np.nan

        nearby_budding_count = 0
        if unassigned_tree is not None:
            query_geom = geom.buffer(BUDDING_DISTANCE_UM)
            for j in strtree_query_indices(unassigned_tree, query_geom, unassigned_lookup):
                if geom.distance(unassigned_polygons[j]) <= BUDDING_DISTANCE_UM:
                    nearby_budding_count += 1

        rec = {
            NICHE_KEY: niche_value,
            "sample_id": cfg["sample_id"],
            "duct_lumen__component_epithelium_area_um2": epi_area,
            "duct_lumen__component_outer_area_um2": outer_area,
            "duct_lumen__lumen_area_um2": lumen_area,
            "duct_lumen__max_lumen_area_um2": max_lumen_area,
            "duct_lumen__n_lumens": n_lumens,
            "duct_lumen__lumen_fraction": lumen_area / outer_area if outer_area > 0 else np.nan,
            "duct_lumen__lumen_to_epithelium_area_ratio": lumen_area / epi_area if epi_area > 0 else np.nan,
            "duct_lumen__outer_equivalent_diameter_um": outer_equiv_diameter,
            "duct_lumen__lumen_equivalent_diameter_um": lumen_equiv_diameter,
            "duct_lumen__epithelial_ring_thickness_um": ring_thickness,
            "duct_lumen__cystic_dilation_proxy": (lumen_area / outer_area) * np.log1p(max_lumen_area) if outer_area > 0 else np.nan,
            "duct_continuity__component_axis_length_um": major,
            "duct_continuity__component_axis_width_um": minor,
            "duct_continuity__component_axis_ratio": axis_ratio,
            "duct_continuity__component_orientation": orientation,
            "duct_continuity__component_area_um2": epi_area,
            "duct_continuity__component_perimeter_um": boundary_length,
            "duct_continuity__cells_per_100um_axis": len(cell_ids) / (major / 100.0) if major and major > 0 else np.nan,
            "interface__epithelial_boundary_length_um": boundary_length,
            "interface__epithelial_boundary_roughness": boundary_roughness,
            "interface__boundary_length_per_area": boundary_length / epi_area if epi_area > 0 else np.nan,
            "interface__boundary_length_per_epithelial_cell": boundary_length / len(cell_ids),
            "interface__nearby_unassigned_epithelial_cells_50um": nearby_budding_count,
            "interface__nearby_unassigned_epithelial_cells_per_100um_boundary": (
                nearby_budding_count / (boundary_length / 100.0) if boundary_length > 0 else np.nan
            ),
        }
        rows.append(rec)
        component_geoms.append(geom)
        component_labels.append(niche_value)
        component_orientations.append(orientation)

    arch_df = pd.DataFrame(rows)
    if arch_df.empty:
        return arch_df

    comp_tree = STRtree(component_geoms)
    comp_lookup = {id(geom): i for i, geom in enumerate(component_geoms)}
    continuity_rows = []
    for i, geom in enumerate(component_geoms):
        distances = []
        alignments = []
        for j in strtree_query_indices(comp_tree, geom.buffer(DUCT_CONTINUITY_RADIUS_UM), comp_lookup):
            if j == i:
                continue
            dist = geom.distance(component_geoms[j])
            if dist <= DUCT_CONTINUITY_RADIUS_UM:
                distances.append(float(dist))
                oi = component_orientations[i]
                oj = component_orientations[j]
                if np.isfinite(oi) and np.isfinite(oj):
                    alignments.append(abs(np.cos(oi - oj)))
        continuity_rows.append(
            {
                NICHE_KEY: component_labels[i],
                "sample_id": cfg["sample_id"],
                "duct_continuity__nearby_components_150um": len(distances),
                "duct_continuity__nearest_component_distance_um": min(distances) if len(distances) > 0 else np.nan,
                "duct_continuity__mean_neighbor_distance_um": float(np.mean(distances)) if len(distances) > 0 else np.nan,
                "duct_continuity__neighbor_orientation_alignment_150um": (
                    float(np.mean(alignments)) if len(alignments) > 0 else np.nan
                ),
            }
        )

    continuity_df = pd.DataFrame(continuity_rows)
    interface_df = summarize_epithelial_interface_features(adata, NICHE_KEY)
    arch_df = arch_df.merge(continuity_df, on=[NICHE_KEY, "sample_id"], how="left")
    arch_df = arch_df.merge(interface_df, on=[NICHE_KEY, "sample_id"], how="left")
    save_df(arch_df, cache_path)
    return arch_df

def score_signed_module(df, positive_cols, negative_cols=None, score_name=None):
    negative_cols = negative_cols or []
    pos = [c for c in positive_cols if c in df.columns]
    neg = [c for c in negative_cols if c in df.columns]
    if not pos and not neg:
        return pd.Series(np.nan, index=df.index, name=score_name)
    frames = []
    signs = []
    for c in pos:
        frames.append(pd.to_numeric(df[c], errors="coerce"))
        signs.append(1.0)
    for c in neg:
        frames.append(pd.to_numeric(df[c], errors="coerce"))
        signs.append(-1.0)
    X = pd.concat(frames, axis=1)
    X = X.apply(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else np.nan)
    score = (X * np.asarray(signs)).mean(axis=1)
    score.name = score_name
    return score

def add_xenium_niche_module_scores(df):
    df = df.copy()
    df["xenium_epithelial_identity_score"] = score_signed_module(
        df,
        [
            "state__ductal_epithelial_score_z__mean",
            "state__EPCAM_expr_z__mean",
            "state__KRT7_expr_z__mean",
            "state__SOX9_expr_z__mean",
        ],
        [
            "state__acinar_epithelial_score_z__mean",
            "state__islet_endocrine_score_z__mean",
        ],
        "xenium_epithelial_identity_score",
    )
    df["xenium_panin_like_remodeling_score"] = score_signed_module(
        df,
        [
            "state__MUC5AC_expr_z__mean",
            "state__TFF2_expr_z__mean",
            "state__TFF3_expr_z__mean",
            "state__CEACAM6_expr_z__mean",
            "state__AGR3_expr_z__mean",
            "geometry__hull_circularity",
            "topology__skeleton_branchpoint_fraction",
        ],
        ["geometry__cell_density_hull"],
        "xenium_panin_like_remodeling_score",
    )
    df["xenium_proliferation_score"] = score_signed_module(
        df,
        [
            "state__proliferation_score_z__mean",
            "state__MKI67_expr_z__mean",
            "state__UBE2C_expr_z__mean",
            "state__TOP2A_expr_z__mean",
        ],
        score_name="xenium_proliferation_score",
    )
    df["xenium_desmoplastic_context_score"] = score_signed_module(
        df,
        [
            "surround_prop__Fibroblasts",
            "surround__Fibroblasts__ACTA2_expr_z__mean",
            "surround__Fibroblasts__PDGFRA_expr_z__mean",
            "surround__Fibroblasts__THY1_expr_z__mean",
            "surround__Fibroblasts__PDPN_expr_z__mean",
        ],
        score_name="xenium_desmoplastic_context_score",
    )
    df["xenium_immune_context_score"] = score_signed_module(
        df,
        [
            "surround_prop__T_cells",
            "surround_prop__B_lineage",
            "surround_prop__Myeloid_cells",
        ],
        score_name="xenium_immune_context_score",
    )
    df["xenium_checkpoint_context_score"] = score_signed_module(
        df,
        [
            "surround__T_cells__FOXP3_expr_z__mean",
            "surround__T_cells__GZMB_expr_z__mean",
            "surround__T_cells__NKG7_expr_z__mean",
            "surround__B_lineage__MZB1_expr_z__mean",
        ],
        score_name="xenium_checkpoint_context_score",
    )
    df["xenium_nuclear_dapi_texture_score"] = score_signed_module(
        df,
        [
            "state__nucleus_boundary_major_minor_axis_ratio_z__mean",
            "state__nucleus_boundary_irregularity_z__mean",
            "state__nucleus_boundary_feret_diameter_max_z__mean",
            "state__dapi_std_z__mean",
            "state__dapi_iqr_z__mean",
            "state__dapi_entropy_z__mean",
            "state__dapi_lacunarity_z__mean",
        ],
        score_name="xenium_nuclear_dapi_texture_score",
    )
    df["xenium_duct_lumen_topology_score"] = score_signed_module(
        df,
        [
            "duct_lumen__lumen_fraction",
            "duct_lumen__max_lumen_area_um2",
            "duct_lumen__outer_equivalent_diameter_um",
            "duct_lumen__cystic_dilation_proxy",
        ],
        score_name="xenium_duct_lumen_topology_score",
    )
    df["xenium_duct_continuity_cancerization_score"] = score_signed_module(
        df,
        [
            "duct_continuity__component_axis_length_um",
            "duct_continuity__component_axis_ratio",
            "duct_continuity__nearby_components_150um",
            "duct_continuity__neighbor_orientation_alignment_150um",
            "topology__skeleton_total_length",
            "topology__skeleton_n_edges",
        ],
        ["duct_continuity__nearest_component_distance_um"],
        "xenium_duct_continuity_cancerization_score",
    )
    df["xenium_epithelial_stromal_interface_disruption_score"] = score_signed_module(
        df,
        [
            "interface__epithelial_boundary_roughness",
            "interface__boundary_length_per_area",
            "interface__nearby_unassigned_epithelial_cells_per_100um_boundary",
            "interface__fibroblast_contact_fraction_hop1",
            "interface__fibroblast_contacts_per_epithelial_cell",
            "interface__non_epithelial_contact_fraction_hop1",
        ],
        score_name="xenium_epithelial_stromal_interface_disruption_score",
    )
    return df
'''
    ),
    md(
        """
## Optional DAPI Pixel Feature Extraction

This is now a real extraction step, not just a placeholder. The safest workflow is:

1. Run a small pilot (`RUN_DAPI_PILOT = True`) to verify speed and image alignment.
2. If the pilot looks good, set `RUN_DAPI_FULL_EPITHELIAL = True` and `DAPI_FULL_MAX_CELLS = None` to extract all ductal epithelial nuclei.
3. The niche-building cell below automatically joins any full epithelial DAPI CSV that exists.

This notebook uses user-verified DAPI sources per sample instead of the generic auto-picker. The selected file is shown in the QC table below before extraction starts.
"""
    ),
    code(
        r'''
RUN_DAPI_PILOT = True
RUN_DAPI_FULL_EPITHELIAL = False
FORCE_DAPI_EXTRACTION = False
DAPI_PILOT_MAX_CELLS = 500
DAPI_FULL_MAX_CELLS = None  # set to an integer for a larger staged run; None means all requested epithelial cells

pixel_size_df = pd.DataFrame(
    [
        {
            **resolve_dapi_image_source(cfg),
            "pixel_size_um": get_xenium_pixel_size_um(cfg),
        }
        for cfg in SAMPLE_CONFIGS
    ]
)
pixel_size_df["dapi_image_path"] = pixel_size_df["dapi_image_path"].astype(str)
display(pixel_size_df)

if RUN_DAPI_PILOT:
    pilot_frames = []
    for cfg in SAMPLE_CONFIGS:
        annotated_path = ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad"
        pilot_path = dapi_feature_path(cfg, suffix=f"pilot_{DAPI_PILOT_MAX_CELLS}")
        pilot_df = extract_dapi_features_for_cfg(
            cfg,
            annotated_path=annotated_path,
            output_path=pilot_path,
            max_cells=DAPI_PILOT_MAX_CELLS,
            force=FORCE_DAPI_EXTRACTION,
            target_tier_a=DAPI_TARGET_TIER_A,
        )
        pilot_df["sample_id"] = cfg["sample_id"]
        pilot_frames.append(pilot_df)
    dapi_pilot_df = pd.concat(pilot_frames, ignore_index=True)
    display(dapi_pilot_df.head())
    display(
        dapi_pilot_df.groupby("sample_id")[
            ["dapi_mean", "dapi_std", "dapi_entropy", "dapi_lacunarity", "dapi_polarity_score"]
        ].describe()
    )

if RUN_DAPI_FULL_EPITHELIAL:
    for cfg in SAMPLE_CONFIGS:
        annotated_path = ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad"
        full_path = dapi_feature_path(cfg, suffix="epithelial")
        extract_dapi_features_for_cfg(
            cfg,
            annotated_path=annotated_path,
            output_path=full_path,
            max_cells=DAPI_FULL_MAX_CELLS,
            force=FORCE_DAPI_EXTRACTION,
            target_tier_a=DAPI_TARGET_TIER_A,
        )
else:
    print("Full epithelial DAPI extraction is off. Set RUN_DAPI_FULL_EPITHELIAL=True when the pilot looks good.")
'''
    ),
    code(
        r'''
FORCE_REBUILD_NICHES = False
sample_feature_frames = []
sample_context_frames = []

for cfg in SAMPLE_CONFIGS:
    feature_path = NICHE_DIR / f"{cfg['sample_id']}_niche_feature_df.pkl"
    context_path = NICHE_DIR / f"{cfg['sample_id']}_surround_context_df.pkl"
    annotated_path = ANNOTATED_DIR / f"{cfg['sample_id']}_annotated.h5ad"
    full_dapi_path = dapi_feature_path(cfg, suffix="epithelial")
    expected_dapi_feature_version = DAPI_FEATURE_VERSION if full_dapi_path.exists() else "none"

    if feature_path.exists() and context_path.exists() and not FORCE_REBUILD_NICHES:
        cached_feature_df = load_df(feature_path)
        cached_context_df = load_df(context_path)
        cache_ok = (
            "xenium_niche_feature_version" in cached_feature_df.columns
            and cached_feature_df["xenium_niche_feature_version"].eq(NICHE_FEATURE_VERSION).all()
            and "xenium_annotation_version" in cached_feature_df.columns
            and cached_feature_df["xenium_annotation_version"].fillna("unknown").eq(EXPECTED_ANNOTATION_VERSION).all()
            and "xenium_dapi_feature_version" in cached_feature_df.columns
            and cached_feature_df["xenium_dapi_feature_version"].fillna("none").eq(expected_dapi_feature_version).all()
        )
        if cache_ok:
            print(f"Using cached niche features for {cfg['sample_id']}")
            sample_feature_frames.append(cached_feature_df)
            sample_context_frames.append(cached_context_df)
            continue
        print(f"Cached niche features are from an older workflow; rebuilding {cfg['sample_id']}.")

    print(f"Building niches for {cfg['sample_id']}")
    adata = sc.read_h5ad(annotated_path)
    adata = add_shape_qc_features(adata)
    adata = add_boundary_shape_features(adata, cfg, force=False)
    if full_dapi_path.exists():
        print(f"Joining DAPI features from {full_dapi_path}")
        adata = add_dapi_features_to_obs(adata, pd.read_csv(full_dapi_path))
    else:
        print(f"No full epithelial DAPI feature file found for {cfg['sample_id']}; continuing without DAPI pixel features.")
    adata = assign_epithelial_components(adata, cfg)

    adata = se.build_cell_graph(
        adata,
        feature_cols=[],
        phenotype_key=None,
        radius=CELL_GRAPH_RADIUS_UM,
        image_key="sample_id",
        x_key="x_centroid",
        y_key="y_centroid",
        auto_log=False,
        scale_features=False,
        compute_weights=False,
        feature_obsm_key="cell_features",
        adjacency_key="cell_graph_connectivities",
        distance_key="cell_graph_distances",
        graph_obs_key="cell_graph_valid",
    )

    state_cols = [c for c in STATE_FEATURE_CANDIDATES if c in adata.obs.columns]
    feature_df = se.summarize_niche_graph_features(
        adata,
        niche_key=NICHE_KEY,
        feature_cols=state_cols,
        state_feature_cols=state_cols,
        phenotype_key="Tier_A",
        image_key="sample_id",
        x_key="x_centroid",
        y_key="y_centroid",
        adjacency_key="cell_graph_connectivities",
        distance_key="cell_graph_distances",
        min_cells=MIN_NICHE_CELLS,
        include_graph_surroundings=True,
        surround_hops=1,
        lightweight=True,
        show_progress=True,
        progress_desc=f"{cfg['sample_id']} niche features",
    )
    feature_df["sample_id"] = cfg["sample_id"]
    feature_df["display_name"] = cfg["display_name"]
    feature_df["disease_group"] = cfg["disease_group"]
    feature_df["xenium_niche_feature_version"] = NICHE_FEATURE_VERSION
    feature_df["xenium_annotation_version"] = adata.uns.get("spatioev_xenium_annotation_version", "unknown")
    feature_df["xenium_dapi_feature_version"] = expected_dapi_feature_version
    feature_df["xenium_dapi_feature_path"] = str(full_dapi_path) if full_dapi_path.exists() else pd.NA
    component_methods = adata.obs["xenium_epithelial_component_method"].dropna().astype(str)
    feature_df["xenium_epithelial_component_method"] = (
        component_methods.mode().iat[0] if len(component_methods) > 0 else "unknown"
    )
    architecture_extension_df = summarize_epithelial_architecture_extensions(
        adata,
        cfg,
        force=FORCE_REBUILD_NICHES,
    )
    if not architecture_extension_df.empty:
        feature_df = feature_df.merge(
            architecture_extension_df,
            on=[NICHE_KEY, "sample_id"],
            how="left",
        )

    phenotype_labels = (
        adata.obs["Tier_A"]
        .value_counts()
        .loc[lambda s: ~s.index.isin(["Unknown", "noise", "unassigned"])]
        .index.tolist()
    )
    context_df = se.summarize_niche_surrounding_context(
        adata,
        niche_key=NICHE_KEY,
        phenotype_key="Tier_A",
        phenotype_labels=phenotype_labels,
        phenotype_feature_map=available_feature_map(adata),
        image_key="sample_id",
        adjacency_key="cell_graph_connectivities",
        surround_hops=SURROUND_HOPS,
        min_cells=MIN_NICHE_CELLS,
        summary_stats=("mean", "median"),
        show_progress=True,
        progress_desc=f"{cfg['sample_id']} surroundings",
    )
    context_df["sample_id"] = cfg["sample_id"]
    context_df["display_name"] = cfg["display_name"]
    context_df["disease_group"] = cfg["disease_group"]
    context_df["xenium_niche_feature_version"] = NICHE_FEATURE_VERSION
    context_df["xenium_annotation_version"] = adata.uns.get("spatioev_xenium_annotation_version", "unknown")
    context_df["xenium_dapi_feature_version"] = expected_dapi_feature_version

    feature_df = add_xenium_niche_module_scores(feature_df.merge(
        context_df.drop(columns=["n_cells"], errors="ignore"),
        on=["sample_id", NICHE_KEY],
        how="left",
        suffixes=("", "__context"),
    ))

    save_df(feature_df, feature_path)
    save_df(context_df, context_path)
    sample_feature_frames.append(feature_df)
    sample_context_frames.append(context_df)

    adata.write_h5ad(NICHE_DIR / f"{cfg['sample_id']}_with_niches.h5ad")
    del adata, feature_df, context_df
    gc.collect()

pooled_niche_feature_df = pd.concat(sample_feature_frames, ignore_index=True)
pooled_surround_context_df = pd.concat(sample_context_frames, ignore_index=True)

save_df(pooled_niche_feature_df, OUTPUT_DIR / "pooled_xenium_niche_feature_df.pkl")
save_df(pooled_surround_context_df, OUTPUT_DIR / "pooled_xenium_surround_context_df.pkl")

print(pooled_niche_feature_df.shape)
pooled_niche_feature_df.head()
'''
    ),
    code(
        r'''
module_cols = [c for c in pooled_niche_feature_df.columns if c.startswith("xenium_") and c.endswith("_score")]

fig, axes = plt.subplots(1, len(module_cols), figsize=(2.4 * len(module_cols), 3), sharey=False)
axes = np.array(axes).reshape(-1)
for ax, col in zip(axes, module_cols):
    sns.boxplot(
        data=pooled_niche_feature_df,
        x="disease_group",
        y=col,
        hue="disease_group",
        palette={"NormalPancreas": "#4daf4a", "PDAC": "#e41a1c"},
        legend=False,
        fliersize=0,
        ax=ax,
    )
    ax.set_title(col.replace("xenium_", "").replace("_score", ""), fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(False)
plt.tight_layout()
plt.show()
'''
    ),
    md(
        """
## Feature Coverage Check Against PanIN/PDAC Morphology

The JCI review emphasizes that PanIN/PDAC progression involves architectural complexity, mucinous ductal programs, nuclear atypia, proliferation, and microenvironment remodeling. This table makes the Xenium feature coverage explicit so we do not over-claim what the dataset can measure.
"""
    ),
    code(
        r'''
feature_coverage_df = pd.DataFrame(
    [
        {
            "axis": "Ductal epithelial identity",
            "covered_now": True,
            "current_features": "EPCAM/KRT/SOX9/MUC/TFF/CEACAM/AGR expression programs when present",
            "missing_or_optional": "KRT19 is absent from some panels; membrane polarity markers from IMC are not available",
        },
        {
            "axis": "PanIN-like mucin/remodeling",
            "covered_now": True,
            "current_features": "MUC5AC, TFF1/2/3, CEACAM6, AGR2/3 expression plus epithelial component geometry",
            "missing_or_optional": "No manual PanIN grade labels yet; should be validated against H&E/ROI if available",
        },
        {
            "axis": "Nuclear atypia / shape",
            "covered_now": True,
            "current_features": "nucleus_area from cells table plus nucleus-boundary circularity, solidity, elongation, Feret diameter, irregularity",
            "missing_or_optional": "DAPI intensity/texture can now be extracted above; run full epithelial extraction before rebuilding niches to include it",
        },
        {
            "axis": "Apicobasal polarity / CK19-NaKATPase relationship",
            "covered_now": False,
            "current_features": "None",
            "missing_or_optional": "Requires multiplexed imaging membrane channels or aligned IF; Xenium transcript panel alone cannot reproduce this",
        },
        {
            "axis": "Architectural complexity",
            "covered_now": True,
            "current_features": "Boundary-proximity epithelial components, hull geometry, topology/skeleton features, duct/lumen topology, duct continuity proxies",
            "missing_or_optional": "2D section limitation remains; true 3D duct connectedness requires serial sections or registration",
        },
        {
            "axis": "Duct/lumen topology",
            "covered_now": True,
            "current_features": "Polygon-union hole/lumen fraction, lumen area, outer duct caliber, epithelial ring thickness, cystic dilation proxy",
            "missing_or_optional": "Depends on cell-boundary polygon quality; H&E would improve mucin/lumen interpretation",
        },
        {
            "axis": "Ductal continuity / cancerization",
            "covered_now": True,
            "current_features": "Component axis length/ratio, nearby epithelial components, nearest component distance, neighbor orientation alignment",
            "missing_or_optional": "This is a 2D local proxy for intraductal spread, not proof of clonal ductal migration",
        },
        {
            "axis": "Epithelial-stromal interface disruption",
            "covered_now": True,
            "current_features": "Union-boundary roughness, boundary length per area, nearby unassigned epithelial cells, immediate fibroblast contact from cell graph",
            "missing_or_optional": "Budding/extrusion calls are proxy features and should be spatially checked in representative regions",
        },
        {
            "axis": "Proliferation",
            "covered_now": True,
            "current_features": "MKI67/TOP2A/UBE2C/CENPF/CDK1 expression when present",
            "missing_or_optional": "Panel-dependent availability",
        },
        {
            "axis": "Desmoplastic/immune microenvironment",
            "covered_now": True,
            "current_features": "Surrounding Tier_A/Tier_B proportions and compartment marker summaries",
            "missing_or_optional": "Annotation should be reviewed using the notebook 01 QC plots before interpreting",
        },
    ]
)
feature_coverage_df
'''
    ),
]


pseudotime_cells = [
    md(
        """
# Pooled Xenium Pancreas/PDAC Niche Pseudotime

This notebook fits a pooled epithelial-niche trajectory across three 10x Xenium PDAC datasets and one nondiseased pancreas dataset.

Interpretation guardrail:

This is a cross-sample **morphologic/transcriptomic spatial continuum**, not literal patient time. The value is that SpatioEv can align normal-like, PanIN-like, proliferative, desmoplastic, and immune-remodeled epithelial niches onto a shared tree.
"""
    ),
    code(COMMON_SETUP),
    code(
        r'''
import elpigraph
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP

NICHE_KEY = "xenium_ductal_epithelium_component"
NICHE_DIR = OUTPUT_DIR / "niche_features"
RESULT_DIR = OUTPUT_DIR / "pseudotime"
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
SIMPLIFIED_NUM_NODES = 24
DETAILED_TREE_MIN_NODES = 40
DETAILED_TREE_MAX_NODES = 100
DETAILED_TREE_NODE_SCALE = 2.5
EXPECTED_NICHE_FEATURE_VERSION = "boundary_components_v3_lumen_continuity_interface"
EXPECTED_DAPI_FEATURE_VERSION = "user_verified_focus_dapi_sources_v2"
EXPECTED_ANNOTATION_VERSION = "cluster_full_panel_v9_xenium_graphclust_io_mucosa_submucosa_k24"

pooled_niche_feature_df = attach_clinical_metadata(load_df(OUTPUT_DIR / "pooled_xenium_niche_feature_df.pkl"))
print(pooled_niche_feature_df.shape)
clinical_sample_metadata_df = clinical_metadata_frame()
display(
    clinical_sample_metadata_df[
        [
            "sample_id",
            "clinical_diagnosis",
            "clinical_stage",
            "clinical_grade",
            "tumor_content_percent",
            "clinical_progression_label",
        ]
    ]
)

version_cols = [
    col for col in [
        "sample_id",
        "xenium_annotation_version",
        "xenium_niche_feature_version",
        "xenium_dapi_feature_version",
    ]
    if col in pooled_niche_feature_df.columns
]
if len(version_cols) > 1:
    version_summary_df = pooled_niche_feature_df[version_cols].drop_duplicates().sort_values("sample_id")
    display(version_summary_df)
else:
    print("No upstream version metadata found. Rerun notebook 02 with the current workflow before final interpretation.")

if "xenium_annotation_version" in pooled_niche_feature_df.columns:
    annotation_versions = set(pooled_niche_feature_df["xenium_annotation_version"].dropna().astype(str))
    if annotation_versions != {EXPECTED_ANNOTATION_VERSION}:
        print(f"Warning: annotation versions differ from expected {EXPECTED_ANNOTATION_VERSION}: {sorted(annotation_versions)}")
else:
    print("Warning: xenium_annotation_version is missing; rerun notebook 02 after notebook 01.")

if "xenium_niche_feature_version" in pooled_niche_feature_df.columns:
    niche_versions = set(pooled_niche_feature_df["xenium_niche_feature_version"].dropna().astype(str))
    if niche_versions != {EXPECTED_NICHE_FEATURE_VERSION}:
        print(f"Warning: niche feature versions differ from expected {EXPECTED_NICHE_FEATURE_VERSION}: {sorted(niche_versions)}")

if "xenium_dapi_feature_version" in pooled_niche_feature_df.columns:
    dapi_versions = set(pooled_niche_feature_df["xenium_dapi_feature_version"].fillna("none").astype(str))
    if dapi_versions == {"none"}:
        print("DAPI full epithelial features are not included yet. This is OK for a first run; run full DAPI extraction in notebook 02 to include nuclear texture.")
    elif dapi_versions != {EXPECTED_DAPI_FEATURE_VERSION}:
        print(f"Warning: DAPI feature versions differ from expected {EXPECTED_DAPI_FEATURE_VERSION}: {sorted(dapi_versions)}")

pooled_niche_feature_df.head()
'''
    ),
    code(
        r'''
MODULE_COLS = [
    "xenium_epithelial_identity_score",
    "xenium_panin_like_remodeling_score",
    "xenium_proliferation_score",
    "xenium_desmoplastic_context_score",
    "xenium_immune_context_score",
    "xenium_checkpoint_context_score",
    "xenium_nuclear_dapi_texture_score",
    "xenium_duct_lumen_topology_score",
    "xenium_duct_continuity_cancerization_score",
    "xenium_epithelial_stromal_interface_disruption_score",
]

HISTOLOGY_PROXY_DEFINITIONS = {
    "histology__normal_duct_like_score": [
        ("xenium_epithelial_identity_score", 1),
        ("state__CFTR_expr_z__mean", 1),
        ("state__FXYD2_expr_z__mean", 1),
        ("state__TM4SF4_expr_z__mean", 1),
        ("state__PROX1_expr_z__mean", 1),
        ("xenium_panin_like_remodeling_score", -1),
        ("xenium_proliferation_score", -1),
        ("xenium_desmoplastic_context_score", -1),
    ],
    "histology__adm_panin_like_score": [
        ("xenium_panin_like_remodeling_score", 1),
        ("state__MUC5AC_expr_z__mean", 1),
        ("state__TFF1_expr_z__mean", 1),
        ("state__TFF2_expr_z__mean", 1),
        ("state__TFF3_expr_z__mean", 1),
        ("state__CEACAM6_expr_z__mean", 1),
        ("state__AGR3_expr_z__mean", 1),
        ("state__SOX9_expr_z__mean", 1),
    ],
    "histology__glandular_architecture_score": [
        ("xenium_epithelial_identity_score", 1),
        ("xenium_duct_lumen_topology_score", 1),
        ("geometry__hull_circularity", 1),
        ("geometry__orientation_coherence", 1),
        ("duct_lumen__lumen_fraction", 1),
        ("duct_lumen__epithelial_ring_thickness_um", 1),
        ("topology__skeleton_branchpoint_fraction", 1),
        ("topology__skeleton_n_edges", 1),
        ("topology__skeleton_total_length", 1),
        ("geometry__cell_density_hull", -1),
    ],
    "histology__ductal_continuity_cancerization_score": [
        ("xenium_duct_continuity_cancerization_score", 1),
        ("duct_continuity__component_axis_length_um", 1),
        ("duct_continuity__component_axis_ratio", 1),
        ("duct_continuity__nearby_components_150um", 1),
        ("duct_continuity__neighbor_orientation_alignment_150um", 1),
        ("duct_continuity__nearest_component_distance_um", -1),
    ],
    "histology__epithelial_stromal_interface_disruption_score": [
        ("xenium_epithelial_stromal_interface_disruption_score", 1),
        ("interface__epithelial_boundary_roughness", 1),
        ("interface__nearby_unassigned_epithelial_cells_per_100um_boundary", 1),
        ("interface__fibroblast_contact_fraction_hop1", 1),
        ("interface__non_epithelial_contact_fraction_hop1", 1),
    ],
    "histology__desmoplastic_tumor_score": [
        ("xenium_desmoplastic_context_score", 1),
        ("xenium_epithelial_stromal_interface_disruption_score", 1),
        ("surround_prop__Fibroblasts", 1),
        ("surround__Fibroblasts__ACTA2_expr_z__mean", 1),
        ("surround__Fibroblasts__PDGFRA_expr_z__mean", 1),
        ("surround__Fibroblasts__THY1_expr_z__mean", 1),
        ("surround__Fibroblasts__PDPN_expr_z__mean", 1),
        ("graph_surround__phenotype_entropy", 1),
    ],
    "histology__immune_inflamed_score": [
        ("xenium_immune_context_score", 1),
        ("xenium_checkpoint_context_score", 1),
        ("surround_prop__T_cells", 1),
        ("surround_prop__B_lineage", 1),
        ("surround_prop__Myeloid_cells", 1),
        ("surround__T_cells__CD3D_expr_z__mean", 1),
        ("surround__B_lineage__CD79A_expr_z__mean", 1),
    ],
    "histology__immune_exclusion_score": [
        ("xenium_immune_context_score", -1),
        ("surround_prop__T_cells", -1),
        ("surround_prop__B_lineage", -1),
        ("surround_prop__Myeloid_cells", -1),
        ("graph_surround__phenotype_entropy", -1),
    ],
    "histology__duodenum_invasion_context_score": [
        ("surround_prop__Mucosa_gland", 1),
        ("surround_prop__Submucosa", 1),
        ("surround__Duodenum_epithelial__DMBT1_expr_z__mean", 1),
        ("surround__Duodenum_epithelial__KRT20_expr_z__mean", 1),
    ],
    "histology__gland_poor_undifferentiated_score": [
        ("histology__glandular_architecture_score", -1),
        ("histology__epithelial_stromal_interface_disruption_score", 1),
        ("xenium_epithelial_identity_score", -1),
        ("xenium_proliferation_score", 1),
        ("histology__immune_exclusion_score", 1),
        ("xenium_desmoplastic_context_score", -1),
    ],
}

def _zscore_series(values):
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or np.isclose(sd, 0):
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd

def add_histology_proxy_scores(df, definitions):
    df = df.copy()
    availability_rows = []
    for score_name, components in definitions.items():
        component_values = []
        used = []
        missing = []
        for col, sign in components:
            if col not in df.columns:
                missing.append(col)
                continue
            component_values.append(sign * _zscore_series(df[col]))
            used.append(col)
        if len(component_values) == 0:
            df[score_name] = np.nan
        else:
            df[score_name] = pd.concat(component_values, axis=1).mean(axis=1, skipna=True)
        availability_rows.append(
            {
                "score": score_name,
                "n_used": len(used),
                "n_missing": len(missing),
                "used_features": ", ".join(used),
                "missing_features": ", ".join(missing),
            }
        )
    return df, pd.DataFrame(availability_rows)

pooled_niche_feature_df, histology_proxy_availability_df = add_histology_proxy_scores(
    pooled_niche_feature_df,
    HISTOLOGY_PROXY_DEFINITIONS,
)
HISTOLOGY_PROXY_COLS = [col for col in HISTOLOGY_PROXY_DEFINITIONS if col in pooled_niche_feature_df.columns]
MODULE_COLS = list(dict.fromkeys(MODULE_COLS + HISTOLOGY_PROXY_COLS))

display(histology_proxy_availability_df)

SUPPLEMENTAL_PATTERNS = [
    "duct_lumen__",
    "duct_continuity__",
    "interface__",
    "geometry__",
    "topology__skeleton",
    "surround_prop__",
    "state__EPCAM",
    "state__KRT7",
    "state__SOX9",
    "state__MUC5AC",
    "state__TFF",
    "state__CEACAM6",
    "state__AGR3",
    "state__MKI67",
    "state__UBE2C",
    "state__TOP2A",
    "state__dapi_",
    "state__nucleus_boundary",
]

def prepare_feature_matrix(df, feature_cols, max_na_frac=0.45, corr_cutoff=0.97):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.isna().mean() <= max_na_frac]
    X = X.apply(lambda col: col.fillna(col.median()), axis=0)
    X = X.loc[:, X.var(ddof=0) > 1e-8]

    priority = [c for c in MODULE_COLS if c in X.columns]
    candidates = [c for c in X.columns if c not in priority]
    corr_abs = X.corr().abs()
    selected = list(priority)
    for col in candidates:
        if not selected:
            selected.append(col)
            continue
        max_corr = corr_abs.loc[col, selected].dropna().max()
        if not np.isfinite(max_corr) or max_corr < corr_cutoff:
            selected.append(col)
    return X[selected].copy(), selected

candidate_cols = []
for col in pooled_niche_feature_df.columns:
    if col in MODULE_COLS:
        candidate_cols.append(col)
    elif any(col.startswith(pattern) for pattern in SUPPLEMENTAL_PATTERNS):
        candidate_cols.append(col)

X_pool, selected_cols = prepare_feature_matrix(pooled_niche_feature_df, candidate_cols)
print(f"Selected {len(selected_cols)} trajectory features")
pd.Series(selected_cols, name="feature").to_frame().head(25)
'''
    ),
    code(
        r'''
TRAJECTORY_SCALING = "block_balanced"

def assign_feature_block(feature):
    if feature.startswith("histology__") or feature in MODULE_COLS:
        return "histology_modules"
    if feature.startswith("duct_lumen__") or feature.startswith("duct_continuity__"):
        return "duct_architecture"
    if feature.startswith("interface__"):
        return "epithelial_stromal_interface"
    if feature.startswith("surround_prop__") or feature.startswith("surround__") or feature.startswith("graph_surround__"):
        return "microenvironment"
    if feature.startswith("state__dapi_") or feature.startswith("state__nucleus_boundary"):
        return "nuclear_morphology"
    if feature.startswith("geometry__") or feature.startswith("topology__"):
        return "architecture_topology"
    if feature.startswith("state__"):
        return "epithelial_state"
    return "other"

def block_balance_feature_matrix(X):
    X_scaled_df = pd.DataFrame(
        StandardScaler().fit_transform(X),
        index=X.index,
        columns=X.columns,
    )
    feature_block_df = pd.DataFrame(
        {
            "feature": X.columns,
            "feature_block": [assign_feature_block(col) for col in X.columns],
        }
    )
    X_balanced_df = X_scaled_df.copy()
    for block, block_features in feature_block_df.groupby("feature_block")["feature"]:
        block_features = block_features.tolist()
        if len(block_features) == 0:
            continue
        # Equalize broad information families so DAPI/shape or context columns
        # cannot dominate only because there are more of them.
        X_balanced_df[block_features] = X_balanced_df[block_features] / np.sqrt(len(block_features))
    return X_balanced_df, feature_block_df

if TRAJECTORY_SCALING == "block_balanced":
    X_scaled_df, feature_block_df = block_balance_feature_matrix(X_pool)
else:
    X_scaled_df = pd.DataFrame(
        StandardScaler().fit_transform(X_pool),
        index=X_pool.index,
        columns=X_pool.columns,
    )
    feature_block_df = pd.DataFrame(
        {
            "feature": X_pool.columns,
            "feature_block": [assign_feature_block(col) for col in X_pool.columns],
        }
    )

display(feature_block_df["feature_block"].value_counts().rename("n_features").reset_index())
X_scaled = X_scaled_df.to_numpy()

pca = PCA(n_components=min(12, X_scaled.shape[1]), random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
X_pca_use = X_pca[:, : min(6, X_pca.shape[1])]

umap_model = UMAP(
    n_neighbors=15,
    min_dist=0.20,
    n_components=2,
    random_state=RANDOM_STATE,
)
X_umap = umap_model.fit_transform(X_pca_use)

embedding_meta_cols = [
    col
    for col in [
        NICHE_KEY,
        "sample_id",
        "display_name",
        "disease_group",
        "clinical_diagnosis",
        "clinical_stage",
        "clinical_grade",
        "clinical_grade_order",
        "tumor_content_percent",
        "clinical_progression_label",
        "clinical_progression_order",
        "clinical_note",
    ]
    if col in pooled_niche_feature_df.columns
]
embedding_df = pooled_niche_feature_df[embedding_meta_cols].copy()
embedding_df["UMAP1"] = X_umap[:, 0]
embedding_df["UMAP2"] = X_umap[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
sns.scatterplot(
    data=embedding_df,
    x="UMAP1",
    y="UMAP2",
    hue="disease_group",
    palette={"NormalPancreas": "#4daf4a", "PDAC": "#e41a1c"},
    s=5,
    linewidth=0,
    alpha=0.65,
    ax=axes[0],
)
axes[0].set_title("Pooled Xenium UMAP by disease")
axes[0].grid(False)
axes[0].legend(frameon=False, fontsize=7)

sns.scatterplot(
    data=embedding_df,
    x="UMAP1",
    y="UMAP2",
    hue="sample_id",
    palette="tab10",
    s=5,
    linewidth=0,
    alpha=0.65,
    ax=axes[1],
)
axes[1].set_title("Pooled Xenium UMAP by sample")
axes[1].grid(False)
axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

scat = axes[2].scatter(
    embedding_df["UMAP1"],
    embedding_df["UMAP2"],
    c=pooled_niche_feature_df["xenium_panin_like_remodeling_score"],
    s=5,
    linewidth=0,
    alpha=0.7,
    cmap="magma",
)
axes[2].set_title("PanIN-like remodeling score")
axes[2].grid(False)
plt.colorbar(scat, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
'''
    ),
    code(
        r'''
reduced_data = X_pca_use
n_niches = reduced_data.shape[0]
num_nodes = int(
    min(
        DETAILED_TREE_MAX_NODES,
        max(DETAILED_TREE_MIN_NODES, np.ceil(np.sqrt(n_niches) * DETAILED_TREE_NODE_SCALE)),
    )
)
num_nodes = int(min(num_nodes, max(2, n_niches - 1)))
num_nodes_simple = int(min(SIMPLIFIED_NUM_NODES, max(2, n_niches - 1)))

pg_tree = elpigraph.computeElasticPrincipalTree(
    X=reduced_data,
    NumNodes=num_nodes,
    Lambda=0.005,
    Mu=0.01,
)[0]

pg_tree_simple = elpigraph.computeElasticPrincipalTree(
    X=reduced_data,
    NumNodes=num_nodes_simple,
    Lambda=0.005,
    Mu=0.01,
)[0]

root_score = pd.to_numeric(pooled_niche_feature_df["xenium_epithelial_identity_score"], errors="coerce")
normal_mask = pooled_niche_feature_df["disease_group"] == "NormalPancreas"
root_mask = normal_mask & (root_score >= root_score[normal_mask].quantile(0.85))
if root_mask.sum() == 0:
    root_mask = normal_mask
root_anchor = reduced_data[root_mask.to_numpy()].mean(axis=0)
source_node = int(np.argmin(np.sum((pg_tree["NodePositions"] - root_anchor) ** 2, axis=1)))
source_node_simple = int(np.argmin(np.sum((pg_tree_simple["NodePositions"] - root_anchor) ** 2, axis=1)))

elpigraph.utils.getPseudotime(
    X=reduced_data,
    PG=pg_tree,
    source=source_node,
    target=None,
)

elpigraph.utils.getPseudotime(
    X=reduced_data,
    PG=pg_tree_simple,
    source=source_node_simple,
    target=None,
)

projection = pg_tree["projection"]
projection_simple = pg_tree_simple["projection"]
result_df = embedding_df.copy()
result_df["xenium_pseudotime"] = pg_tree["pseudotime"]
result_df["xenium_node_id"] = projection["node_id"]
result_df["xenium_edge_id"] = projection["edge_id"]
result_df["xenium_simple_pseudotime"] = pg_tree_simple["pseudotime"]
result_df["xenium_simple_node_id"] = projection_simple["node_id"]
result_df["xenium_simple_edge_id"] = projection_simple["edge_id"]
for col in MODULE_COLS:
    if col in pooled_niche_feature_df.columns:
        result_df[col] = pooled_niche_feature_df[col].values

print("NumNodes (detailed tree):", num_nodes)
print("NumNodes (simplified tree):", num_nodes_simple)
print("Root niches:", int(root_mask.sum()))
print("Detailed root node:", source_node)
print("Simplified root node:", source_node_simple)
result_df.head()
'''
    ),
    code(
        r'''
def tree_edges(pg_tree):
    edges = np.asarray(pg_tree["Edges"][0], dtype=int)
    return [tuple(map(int, edge)) for edge in edges]

def node_graph(pg_tree):
    G = nx.Graph()
    G.add_edges_from(tree_edges(pg_tree))
    return G

def infer_branch_labels(pg_tree, result_df, source_node, node_col="xenium_node_id"):
    G = node_graph(pg_tree)
    degrees = dict(G.degree())
    branch_nodes = [n for n, d in degrees.items() if d >= 3]
    hub = max(branch_nodes, key=lambda n: degrees[n]) if branch_nodes else source_node
    trunk_path = nx.shortest_path(G, source=source_node, target=hub)

    branch_paths = {}
    branch_i = 1
    for neighbor in sorted(G.neighbors(hub)):
        if neighbor in trunk_path:
            continue
        component = nx.node_connected_component(nx.Graph(G.copy().subgraph(set(G.nodes) - {hub})), neighbor)
        leaves = sorted([n for n in component if G.degree(n) == 1])
        if len(leaves) <= 1:
            end = leaves[0] if leaves else neighbor
            branch_paths[f"branch {branch_i}"] = [hub] + nx.shortest_path(G, source=neighbor, target=end)
            branch_i += 1
            continue
        for leaf in leaves:
            branch_paths[f"branch {branch_i}"] = [hub] + nx.shortest_path(G, source=neighbor, target=leaf)
            branch_i += 1

    node_to_branch = {n: "trunk" for n in trunk_path}
    for label, path in branch_paths.items():
        for n in path[1:]:
            node_to_branch[n] = label

    labels = result_df[node_col].map(node_to_branch).fillna("other")
    return labels.astype(str), {"hub_node": hub, "trunk_path": trunk_path, "branch_paths": branch_paths}

result_df["major_branch"], branch_defs = infer_branch_labels(
    pg_tree_simple,
    result_df,
    source_node_simple,
    node_col="xenium_simple_node_id",
)
print(branch_defs)
'''
    ),
    code(
        r'''
def compute_umap_tree_nodes(pg_tree, X_pca_use, X_umap):
    node_positions = pg_tree["NodePositions"]
    nbrs = NearestNeighbors(n_neighbors=min(60, len(X_pca_use))).fit(X_pca_use)
    node_umap = []
    for node_pos in node_positions:
        _, idx = nbrs.kneighbors(node_pos.reshape(1, -1))
        node_umap.append(np.median(X_umap[idx[0]], axis=0))
    return np.asarray(node_umap)

def plot_tree_overlay(ax, result_df, pg_tree, node_umap, color_col, palette=None, point_size=5):
    sns.scatterplot(
        data=result_df,
        x="UMAP1",
        y="UMAP2",
        hue=color_col,
        palette=palette,
        s=point_size,
        linewidth=0,
        alpha=0.65,
        ax=ax,
    )
    for u, v in tree_edges(pg_tree):
        ax.plot(
            [node_umap[u, 0], node_umap[v, 0]],
            [node_umap[u, 1], node_umap[v, 1]],
            color="#222222",
            linewidth=1.0,
            alpha=0.75,
        )
    for node, xy in enumerate(node_umap):
        ax.text(xy[0], xy[1], str(node), fontsize=6, color="black")
    ax.grid(False)

node_umap = compute_umap_tree_nodes(pg_tree_simple, reduced_data, X_umap)
branch_order = ["trunk"] + sorted([b for b in result_df["major_branch"].unique() if b not in {"trunk", "other"}]) + ["other"]
branch_palette = dict(zip(branch_order, sns.color_palette("tab20", n_colors=len(branch_order))))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
scat = axes[0].scatter(
    result_df["UMAP1"],
    result_df["UMAP2"],
    c=result_df["xenium_pseudotime"],
    s=5,
    linewidth=0,
    cmap="viridis",
    alpha=0.75,
)
axes[0].set_title("Xenium UMAP by pseudotime")
axes[0].grid(False)
plt.colorbar(scat, ax=axes[0], fraction=0.046, pad=0.04)

plot_tree_overlay(
    axes[1],
    result_df,
    pg_tree_simple,
    node_umap,
    color_col="major_branch",
    palette=branch_palette,
    point_size=5,
)
axes[1].set_title("Pooled UMAP with simplified tree/branches")
axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

sns.boxplot(
    data=result_df,
    x="disease_group",
    y="xenium_pseudotime",
    hue="disease_group",
    palette={"NormalPancreas": "#4daf4a", "PDAC": "#e41a1c"},
    legend=False,
    fliersize=0,
    ax=axes[2],
)
axes[2].set_title("Pseudotime by disease group")
axes[2].grid(False)

plt.tight_layout()
plt.show()
'''
    ),
    code(
        r'''
def lowess_or_line(x, y, frac=0.3):
    import statsmodels.api as sm
    tmp = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(tmp) < 20:
        return tmp["x"].to_numpy(), tmp["y"].to_numpy()
    out = sm.nonparametric.lowess(tmp["y"], tmp["x"], frac=frac, return_sorted=True)
    return out[:, 0], out[:, 1]

trend_features = [
    "xenium_epithelial_identity_score",
    "xenium_panin_like_remodeling_score",
    "xenium_proliferation_score",
    "xenium_desmoplastic_context_score",
    "xenium_immune_context_score",
    "xenium_checkpoint_context_score",
]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.ravel()
for ax, feature in zip(axes, trend_features):
    if feature not in result_df.columns:
        ax.axis("off")
        continue
    for disease, color in {"NormalPancreas": "#4daf4a", "PDAC": "#e41a1c"}.items():
        sub = result_df[result_df["disease_group"] == disease]
        x, y = lowess_or_line(sub["xenium_pseudotime"], sub[feature], frac=0.35)
        ax.plot(x, y, color=color, linewidth=1.8, label=disease)
    ax.set_title(feature.replace("xenium_", "").replace("_score", ""), fontsize=8)
    ax.set_xlabel("Xenium pseudotime")
    ax.set_ylabel("score")
    ax.grid(False)
axes[0].legend(frameon=False, fontsize=7)
plt.tight_layout()
plt.show()
'''
    ),
    md(
        """
## Sample Effect Diagnostic and Batch-Aware Sensitivity Trajectory

The pooled Xenium datasets can separate by sample because disease, tissue region, panel content, and technical acquisition are partly confounded. This section does **not** replace the original trajectory. It adds a sensitivity trajectory where selected features are z-scored within each sample before PCA/tree fitting, and the root is anchored by normal-like epithelial niches from all samples.

This is especially useful for checking residual normal pancreas regions inside PDAC samples. If those niches are late only in the original trajectory but early/intermediate after sample-centering, the original result is likely influenced by sample-specific feature shifts.
"""
    ),
    code(
        r'''
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import f_classif

def minmax(values):
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(hi - lo) or np.isclose(hi, lo):
        return np.full(values.shape, np.nan)
    return (values - lo) / (hi - lo)

def knn_purity(embedding, labels, n_neighbors=15):
    labels = np.asarray(labels).astype(str)
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(embedding))).fit(embedding)
    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    return (labels[idx] == labels[:, None]).mean(axis=1)

sample_ids = pooled_niche_feature_df["sample_id"].astype(str)
disease_ids = pooled_niche_feature_df["disease_group"].astype(str)
same_sample_fraction = knn_purity(X_pca_use, sample_ids, n_neighbors=15)
same_disease_fraction = knn_purity(X_pca_use, disease_ids, n_neighbors=15)

sample_effect_qc_df = pd.DataFrame(
    {
        "sample_id": sample_ids,
        "disease_group": disease_ids,
        "same_sample_fraction": same_sample_fraction,
        "same_disease_fraction": same_disease_fraction,
        "xenium_pseudotime": result_df["xenium_pseudotime"],
    }
)

feature_sample_F, _ = f_classif(
    StandardScaler().fit_transform(X_pool),
    sample_ids.to_numpy(),
)
sample_discriminating_feature_df = (
    pd.DataFrame({"feature": X_pool.columns, "sample_F": feature_sample_F})
    .sort_values("sample_F", ascending=False)
    .reset_index(drop=True)
)

display(
    sample_effect_qc_df.groupby(["sample_id", "disease_group"])[
        ["same_sample_fraction", "same_disease_fraction", "xenium_pseudotime"]
    ]
    .median()
    .round(3)
)
display(sample_discriminating_feature_df.head(20))
'''
    ),
    code(
        r'''
def sample_center_feature_matrix(X, sample_ids):
    X_centered = X.copy()
    sample_ids = pd.Series(sample_ids, index=X.index).astype(str)
    for sample_id in sample_ids.unique():
        idx = sample_ids == sample_id
        mu = X_centered.loc[idx].mean(axis=0)
        sd = X_centered.loc[idx].std(axis=0, ddof=0).replace(0, np.nan)
        X_centered.loc[idx] = (X_centered.loc[idx] - mu) / sd
    return X_centered.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def build_normal_like_root_mask(df):
    epi = pd.to_numeric(df["xenium_epithelial_identity_score"], errors="coerce")
    normal_mask = df["disease_group"].astype(str).eq("NormalPancreas")
    root_normal = normal_mask & (epi >= epi[normal_mask].quantile(0.85))

    normal_like = epi >= epi.quantile(0.50)
    for feature, quantile in [
        ("xenium_panin_like_remodeling_score", 0.40),
        ("xenium_proliferation_score", 0.50),
        ("xenium_desmoplastic_context_score", 0.40),
        ("xenium_immune_context_score", 0.40),
    ]:
        if feature in df.columns:
            vals = pd.to_numeric(df[feature], errors="coerce")
            normal_like &= vals <= vals.quantile(quantile)

    root_mask = root_normal | normal_like
    if root_mask.sum() < 20:
        root_mask = root_normal
    if root_mask.sum() == 0:
        root_mask = normal_mask
    return root_mask.fillna(False)

X_pool_sample_centered = sample_center_feature_matrix(X_pool, pooled_niche_feature_df["sample_id"])
X_sample_centered_scaled_df, sample_centered_feature_block_df = block_balance_feature_matrix(X_pool_sample_centered)
X_sample_centered_scaled = X_sample_centered_scaled_df.to_numpy()
X_pca_sample_centered = PCA(
    n_components=min(12, X_sample_centered_scaled.shape[1]),
    random_state=RANDOM_STATE,
).fit_transform(X_sample_centered_scaled)
X_pca_sample_centered_use = X_pca_sample_centered[:, : min(6, X_pca_sample_centered.shape[1])]

X_umap_sample_centered = UMAP(
    n_neighbors=15,
    min_dist=0.20,
    n_components=2,
    random_state=RANDOM_STATE,
).fit_transform(X_pca_sample_centered_use)

pg_tree_sample_centered = elpigraph.computeElasticPrincipalTree(
    X=X_pca_sample_centered_use,
    NumNodes=num_nodes,
    Lambda=0.005,
    Mu=0.01,
)[0]

root_mask_sample_centered = build_normal_like_root_mask(pooled_niche_feature_df)
root_anchor_sample_centered = X_pca_sample_centered_use[root_mask_sample_centered.to_numpy()].mean(axis=0)
source_node_sample_centered = int(
    np.argmin(np.sum((pg_tree_sample_centered["NodePositions"] - root_anchor_sample_centered) ** 2, axis=1))
)
elpigraph.utils.getPseudotime(
    X=X_pca_sample_centered_use,
    PG=pg_tree_sample_centered,
    source=source_node_sample_centered,
    target=None,
)

result_df["UMAP1_sample_centered"] = X_umap_sample_centered[:, 0]
result_df["UMAP2_sample_centered"] = X_umap_sample_centered[:, 1]
result_df["xenium_pseudotime_sample_centered"] = pg_tree_sample_centered["pseudotime"]
result_df["xenium_pseudotime_norm"] = minmax(result_df["xenium_pseudotime"])
result_df["xenium_pseudotime_sample_centered_norm"] = minmax(result_df["xenium_pseudotime_sample_centered"])
result_df["xenium_node_id_sample_centered"] = pg_tree_sample_centered["projection"]["node_id"]
result_df["xenium_edge_id_sample_centered"] = pg_tree_sample_centered["projection"]["edge_id"]
result_df["normal_like_root_candidate"] = root_mask_sample_centered.to_numpy()

print("Sample-centered root node:", source_node_sample_centered)
print("Sample-centered root candidates by sample:")
print(pooled_niche_feature_df.loc[root_mask_sample_centered, "sample_id"].value_counts().to_string())

display(
    result_df.groupby(["sample_id", "disease_group"])[
        ["xenium_pseudotime", "xenium_pseudotime_sample_centered"]
    ]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .round(3)
)
'''
    ),
    code(
        r'''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

s0 = axes[0].scatter(
    result_df["UMAP1_sample_centered"],
    result_df["UMAP2_sample_centered"],
    c=result_df["xenium_pseudotime_sample_centered"],
    s=5,
    linewidth=0,
    alpha=0.75,
    cmap="viridis",
)
axes[0].set_title("Sample-centered UMAP by pseudotime")
axes[0].grid(False)
plt.colorbar(s0, ax=axes[0], fraction=0.046, pad=0.04)

sns.scatterplot(
    data=result_df,
    x="xenium_pseudotime_norm",
    y="xenium_pseudotime_sample_centered_norm",
    hue="sample_id",
    palette="tab10",
    s=10,
    linewidth=0,
    alpha=0.55,
    ax=axes[1],
)
axes[1].plot([0, 1], [0, 1], color="#333333", linewidth=1, linestyle="--")
axes[1].set_xlabel("Original pseudotime, normalized")
axes[1].set_ylabel("Sample-centered pseudotime, normalized")
axes[1].set_title("Original vs sample-centered pseudotime")
axes[1].grid(False)
axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

plot_df = result_df.melt(
    id_vars=["sample_id", "disease_group"],
    value_vars=["xenium_pseudotime", "xenium_pseudotime_sample_centered"],
    var_name="trajectory",
    value_name="pseudotime",
)
sns.boxplot(
    data=plot_df,
    x="sample_id",
    y="pseudotime",
    hue="trajectory",
    fliersize=0,
    ax=axes[2],
)
axes[2].set_title("Pseudotime distributions by sample")
axes[2].tick_params(axis="x", rotation=30)
axes[2].grid(False)
axes[2].legend(frameon=False, fontsize=6)

plt.tight_layout()
plt.show()
'''
    ),
    md(
        """
## Clinical Metadata Sanity Check

The clinical labels are **not** used to fit the trajectory. They are added here as an external interpretation layer: if the morphology/transcriptome-driven trajectory is sensible, the clinical contexts should show interpretable enrichment patterns without requiring a perfectly linear stage order.
"""
    ),
    code(
        r'''
clinical_plot_df = result_df.copy()
clinical_plot_df["clinical_progression_label"] = pd.Categorical(
    clinical_plot_df["clinical_progression_label"],
    categories=CLINICAL_LABEL_ORDER,
    ordered=True,
)

clinical_summary_df = (
    clinical_plot_df.groupby(
        [
            "sample_id",
            "clinical_progression_label",
            "clinical_diagnosis",
            "clinical_stage",
            "clinical_grade",
            "tumor_content_percent",
        ],
        observed=True,
        dropna=False,
    )
    .agg(
        n_niches=(NICHE_KEY, "nunique"),
        pseudotime_median=("xenium_pseudotime", "median"),
        pseudotime_q25=("xenium_pseudotime", lambda x: np.nanquantile(x, 0.25)),
        pseudotime_q75=("xenium_pseudotime", lambda x: np.nanquantile(x, 0.75)),
        sample_centered_pseudotime_median=("xenium_pseudotime_sample_centered", "median"),
        sample_centered_pseudotime_q25=("xenium_pseudotime_sample_centered", lambda x: np.nanquantile(x, 0.25)),
        sample_centered_pseudotime_q75=("xenium_pseudotime_sample_centered", lambda x: np.nanquantile(x, 0.75)),
    )
    .reset_index()
    .sort_values("clinical_progression_label")
)
display(clinical_summary_df)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, y_col, title in [
    (axes[0], "xenium_pseudotime", "Original pooled pseudotime by clinical context"),
    (axes[1], "xenium_pseudotime_sample_centered", "Sample-centered pseudotime by clinical context"),
]:
    sns.boxplot(
        data=clinical_plot_df,
        x="clinical_progression_label",
        y=y_col,
        hue="clinical_progression_label",
        order=CLINICAL_LABEL_ORDER,
        palette=CLINICAL_PALETTE,
        legend=False,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=clinical_plot_df.sample(min(len(clinical_plot_df), 1200), random_state=RANDOM_STATE),
        x="clinical_progression_label",
        y=y_col,
        order=CLINICAL_LABEL_ORDER,
        color="black",
        size=1.0,
        alpha=0.12,
        jitter=0.25,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Pseudotime")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(False)

plt.tight_layout()
plt.show()

clinical_branch_occupancy_df = (
    clinical_plot_df.groupby(["clinical_progression_label", "major_branch"], observed=True)
    .size()
    .rename("n")
    .reset_index()
)
clinical_branch_occupancy_df["fraction_within_clinical_context"] = (
    clinical_branch_occupancy_df["n"]
    / clinical_branch_occupancy_df.groupby("clinical_progression_label", observed=True)["n"].transform("sum")
)
branch_occupancy_heatmap_df = (
    clinical_branch_occupancy_df.pivot(
        index="major_branch",
        columns="clinical_progression_label",
        values="fraction_within_clinical_context",
    )
    .reindex(index=branch_order, columns=CLINICAL_LABEL_ORDER)
    .fillna(0)
)

fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, 0.4 * len(branch_occupancy_heatmap_df))))
sns.heatmap(
    branch_occupancy_heatmap_df,
    cmap="YlOrRd",
    vmin=0,
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "Fraction within clinical context"},
    ax=axes[0],
)
axes[0].set_title("Automatic branch occupancy by clinical context")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

sns.barplot(
    data=clinical_branch_occupancy_df,
    x="major_branch",
    y="fraction_within_clinical_context",
    hue="clinical_progression_label",
    hue_order=CLINICAL_LABEL_ORDER,
    palette=CLINICAL_PALETTE,
    ax=axes[1],
)
axes[1].set_title("Clinical context distribution across branches")
axes[1].set_xlabel("")
axes[1].set_ylabel("Fraction within clinical context")
axes[1].tick_params(axis="x", rotation=35)
axes[1].grid(False)
axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

clinical_score_cols = [
    col
    for col in [
        "histology__normal_duct_like_score",
        "histology__adm_panin_like_score",
        "histology__glandular_architecture_score",
        "histology__ductal_continuity_cancerization_score",
        "histology__epithelial_stromal_interface_disruption_score",
        "histology__desmoplastic_tumor_score",
        "histology__immune_inflamed_score",
        "histology__immune_exclusion_score",
        "histology__gland_poor_undifferentiated_score",
        "histology__duodenum_invasion_context_score",
    ]
    if col in clinical_plot_df.columns
]
clinical_score_median_df = (
    clinical_plot_df.groupby("clinical_progression_label", observed=True)[clinical_score_cols]
    .median()
    .reindex(CLINICAL_LABEL_ORDER)
)
clinical_score_heatmap_df = clinical_score_median_df.apply(_zscore_series, axis=0)
clinical_score_heatmap_df.columns = [
    col.replace("histology__", "").replace("_score", "").replace("_", " ")
    for col in clinical_score_heatmap_df.columns
]

plt.figure(figsize=(9.5, 3.4))
sns.heatmap(
    clinical_score_heatmap_df,
    cmap="RdBu_r",
    center=0,
    linewidths=0.25,
    linecolor="white",
    cbar_kws={"label": "Clinical-context median z-score"},
)
plt.title("Histology proxy enrichment by clinical context")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()

save_df(clinical_sample_metadata_df, RESULT_DIR / "xenium_clinical_sample_metadata.csv")
save_df(clinical_summary_df, RESULT_DIR / "xenium_clinical_pseudotime_summary.csv")
save_df(clinical_branch_occupancy_df, RESULT_DIR / "xenium_clinical_branch_occupancy.csv")
save_df(clinical_score_median_df.reset_index(), RESULT_DIR / "xenium_clinical_histology_score_medians.csv")
'''
    ),
    md(
        """
## Intrinsic Epithelial Sensitivity Trajectory

The microenvironment-aware trajectory is useful for disease-state mapping, but fibrosis and immune exclusion can dominate sample separation. This intrinsic sensitivity trajectory uses only epithelial/histology modules, epithelial state, duct architecture, architecture/topology, and nuclear morphology blocks after within-sample centering.
"""
    ),
    code(
        r'''
INTRINSIC_FEATURE_BLOCKS = {
    "histology_modules",
    "epithelial_state",
    "duct_architecture",
    "architecture_topology",
    "nuclear_morphology",
}
intrinsic_feature_cols = feature_block_df.loc[
    feature_block_df["feature_block"].isin(INTRINSIC_FEATURE_BLOCKS),
    "feature",
].tolist()
intrinsic_feature_cols = [col for col in intrinsic_feature_cols if col in X_pool.columns]

if len(intrinsic_feature_cols) < 5:
    print("Too few intrinsic features for intrinsic sensitivity trajectory.")
    source_node_intrinsic = None
else:
    X_intrinsic = X_pool[intrinsic_feature_cols].copy()
    X_intrinsic_sample_centered = sample_center_feature_matrix(
        X_intrinsic,
        pooled_niche_feature_df["sample_id"],
    )
    X_intrinsic_scaled_df, intrinsic_feature_block_df = block_balance_feature_matrix(X_intrinsic_sample_centered)
    X_intrinsic_scaled = X_intrinsic_scaled_df.to_numpy()
    X_intrinsic_pca = PCA(
        n_components=min(12, X_intrinsic_scaled.shape[1]),
        random_state=RANDOM_STATE,
    ).fit_transform(X_intrinsic_scaled)
    X_intrinsic_pca_use = X_intrinsic_pca[:, : min(6, X_intrinsic_pca.shape[1])]

    pg_tree_intrinsic = elpigraph.computeElasticPrincipalTree(
        X=X_intrinsic_pca_use,
        NumNodes=num_nodes,
        Lambda=0.005,
        Mu=0.01,
    )[0]
    root_anchor_intrinsic = X_intrinsic_pca_use[root_mask_sample_centered.to_numpy()].mean(axis=0)
    source_node_intrinsic = int(
        np.argmin(np.sum((pg_tree_intrinsic["NodePositions"] - root_anchor_intrinsic) ** 2, axis=1))
    )
    elpigraph.utils.getPseudotime(
        X=X_intrinsic_pca_use,
        PG=pg_tree_intrinsic,
        source=source_node_intrinsic,
        target=None,
    )
    result_df["xenium_pseudotime_intrinsic_sample_centered"] = pg_tree_intrinsic["pseudotime"]
    result_df["xenium_pseudotime_intrinsic_sample_centered_norm"] = minmax(
        result_df["xenium_pseudotime_intrinsic_sample_centered"]
    )
    result_df["xenium_node_id_intrinsic_sample_centered"] = pg_tree_intrinsic["projection"]["node_id"]
    result_df["xenium_edge_id_intrinsic_sample_centered"] = pg_tree_intrinsic["projection"]["edge_id"]

    print("Intrinsic feature blocks:")
    display(intrinsic_feature_block_df["feature_block"].value_counts().rename("n_features").reset_index())
    print("Intrinsic root node:", source_node_intrinsic)
    display(
        result_df.groupby(["sample_id", "disease_group"])[
            ["xenium_pseudotime_sample_centered", "xenium_pseudotime_intrinsic_sample_centered"]
        ]
        .median()
        .round(3)
    )
'''
    ),
    md(
        """
## Automatic Branch Annotation

Branch IDs are assigned from the simplified tree structure. The table below does not hard-code biology; it summarizes each branch by sample composition, pseudotime, and histology proxy enrichment, then proposes a suggested biological identity for review.
"""
    ),
    code(
        r'''
BRANCH_BIOLOGY_SCORE_COLS = [
    col
    for col in [
        "histology__normal_duct_like_score",
        "histology__adm_panin_like_score",
        "histology__glandular_architecture_score",
        "histology__ductal_continuity_cancerization_score",
        "histology__epithelial_stromal_interface_disruption_score",
        "histology__desmoplastic_tumor_score",
        "histology__immune_inflamed_score",
        "histology__immune_exclusion_score",
        "histology__duodenum_invasion_context_score",
        "histology__gland_poor_undifferentiated_score",
        "xenium_epithelial_identity_score",
        "xenium_panin_like_remodeling_score",
        "xenium_proliferation_score",
        "xenium_desmoplastic_context_score",
        "xenium_immune_context_score",
        "xenium_checkpoint_context_score",
        "xenium_nuclear_dapi_texture_score",
        "xenium_duct_lumen_topology_score",
        "xenium_duct_continuity_cancerization_score",
        "xenium_epithelial_stromal_interface_disruption_score",
    ]
    if col in result_df.columns
]

SAMPLE_HISTOLOGY_CONTEXT = {
    "normal_nondiseased_v1": "normal pancreas reference",
    "pdac_addon_v1": "Grade I-II adenocarcinoma, 50% tumor; earlier PDAC with preserved lobular architecture, interlobular fibrosis, glandular tumor/PanIN regions",
    "pdac_pancreas_v1": "Stage III adenocarcinoma; mixed residual normal pancreas, ADM/PanIN, and desmoplastic glandular tumor regions",
    "pdac_io_v1": "Stage IIB Grade 3 PDAC; less fibrotic, gland-poor/undifferentiated-looking, duodenum-invasive, immune-exclusive tumor",
}

def branch_score_enrichment(df, score_cols):
    out = {}
    for col in score_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        out[col] = vals.median()
    return out

def summarize_auto_branches(df, branch_col="major_branch"):
    rows = []
    score_median_all = {
        col: pd.to_numeric(df[col], errors="coerce").median()
        for col in BRANCH_BIOLOGY_SCORE_COLS
    }
    score_sd_all = {
        col: pd.to_numeric(df[col], errors="coerce").std(ddof=0)
        for col in BRANCH_BIOLOGY_SCORE_COLS
    }

    for branch, sub in df.groupby(branch_col, observed=True):
        sample_fracs = sub["sample_id"].value_counts(normalize=True)
        disease_fracs = sub["disease_group"].value_counts(normalize=True)
        row = {
            "branch": branch,
            "n_niches": len(sub),
            "fraction_of_all_niches": len(sub) / len(df),
            "median_original_pseudotime": pd.to_numeric(sub["xenium_pseudotime"], errors="coerce").median(),
            "median_sample_centered_pseudotime": pd.to_numeric(sub["xenium_pseudotime_sample_centered"], errors="coerce").median(),
            "dominant_sample": sample_fracs.index[0],
            "dominant_sample_fraction": sample_fracs.iloc[0],
            "dominant_sample_histology_context": SAMPLE_HISTOLOGY_CONTEXT.get(sample_fracs.index[0], ""),
            "dominant_disease_group": disease_fracs.index[0],
            "dominant_disease_fraction": disease_fracs.iloc[0],
        }
        for sample_id, frac in sample_fracs.items():
            row[f"sample_frac__{sample_id}"] = frac
        for col in BRANCH_BIOLOGY_SCORE_COLS:
            median_val = pd.to_numeric(sub[col], errors="coerce").median()
            sd = score_sd_all[col]
            row[f"{col}__median"] = median_val
            row[f"{col}__z_enrichment"] = (
                (median_val - score_median_all[col]) / sd
                if np.isfinite(sd) and not np.isclose(sd, 0)
                else np.nan
            )
        rows.append(row)

    branch_summary = pd.DataFrame(rows)
    branch_summary["branch"] = pd.Categorical(branch_summary["branch"], categories=branch_order, ordered=True)
    return branch_summary.sort_values(["branch"]).reset_index(drop=True)

def row_score(row, name):
    return row.get(f"{name}__z_enrichment", np.nan)

def suggest_branch_biology(row):
    sample_io = row.get("sample_frac__pdac_io_v1", 0.0)
    sample_pancreas = row.get("sample_frac__pdac_pancreas_v1", 0.0)
    sample_addon = row.get("sample_frac__pdac_addon_v1", 0.0)
    sample_normal = row.get("sample_frac__normal_nondiseased_v1", 0.0)

    # Histology-informed sample priors are only used when a branch is strongly
    # dominated by a reviewed sample and the proxy scores do not contradict the
    # review. The tree structure and branch assignment remain automatic.
    if sample_io >= 0.65:
        immune_inflamed = row_score(row, "histology__immune_inflamed_score")
        desmoplastic = row_score(row, "histology__desmoplastic_tumor_score")
        duodenum_context = row_score(row, "histology__duodenum_invasion_context_score")
        if (
            (pd.isna(immune_inflamed) or immune_inflamed < 0.35)
            and (pd.isna(desmoplastic) or desmoplastic < 0.45)
        ):
            label = "gland-poor / undifferentiated immune-excluded tumor"
            if pd.notna(duodenum_context) and duodenum_context >= -0.10:
                label += " with duodenum-invasion context"
            return label

    if sample_addon >= 0.55:
        adm_panin = row_score(row, "histology__adm_panin_like_score")
        glandular = row_score(row, "histology__glandular_architecture_score")
        if (
            (pd.notna(adm_panin) and adm_panin > 0.20)
            or (pd.notna(glandular) and glandular > 0.25)
        ):
            return "early disease / PanIN-glandular lobule-preserved branch"

    if sample_pancreas >= 0.55:
        desmoplastic = row_score(row, "histology__desmoplastic_tumor_score")
        glandular = row_score(row, "histology__glandular_architecture_score")
        if pd.notna(desmoplastic) and desmoplastic > 0.35:
            if pd.notna(glandular) and glandular > 0.10:
                return "desmoplastic glandular tumor branch"
            return "desmoplastic tumor / stromal-remodeled"

    candidates = {
        "normal / residual normal duct-like": row_score(row, "histology__normal_duct_like_score") + sample_normal,
        "ADM / PanIN-like remodeling": row_score(row, "histology__adm_panin_like_score") + 0.3 * (sample_addon + sample_pancreas),
        "glandular architecture / differentiated tumor": row_score(row, "histology__glandular_architecture_score") + 0.2 * sample_pancreas,
        "ductal continuity / cancerization-like spread": row_score(row, "histology__ductal_continuity_cancerization_score"),
        "interface-disrupted budding / stromal-contact state": row_score(row, "histology__epithelial_stromal_interface_disruption_score"),
        "desmoplastic tumor / stromal-remodeled": row_score(row, "histology__desmoplastic_tumor_score") + 0.3 * sample_pancreas,
        "immune-inflamed epithelial niche": row_score(row, "histology__immune_inflamed_score"),
        "immune-excluded gland-poor / undifferentiated-like": (
            row_score(row, "histology__gland_poor_undifferentiated_score")
            + row_score(row, "histology__immune_exclusion_score")
            + 0.4 * sample_io
        ),
        "duodenum-invasion-associated context": row_score(row, "histology__duodenum_invasion_context_score") + 0.4 * sample_io,
    }
    candidates = {k: v for k, v in candidates.items() if pd.notna(v)}
    if len(candidates) == 0:
        return "mixed / unclear"
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    if ranked[0][1] < 0.35:
        return "mixed / unclear"
    if len(ranked) > 1 and ranked[1][1] > ranked[0][1] - 0.25:
        return f"{ranked[0][0]} + {ranked[1][0]}"
    return ranked[0][0]

branch_biology_summary_df = summarize_auto_branches(result_df)
branch_biology_summary_df["suggested_biology"] = branch_biology_summary_df.apply(suggest_branch_biology, axis=1)

top_feature_rows = []
for _, row in branch_biology_summary_df.iterrows():
    enrichments = []
    for col in BRANCH_BIOLOGY_SCORE_COLS:
        enrichments.append((col, row.get(f"{col}__z_enrichment", np.nan)))
    enrichments = [(col, val) for col, val in enrichments if pd.notna(val)]
    top = sorted(enrichments, key=lambda item: abs(item[1]), reverse=True)[:5]
    top_feature_rows.append("; ".join(f"{col}:{val:+.2f}" for col, val in top))
branch_biology_summary_df["top_enriched_scores"] = top_feature_rows

display_cols = [
    "branch",
    "suggested_biology",
    "n_niches",
    "dominant_sample",
    "dominant_sample_histology_context",
    "dominant_sample_fraction",
    "median_sample_centered_pseudotime",
    "top_enriched_scores",
]
display(branch_biology_summary_df[display_cols])
'''
    ),
    code(
        r'''
branch_heatmap_cols = [f"{col}__z_enrichment" for col in BRANCH_BIOLOGY_SCORE_COLS]
branch_heatmap_df = branch_biology_summary_df.set_index("branch")[branch_heatmap_cols].copy()
branch_heatmap_df.columns = [
    col.replace("__z_enrichment", "").replace("histology__", "").replace("xenium_", "").replace("_score", "")
    for col in branch_heatmap_df.columns
]

fig, axes = plt.subplots(1, 2, figsize=(14, max(3.5, 0.45 * len(branch_heatmap_df))))
sns.heatmap(
    branch_heatmap_df,
    cmap="RdBu_r",
    center=0,
    linewidths=0.2,
    linecolor="white",
    cbar_kws={"label": "Branch median z-enrichment"},
    ax=axes[0],
)
axes[0].set_title("Automatic branch histology enrichment")
axes[0].set_xlabel("")
axes[0].set_ylabel("")

branch_sample_comp_df = (
    result_df.groupby(["major_branch", "sample_id"], observed=True)
    .size()
    .rename("n")
    .reset_index()
)
branch_sample_comp_df["fraction"] = (
    branch_sample_comp_df["n"]
    / branch_sample_comp_df.groupby("major_branch", observed=True)["n"].transform("sum")
)
sns.barplot(
    data=branch_sample_comp_df,
    x="major_branch",
    y="fraction",
    hue="sample_id",
    palette="tab10",
    ax=axes[1],
)
axes[1].set_title("Sample composition by automatic branch")
axes[1].tick_params(axis="x", rotation=35)
axes[1].grid(False)
axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()
'''
    ),
    code(
        r'''
def plot_sample_spatial_pseudotime(sample_id, max_background=80000):
    adata_path = NICHE_DIR / f"{sample_id}_with_niches.h5ad"
    if not adata_path.exists():
        print(f"Missing niche AnnData for {sample_id}: {adata_path}")
        return
    adata = sc.read_h5ad(adata_path, backed="r")
    obs = adata.obs[["x_centroid", "y_centroid", NICHE_KEY, "Tier_A"]].copy()
    adata.file.close()

    lookup_cols = [NICHE_KEY, "xenium_pseudotime", "major_branch"]
    has_sample_centered_pt = "xenium_pseudotime_sample_centered" in result_df.columns
    if has_sample_centered_pt:
        lookup_cols.append("xenium_pseudotime_sample_centered")
    lookup = result_df[result_df["sample_id"] == sample_id][lookup_cols].drop_duplicates()
    obs = obs.merge(lookup, on=NICHE_KEY, how="left")
    rng = np.random.default_rng(42)
    bg = obs
    if len(bg) > max_background:
        bg = bg.iloc[rng.choice(len(bg), size=max_background, replace=False)]
    niche = obs[obs["xenium_pseudotime"].notna()].copy()

    n_panels = 3 if has_sample_centered_pt else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    axes = np.asarray(axes).reshape(-1)
    axes[0].scatter(bg["x_centroid"], bg["y_centroid"], s=0.2, color="#d9d9d9", alpha=0.25, linewidth=0)
    sc0 = axes[0].scatter(
        niche["x_centroid"],
        niche["y_centroid"],
        c=niche["xenium_pseudotime"],
        s=0.8,
        cmap="viridis",
        linewidth=0,
        alpha=0.85,
    )
    axes[0].set_title(f"{sample_id}: epithelial niche pseudotime")
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(False)
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    panel_idx = 1
    if has_sample_centered_pt:
        axes[panel_idx].scatter(bg["x_centroid"], bg["y_centroid"], s=0.2, color="#d9d9d9", alpha=0.25, linewidth=0)
        sc1 = axes[panel_idx].scatter(
            niche["x_centroid"],
            niche["y_centroid"],
            c=niche["xenium_pseudotime_sample_centered"],
            s=0.8,
            cmap="viridis",
            linewidth=0,
            alpha=0.85,
        )
        axes[panel_idx].set_title(f"{sample_id}: sample-centered pseudotime")
        axes[panel_idx].invert_yaxis()
        axes[panel_idx].set_aspect("equal", adjustable="box")
        axes[panel_idx].grid(False)
        plt.colorbar(sc1, ax=axes[panel_idx], fraction=0.046, pad=0.04)
        panel_idx += 1

    axes[panel_idx].scatter(bg["x_centroid"], bg["y_centroid"], s=0.2, color="#d9d9d9", alpha=0.25, linewidth=0)
    sns.scatterplot(
        data=niche,
        x="x_centroid",
        y="y_centroid",
        hue="major_branch",
        palette=branch_palette,
        s=1.0,
        linewidth=0,
        alpha=0.85,
        ax=axes[panel_idx],
        legend=False,
    )
    axes[panel_idx].set_title(f"{sample_id}: major branch")
    axes[panel_idx].invert_yaxis()
    axes[panel_idx].set_aspect("equal", adjustable="box")
    axes[panel_idx].grid(False)

    plt.tight_layout()
    plt.show()

for cfg in SAMPLE_CONFIGS:
    plot_sample_spatial_pseudotime(cfg["sample_id"])
'''
    ),
    code(
        r'''
save_df(result_df, RESULT_DIR / "xenium_pseudotime_result_df.pkl")
pd.Series(selected_cols, name="feature").to_csv(RESULT_DIR / "xenium_pseudotime_selected_features.csv", index=False)
save_df(feature_block_df, RESULT_DIR / "xenium_pseudotime_feature_blocks.csv")
save_df(histology_proxy_availability_df, RESULT_DIR / "xenium_histology_proxy_availability.csv")
save_df(sample_effect_qc_df, RESULT_DIR / "xenium_sample_effect_qc.csv")
save_df(sample_discriminating_feature_df, RESULT_DIR / "xenium_sample_discriminating_features.csv")
save_df(branch_biology_summary_df, RESULT_DIR / "xenium_branch_biology_summary.csv")
with open(RESULT_DIR / "xenium_branch_definitions.json", "w") as f:
    json.dump(branch_defs, f, indent=2)
with open(RESULT_DIR / "xenium_tree_settings.json", "w") as f:
    json.dump(
        {
            "n_niches": int(n_niches),
            "detailed_num_nodes": int(num_nodes),
            "simplified_num_nodes": int(num_nodes_simple),
            "detailed_root_node": int(source_node),
            "simplified_root_node": int(source_node_simple),
            "sample_centered_root_node": int(source_node_sample_centered),
            "sample_centered_root_niches": int(root_mask_sample_centered.sum()),
            "intrinsic_sample_centered_root_node": (
                int(source_node_intrinsic) if source_node_intrinsic is not None else None
            ),
            "trajectory_scaling": TRAJECTORY_SCALING,
            "detailed_tree_min_nodes": int(DETAILED_TREE_MIN_NODES),
            "detailed_tree_max_nodes": int(DETAILED_TREE_MAX_NODES),
            "detailed_tree_node_scale": float(DETAILED_TREE_NODE_SCALE),
        },
        f,
        indent=2,
    )

result_df.head()
'''
    ),
]


handoff_text = """# Xenium Pancreas Pseudotime Handoff

Generated notebooks:

- `notebooks/07_xenium_00_data_audit_and_spatialdata.ipynb`
- `notebooks/07_xenium_01_cell_annotation.ipynb`
- `notebooks/07_xenium_02_epithelial_niche_features.ipynb`
- `notebooks/07_xenium_03_pooled_pseudotime.ipynb`

Raw data root:

- `/Volumes/Shihong_5/for_spatioev/pancreas_Xenium_example_data_from_10X`

Workflow outputs:

- `/Users/shihongwu/SpatioEv/data/xenium_pancreas_10x`

Current implementation choices:

- Uses the four user-provided Xenium pancreas `outs` folders only.
- Treats `Xenium_V1_human_Pancreas_FFPE_outs`, `Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs`, and `Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs` as PDAC.
- Treats `Xenium_V1_hPancreas_nondiseased_section_outs` as normal pancreas.
- SpatialData is installed in `spatioev_env`; notebook 00 includes a SpatialData conversion cell.
- The Scanpy/H5/CSV fallback remains because it is faster for tabular annotation/modeling.
- Installed SpatialData stack tested in `spatioev_env`: `spatialdata==0.5.0`, `spatialdata-io==0.2.0`, `spatialdata-plot==0.2.14`, `pyarrow==24.0.0`, `zarr==2.18.7`, `anndata==0.10.9`, `numpy==1.26.4`.
- `spatialdata_io.xenium(...)` was smoke-tested on `Xenium_V1_hPancreas_nondiseased_section_outs` with lightweight `cell_circles` and returned a valid SpatialData object.
- `pip check` still reports metadata conflicts because scimap pins older `dask`/`zarr`, while SpatialData needs newer versions. Imports for `scimap`, `spatioev`, `scanpy`, `elpigraph`, `spatialdata`, and `spatialdata_io.xenium` were tested successfully.
- Annotates cells independently per sample with all retained Xenium genes/probes. The preferred annotation unit is the 10x precomputed gene-expression graph cluster from `analysis.tar.gz`; Scanpy Leiden is still computed and shown as an independent QC/fallback clustering.
- The annotation notebook stores 10x graphclust, 10x k-means-10, and any available 10x `cell_groups.csv` labels in `adata.obs` for review.
- Cluster labels are assigned from graphclust-level marker-program scores, unsupervised top-marker rules, marker dotplots, spatial QC, and a curatable cluster-review CSV.
- The annotation cache is versioned as `cluster_full_panel_v9_xenium_graphclust_io_mucosa_submucosa_k24`, so older labels are not silently reused.
- Includes a curated `pdac_io_v1` graphclust correction from Xenium Explorer review: graphclust 2 is treated as mucosa gland and graphclust 17 as submucosa, preventing those cells from entering pancreatic ductal niches.
- Broad cluster-level duodenum calls are conservative. When a sample has `CDX2/REG4/DMBT1/TMPRSS2`, the annotation notebook additionally runs epithelial-only unsupervised PCA + MiniBatchKMeans refinement to split true duodenum-like cells from intestinal-like/PanIN-like ductal epithelium.
- Uses stable Tier_A/Tier_B palettes across annotation composition, UMAP, and spatial QC plots.
- Builds ductal/tumor epithelial connected-component niches from 10x `cell_boundaries.parquet` boundary proximity first, with a centroid-radius fallback.
- Adds cell/nucleus boundary shape summaries, including circularity, solidity, elongation, Feret diameter, and boundary irregularity.
- Adds a runnable Xenium DAPI pixel-feature extraction path using `nucleus_boundaries.parquet` plus user-verified focus morphology images. The notebook defaults to a pilot run before full epithelial extraction.
- Summarizes epithelial niche graph morphology, state markers, surrounding cell context, duct/lumen topology, ductal-continuity/cancerization proxies, and epithelial-stromal interface disruption proxies.
- Fits a pooled ElPiGraph pseudotime trajectory from niche-level Xenium features. The detailed tree node count is adaptive (`min(100, max(40, ceil(sqrt(n_niches) * 2.5)))`), while the simplified tree used for major branch labels defaults to 24 nodes.
- Adds histology proxy scores for normal-duct-like, ADM/PanIN-like, glandular architecture, ductal-continuity/cancerization-like spread, epithelial-stromal interface disruption, desmoplastic tumor, immune-inflamed, immune-excluded, duodenum-invasion context, and gland-poor/undifferentiated-like states.
- Uses block-balanced feature scaling so broad feature families (histology modules, epithelial state, architecture/topology, nuclear morphology, microenvironment) contribute more evenly to the pooled trajectory.
- Adds an intrinsic epithelial sensitivity trajectory using within-sample centered epithelial, architecture, and nuclear morphology features only.
- Adds automatic branch annotation summaries: branch structure is inferred from the simplified tree, then each branch receives sample composition, histology-score enrichment, and a suggested biological identity for manual review.

Important caveat:

- The public 10x datasets use different panels. The pooled trajectory must prioritize shared or sufficiently available features and should be interpreted as a cross-sample spatial/transcriptional continuum, not literal patient time.
- DAPI intensity/texture features are optional but implemented. Run the pilot first, then set `RUN_DAPI_FULL_EPITHELIAL=True` before rebuilding niches if the pilot QC looks good.

If continuing in a new Codex chat, say:

> Continue from `/Users/shihongwu/SpatioEv/docs/xenium_pseudotime_handoff.md`. Please inspect and help run/refine the Xenium notebooks.
"""


def main():
    write_notebook(
        NOTEBOOK_DIR / "07_xenium_00_data_audit_and_spatialdata.ipynb",
        audit_cells,
    )
    write_notebook(
        NOTEBOOK_DIR / "07_xenium_01_cell_annotation.ipynb",
        annotation_cells,
    )
    write_notebook(
        NOTEBOOK_DIR / "07_xenium_02_epithelial_niche_features.ipynb",
        niche_cells,
    )
    write_notebook(
        NOTEBOOK_DIR / "07_xenium_03_pooled_pseudotime.ipynb",
        pseudotime_cells,
    )

    DOCS_DIR.mkdir(exist_ok=True)
    handoff_path = DOCS_DIR / "xenium_pseudotime_handoff.md"
    handoff_path.write_text(handoff_text, encoding="utf-8")
    print(f"Wrote {handoff_path}")


if __name__ == "__main__":
    main()
