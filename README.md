# SpatioEv

SpatioEv is a Python toolbox for reusable spatial analysis over multiplexed
imaging data, with extensions for spatial transcriptomics workflows such as
Xenium. It focuses on per-cell spatial features, neighborhood organization,
cell-cell and cell-ECM interactions, niche graph summaries, and trajectory-ready
feature engineering.

The repository currently contains the package source, tests, cleaned tutorials,
a generated function catalog, workflow scripts, and manuscript source drafts.
Large local analysis data and generated manuscript binaries are intentionally
ignored for GitHub upload; see [Data Policy](docs/data_policy.md).

## Main Capabilities

- Segmentation QC from cell and nuclear morphology summaries.
- Marker and morphology feature construction for phenotype models.
- Tile, radius, kNN, KDE, and phenotype-specific density summaries.
- Ripley and Moran families for point-pattern and feature autocorrelation.
- Source-centered phenotype interaction and pseudotime dynamics.
- Spatial niche boundaries, cell graphs, and niche graph feature tables.
- Pseudotime-ready feature matrices, block balancing, branch annotation, and
  pseudotime trend tables.
- Cell-ECM neighborhood summaries and collagen/fiber interaction modules.
- Xenium-compatible annotation, niche, DAPI, and pseudotime support.

The WGCNA-like workflow from earlier manuscript drafts is not part of the
current public package API yet.

## Installation

Create a fresh environment and install the package in editable mode:

```bash
conda create -n spatioev python=3.11
conda activate spatioev
pip install -e .
```

Optional extras:

```bash
pip install -e ".[scanpy]"       # clustering and Scanpy plotting
pip install -e ".[viewer]"       # scimap/Napari interactive viewers
pip install -e ".[spatialdata]"  # SpatialData and Squidpy workflows
pip install -e ".[trajectory]"   # UMAP and ElPiGraph trajectory notebooks
pip install -e ".[dev]"          # tests and developer tools
```

The repository `environment.yml` is a pinned export of a working environment
that includes scimap/Napari, SpatialData, and `dask==2024.11.2`. If you install
optional extras manually without pins, keep the `viewer` and `spatialdata`
workflows in separate environments because their latest upstream dependency
ranges may not resolve together:

```bash
conda env create -f environment-viewer.yml
conda env create -f environment-spatialdata.yml
```

Quick import check:

```python
import spatioev

print(spatioev.__version__)
```

SpatioEv uses lazy imports at the top level, so `import spatioev` does not load
Scanpy, scimap, Napari, or other heavy optional packages until the relevant
function is called.

## Public API Style

SpatioEv now provides Scimap-inspired public namespaces:

```python
import spatioev as sv

sv.pp  # preprocessing and QC
sv.tl  # analysis tools: statistics, niches, pseudotime, ECM
sv.pl  # plotting
sv.hl  # reusable helper functions
sv.xe  # Xenium/spatial-transcriptomics helpers
sv.io  # input/output
```

For example, `sv.tl.morans_i`, `sv.tl.cross_ripleys_k_by_phenotype`,
`sv.tl.cell_to_fiber_distance`, and `sv.hl.tree_edges` are all available as
stable public entry points.

Historical implementation modules are kept under `spatioev.archive` for source
organization. New user-facing code should use the public namespaces above.

## Repository Layout

```text
spatioev/       Python package source
tests/          Reproducible smoke and example-data tests
tutorials/      Clean tutorial notebooks
docs/           Data policy, testing notes, and release checklist
mkdocs.yml      Documentation site navigation
scripts/        Analysis utilities used to generate or integrate workflows
notebooks/      Historical/development notebooks kept as local provenance
```

The local `data/`, `background/`, `results/`, and `notebooks/results/`
directories, plus generated manuscript figures and office-document binaries,
are not intended for GitHub upload because they contain large raw and derived
analysis files.

## Testing

On this workstation, disable third-party pytest plugin autoload because a
Napari plugin import can fail before SpatioEv tests begin:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

The test suite includes:

- deterministic toy-data tests for the core package API;
- a local smoke test using `data/exp_2/34434_1_adata.h5ad` when present;
- graceful skipping of local-data tests in lightweight GitHub checkouts.

Current local verification: `14 passed`.

## Documentation Site

The MkDocs-style documentation scaffold is in [docs](docs), with navigation in
[mkdocs.yml](mkdocs.yml). Build it locally with:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The website-style tutorial landing page is
[Mastering Spatial Evolution Analysis with SpatioEv](docs/tutorials/md/spatial_evolution_spatioev.md).

## Tutorials

The cleaned tutorial series is in [tutorials](tutorials):

1. [Data Model and Function Catalog](tutorials/00_data_model_and_function_catalog.ipynb)
2. [QC, Preprocessing, and Pixel Features](tutorials/01_qc_preprocessing_pixel_features.ipynb)
3. [Phenotyping, SVM, and Annotation Refinement](tutorials/02_phenotyping_svm_annotation_refinement.ipynb)
4. [Density, Interaction, and Spatial Statistics](tutorials/03_density_interaction_spatial_statistics.ipynb)
5. [Niche Boundaries and Cell Graphs](tutorials/04_niche_boundaries_cell_graphs.ipynb)
6. [ECM-Cell Interactions](tutorials/05_ecm_cell_interactions.ipynb)
7. [Pseudotime, Xenium, and Manuscript Figures](tutorials/06_pseudotime_xenium_manuscript_figures.ipynb)

Each notebook uses the existing local example dataset when available and falls
back to a small synthetic AnnData object, so the tutorials remain runnable after
GitHub upload.

The full generated API guide is in
[docs/function_catalog.md](docs/function_catalog.md), with a CSV version at
[docs/function_catalog.csv](docs/function_catalog.csv).

## Manuscript and Figures

The current manuscript source artifacts are in [manuscript](manuscript). The
generated figure binaries and Word/PowerPoint exports are kept locally and
ignored by Git because they are large reproducible outputs.

- `SpatioEv_publication_manuscript.md`
- figure source tables under `manuscript/analysis_tables/`
- local generated figures under `manuscript/figures/`

Regenerate figures and manuscript text with:

```bash
python scripts/generate_manuscript_figures.py
/Users/shihongwu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/write_publication_manuscript.py
```

## Minimal Example

```python
import anndata as ad
import numpy as np
import pandas as pd
import spatioev as sv

from spatioev.config import QCConfig

adata = ad.AnnData(
    X=np.random.normal(size=(6, 2)),
    obs=pd.DataFrame(
        {
            "label": range(6),
            "area": [80, 120, 140, 200, 95, 110],
            "nc_ratio": [0.3, 0.4, 0.6, 0.7, 0.2, 0.5],
            "X_centroid": [0, 10, 20, 30, 40, 50],
            "Y_centroid": [0, 5, 10, 10, 5, 0],
            "imageid": ["img1"] * 6,
            "phenotype": ["duct", "duct", "immune", "immune", "stromal", "stromal"],
        }
    ),
)

adata = sv.pp.run_segmentation_qc(adata, QCConfig(pixel_size=0.325))
tiles = sv.tl.assign_tiles(adata, tile_size=20)
density = sv.tl.compute_general_density(tiles, tile_size=20)
spatial_i = sv.tl.morans_i(
    adata.obs[["X_centroid", "Y_centroid"]],
    adata.obs["area"],
)
```

## Citation

If you use SpatioEv before a formal manuscript citation is available, cite the
GitHub repository and include the commit hash used for analysis.
