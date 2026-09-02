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
- A staged interface for CellSAM-to-AnnData conversion, broad clustering,
  marker autogating, and SCIMAP subset phenotyping, including multi-FOV TMA
  projects split across multiple ARK working directories.

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
pip install -e ".[apps]"         # all four interactive analysis workflows
pip install -e ".[spatialdata]"  # SpatialData and Squidpy workflows
pip install -e ".[trajectory]"   # UMAP and ElPiGraph trajectory notebooks
pip install -e ".[dev]"          # tests and developer tools
```

The repository `environment.yml` plus `requirements-spatioev_env.txt` recreate
a pinned working environment that includes scimap/Napari, SpatialData, and
`dask==2024.11.2`. The pip requirements are installed with `--no-deps` so pip
uses the same pinned package set instead of re-solving upstream dependency
metadata:

```bash
conda env create -f environment.yml
conda activate spatioev_env
python -m pip install --no-deps -r requirements-spatioev_env.txt
```

Quick import check:

```python
import spatioev

print(spatioev.__version__)
```

SpatioEv uses lazy imports at the top level, so `import spatioev` does not load
Scanpy, scimap, Napari, or other heavy optional packages until the relevant
function is called.

Launch the complete image-analysis workflow with:

```bash
spatioev ui --project-root /path/to/project
```

The interface opens at `http://localhost:8501` by default. See the
[interactive workflow guide](docs/workflow_apps.md) for stage inputs, outputs,
templates, and reproducibility notes.

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

All user-facing code should use the public namespaces above.

## Repository Layout

```text
spatioev/           Python package source (the installable library)
spatioev/apps/      Four-stage Streamlit interface
spatioev/workflows/ Reusable workflow engines and Napari review tools
tests/              Test suite
docs/               Documentation sources (MkDocs)
scripts/            Standalone data-conversion utilities
tools/              Repository maintenance tooling
examples/           Runnable tutorial notebooks (synthetic data, CI-executed)
paper/              Analysis notebooks and figure scripts for the manuscript
```

`paper/` holds study-specific analysis and is **not** part of the installable
package — it is excluded from both the wheel and the sdist. Only `spatioev/`
is distributed.

The local `data/`, `background/`, `results/`, and `outputs/` directories, plus
generated figures and office-document binaries, are excluded from version
control because they contain large raw and derived analysis files.

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

Tests that depend on large local datasets skip automatically when the data is
absent, so the suite runs in a clean checkout.

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

Three runnable notebooks live in [examples](examples). They execute end to end
on synthetic data — no download required — and are run in CI on every push:

1. [Quick start](examples/01_quickstart.ipynb) — QC, tile density, Moran's I, Ripley's K
2. [Spatial niches](examples/02_spatial_niches.ipynb) — components, boundaries, composition
3. [ECM–cell analysis](examples/03_ecm_cell_analysis.ipynb) — fibre links, orientation, coupling

The narrative tutorial is
[Mastering Spatial Evolution Analysis with SpatioEv](docs/tutorials/md/spatial_evolution_spatioev.md).

The full generated API guide is in
[docs/function_catalog.md](docs/function_catalog.md), with a CSV version at
[docs/function_catalog.csv](docs/function_catalog.csv).

## Manuscript and Figures

Analysis notebooks and figure-generation scripts for the accompanying manuscript
live under [paper](paper):

- `paper/notebooks/` — analysis notebooks
- `paper/figures/` — per-panel figure scripts
- `paper/scripts/` — figure and manuscript generators

These depend on local raw data that is not distributed. Generated figure
binaries and office-document exports are ignored by Git because they are large
reproducible outputs.

```bash
python paper/scripts/generate_manuscript_figures.py
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
