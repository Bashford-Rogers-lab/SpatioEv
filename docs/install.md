# Installation

Install SpatioEv in editable mode from the repository root:

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
pip install -e ".[docs]"         # MkDocs documentation site
pip install -e ".[dev]"          # tests and developer tools
```

The repository `environments.yml` is a pinned export of a working environment
that includes scimap/Napari, SpatialData, and `dask==2024.11.2`. It uses
`pip --no-deps` so pip installs the same pinned package set instead of
re-solving upstream dependency metadata.

Quick check:

```python
import spatioev as sv

print(sv.__version__)
print(sv.tl.morans_i)
```
