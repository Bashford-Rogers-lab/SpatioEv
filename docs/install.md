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
pip install -e ".[apps]"         # complete staged workflow interface
pip install -e ".[spatialdata]"  # SpatialData and Squidpy workflows
pip install -e ".[trajectory]"   # UMAP and ElPiGraph trajectory notebooks
pip install -e ".[docs]"         # MkDocs documentation site
pip install -e ".[dev]"          # tests and developer tools
```

The `apps` extra follows the NumPy, Dask, Zarr, and numcodecs compatibility
ranges required by SCIMAP 2.3.x. Create the environment with Python 3.11 for
the most predictable combination of Streamlit, Scanpy, SCIMAP, and Napari
dependencies.

The repository `environment.yml` plus `requirements-spatioev_env.txt` recreate
a pinned working environment that includes scimap/Napari, SpatialData, and
`dask==2024.11.2`. Create the conda environment first, then install the pinned
pip requirements with `--no-deps` so pip uses the same package set instead of
re-solving upstream dependency metadata:

```bash
conda env create -f environment.yml
conda activate spatioev_env
python -m pip install --no-deps -r requirements-spatioev_env.txt
```

Quick check:

```python
import spatioev as sv

print(sv.__version__)
print(sv.tl.morans_i)
```

Launch the interface after installing the `apps` extra:

```bash
spatioev ui --project-root /path/to/project
```
