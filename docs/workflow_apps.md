# Interactive image-analysis workflows

SpatioEv packages four linked interfaces for taking CellSAM quantification
tables through broad cell annotation, marker gating, and prior-knowledge subset
phenotyping. Each stage writes ordinary CSV, JSON, PNG, and H5AD artifacts, so
results remain inspectable outside the interface.

## Install and launch

Install the application dependencies from a checkout:

```bash
conda create -n spatioev python=3.11
conda activate spatioev
pip install -e ".[apps]"
spatioev ui --project-root /path/to/project
```

The default address is `http://localhost:8501`. Use another port when needed:

```bash
spatioev ui --project-root /path/to/project --port 8510
```

The project root is only a starting location for the file selectors. Every
input and output path can be changed in its workflow page. The same default can
also be supplied with the `SPATIOEV_PROJECT_ROOT` environment variable.

## Workflow order

1. **Prepare AnnData** converts the two CellSAM expression tables into one H5AD.
   It removes the duplicate `cell_size` field, preserves observation metadata,
   orders markers from the OME-TIFF channel metadata, places the second matrix
   in a named layer, and writes `uns["all_markers"]`.
2. **Broad clustering** performs unsupervised clustering, supports spatial
   Napari review, and exports broad tissue-population annotations.
3. **Marker autogating** combines a marker-condition questionnaire with
   distribution diagnostics and a strategy profile. Calculated gates remain
   adjustable in the original-image Napari overlay before accepted gates and a
   gated H5AD are written.
4. **Subset phenotyping** selects any broad population and applies a user-chosen
   SCIMAP phenotype workflow. It writes subset and optional full-tissue H5ADs,
   tables, publication-ready heatmaps, spatial plots, and image overlays.

## Templates

The package includes an HCC Phenocycler gating strategy and immune phenotype
workflow as worked examples. They are useful starting points for matching
panels, not universal biological truth. For a different panel, start from a
blank marker questionnaire or an existing condition CSV, review the inferred
expression distributions, and supply a phenotype workflow whose marker columns
match the H5AD.

## Reproducibility

Long-running operations execute in background Python processes and record a
configuration JSON, status JSON, and log next to their outputs. Napari review
writes reviewed gate tables separately from calculated gates. Keep these files
with the final H5AD to preserve both the automated starting point and manual
decisions.

The workflow engines are importable from `spatioev.workflows` and can also be
launched as modules, for example:

```bash
python -m spatioev.workflows.cellsam --help
python -m spatioev.workflows.marker_gating --help
python -m spatioev.workflows.marker_gating_review --help
```
