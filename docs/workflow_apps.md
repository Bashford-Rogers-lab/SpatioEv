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
   OME-TIFF channels define the marker matrix and its order; unmatched table
   columns are preserved as observation metadata regardless of their CSV
   position. An editable schema review supports marker, cell-ID, coordinate,
   FOV/group, metadata, and ignored roles plus manual marker-to-channel
   assignments. The converter removes `cell_size` only when it duplicates an
   area field, normalizes common centroid aliases to `X_centroid` and
   `Y_centroid`, writes `obsm["spatial"]`, places the second matrix in a named
   layer, and writes `uns["all_markers"]`.
2. **Broad clustering** performs unsupervised clustering, supports spatial
   Napari review, and exports broad tissue-population annotations.
3. **Marker autogating** combines a marker-condition questionnaire with
   distribution diagnostics and a strategy profile. Calculated gates remain
   adjustable in the original-image Napari overlay before accepted gates and a
   gated H5AD are written.
4. **Subset phenotyping** selects any broad population and applies a user-chosen
   SCIMAP phenotype workflow. It writes subset and optional full-tissue H5ADs,
   tables, publication-ready heatmaps, spatial plots, and image overlays.

## TMA and multi-FOV datasets

Choose **Multi-FOV / TMA** on the Prepare AnnData page when one specimen has
multiple `ark_wdir*` batches and one OME-TIFF per FOV. The importer:

- discovers complete ARK cell-table pairs across batches;
- lets users assign the CSV used for `adata.X` and the CSV used for the named
  layer, applying those filenames to every discovered ARK batch;
- reads channel order from a marker CSV when dearrayed OME channels are unnamed;
- combines paired `whole_cell` and `nuclear` rows into one cell observation;
- creates unique cell IDs from dataset, FOV, and segmentation label;
- sets `obs["imageid"]` to the FOV value for per-image SCIMAP rescaling;
- stores `obs["dataset_id"]` for the slide-level identity; and
- stores an FOV-to-file table in `uns["image_manifest"]`.

The clustering, marker-gating, and subset-phenotyping pages accept either one
OME-TIFF or a folder of FOV OME-TIFFs. For a TMA, choose the FOV used for napari
or original-image overlay review. Calculations still use the complete AnnData;
only image overlays are restricted to the selected FOV. Multi-FOV spatial QC is
faceted by `imageid` so cores with local coordinate systems are not stacked.

The clustering page also provides a **Clustering scope** control. **All FOVs
jointly** produces one shared embedding and annotation system. **Selected FOV
only** subsets the AnnData before normalization, PCA, neighbor graph, Leiden,
refinement, and export. FOV-specific artifacts include the FOV in every
filename and the exported H5AD contains only that FOV, so joint and per-FOV
analyses can coexist without overwriting one another.

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

The AnnData conversion schema is also written beside the H5AD as
`*.conversion_schema.json`. This records every reviewed column role and manual
marker-to-channel assignment, allowing the same interpretation to be audited
or reused when CSV column order changes.

The workflow engines are importable from `spatioev.workflows` and can also be
launched as modules, for example:

```bash
python -m spatioev.workflows.cellsam --help
python -m spatioev.workflows.cellsam_tma --help
python -m spatioev.workflows.marker_gating --help
python -m spatioev.workflows.marker_gating_review --help
```
