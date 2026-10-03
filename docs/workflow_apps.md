# Interactive image-analysis workflows

SpatioEv packages linked interfaces for taking CellSAM quantification
tables through broad cell annotation, marker gating, prior-knowledge subset
phenotyping, ECM fiber segmentation, tissue regions, co-localisation and
cohort comparison. Each stage writes ordinary CSV, JSON, PNG, and H5AD artifacts, so
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

If SCIMAP or Napari fails with `No module named 'pkg_resources'`, the
environment has setuptools 82 or newer while `mpl-scatter-density` still uses
that removed compatibility module. Repair an existing environment and refresh
the editable installation with:

```bash
python -m pip install "setuptools>=68,<82"
python -m pip install -e ".[apps]"
```

The project root is only a starting location for the file selectors. Every
input and output path can be changed in its workflow page. The same default can
also be supplied with the `SPATIOEV_PROJECT_ROOT` environment variable.

## Workflow order

1. **Prepare AnnData** converts the two CellSAM expression tables into one H5AD.
   OME-TIFF channels define the marker matrix and its order by default;
   supplying the optional **Marker order CSV** overrides that, so the CSV's
   row order becomes the order of `var_names`. Use it when the image has
   lost its channel names, or when the stored names disagree with the panel
   that was actually run. Unmatched table columns are preserved as
   observation metadata regardless of their CSV position, and are now named
   in a warning rather than dropped from the matrix silently.
   An editable schema review supports marker, cell-ID, coordinate,
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

The spatial-analysis stages start from the phenotyped H5AD and the original
OME-TIFF. Every stage writes CSV tables, PNG plots and a manifest JSON under
`results/<sample>_<stage>/`, which the next stage and the cohort comparison
pick up by default.

5. **Fiber segmentation** segments ECM fibers in one or more matrix channels
   (e.g. COL1, FN) with the ark-analysis method (Angelo Lab, MIT licence):
   blur, CLAHE, Frangi ridges, a distance-map watershed and small-object
   removal. A crop preview shows every step for parameter tuning. Images up to
   the processing tile size (4096 px by default) are segmented exactly as ark
   does; whole slides are processed in overlapping tiles with the intensity
   scale, Frangi `gamma` and Otsu thresholds computed once over the full image.
   Outputs: a label TIFF per image and channel, one fiber table (centroid,
   size, orientation in degrees from +x, alignment score), and whole-image and
   tile summaries. Alignment uses axial angle differences by default; ark's
   original score, which treats nearly horizontal fibers at +89 and -89 degrees
   as misaligned, is available as an option. Frangi finds ridges, so filled
   matrix (collagen cut across, dense patches) is missed; **Also count bright
   matrix** adds pixels no fiber covers that are in the channel's brightest
   multi-Otsu class or, with a local window (65 µm suits the 0.325 µm TMA
   cores), brighter than their surroundings and above a floor. These objects
   are marked `object_type = bright_matrix` in the fiber table and magenta in
   the QC images; they count towards matrix area but not towards fiber counts,
   shape or alignment. The same stage measures TWOMBLI-style
   architecture on the fiber skeleton: total length, endpoints, branchpoints,
   curvature, fractal dimension, lacunarity, alignment coherency and % high-
   density matrix, per image and per summary tile. Values follow TWOMBLI's
   definitions but use SpatioEv's segmentation, so they compare between
   SpatioEv samples rather than with Fiji output. Fractal dimension depends on
   the size of the area measured; compare it between images or regions of
   similar size.
6. **Tissue regions** groups tumour cells into nests, outlines each nest,
   expands it by the envelope width (µm) and labels every cell `tumour`,
   `envelope` or `stroma` by location. It writes the phenotype composition of
   each region and, given a fiber manifest, matrix area fraction, fiber density,
   alignment and the TWOMBLI-style architecture inside each region, measured
   within tissue only.
7. **Co-localisation** analyses each image separately. Cell–cell: cross-Ripley
   `L(r) - r` with a 95% label-permutation envelope for each source → target
   pair, plus per-region neighbour ratios. Cell–matrix: distance from each cell
   to the nearest matrix pixel and the matrix fraction around it, a
   label-permutation test per phenotype, and the association between dense or
   aligned matrix tiles and each phenotype.
8. **Cohort comparison** reads a sample sheet (`sample_id`, `group`, optional
   `imageid` for TMA cores, `patient_id`, `project_root`), collects every
   per-sample summary, averages per patient and tests each feature between
   groups (Mann–Whitney U or Kruskal–Wallis, Benjamini–Hochberg q-values). It
   reports the smallest p-value the group sizes allow, so a 3 vs 3 design is
   flagged as unable to reach p < 0.05.

## Classifying cells in QuPath instead

For large cohorts (e.g. hundreds of TMA cores) cell types can be assigned in
QuPath instead of stages 01-03, using SpatioEv's segmentation and
quantification. Cells move between the two by `cell_id` (`<core>_<label>`),
never by position: one import into QuPath and one export back per core. The
**QuPath bridge** page runs the SpatioEv side, or from a terminal:

```bash
spatioev qupath prepare /path/to/TMA_folder
spatioev qupath scripts ~/QuPath/v0.7/scripts
spatioev qupath collect /path/to/TMA_folder
```

1. `prepare` writes `<core>/qupath/<core>_cells.geojson` for every core folder
   (one holding `segmentation/` and `quantification/`): each cell's exact
   pixel-edge outline, its matched nucleus (from `matched_nuclear_label`),
   `cell_id` as the object name, a stable object ID derived from it, and the
   marker measurements (bare marker name = whole-cell mean, `<marker> nucleus`
   = nuclear mean).
2. In QuPath, add each core's `background/<core>.ome.tiff` to a project (the
   image the fiber segmentation reads, so coordinates line up) and run
   `spatioev_import_cells.groovy` with *Automate > Run for project*. Images
   that already hold the cells are skipped, so a second run cannot wipe
   classifications.
3. Classify. Derived classes such as `Fibroblast: WNT5A+` are kept.
   Annotations whose class starts with `Exclude`, `Ignore` or `Artefact` remove
   the cells inside them; other classified annotations are recorded per cell.
4. Run `spatioev_export_classes.groovy` for the project, then `collect`, which
   writes `<core>/qupath/<core>_phenotyped.h5ad` and a class-count table.
   Re-run only these two after changing the classifier.

Stages 04-07 then take the QuPath-phenotyped AnnData directly: set the project
root to the TMA folder and the sample ID to the core. Use the `qupath_class`
column to keep derived classes separate, or `phenotype` for base classes.
QuPath reports centroids from pixel corners, 0.5 px right of and below
SpatioEv's pixel-centre centroids; joining by `cell_id` makes this irrelevant.

## TMA and multi-FOV datasets

Choose **Multi-FOV / TMA** on the Prepare AnnData page when one specimen has
multiple `ark_wdir*` batches and one OME-TIFF per FOV. The importer:

- discovers complete ARK cell-table pairs across batches;
- lets users assign the CSV used for `adata.X` and the CSV used for the named
  layer, applying those filenames to every discovered ARK batch;
- takes the channel order from the marker order CSV, whose *row* order is
  authoritative -- a `channel_number` column that disagrees with the row
  order is rejected, and named OME channels that disagree are overridden
  with a warning;
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
python -m spatioev.workflows.fiber_segmentation --help
python -m spatioev.workflows.tissue_regions --help
python -m spatioev.workflows.colocalization --help
python -m spatioev.workflows.cohort_comparison --help
```
