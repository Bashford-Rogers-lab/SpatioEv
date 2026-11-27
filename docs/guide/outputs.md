# Output files explained

Everything SpatioEv writes is an ordinary file you can open: tables are
**CSV** (open in Excel), pictures are **PNG**, and the run settings are
**JSON** (plain text). The `.h5ad` and `.tiff` files are for SpatioEv itself
(and Python, QuPath or Fiji).

For a core called `CORE` in project root `TMA_072`:

```text
TMA_072/
├── CORE/qupath/                         cell types (step 5)
└── results/
    ├── CORE_fiber_segmentation/         matrix fibers (step 6)
    ├── CORE_tissue_regions/             tissue regions (step 7)
    ├── CORE_colocalization/             co-localisation (step 8)
    ├── cohort_comparison/               group comparison (step 9)
    └── batch_<stage>_summary.csv        the last "Run for every core" of each stage
```

Every results folder holds a `…_manifest.json` (the list of files written)
and a `logs/` folder with the run's bookkeeping:

- `logs/CORE_<stage>_config.json`: the exact settings used;
- `logs/CORE_<stage>_status.json` and `logs/CORE_<stage>.log`: progress and
  messages, useful when asking for help.

A new run of the same stage replaces them.

Quality-control pictures are in a `qc/` sub-folder. Intermediate tables that
you rarely need are in `details/` or `checkpoints/`.

Distances are in micrometres (`_um`), areas in square micrometres (`_um2`)
or square millimetres (`_mm2`). Columns ending in `_px` are in pixels.

## Cell types

`CORE/qupath/`

| File | What it is |
|---|---|
| `CORE_cells.geojson` | The cells handed to QuPath (outlines and measurements). |
| `CORE_cell_classes.csv` | The class QuPath gave every cell, keyed by `cell_id` (`CORE_<label>`). |
| `CORE_annotations.geojson` | The annotations you drew in QuPath. |
| `CORE_phenotyped.h5ad` | Cells with marker levels, positions and class, used by steps 7–8. Its `qupath_class` column holds the full class (`Fibroblast: WNT5A+`); `phenotype` holds the base class (`Fibroblast`). |

## Cell types by clustering and gating (steps 00–03)

For a multi-FOV TMA with sample ID `ST_TMA_01` ([step 5b](clustering_gating.md)):

```text
ST_TMA_01/
├── data/
│   ├── ST_TMA_01_adata.h5ad                     step 00: all cells, all markers
│   ├── ST_TMA_01_adata_qc/                      step 00 checks
│   └── ST_TMA_01_marker_condition_for_gating.csv   staining answers (optional)
└── results/
    ├── ST_TMA_01_clustering_workflow/           step 01
    ├── ST_TMA_01_marker_gating_qc/              step 02
    └── ST_TMA_01_scimap_phenotyping_interface/  step 03
```

| File | What it is |
|---|---|
| `data/ST_TMA_01_adata.h5ad` | Every cell. `adata.X` holds the arcsinh values with columns in `markers_bs.csv` order (= image-plane order); the layer `size_normalized` holds the size-normalised values. |
| `data/ST_TMA_01_adata_qc/marker_channel_order.csv` | Which marker was taken for each image plane. Check this first if markers look swapped. |
| `data/ST_TMA_01_adata_qc/cells_per_fov.csv` | Cells per FOV. |
| `…_clustering_workflow/ST_TMA_01_phenotyped.h5ad` | All cells with the broad label in `annotation`. Steps 02 and 03 read it. |
| `…_clustering_workflow/ST_TMA_01_level0_annotation_mapping.csv` | The name you gave each cluster. |
| `…_clustering_workflow/qc/` | Cluster heatmap, UMAP and cluster summary. `checkpoints/` holds the clustered file before naming. |
| `…_marker_gating_qc/ST_TMA_01_scimap_manual_gates.csv` | The gates (`markers,gates`, log1p of the gate layer), as used by step 03. |
| `…_marker_gating_qc/ST_TMA_01_selected_marker_gates.csv` | The same gates with the method, positive fraction and review flags. |
| `…_marker_gating_qc/ST_TMA_01_gating_manifest.json` | Includes `gate_layer`, the expression values the gates belong to. |
| `…_marker_gating_qc/details/` | All candidate gates, and `*_previous.csv`: the gate tables before the last calculation or napari export. |
| `…_scimap_phenotyping_interface/ST_TMA_01_broad_plus_<run>_phenotyped.h5ad` | All cells, with the broad labels and this run's cell types combined in the *Combined final column*. Use the last run's file downstream. |
| `…_scimap_phenotyping_interface/ST_TMA_01_<run>_scimap_phenotype_counts.csv` | Number and fraction of cells per cell type in that run. |
| `…_scimap_phenotyping_interface/qc/` | Marker heatmap, spatial maps, image overlays and `*_gate_rescale_agreement.csv` per run. |

## Matrix fibers

`results/CORE_fiber_segmentation/`

| File | What it is |
|---|---|
| `CORE_fiber_image_stats.csv` | **One row per matrix channel**: the whole-core summary. |
| `CORE_matrix_architecture.csv` | One row per channel: TWOMBLI-style architecture of the whole core. |
| `CORE_fiber_tile_stats.csv` | The same summaries per 150 µm tile (with `tile_y`, `tile_x` in pixels). |
| `CORE_fiber_table.csv` | One row per fiber or bright-matrix piece: position, size, orientation, alignment. |
| `labels/CORE__<channel>_fiber_labels.tiff` | The segmentation itself, one image per channel. |
| `qc/CORE__<channel>_fiber_overlay.png` | The quality-control pictures. |

Main columns of the image and tile summaries:

| Column | Meaning |
|---|---|
| `area_fraction_pct` | Matrix area as % of the image (or tile). Includes bright matrix. For % of *tissue*, use the tissue-regions output. |
| `n_fibers`, `fibers_per_mm2` | Number and density of ridge fibers (bright-matrix pieces are counted separately in `n_bright_matrix`). |
| `avg_major_axis_length`, `avg_minor_axis_length` | Mean fiber length and width, in pixels. |
| `avg_alignment_score` | How differently each fiber points from its neighbours. **Lower = more aligned.** |
| `orientation_coherence` | 0 = fibers point every way, 1 = all parallel. |

Architecture columns (`CORE_matrix_architecture.csv`):

| Column | Meaning |
|---|---|
| `length_density_mm_per_mm2` | Total fiber length per tissue area. |
| `branchpoints_per_mm2`, `endpoints_per_mm2` | How branched, and how fragmented, the network is. |
| `mean_branch_length_um` | Average length between branch points. |
| `curvature_per_um` | How curved the fibers are. |
| `fiber_thickness_um` | Average fiber thickness. |
| `alignment_coherency` | Alignment measured from the image texture, 0 to 1. |
| `fractal_dimension` | How completely the matrix fills space, between 1 and 2. Compare only areas of similar size. |
| `lacunarity_gliding` | How unevenly matrix and gaps are spread (higher = more clumped). |
| `hdm_pct` | % of pixels that are high-density matrix (TWOMBLI's HDM). |

## Tissue regions

`results/CORE_tissue_regions/`

| File | What it is |
|---|---|
| `CORE_region_composition.csv` | Per region (`tissue_region`: all / tumour / envelope / stroma) and cell type: `count`, `region_total` and `proportion`. |
| `CORE_fiber_region_stats.csv` | Matrix and architecture per channel and region (columns as above). |
| `CORE_region_areas.csv` | Area of each region in mm². |
| `CORE_tissue_regions.png`, `CORE_region_composition.png` | The region map and composition chart. |
| `CORE_region_boundaries.geojson` | Nest and envelope outlines (GeoJSON, in pixel coordinates). |
| `CORE_tissue_regions.h5ad` | The cells with their region, used by step 8. |

## Co-localisation

`results/CORE_colocalization/`

| File | What it is |
|---|---|
| `CORE_cell_cell_curves.csv` / `.png` | For each question and radius: `L_minus_r_um`, the chance band (`envelope_low_um`, `envelope_high_um`), and whether the curve is `above_envelope` or `below_envelope`. |
| `CORE_cell_cell_local.csv` | Per region: `mean_neighbour_ratio` (1 = as expected by chance) and `fraction_source_with_excess`. |
| `CORE_cell_matrix_summary.csv` | Per channel, region and cell type: `median_distance_to_matrix_um`, `mean_matrix_fraction`, `fraction_in_contact`. |
| `CORE_cell_matrix_enrichment.csv` | Does a cell type sit in more matrix than cells in general? `z_score`, `p_value`. |
| `CORE_matrix_tile_association.csv` | Cell types in dense or aligned matrix tiles: `spearman_rho`, `enrichment_ratio`. |
| `CORE_cell_matrix_per_cell.csv` | Every cell's distance to matrix and local matrix fraction. |

## Compare groups

`results/cohort_comparison/` (file names start with the comparison name)

| File | What it is |
|---|---|
| `<name>_group_comparison.csv` | **The main result**: one row per measurement with medians per group, `p_value` and `q_value`. Sorted with the most significant first. |
| `<name>_unit_values.csv` | One row per patient (unit), one column per measurement. Use it to make your own plots. |
| `<name>_features_long.csv` | Every measurement for every core before averaging per patient. |
| `<name>_top_features.png` | The plot of the most different measurements. |
| `<name>_inputs_found.csv` | Which result files were found for each sample. Check it if a core seems to be missing. |

Measurement names (`feature`) are built from their parts, separated by `|`.
For example, `composition | stroma | Fibroblast: WNT5A+` is the share of
WNT5A+ fibroblasts among stroma cells, and
`matrix | COL1 | envelope | area_fraction_pct` is the COL1 area fraction in
the envelope.
