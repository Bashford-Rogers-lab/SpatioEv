# Changelog

All notable changes to SpatioEv will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Step-by-step guide** (`docs/guide/`, a new section of the documentation
  site) for users with no coding experience: installing on a Mac, organising
  data, opening the app, cell typing in QuPath, every analysis stage, output
  files and troubleshooting, with screenshots of the synthetic demo data.
  `scripts/guide_screenshots.py` regenerates the app screenshots. Images open
  full size on click (`mkdocs-glightbox`, added to the `docs` extra).
- **Run for every core** (`spatioev.workflows.batch`, `spatioev batch`, and a
  panel under each finished run on pages 04–06): repeats one core's fiber
  segmentation, tissue regions or co-localisation run for every core folder
  with identical settings, changing only the core name in the paths. Cores
  with a result are skipped, so an interrupted batch resumes; cores missing
  an input (for example no QuPath classes yet) are reported, not failed; a run
  whose paths lack the core name is refused, since every core would write to
  the same place.
- **Synthetic demo data** (`spatioev.io.make_demo_tma`, `spatioev demo`):
  multiplexed-IF TMA cores in the real folder layout (17-channel OME-TIFF,
  whole-cell and nuclear masks, cell tables measured from them, a sample
  sheet), with built-in differences between two groups. `--with-classes`
  also writes QuPath-style classes so the QuPath step can be skipped.
- The cohort page's sample sheet starts with one row per core folder found.
- **Fiber segmentation** (`spatioev.pp.fibers`, interface stage 04), adapted
  from ark-analysis (MIT; licence shipped as `spatioev/pp/LICENSE.ark-analysis`).
  `segment_fibers` reproduces ark's label image exactly on images that fit in
  one tile; `segment_fibers_tiled` handles whole slides with the intensity
  scale, Frangi `gamma` and multi-Otsu thresholds computed once over the full
  image (on a TMA core: area fraction identical to whole-image, fiber-pixel
  IoU 0.97). `calculate_fiber_alignment` uses a KD-tree instead of ark's
  all-pairs distance matrix and axial angle differences by default
  (`axial=False` restores ark's score, which rates nearly horizontal fibers as
  misaligned). Fiber tables use SpatioEv conventions (`X_centroid` = column,
  `orientation` in degrees from +x) and are indexed by `fiber_id`, so they
  feed `spatioev.tl.ecm` directly.
- **Bright matrix** (`include_bright`, optional local window and floor): adds
  filled matrix that Frangi's ridge filter misses (cross-cut collagen, dense
  patches) as extra objects marked `object_type = bright_matrix`. They count
  towards matrix area; fiber counts, shape averages, alignment and orientation
  coherence use the ridge fibers only. The brightness threshold ignores a few
  saturated spots (it is recomputed on the clipped histogram when its top
  class holds under 0.1% of pixels). QC images show whole-image and
  automatically picked crops, ridge fibers in teal and bright matrix in
  magenta.
- **Fiber summaries** (`spatioev.tl.ecm.fiber_stats`): ark's density, tile and
  per-image statistics for any image size, plus `area_fraction_pct`,
  orientation coherence, `fiber_region_stats` (matrix inside each tissue
  region), `cell_matrix_proximity` (distance to the nearest matrix pixel and
  local matrix fraction), `cell_matrix_enrichment` (label-permutation test) and
  `tile_cell_matrix_association` (phenotypes in dense / aligned matrix tiles).
- **Tissue regions** (`spatioev.tl.niche.regions`, stage 05):
  `define_tissue_regions` labels cells tumour / envelope / stroma from tumour
  nests; `summarize_region_composition` covers every region including stroma;
  `rasterize_tissue_regions`, and GeoJSON read/write for nest polygons. The
  boundary grid defaults to half the median cell spacing, so nests are solid
  in pixel as well as micrometre coordinates.
- **Co-localisation** (stage 06): per-image cross-Ripley curves with a
  label-permutation envelope, per-region neighbour ratios, and the cell–matrix
  measures above.
- **Cohort comparison** (`spatioev.tl.compare`, stage 07): per-patient
  aggregation, Mann–Whitney U / Kruskal–Wallis with Benjamini–Hochberg q-values,
  and the minimum achievable p-value for the group sizes.
- Plots: `sv.pl.plot_fiber_segmentation_steps`, `plot_fiber_overlay`,
  `plot_tissue_regions`.
- **QuPath bridge** (`spatioev.io.qupath`, `spatioev qupath ...`, interface
  page 03): classify cells in QuPath on SpatioEv's segmentation with one
  import and one export per core, keyed on `cell_id`. `prepare` writes QuPath
  cell GeoJSON with exact pixel-edge outlines, the matched nucleus, `cell_id`
  names, deterministic object IDs and marker measurements;
  `spatioev_import_cells.groovy` / `spatioev_export_classes.groovy` (shipped in
  `spatioev/resources/qupath`) run per image or for a whole project; `collect`
  builds a phenotyped AnnData per core, applying Exclude annotations. Replaces
  hand-edited scripts that paired each cell with the nucleus of the same label
  number (wrong for 99.7 % of cells, as the masks are numbered independently),
  matched measurements by rounded centroid (off by a systematic 0.5 px) and
  clamped labels to 16 bits. Tested end to end against QuPath 0.7.
- **TWOMBLI-style matrix architecture** (`spatioev.tl.ecm.architecture`):
  total fiber length, endpoints, branchpoints, mean branch length, curvature,
  box-counting fractal dimension, lacunarity, alignment (structure-tensor
  coherency) and % high-density matrix, following the definitions in TWOMBLI
  v1, AnaMorf and IAClassLibrary (re-implemented; no TWOMBLI code used). Measured
  from one skeleton of each image and summed per tissue region or tile, so
  region edges do not create fiber ends. Stage 04 reports them per image and
  tile, stage 05 per region, stage 06 relates them to phenotypes per tile, and
  stage 07 tests them between groups. Differences from TWOMBLI, documented in
  the module: the mask is SpatioEv's segmentation rather than Ridge Detection;
  curvature uses every fiber path (AnaMorf's whole-image mode uses only the
  longest); AnaMorf's lacunarity is shown to equal `|1/p - 2|` for coverage
  `p` and is reported beside a gliding-box lacunarity; the % HDM intensity
  range uses robust percentiles instead of the raw minimum and maximum.
- Optional **Marker order CSV** on the single-image half of *Prepare AnnData*
  (`--marker-manifest`, `ConversionPlan.marker_manifest`). When supplied it
  defines the marker order instead of the OME channel names, so `var_names`
  follows the CSV's row order. This makes single images with unnamed channels
  convertible at all: previously they failed with "OME channels without an
  expression column: ['C0', 'C1', ...]", because no table column can match a
  generic plane name.

### Changed
- Interface clean-ups found while writing the guide: the QuPath bridge's
  prepare table no longer shows each file's full path (it pushed the cell
  counts off screen); fiber segmentation leaves *Cell AnnData* empty instead
  of suggesting a file that does not exist; the fiber preview's last panel
  counts ridge fibers and bright-matrix pieces separately, like the summary.
- `FRIEND_INSTALL_INSTRUCTIONS.md` now points to the guide's install page.
- **Python 3.10 is no longer supported**; the floor is now 3.11. The `ui`,
  `apps` and `dev` extras pin `zarr>=3.1` because `tifffile.aszarr()` requires
  zarr 3, and no zarr release from 3.1 onward supports 3.10 (3.2+ already
  require 3.12). Since the pin landed on 12 August the 3.10 matrix entry could
  not resolve its dependencies, so `requires-python = ">=3.10"` and the 3.10
  classifier were advertising support that could not be installed. The CI
  matrix, the ruff target and the now-dead `tomli` backport dependency follow.
- The marker order CSV's **row order** is now authoritative in both conversion
  paths. `read_marker_manifest` no longer re-sorts by `channel_number`.
  `channel_number` is interpreted by what it contains: *unique* values are a
  global plane index and must ascend with the rows, so a shuffled index is
  rejected with an error naming the offending rows instead of silently
  reordering the markers away from the order the file shows. *Repeated* values
  are cycle-local — a CODEX/PhenoCycler panel restarts the count at 1 every
  imaging round — so they make no claim about global order and are kept as
  metadata while the row order stands.
- A named OME image whose channel order disagrees with the marker order CSV is
  now overridden with a warning instead of aborting the TMA conversion. A
  differing number of planes is still an error — that is a different panel.
- `read_marker_manifest` is shared by both paths (defined in
  `workflows.cellsam`, re-exported from `workflows.cellsam_tma`), so the two
  cannot drift apart on what a marker order CSV means.
- The TMA path now matches marker names to cell-table columns on the same
  normalised key the single-image path has always used, via the shared
  `resolve_marker_columns`. Requiring an exact string there meant a marker CSV
  saying `NAKATPASE` failed against a `NaKATPase` column on one path and
  succeeded on the other, for the same pair of files. `var_names` keep the CSV
  spelling, `var["source_column"]` records the column each marker came from,
  and a name matched only after normalisation is reported as a warning. A
  marker whose key hits more than one column is an error rather than a guess.

### Fixed
- `dask[array]` was missing from the `dev` extra, but
  `workflows.marker_gating_review` imports it at module level. `pip install -e
  ".[dev]" && pytest` — what CI runs — therefore died collecting
  `test_gate_range.py`, and a collection error aborts the whole session, so the
  entire test job failed rather than one file. Present since the gate-slider
  work of 19 August, which also meant the Streamlit AppTest suite had not
  actually run in CI since then.
- Both converters now write the H5AD to a temporary file and move it into
  place. Writing straight to the destination left a truncated file there when
  a run was interrupted; HDF5 keeps its superblock, so the file still looked
  openable and only failed later in the clustering page with h5py's `KeyError:
  'Unable to synchronously open object (unable to determine object type)'`. An
  interrupted run now leaves the previous good file, or nothing.
- The clustering page checks the H5AD is readable before loading it and says
  what is wrong -- truncated, empty, or not HDF5 at all (an undownloaded cloud
  placeholder) -- instead of surfacing an h5py traceback.
- Expression columns with no matching image channel were moved to `.obs` with
  no error and no warning, silently yielding fewer markers than the table
  offered. They are now named in a warning.
- `uns["all_markers"]` follows the marker order CSV when one is supplied, so it
  no longer disagrees with `var_names` on the single-image path.

---

## [0.2.0] — 2026-08-02

Repository reorganisation, dependency reduction and performance work. No
scientific results change: every optimisation in this release was verified to
reproduce the previous numbers exactly.

### Added
- Ground-truth numerical tests for `tl.stats`, `tl.ecm` and `tl.niche`, pinning
  the statistics against analytic values (Ripley `K_expected == pi*r^2`,
  Moran's I sign on gradients and checkerboards, `E[I] = -1/(n-1)`, permutation
  nulls centred on expectation, exact convex-hull and Minkowski areas).
- `spatioev._core` — shared internals: `knn_weights`, `get_coords`,
  `per_image`, `per_image_table`, `require_obs_columns`, `require`.
- `spatioev._vendor.scimap` — vendored `rescale` and `phenotype_cells`
  (MIT, Laboratory of Systems Pharmacology @ Harvard).
- `examples/` — three notebooks that run end to end on synthetic data and are
  executed in CI.
- CI now runs ruff, a 3.10/3.11/3.12 x ubuntu/macos matrix, coverage, and a
  packaging job that installs the built wheel outside the source tree.

### Changed
- `tl.stats`, `tl.density`, `tl.niche`, `tl.ecm` and `tl.pseudotime` are now
  packages rather than single large modules. Import paths are unchanged.
- Density plotting moved from `tl.density` to `pl`.
- `apps` extra composes `spatioev[scanpy,ui]`; `ui` no longer pins
  `zarr==2.10.3` or an old dask. New `gating` extra owns the scimap
  dependency; `viewer` is retained as an alias.
- Version is single-sourced from `spatioev.__version__`.

### Removed
- `spatioev.archive` (47 modules). It contained no implementations — 38 were
  three-line star-import shims forwarding into the live modules, and the rest
  duplicated the public API surface. Everything it exposed remains available
  from `tl`/`pp`/`pl`/`xe`/`hl`.
- Manuscript analysis material moved out of the package tree to `paper/`,
  which is excluded from both wheel and sdist.

### Fixed
- `tl.ecm.cross_ripleys_k` was unreachable through the public namespace: the
  `ecm_cross_ripleys_k` alias pointed at `cross_ripleys_k_permutation_envelope`,
  and `sv.tl.cross_ripleys_k` resolved to the `tl.stats` function instead.
- `compute_convex_hull_area` existed twice with divergent NaN behaviour; the
  canonical implementation now drops non-finite rows rather than raising
  inside Qhull.
- `tests/workflows/test_apps.py` imported `tomllib`, which is stdlib only from
  Python 3.11, so the suite could not run on 3.10 despite
  `requires-python = ">=3.10"`.
- `tifffile` and `zarr` were required by four workflow modules but declared
  only in the `ui` extra, so a single uncollectable module aborted the entire
  test session under `pip install -e ".[dev]"`.
- The vendored MIT licence now ships in the wheel and sdist.
- 78 undefined-name errors: five modules annotated with `ad.AnnData` without
  importing anndata under any guard.

### Performance
No result changes; all verified against the previous implementation.
- Moran's I permutation tests build the spatial weight matrix once and
  evaluate simulations in blocks: **~30-40x** faster
  (10,000 cells / 199 sims: 0.313s -> 0.011s). Identical to 8.3e-17.
- ECM Moran statistics are sparse throughout. The cross-Moran helper
  previously built two dense n x n matrices (`W.toarray()` and
  `np.outer(x, y)`), ~1.6 GB of temporaries per call at 10,000 fibres, inside
  a 999-iteration loop: **~28x** faster and O(n*k) instead of O(n^2).
- `assign_cells_to_niche_regions` uses vectorised Shapely 2 predicates
  instead of a per-cell Python loop: **~35-49x** faster
  (40,000 cells / 5 niches: 4.049s -> 0.083s).

---

## [0.1.0] — 2025-05-01

Initial alpha release accompanying the SpatioEv manuscript submission.

### Added
- **Segmentation QC** — area and nucleus/cytoplasm ratio filtering
  (`spatioev.pp.run_segmentation_qc`).
- **Normalization** — per-marker z-score normalization and obs feature
  construction (`spatioev.pp.zscore_normalize`, `add_obs_from_var`).
- **Pixel features** — per-cell Haralick texture, entropy, lacunarity,
  polarity, moment of inertia, and channel correlation extraction from
  multiplexed image stacks (`spatioev.pp.extract_cell_pixel_features`).
- **Spatial preprocessing** — coordinate validation, convex hull estimation,
  tissue area computation, edge cell detection.
- **Phenotyping** — Leiden/Louvain clustering, SVM-based phenotype
  classification with marker and morphology features.
- **Density analysis** — tile, KDE, kNN, radius, and phenotype-specific
  density summaries; source–target interaction density.
- **Ripley statistics** — global and cross-phenotype Ripley K/L curves with
  Monte Carlo envelope testing.
- **Moran statistics** — global and local Moran's I; cross-feature Moran's I
  and local cross-Moran quadrant classification.
- **Niche analysis** — DBSCAN/HDBSCAN niche detection, Shapely boundary
  construction, cell proximity graphs, and niche graph feature tables.
- **ECM–cell interactions** — fiber linking, ECM spatial statistics, bipartite
  ECM graphs, invasion scoring, and ECM neighborhood clustering.
- **Spatial pseudotime** — niche-level feature matrix preparation, block
  balancing, ElPiGraph branch annotation, pseudotime dynamics, and trend
  analysis.
- **Xenium support** — marker-set scoring, histology module scoring, and DAPI
  nuclear feature extraction for 10x Xenium data.
- Lazy-import top-level namespace (`spatioev.pp`, `spatioev.tl`, etc.)
  compatible with lightweight environments.
- MkDocs documentation scaffold with function catalog and tutorial notebooks.
- GitHub Actions CI workflow running the full test suite.

[Unreleased]: https://github.com/Bashford-Rogers-lab/SpatioEv/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bashford-Rogers-lab/SpatioEv/releases/tag/v0.1.0
