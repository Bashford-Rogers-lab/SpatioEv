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
- **The marker order CSV is found automatically** in both conversion modes:
  `markers_bs.csv`, `marker_bs.csv` or `markers.csv` in a `background/`
  folder or beside the data, searched from the cell-table folder, the images
  and the project folder up to three levels (`find_marker_csv`). A TMA
  requires one; for a single image `--no-auto-marker-csv` falls back to the
  OME channel names. Its row order defines `adata.X`, so
  column *n* is image plane *n*. An optional `remove` column keeps a plane out
  of `adata.X`, and `adata.var["channel_index"]` records each marker's plane.
  The page shows which file was used and a plane-by-plane table before
  building. `marker_order_warning` recognises AnnData files from 0.2.0 and
  earlier, which sorted a cycle-based panel by its cycle-local
  `channel_number` and so paired most markers with the wrong plane; the
  clustering and gating pages ask for those to be rebuilt.
- **Per-image gates for SCIMAP phenotyping** (stage 03): the gate CSV may hold
  `markers,gates` or one column per image ID (`markers,fov1,fov2,...`), on
  `adata.X` or a named **Gate layer** (stage 02 records the layer it gated and
  stage 03 fills it in). Images without a gate stop the run with the missing
  marker/image pairs, take the marker's median gate, or use scimap's GMM
  estimate. Workflow marker names that differ only in a numeric suffix or case
  (`CLDN11_1` for `CLDN11_2`) are matched and reported, and the run can be
  limited to chosen image IDs. Phenotyping runs can be chained: a second
  population is phenotyped on the first run's full-tissue H5AD.
- Stage 03 explains two scimap behaviours that silently changed results:
  `workflow_rule_notes` flags rows where a lone `pos`/`neg` marker is only
  averaged into the score (so cells that break it can still win; use
  `allpos`/`allneg`), and the per-image minimum cell count
  (**Minimum cells per phenotype per image**, now allowed to be 0) reports how
  many cells of which phenotypes it turned into `Unknown`.
- **Synthetic testis TMA** (`spatioev.io.demo_testis.make_demo_testis_tma`,
  `spatioev demo <folder> --kind testis`): six FOVs over three ARK working
  directories, unnamed image planes, a cycle-based `markers_bs.csv`,
  alphabetical cell-table columns with nuclear rows, FOV-to-FOV staining
  differences, phenotype workflows, a partial per-FOV gate table, staining
  answers and the true cell type, compartment and tubule of every cell. An
  end-to-end test runs stages 00–03 on it and checks the result against the
  truth (structures 100 %, cell types ≥ 95 %; 99 % in practice). NaKATPase
  stains tubule cells four times as strongly as interstitial cells, as on
  real cores.
- Guide page **5b. Cell types by clustering and gating**
  (`docs/guide/clustering_gating.md`): stages 00–03, tubule detection and
  composition, and co-localisation on the testis demo, with an output-file
  section and troubleshooting rows.
- The gating page loads saved questionnaire answers: those of the last
  gating run, else `data/<sample>_marker_condition_for_gating.csv`.
- **Z-score within each FOV** (page 01, `scale_within_image` in the clustering
  config, `zscore_normalize(adata, groupby=...)`): scales every marker
  separately in each FOV before clustering. On a 28-core testis TMA, scaling
  over all cells gave large clusters drawn mostly from one or two cores (81 %
  of a 7,700-cell cluster from two FOVs) and called many cells inside tubules
  myoid. Scaled within each FOV, no cluster over 2,000 cells drew more than
  29 % of its cells from one FOV, and the myoid calls lay on the SMA rings.
- **Average markers over neighbouring cells** (page 01; `neighbourhood_um`,
  `pixel_size_um` and `per_cell_markers` in the Level-0 config): each
  clustering marker is replaced by its mean over the cells within a radius in
  the same FOV, so clusters follow tissue structures rather than single cells.
  Markers of one-cell-thick structures (SMA, MYH11 suggested) can stay per
  cell. Off by default. The page asks for the pixel size, since some scanners
  write a wrong one into the OME-TIFF, and warns when cells had almost no
  neighbour within the radius. Refinement and export use each cell's own
  values. 0.1 s for 104,000 cells. On five imaged cores of a 28-core testis
  TMA, it improved the structure map in some cores and not in others, so it
  is an option to compare, not a default.
- **Testis structures marker preset** (page 01): GFRA1, PIWIL4, NaKATPase,
  AMH, SOX9, MAGEA4, MYH11, SMA, CYP17A1 and STAR, matched ignoring `_1`/`_2`
  suffixes, and chosen automatically for a panel holding most of them. They
  were picked from the images of five cores of a 28-core prepubertal testis
  TMA: GFRA1, PIWIL4, NaKATPase and AMH stain only inside tubules in every
  core. DMRT1 (Sertoli and germ cells alike), SYCP3 and Acrosin (no meiotic
  cells before puberty) and broad stains (Vimentin, NF2F2, CD45, CD68) are
  left out; the help text says to add SYCP3 and Acrosin for pubertal or adult
  testes. On the demo, with SYCP3 and Acrosin added, 100 % of cells get the
  right structure and 99.3 % the right type (98.2 % without them).
- **Testis refinements** (page 01): refining `Tubule` starts from SOX9,
  MAGEA4, GFRA1, PIWIL4, AMH and DMRT1, and refining `Interstitium` from SMA,
  MYH11, CYP17A1, STAR, CYP11A1 and CD68, both at resolution 0.5, when the
  panel has at least three of them. On the real TMA, naming Myoid and Leydig
  straight from Level-0 put 43 % of hand-gated myoid cells in Leydig-like
  clusters. Naming Tubule/Interstitium first matched hand gates for 86 % of
  cells, and refining the interstitium put Leydig and myoid cells in separate
  clusters. The annotation drop-downs gain `Interstitium`, `Sertoli`,
  `Spermatogonia`, `Spermatocyte`, `Spermatid`, `Myoid`, `Leydig` and
  `Other interstitium` next to `Tubule`.
- **Remove each cell's overall brightness** (page 01 refinements;
  `remove_brightness` in a refinement's config,
  `clustering.remove_cell_brightness`): subtracts each cell's mean over the
  refinement markers after z-scoring, so clusters follow which markers stand
  out rather than how bright a cell is. On the real TMA one shared factor
  made up 42 % of the variation across the preset's markers. It raised the
  best-case Sertoli/spermatogonia split from 81 % to 89 % (balanced, against
  hand gates). On by default only for a testis `Tubule` refinement; it needs
  z-scored markers.
- Only some FOVs need an image. A TMA's images are about 1 GB each, so often
  only a few are copied locally: Prepare AnnData converts every FOV from its
  cell table and warns which have no image, the FOV pickers on pages 01–03
  list the FOVs with one, and image QC and overlays are skipped (with a
  note) for a FOV without one. No matching image at all is still an error.

### Changed

- Cluster heatmaps (page 01) print every cluster mean and cap the colours of
  z-scored ones at ±3. On a real TMA, three clusters of 15–130 cells with an
  empty channel averaged −7 and stretched the scale until the ±1 differences
  that name the other clusters looked white.
- **Tidier output folders.** Every stage writes its run bookkeeping
  (`*_config.json`, `*_status.json`, the worker log, now `<run>.log`) to a
  `logs/` folder inside its output folder, figures to `qc/`, and
  intermediate tables and checkpoints to `details/` and `checkpoints/`. The
  top of each folder holds only the results the next stage reads. Older runs'
  files are still found where they are. Gate calculation and napari export
  write one canonical set of gate tables and keep the replaced ones as
  `details/*_previous.csv` instead of adding `_web` / `_interactive` copies.
  The conversion schema moved to `logs/<h5ad name>_conversion_config.json`.
- Stage folder names are defined once (`marker_gating.stage_output_dirs`), so
  the home page, the tissue-regions page and the phenotyping page agree on
  where stage 01–03 results live. Co-localisation and tissue regions start
  from the newest phenotyping result when there is no other input.
- *Prepare AnnData* opens in *Multi-FOV / TMA* mode for a project folder with
  `ark_wdir*` folders and a `dearray/` folder.
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
- **Automatic gates no longer call a marker's main population positive.** When
  the 3-component GMM "high" gate would call more than half of a marker's
  cells positive, the upper-tail q75 gate is used instead, and the marker is
  flagged `3GMM_high_is_majority` for review
  (`marker_gating.GMM3_HIGH_MAX_POSITIVE_FRACTION`). On a 28-core testis TMA
  the old choice called 69 % of cells CD68-positive (0.65 % by hand) and 80 %
  GFRA1-positive (13 % by hand). Across the four hand-gated cores, the share of
  cells on which the automatic gates agreed with the hand-set ones rose from
  82 % to 94 %. An upper-tail gate calls at most about a quarter of cells
  positive, so a marker that really is in most cells (AMH and Vimentin in that
  TMA) is now called too rarely: move its gate in napari review. A strategy
  profile that names `3component_gmm_high` still gets that gate.

### Fixed
- The home page looked for the stage 00 AnnData of a TMA project in the
  single-image location, so it always showed *Not started*.
- Spatial QC on a TMA showed every FOV's cells over one FOV's image; it now
  overlays only the chosen FOV. Crop windows on images smaller than the crop
  size no longer get negative offsets.
- Spatial scatter plots size their points by the number of cells per panel,
  so small FOVs are no longer drawn with near-invisible dots.
- Warnings that buried the logs are fixed at the source rather than hidden:
  pandas dtype and chained-assignment warnings in the vendored scimap
  `rescale` and `phenotype_cells` (marked as modified from upstream), and
  scanpy's Leiden-backend `FutureWarning`. Results are unchanged.
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
- Clustering failed on every sample with more than 8,192 cells, i.e. every real
  TMA, with `AttributeError: 'csr_array' object has no attribute 'getnnz'`
  (then, past that, a numba `TypingError` from umap). scanpy 1.11.5 switches to
  pynndescent above that size and cannot read what pynndescent 0.7 returns, nor
  pass its trimmed result to umap-learn 0.6; these are the versions a new
  install gets. `cluster_cells` now runs the same pynndescent search itself and
  hands scanpy the graph in the form it expects. The demo data was too small
  to show it; a regression test now clusters 8,292 cells.

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
