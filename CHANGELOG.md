# Changelog

All notable changes to SpatioEv will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Optional **Marker order CSV** on the single-image half of *Prepare AnnData*
  (`--marker-manifest`, `ConversionPlan.marker_manifest`). When supplied it
  defines the marker order instead of the OME channel names, so `var_names`
  follows the CSV's row order. This makes single images with unnamed channels
  convertible at all: previously they failed with "OME channels without an
  expression column: ['C0', 'C1', ...]", because no table column can match a
  generic plane name.

### Changed
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
