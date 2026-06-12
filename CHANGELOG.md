# Changelog

All notable changes to SpatioEv will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Proper submodule structure: `spatioev.tl.stats`, `spatioev.tl.density`,
  `spatioev.tl.niche`, `spatioev.tl.ecm`, `spatioev.tl.pseudotime`,
  `spatioev.tl.phenotype`, `spatioev.tl.ml`.
- Proper `pp` submodules: `spatioev.pp.qc`, `spatioev.pp.normalize`,
  `spatioev.pp.pixel`, `spatioev.pp.spatial_prep`.
- Proper `pl` submodules: `spatioev.pl.qc`, `spatioev.pl.spatial`.
- Proper `xe` submodules: `spatioev.xe.annotation`, `spatioev.xe.features`.
- NumPy-style docstrings across all public modules.
- `CONTRIBUTING.md`, `CITATION.cff`, and `CHANGELOG.md`.
- Supplementary tables moved from root to `data/supplementary/`.

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
