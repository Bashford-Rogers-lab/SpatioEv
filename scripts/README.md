# Scripts

This directory contains reproducible workflow utilities that sit outside the
installable `spatioev` package.

## Tutorial and Documentation

- `write_tutorial_notebooks.py` regenerates the tutorial notebook series and
  the public function catalog in `docs/`.
- `audit_notebook_compatibility.py` validates historical notebooks without
  executing heavyweight workflows and checks that their public `spatioev` API
  references still resolve.

## Manuscript Workflows

- `generate_manuscript_figures.py` regenerates the current publication figure
  panels.
- `write_publication_manuscript.py` regenerates the current manuscript draft.
- Older manuscript/storyboard scripts are retained as provenance for previous
  analysis iterations.

## Xenium Integration

- `extract_xenium_dapi_features.py` extracts Xenium DAPI morphology features.
- `run_xenium_banksy*.py`, `integrate_xenium_banksy_to_pseudotime.py`, and
  `integrate_xenium_spatialcellchat_to_pseudotime.py` reproduce downstream
  spatial-transcriptomics integration analyses.

## Image Tiling

Cluster and local helpers for tiling large OME-TIFF channels live in
`scripts/image_tiling/`.
