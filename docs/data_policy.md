# Data Policy

This working directory contains approximately 129 GB of local analysis data,
including multiplexed imaging H5AD files, OME-TIFF images, segmentation masks,
Xenium SpatialData Zarr stores, pickled intermediate objects, and rendered
figures. These files are useful for local method development but are not
appropriate for direct GitHub upload.

## What Is Ignored

The root `.gitignore` excludes:

- `data/`
- `background/`
- `results/`
- `notebooks/results/`
- generated manuscript binaries and figures in `manuscript/figures/`,
  `manuscript/*.docx`, and `manuscript/*.pptx`
- `*.h5ad`, `*.zarr/`, `*.ome.tif`, `*.tif`, `*.pkl`, and other large outputs
- generated Python, Jupyter, operating-system, and editor artifacts

## Recommended GitHub Strategy

Keep the package repository lightweight and reproducible:

1. Commit package source, tests, tutorials, docs, scripts, `pyproject.toml`, and
   the license.
2. Do not commit raw imaging data or generated analysis outputs.
3. Publish small demo data separately only if redistribution is permitted.
4. Add download instructions, checksums, or accession links for public datasets.
5. Use Git LFS only for small curated binary examples that must live beside the
   code.

## Local Example Data

The test suite can use the existing local example file:

```text
data/exp_2/34434_1_adata.h5ad
```

That test is skipped automatically when the file is absent, so GitHub Actions
and clean checkouts do not require private or bulky data.
