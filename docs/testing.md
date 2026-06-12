# Testing

Run the local smoke tests from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

The `PYTEST_DISABLE_PLUGIN_AUTOLOAD` prefix avoids unrelated third-party pytest
plugins in the workstation environment. It is not required in a clean CI image
unless that image also auto-loads GUI or Napari plugins.

## Test Coverage

The current tests cover:

- lightweight top-level imports and lazy optional dependencies;
- Scimap-style public namespaces (`pp`, `tl`, `pl`, `hl`, `xe`);
- lazy exposure of ECM APIs without importing statsmodels at package import
  time;
- segmentation QC;
- marker and morphology feature matrix construction;
- density summaries;
- local radius and kNN density;
- phenotype interaction density;
- Ripley and Moran statistics;
- source-centered interaction summaries;
- pseudotime binning and interaction dynamics;
- a smoke run on the existing local `exp_2` H5AD file when available.

The tests are intentionally small and deterministic. Long-running notebooks and
full-resolution image extraction workflows should be validated separately before
publication or a release tag.

Current local verification:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
# 14 passed
```

## Notebook Compatibility Audit

The historical development notebooks are validated with a lightweight static
audit:

```bash
python scripts/audit_notebook_compatibility.py --report docs/notebook_compatibility_audit.md
```

This check validates notebook JSON, parses Python code cells, and confirms that
public `spatioev` references still resolve after API reorganization. It does not
execute heavyweight image, Xenium, or trajectory workflows. Full execution still
requires the optional workflow-specific stack and the original local datasets.
Install `spatioev[dev]` first if `nbformat` is not already available.

Current notebook audit:

```text
26 notebooks scanned
26/26 compatibility checks passed
0 Python syntax errors
0 missing spatioev public API references
```

The current lightweight environment is missing optional notebook dependencies
used by some workflows: `elpigraph`, `scimap`, `shapely`, and
`spatialdata_io`.
