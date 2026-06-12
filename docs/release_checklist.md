# Release Checklist

Use this checklist before uploading or tagging SpatioEv on GitHub.

## Repository Hygiene

- Confirm the repository is initialized with Git.
- Confirm `.gitignore` excludes local data, generated results, caches, and
  operating-system artifacts.
- Confirm `spatioev.egg-info/`, `__pycache__/`, `.DS_Store`, and Finder `Icon`
  files are not committed.
- Confirm generated manuscript binaries/figures and notebook result caches stay
  out of the package repository.
- Confirm no private paths, credentials, or unpublished patient-identifiable
  data are committed.
- Keep the repository root limited to package metadata, docs, source, tests,
  tutorials, and workflow scripts.

## Package Checks

```bash
python -m compileall -q spatioev
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python -m build
```

If `python -m build` is unavailable, install the developer extra first:

```bash
pip install -e ".[dev]"
```

## Documentation Checks

- Regenerate the function catalog and tutorial notebooks:

```bash
python scripts/write_tutorial_notebooks.py
```

- Build or preview the documentation site:

```bash
mkdocs serve
```

- Run the tutorial notebooks against the local example data.
- Run the tutorial notebooks without local data to confirm the synthetic
  fallback works.
- Regenerate manuscript figures and manuscript text:

```bash
python scripts/generate_manuscript_figures.py
/Users/shihongwu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/write_publication_manuscript.py
```

- Update README badges, repository URL, and citation once the GitHub repository
  and manuscript citation are final.

## Data Checks

- Keep raw imaging, H5AD, Zarr, and derived result files outside Git history.
- Publish redistributable example data through an approved archive or release
  asset with checksums.
- Record the exact data version used for each manuscript figure.

## Final Cleanup

```bash
find . -name ".DS_Store" -delete
find . -name "Icon?" -delete
find . -name "__pycache__" -type d -prune -exec rm -rf "{}" "+"
rm -rf .pytest_cache spatioev.egg-info build dist
```

Because this working directory contains local data and generated outputs, create
the GitHub repository from Git-tracked files rather than uploading the whole
folder as a zip archive.
