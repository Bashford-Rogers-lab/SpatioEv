# Publication Reproducibility

Before releasing a figure or manuscript result:

1. Record the data source and version.
2. Run the relevant tutorial or manuscript-generation script from a clean
   environment.
3. Save tables into an ignored results directory.
4. Commit source code, notebooks, docs, and small reproducibility metadata.
5. Keep raw images, H5AD files, Zarr stores, and generated figures out of Git
   unless they are explicitly approved as public demo assets.

Core checks:

```bash
python -m compileall -q spatioev
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python -m build
```

