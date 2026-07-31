# Notebook Compatibility Audit

This audit validates notebook structure and checks that notebook references to
the public `spatioev` API still resolve. It does not execute heavyweight
image, Xenium, or trajectory workflows.

- Notebooks scanned: 4
- Compatibility checks passed: 4/4
- Python syntax errors: 0
- Missing `spatioev` public API references: 0

| Notebook | Code Cells | SpatioEv Refs | Missing Optional Imports | Status |
| --- | ---: | ---: | --- | --- |
| `notebooks/00_dev_seg_qc_testing.ipynb` | 7 | 5 | - | pass |
| `notebooks/01_dev_clustering_based_phenotyping_test.ipynb` | 6 | 5 | - | pass |
| `notebooks/01_dev_scimap_phenotype_workflow.ipynb` | 6 | 5 | - | pass |
| `notebooks/02_dev_SVM_phenotype_probility.ipynb` | 6 | 5 | - | pass |
