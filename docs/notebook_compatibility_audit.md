# Notebook Compatibility Audit

This audit validates notebook structure and checks that notebook references to
the public `spatioev` API still resolve. It does not execute heavyweight
image, Xenium, or trajectory workflows.

- Notebooks scanned: 26
- Compatibility checks passed: 26/26
- Python syntax errors: 0
- Missing `spatioev` public API references: 0

| Notebook | Code Cells | SpatioEv Refs | Missing Optional Imports | Status |
| --- | ---: | ---: | --- | --- |
| `notebooks/00_dev_seg_qc_testing.ipynb` | 10 | 0 | - | pass |
| `notebooks/01_dev_clustering_based_phenotyping_test.ipynb` | 21 | 0 | - | pass |
| `notebooks/01_dev_scimap_phenotype_workflow.ipynb` | 28 | 0 | scimap | pass |
| `notebooks/02_dev_SVM_phenotype_probility.ipynb` | 25 | 0 | scimap | pass |
| `notebooks/03_dev_general_density.ipynb` | 11 | 0 | scimap | pass |
| `notebooks/03_dev_local_density_KNN.ipynb` | 9 | 0 | scimap | pass |
| `notebooks/03_dev_local_density_radius.ipynb` | 11 | 0 | scimap | pass |
| `notebooks/04_dev_spatial_stats_exp3.ipynb` | 13 | 0 | scimap | pass |
| `notebooks/04_dev_spatial_stats.ipynb` | 46 | 0 | scimap | pass |
| `notebooks/04_global_organization_PDAC_IgG4AIP.ipynb` | 19 | 0 | - | pass |
| `notebooks/05_dev_spatial_niche_boundaries.ipynb` | 12 | 0 | scimap | pass |
| `notebooks/06_dev_graph_pseudotime_v2_combined_exp_2_3_4_5.ipynb` | 30 | 0 | elpigraph | pass |
| `notebooks/06_dev_graph_pseudotime_v2_exp_2.ipynb` | 32 | 11 | elpigraph | pass |
| `notebooks/06_dev_graph_pseudotime_v2_exp_3.ipynb` | 32 | 0 | elpigraph, scimap | pass |
| `notebooks/06_dev_graph_pseudotime_v2_exp_4.ipynb` | 32 | 0 | elpigraph, scimap | pass |
| `notebooks/06_dev_graph_pseudotime_v2_exp_5.ipynb` | 33 | 0 | elpigraph, scimap | pass |
| `notebooks/07_xenium_00_data_audit_and_spatialdata.ipynb` | 5 | 0 | spatialdata_io | pass |
| `notebooks/07_xenium_01_cell_annotation.ipynb` | 11 | 0 | - | pass |
| `notebooks/07_xenium_02_epithelial_niche_features.ipynb` | 6 | 1 | shapely | pass |
| `notebooks/07_xenium_03_pooled_pseudotime.ipynb` | 17 | 0 | elpigraph | pass |
| `notebooks/08_RA_OA_ECM_cell_00_prepare_links.ipynb` | 9 | 0 | - | pass |
| `notebooks/08_RA_OA_ECM_cell_05_chp_density_micro_holes_col6_dark_zone_segmentation.ipynb` | 12 | 0 | - | pass |
| `notebooks/08_trajectory_microenvironment_interactions.ipynb` | 10 | 0 | - | pass |
| `notebooks/09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb` | 17 | 0 | - | pass |
| `notebooks/09_xenium_banksy_pseudotime_integration.ipynb` | 5 | 0 | - | pass |
| `notebooks/10_xenium_spatialcellchat_pseudotime_integration.ipynb` | 5 | 0 | - | pass |

## Environment Notes

The following imports were referenced by one or more notebooks but were not
installed in the current lightweight test environment:

`elpigraph`, `scimap`, `shapely`, `spatialdata_io`.

Install the relevant optional analysis stack before full notebook execution.
