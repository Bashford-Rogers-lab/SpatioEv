# Xenium Pancreas Pseudotime Handoff

Generated notebooks:

- `notebooks/07_xenium_00_data_audit_and_spatialdata.ipynb`
- `notebooks/07_xenium_01_cell_annotation.ipynb`
- `notebooks/07_xenium_02_epithelial_niche_features.ipynb`
- `notebooks/07_xenium_03_pooled_pseudotime.ipynb`

Raw data root:

- `/Volumes/Shihong_5/for_spatioev/pancreas_Xenium_example_data_from_10X`

Workflow outputs:

- `/Users/shihongwu/SpatioEv/data/xenium_pancreas_10x`

Current implementation choices:

- Uses the four user-provided Xenium pancreas `outs` folders only.
- Treats `Xenium_V1_human_Pancreas_FFPE_outs`, `Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE_outs`, and `Xenium_V1_hPancreas_Cancer_Add_on_FFPE_outs` as PDAC.
- Treats `Xenium_V1_hPancreas_nondiseased_section_outs` as normal pancreas.
- SpatialData is installed in `spatioev_env`; notebook 00 includes a SpatialData conversion cell.
- The Scanpy/H5/CSV fallback remains because it is faster for tabular annotation/modeling.
- Installed SpatialData stack tested in `spatioev_env`: `spatialdata==0.5.0`, `spatialdata-io==0.2.0`, `spatialdata-plot==0.2.14`, `pyarrow==24.0.0`, `zarr==2.18.7`, `anndata==0.10.9`, `numpy==1.26.4`.
- `spatialdata_io.xenium(...)` was smoke-tested on `Xenium_V1_hPancreas_nondiseased_section_outs` with lightweight `cell_circles` and returned a valid SpatialData object.
- `pip check` still reports metadata conflicts because scimap pins older `dask`/`zarr`, while SpatialData needs newer versions. Imports for `scimap`, `spatioev`, `scanpy`, `elpigraph`, `spatialdata`, and `spatialdata_io.xenium` were tested successfully.
- Annotates cells independently per sample with all retained Xenium genes/probes. The preferred annotation unit is the 10x precomputed gene-expression graph cluster from `analysis.tar.gz`; Scanpy Leiden is still computed and shown as an independent QC/fallback clustering.
- The annotation notebook stores 10x graphclust, 10x k-means-10, and any available 10x `cell_groups.csv` labels in `adata.obs` for review.
- Cluster labels are assigned from graphclust-level marker-program scores, unsupervised top-marker rules, marker dotplots, spatial QC, and a curatable cluster-review CSV.
- The annotation cache is versioned as `cluster_full_panel_v9_xenium_graphclust_io_mucosa_submucosa_k24`, so older labels are not silently reused.
- Includes a curated `pdac_io_v1` graphclust correction from Xenium Explorer review: graphclust 2 is treated as mucosa gland and graphclust 17 as submucosa, preventing those cells from entering pancreatic ductal niches.
- Broad cluster-level duodenum calls are conservative. When a sample has `CDX2/REG4/DMBT1/TMPRSS2`, the annotation notebook additionally runs epithelial-only unsupervised PCA + MiniBatchKMeans refinement to split true duodenum-like cells from intestinal-like/PanIN-like ductal epithelium.
- Uses stable Tier_A/Tier_B palettes across annotation composition, UMAP, and spatial QC plots.
- Builds ductal/tumor epithelial connected-component niches from 10x `cell_boundaries.parquet` boundary proximity first, with a centroid-radius fallback.
- Adds cell/nucleus boundary shape summaries, including circularity, solidity, elongation, Feret diameter, and boundary irregularity.
- Adds a runnable Xenium DAPI pixel-feature extraction path using `nucleus_boundaries.parquet` plus user-verified focus morphology images. The notebook defaults to a pilot run before full epithelial extraction.
- Summarizes epithelial niche graph morphology, state markers, surrounding cell context, duct/lumen topology, ductal-continuity/cancerization proxies, and epithelial-stromal interface disruption proxies.
- Fits a pooled ElPiGraph pseudotime trajectory from niche-level Xenium features. The detailed tree node count is adaptive (`min(100, max(40, ceil(sqrt(n_niches) * 2.5)))`), while the simplified tree used for major branch labels defaults to 24 nodes.
- Adds histology proxy scores for normal-duct-like, ADM/PanIN-like, glandular architecture, ductal-continuity/cancerization-like spread, epithelial-stromal interface disruption, desmoplastic tumor, immune-inflamed, immune-excluded, duodenum-invasion context, and gland-poor/undifferentiated-like states.
- Uses block-balanced feature scaling so broad feature families (histology modules, epithelial state, architecture/topology, nuclear morphology, microenvironment) contribute more evenly to the pooled trajectory.
- Adds an intrinsic epithelial sensitivity trajectory using within-sample centered epithelial, architecture, and nuclear morphology features only.
- Adds automatic branch annotation summaries: branch structure is inferred from the simplified tree, then each branch receives sample composition, histology-score enrichment, and a suggested biological identity for manual review.

Important caveat:

- The public 10x datasets use different panels. The pooled trajectory must prioritize shared or sufficiently available features and should be interpreted as a cross-sample spatial/transcriptional continuum, not literal patient time.
- DAPI intensity/texture features are optional but implemented. Run the pilot first, then set `RUN_DAPI_FULL_EPITHELIAL=True` before rebuilding niches if the pilot QC looks good.

If continuing in a new Codex chat, say:

> Continue from `/Users/shihongwu/SpatioEv/docs/xenium_pseudotime_handoff.md`. Please inspect and help run/refine the Xenium notebooks.
