# Tutorials

SpatioEv tutorials are organized around analysis goals rather than isolated
function calls.

## Recommended Learning Path

1. Set up SpatioEv and load AnnData inputs.
2. Run segmentation QC and prepare cell-level features.
3. Explore density, interaction, Ripley, and Moran statistics.
4. Build cell graphs and ductal epithelial niche features.
5. Infer spatial pseudotime in PDAC sample 34434.
6. Extend pseudotime across pooled multiplexed-imaging samples.
7. Transfer the feature logic to Xenium.
8. Analyze ECM-cell links, ECM statistics, and ECM neighborhoods.
9. Export tables and regenerate manuscript figures.

The current notebook tutorials are in the repository [`examples/`](https://github.com/Bashford-Rogers-lab/SpatioEv/tree/main/examples) directory.
The Markdown tutorial below is the public website-style version.

- [Mastering Spatial Evolution Analysis](md/spatial_evolution_spatioev.md)

## Runnable examples

Three notebooks under `examples/` execute end to end on synthetic data, so
they need no download and are run in CI on every push:

1. `01_quickstart.ipynb` — QC, tile density, Moran's I, Ripley's K
2. `02_spatial_niches.ipynb` — components, boundaries, region assignment, composition
3. `03_ecm_cell_analysis.ipynb` — cell–fibre links, orientation, ECM–cell coupling
