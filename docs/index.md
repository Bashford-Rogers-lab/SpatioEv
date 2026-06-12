# SpatioEv

SpatioEv is a Python toolbox for spatial evolution analysis over multiplexed
imaging and spatial-transcriptomics data. It focuses on the features that make
static tissue images biologically dynamic: spatial statistics, morphology,
topology, cell-graph neighborhoods, ECM-cell organization, and
trajectory-ready epithelial niche programs.

The public API is organized in a Scimap-inspired style:

```python
import spatioev as sv

sv.pp   # preprocessing and QC
sv.tl   # analysis tools
sv.pl   # plotting
sv.hl   # helper functions
sv.xe   # Xenium/spatial-transcriptomics helpers
sv.io   # input/output
```

Historical implementation modules live under `spatioev.archive`. New user code
should use the public namespaces above rather than importing implementation
modules directly.

## What Makes SpatioEv Different

- Morphology and topology features extracted from epithelial niches.
- Cell-graph summaries that capture niche architecture and surroundings.
- Spatial pseudotime from static multiplexed images.
- PDAC pathology module scoring for ductal niche evolution.
- Ripley and Moran spatial statistics for cells, phenotypes, and ECM fibers.
- ECM-cell distance, density, autocorrelation, and neighborhood analysis.
- Xenium-compatible annotation and histology-module helper workflows.
