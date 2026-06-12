# Niche Graph Pseudotime

SpatioEv's pseudotime workflow treats each epithelial niche as an observation.
The feature matrix combines pathology-inspired module scores, graph topology,
cell-state summaries, boundary/core contrasts, and surrounding microenvironment
features. A principal-tree model can then order niches along a spatial
progression coordinate.

This approach is designed for static tissue images where temporal evolution is
inferred from recurrent spatial states rather than sampled directly.

