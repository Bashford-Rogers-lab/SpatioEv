"""Reusable helpers for spatial pseudotime workflows.

Split into submodules by stage of the analysis:

    features   feature-matrix construction, tree handling, branch annotation
    dynamics   spatial interaction dynamics along pseudotime
    trends     trend and transition summaries

These helpers intentionally avoid importing heavy optional trajectory
packages (ElPiGraph, UMAP) at module import time. If a workflow needs them,
fit those models in the caller and pass the result in.

All names remain importable directly from this package, so
``from spatioev.tl.pseudotime import infer_branch_labels`` is unchanged.
"""

from __future__ import annotations

from .dynamics import (
    assign_pseudotime_bins,
    compute_epithelial_centered_interaction_dynamics,
    summarize_epithelial_interaction_dynamics,
)
from .features import (
    # Not in __all__ (it is data, not a callable), but re-exported so that
    # `from spatioev.tl.pseudotime import DEFAULT_FEATURE_BLOCK_RULES` works
    # exactly as it did before the split.
    DEFAULT_FEATURE_BLOCK_RULES as DEFAULT_FEATURE_BLOCK_RULES,
)
from .features import (
    FeatureMatrixResult,
    assign_feature_blocks,
    block_balance_feature_matrix,
    infer_branch_labels,
    minmax_scale,
    node_graph,
    prepare_pseudotime_feature_matrix,
    project_tree_nodes_to_embedding,
    sample_center_feature_matrix,
    score_signed_feature_module,
    summarize_branches,
    tree_edges,
    zscore_series,
)
from .trends import (
    add_branch_time_bins,
    benjamini_hochberg,
    branch_time_feature_matrix,
    compute_feature_trend_table,
    find_branch_transition_features,
)

__all__ = [
    "prepare_pseudotime_feature_matrix",
    "block_balance_feature_matrix",
    "sample_center_feature_matrix",
    "infer_branch_labels",
    "summarize_branches",
    "project_tree_nodes_to_embedding",
    "score_signed_feature_module",
    "assign_pseudotime_bins",
    "compute_epithelial_centered_interaction_dynamics",
    "summarize_epithelial_interaction_dynamics",
    "compute_feature_trend_table",
    "add_branch_time_bins",
    "branch_time_feature_matrix",
    "find_branch_transition_features",
    # helper / building-block functions also defined in this package
    "FeatureMatrixResult",
    "zscore_series",
    "minmax_scale",
    "assign_feature_blocks",
    "tree_edges",
    "node_graph",
    "benjamini_hochberg",
]
