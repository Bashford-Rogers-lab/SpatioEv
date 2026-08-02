from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from spatioev.hl import score_signed_feature_module, tree_edges
from spatioev.tl import (
    add_branch_time_bins,
    block_balance_feature_matrix,
    branch_time_feature_matrix,
    compute_feature_trend_table,
    infer_branch_labels,
    prepare_pseudotime_feature_matrix,
    summarize_branches,
)
from spatioev.xe import (
    assign_labels_from_marker_rules,
    compute_marker_set_scores,
    score_xenium_histology_modules,
    summarize_cluster_marker_scores,
)


def test_prepare_and_balance_pseudotime_feature_matrix():
    df = pd.DataFrame(
        {
            "pdac_dysplasia_score": np.linspace(0, 1, 20),
            "pdac_dysplasia_score_copy": np.linspace(0, 1, 20) + 1e-6,
            "state__Ki67_expr_z__mean": np.linspace(1, 3, 20),
            "surround_prop__Fibroblasts": np.linspace(0.2, 0.8, 20),
            "mostly_missing": [np.nan] * 18 + [1.0, 2.0],
            "constant": 1.0,
        }
    )

    result = prepare_pseudotime_feature_matrix(
        df,
        priority_features=["pdac_dysplasia_score"],
        correlation_threshold=0.95,
    )

    assert "pdac_dysplasia_score" in result.selected_features
    assert "mostly_missing" not in result.selected_features
    assert "constant" not in result.selected_features
    assert result.matrix.index.equals(df.index)

    balanced, block_df = block_balance_feature_matrix(result.matrix, return_blocks=True)
    assert balanced.shape == result.matrix.shape
    assert set(block_df.columns) == {"feature", "feature_block"}
    assert np.isfinite(balanced.to_numpy()).all()


def test_module_scoring_tree_branches_and_trends():
    df = pd.DataFrame(
        {
            "good_feature": np.linspace(0, 1, 40),
            "bad_feature": np.linspace(1, 0, 40),
            "pseudotime": np.linspace(0, 1, 40),
            "node_id": [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8 + [4] * 8,
        }
    )
    score = score_signed_feature_module(
        df,
        [(1, ["good_feature"]), (-1, ["bad_feature"])],
        min_features=2,
    )
    assert score.notna().all()
    assert score.iloc[-1] > score.iloc[0]

    tree = nx.Graph([(0, 1), (1, 2), (2, 3), (2, 4)])
    labels, metadata = infer_branch_labels(tree, df, source_node=0, node_col="node_id")
    assert metadata["hub_node"] == 2
    assert {"trunk", "branch 1", "branch 2"}.issubset(set(labels))

    with_labels = df.assign(branch=labels)
    branch_summary = summarize_branches(
        with_labels,
        branch_col="branch",
        pseudotime_col="pseudotime",
        score_cols=["good_feature"],
    )
    assert not branch_summary.empty
    assert "good_feature__z_enrichment" in branch_summary.columns

    trends = compute_feature_trend_table(
        df,
        ["good_feature", "bad_feature"],
        pseudotime_col="pseudotime",
        min_n=10,
    )
    assert {"spearman_fdr", "mannwhitney_fdr"}.issubset(trends.columns)

    binned = add_branch_time_bins(with_labels, "branch", "pseudotime", n_bins=2)
    matrix = branch_time_feature_matrix(
        binned,
        ["good_feature", "bad_feature"],
        branch_col="branch",
        time_bin_col="branch_time_bin",
    )
    assert not matrix.empty


def test_principal_tree_edges_accept_elpigraph_metadata_container():
    pg_tree = {
        "Edges": [
            np.array([[0, 1], [1, 2], [2, 3]]),
            np.array([0.1, 0.2, 0.3]),
            {"metadata": "ignored"},
        ]
    }
    assert tree_edges(pg_tree) == [(0, 1), (1, 2), (2, 3)]


def test_xenium_marker_and_histology_helpers():
    expression = pd.DataFrame(
        {
            "EPCAM": [5, 4, 0, 0],
            "KRT19": [3, 4, 0, 0],
            "PTPRC": [0, 0, 5, 4],
            "COL1A1": [0, 1, 0, 1],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    marker_scores = compute_marker_set_scores(
        expression,
        {"epithelial_score": ["EPCAM", "KRT19"], "immune_score": ["PTPRC"]},
    )
    summary = summarize_cluster_marker_scores(marker_scores, ["a", "a", "b", "b"])
    labeled = assign_labels_from_marker_rules(
        summary,
        {
            "epithelial": {"epithelial_score": 0.2},
            "immune": {"immune_score": 0.2},
        },
    )
    assert set(labeled["suggested_label"]) == {"epithelial", "immune"}

    feature_table = pd.DataFrame(
        {
            "duct_lumen__fraction": [1.0, 0.2, 0.1],
            "duct_continuity__score": [1.0, 0.4, 0.2],
            "interface__stromal_fraction": [0.0, 0.5, 1.0],
            "state__dapi_area_mean": [1.0, 2.0, 3.0],
            "state__nucleus_boundary_irregularity_mean": [0.1, 0.4, 0.8],
        }
    )
    histology_scores = score_xenium_histology_modules(feature_table, min_features=2)
    assert "histology__ductal_integrity_score" in histology_scores
    assert histology_scores.notna().any().any()
