from __future__ import annotations

import sys


def test_scimap_style_namespaces_are_available_without_heavy_imports():
    sys.modules.pop("scanpy", None)
    sys.modules.pop("scimap", None)
    sys.modules.pop("napari", None)

    import spatioev as sv

    assert sv.pp.__name__ == "spatioev.pp"
    assert sv.tl.__name__ == "spatioev.tl"
    assert sv.pl.__name__ == "spatioev.pl"
    assert sv.hl.__name__ == "spatioev.hl"
    assert sv.xe.__name__ == "spatioev.xe"
    assert "scanpy" not in sys.modules
    assert "scimap" not in sys.modules
    assert "napari" not in sys.modules


def test_public_namespaces_expose_core_spatioev_tools():
    import spatioev as sv

    assert callable(sv.pp.run_segmentation_qc)
    assert callable(sv.pp.extract_cell_pixel_features)
    assert callable(sv.tl.ripleys_k)
    assert callable(sv.tl.morans_i_permutation_test)
    assert callable(sv.tl.cross_morans_i_feature_matrix)
    assert callable(sv.tl.build_niche_feature_table)
    assert callable(sv.tl.prepare_pseudotime_feature_matrix)
    assert callable(sv.tl.cell_to_fiber_distance)
    assert callable(sv.tl.ecm_cross_ripleys_k)
    assert callable(sv.pl.plot_spatial_category)
    assert callable(sv.hl.tree_edges)
    assert callable(sv.hl.benjamini_hochberg)
    assert callable(sv.xe.compute_marker_set_scores)


def test_spatial_stats_are_exposed_through_public_tools_namespace():
    import spatioev as sv

    assert not hasattr(sv, "spatial")
    assert callable(sv.tl.morans_i_permutation_test)
    assert callable(sv.tl.morans_i_by_image_permutation_test)
