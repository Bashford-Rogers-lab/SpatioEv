from __future__ import annotations

import sys


def test_top_level_import_is_lightweight():
    sys.modules.pop("scanpy", None)
    sys.modules.pop("scimap", None)
    sys.modules.pop("napari", None)

    import spatioev

    assert spatioev.__version__
    assert "scanpy" not in sys.modules
    assert "scimap" not in sys.modules
    assert "napari" not in sys.modules


def test_lazy_core_function_access():
    import spatioev

    assert callable(spatioev.morans_i)
    assert callable(spatioev.compute_general_density)


def test_ecm_api_imports_without_statsmodels_side_effect():
    sys.modules.pop("statsmodels.formula.api", None)

    import spatioev

    assert callable(spatioev.build_cell_fiber_links)
    assert callable(spatioev.tl.cell_to_fiber_distance)
    assert callable(spatioev.tl.spatial_mixed_model)
    assert "statsmodels.formula.api" not in sys.modules
