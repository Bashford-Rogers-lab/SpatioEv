"""Tests verifying that all new submodule files are importable and that
every public function is reachable via both the submodule path and the
namespace facade (sv.tl.X, sv.pp.X, sv.pl.X, sv.xe.X).
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_module_and_all(module_path: str) -> list[str]:
    """Import *module_path*, assert it has __all__, and return __all__."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "__all__"), f"{module_path} is missing __all__"
    return list(mod.__all__)


def _all_callable(module_path: str):
    """Every name in __all__ must be callable or a class."""
    mod = importlib.import_module(module_path)
    names = mod.__all__
    for name in names:
        obj = getattr(mod, name, None)
        assert obj is not None, f"{module_path}.{name} not found"
        assert callable(obj), f"{module_path}.{name} is not callable"


# ---------------------------------------------------------------------------
# tl submodules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("submod", [
    "spatioev.tl.stats",
    "spatioev.tl.density",
    "spatioev.tl.niche",
    "spatioev.tl.ecm",
    "spatioev.tl.pseudotime",
    "spatioev.tl.phenotype",
    "spatioev.tl.ml",
])
def test_tl_submodule_has_all(submod):
    names = _check_module_and_all(submod)
    assert len(names) > 0, f"{submod}.__all__ is empty"


@pytest.mark.parametrize("submod", [
    "spatioev.tl.stats",
    "spatioev.tl.density",
    "spatioev.tl.niche",
    "spatioev.tl.ecm",
    "spatioev.tl.pseudotime",
    "spatioev.tl.phenotype",
    "spatioev.tl.ml",
])
def test_tl_submodule_all_callable(submod):
    _all_callable(submod)


# ---------------------------------------------------------------------------
# pp submodules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("submod", [
    "spatioev.pp.qc",
    "spatioev.pp.normalize",
    "spatioev.pp.pixel",
    "spatioev.pp.spatial_prep",
])
def test_pp_submodule_has_all(submod):
    names = _check_module_and_all(submod)
    assert len(names) > 0, f"{submod}.__all__ is empty"


@pytest.mark.parametrize("submod", [
    "spatioev.pp.qc",
    "spatioev.pp.normalize",
    "spatioev.pp.spatial_prep",
])
def test_pp_submodule_all_callable(submod):
    _all_callable(submod)


# ---------------------------------------------------------------------------
# pl submodules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("submod", [
    "spatioev.pl.qc",
    "spatioev.pl.spatial",
])
def test_pl_submodule_has_all(submod):
    names = _check_module_and_all(submod)
    assert len(names) > 0, f"{submod}.__all__ is empty"


@pytest.mark.parametrize("submod", [
    "spatioev.pl.qc",
    "spatioev.pl.spatial",
])
def test_pl_submodule_all_callable(submod):
    _all_callable(submod)


# ---------------------------------------------------------------------------
# xe submodules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("submod", [
    "spatioev.xe.annotation",
    "spatioev.xe.features",
])
def test_xe_submodule_has_all(submod):
    names = _check_module_and_all(submod)
    assert len(names) > 0, f"{submod}.__all__ is empty"


@pytest.mark.parametrize("submod", [
    "spatioev.xe.annotation",
    "spatioev.xe.features",
])
def test_xe_submodule_all_callable(submod):
    _all_callable(submod)


# ---------------------------------------------------------------------------
# Namespace facade spot-checks (sv.tl / sv.pp / sv.pl / sv.xe)
# ---------------------------------------------------------------------------

def test_tl_facade_resolves_key_functions():
    import spatioev as sv
    assert callable(sv.tl.morans_i)
    assert callable(sv.tl.ripleys_k_by_phenotype)
    assert callable(sv.tl.cross_ripleys_k_by_phenotype)
    assert callable(sv.tl.build_cell_graph)
    assert callable(sv.tl.build_niche_feature_table)
    assert callable(sv.tl.prepare_pseudotime_feature_matrix)
    assert callable(sv.tl.build_cell_fiber_links)
    assert callable(sv.tl.compute_invasion_score)
    assert callable(sv.tl.cluster_cells)
    assert callable(sv.tl.run_svm_phenotyping)


def test_pp_facade_resolves_key_functions():
    import spatioev as sv
    assert callable(sv.pp.run_segmentation_qc)
    assert callable(sv.pp.generate_qc_summary)
    assert callable(sv.pp.zscore_normalize)
    assert callable(sv.pp.add_obs_from_var)
    assert callable(sv.pp.validate_spatial_coordinates)
    assert callable(sv.pp.extract_cell_pixel_features)


def test_pl_facade_resolves_key_functions():
    import spatioev as sv
    assert callable(sv.pl.spatial_scatter_plot)
    assert callable(sv.pl.plot_niche_boundaries)
    assert callable(sv.pl.plot_cluster_heatmap)
    assert callable(sv.pl.plot_area_distribution)


def test_xe_facade_resolves_key_functions():
    import spatioev as sv
    assert callable(sv.xe.compute_marker_set_scores)
    assert callable(sv.xe.score_xenium_histology_modules)
    assert callable(sv.xe.available_feature_map)


# ---------------------------------------------------------------------------
# Submodule-level direct imports (sv.tl.stats.morans_i etc.)
# ---------------------------------------------------------------------------

def test_direct_submodule_import_stats():
    from spatioev.tl.stats import cross_ripleys_k, morans_i, ripleys_k_by_phenotype
    assert callable(morans_i)
    assert callable(ripleys_k_by_phenotype)
    assert callable(cross_ripleys_k)


def test_direct_submodule_import_density():
    from spatioev.tl.density import (
        assign_tiles,
        compute_general_density,
        phenotype_interaction_density,
    )
    assert callable(assign_tiles)
    assert callable(compute_general_density)
    assert callable(phenotype_interaction_density)


def test_direct_submodule_import_niche():
    from spatioev.tl.niche import (
        build_cell_graph,
        build_niche_boundaries,
        build_niche_feature_table,
    )
    assert callable(build_cell_graph)
    assert callable(build_niche_boundaries)
    assert callable(build_niche_feature_table)


def test_direct_submodule_import_pseudotime():
    from spatioev.tl.pseudotime import (
        assign_pseudotime_bins,
        prepare_pseudotime_feature_matrix,
    )
    assert callable(prepare_pseudotime_feature_matrix)
    assert callable(assign_pseudotime_bins)


def test_direct_submodule_import_ecm():
    from spatioev.tl.ecm import build_cell_fiber_links, compute_invasion_score
    assert callable(build_cell_fiber_links)
    assert callable(compute_invasion_score)


def test_direct_submodule_import_pp():
    from spatioev.pp.normalize import add_obs_from_var, zscore_normalize
    from spatioev.pp.qc import generate_qc_summary, run_segmentation_qc
    from spatioev.pp.spatial_prep import validate_spatial_coordinates
    assert callable(run_segmentation_qc)
    assert callable(generate_qc_summary)
    assert callable(zscore_normalize)
    assert callable(add_obs_from_var)
    assert callable(validate_spatial_coordinates)


def test_direct_submodule_import_xe():
    from spatioev.xe.annotation import compute_marker_set_scores
    from spatioev.xe.features import score_xenium_histology_modules
    assert callable(compute_marker_set_scores)
    assert callable(score_xenium_histology_modules)


# ---------------------------------------------------------------------------
# Config is importable and correct types
# ---------------------------------------------------------------------------

def test_config_dataclasses():
    from spatioev.config import ClusteringConfig, QCConfig
    cfg = QCConfig(pixel_size=0.5)
    assert cfg.pixel_size == 0.5
    assert cfg.min_area_um2 == 10
    cl = ClusteringConfig(markers=["CD8", "Ki67"])
    assert cl.markers == ["CD8", "Ki67"]
    assert cl.resolution == 0.5
