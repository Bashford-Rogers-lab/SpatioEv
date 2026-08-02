"""Numerical ground-truth tests for :mod:`spatioev.tl.niche`.

``tl/niche.py`` is the largest module in the package (~4,000 lines) and
previously had zero functional coverage — its 19 public functions were only
asserted to be ``callable``. These tests use hand-constructed geometry whose
correct answers are known exactly (component counts, hull areas, buffered
areas, composition proportions), so the module can be split and optimised
without silently changing published numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatioev.tl import niche

shapely = pytest.importorskip("shapely", reason="niche geometry requires Shapely")
anndata = pytest.importorskip("anndata")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _square_block(cx: float, cy: float, half: float = 20.0, n: int = 5) -> np.ndarray:
    """An n x n lattice of points spanning [cx-half, cx+half] in both axes."""
    g = np.linspace(-half, half, n)
    xs, ys = np.meshgrid(g, g)
    return np.column_stack([xs.ravel() + cx, ys.ravel() + cy])


def _adata_from(coords: np.ndarray, **obs_cols):
    obs = pd.DataFrame(
        {"X_centroid": coords[:, 0], "Y_centroid": coords[:, 1], "imageid": "img1"}
    )
    for key, value in obs_cols.items():
        obs[key] = value
    adata = anndata.AnnData(X=np.zeros((len(obs), 1)))
    adata.obs = obs
    adata.obs_names = [f"c{i}" for i in range(len(obs))]
    return adata


@pytest.fixture
def three_blobs():
    """Three 5x5 blocks of 25 cells, separated by 500 units — unambiguously 3 groups."""
    coords = np.vstack([_square_block(0, 0), _square_block(500, 0), _square_block(0, 500)])
    return _adata_from(coords, label="tumour")


# --------------------------------------------------------------------------- #
# Spatial component clustering
# --------------------------------------------------------------------------- #


def test_components_recovers_exactly_three_well_separated_blobs(three_blobs):
    out = niche.cluster_spatial_components(
        three_blobs, label_key="label", label_value="tumour", radius=100
    )
    counts = out.obs["tumour_component"].value_counts()

    assert len(counts) == 3, f"expected 3 components, got {len(counts)}"
    assert set(counts.to_numpy()) == {25}, "each blob holds exactly 25 cells"


def test_components_merge_when_the_radius_spans_the_gap(three_blobs):
    """A radius larger than the 500-unit separation must merge the blobs."""
    merged = niche.cluster_spatial_components(
        three_blobs, label_key="label", label_value="tumour", radius=1000
    )
    assert merged.obs["tumour_component"].nunique() == 1


def test_components_are_confined_to_the_labelled_cells():
    """Cells not carrying the target label must not join a component."""
    coords = np.vstack([_square_block(0, 0), _square_block(500, 0)])
    labels = ["tumour"] * 25 + ["stroma"] * 25
    adata = _adata_from(coords, label=labels)

    out = niche.cluster_spatial_components(
        adata, label_key="label", label_value="tumour", radius=100
    )
    assigned = out.obs.loc[out.obs["label"] == "tumour", "tumour_component"]
    others = out.obs.loc[out.obs["label"] == "stroma", "tumour_component"]

    assert assigned.nunique() == 1, "the single tumour blob is one component"
    assert not assigned.eq("unassigned").any()
    # Non-target cells carry the "unassigned" sentinel rather than NaN.
    assert (others == "unassigned").all()


def test_dbscan_niches_also_recover_the_three_blobs(three_blobs):
    out = niche.cluster_spatial_niches(
        three_blobs, label_key="label", label_value="tumour", eps=100, min_samples=3
    )
    counts = out.obs["tumour_component"].value_counts()
    assert len(counts) == 3
    assert set(counts.to_numpy()) == {25}


# --------------------------------------------------------------------------- #
# Boundary geometry — exact areas
# --------------------------------------------------------------------------- #


@pytest.fixture
def blob_with_boundaries(three_blobs):
    out = niche.cluster_spatial_components(
        three_blobs, label_key="label", label_value="tumour", radius=100
    )
    boundaries = niche.build_niche_boundaries(
        out, component_key="tumour_component", min_cluster_size=5, method="convex_hull"
    )
    return out, boundaries


def test_convex_hull_boundary_area_is_exact(blob_with_boundaries):
    """Each blob spans 40 x 40 units, so its convex hull area is exactly 1600."""
    _, boundaries = blob_with_boundaries

    assert len(boundaries) == 3
    np.testing.assert_allclose(boundaries["area"].to_numpy(), 1600.0, rtol=1e-12)
    assert (boundaries["n_cells"] == 25).all()


def test_boundary_bounds_match_the_input_extent(blob_with_boundaries):
    _, boundaries = blob_with_boundaries
    by_component = boundaries.set_index("tumour_component")["bounds"].to_dict()

    expected = {(-20.0, -20.0, 20.0, 20.0), (480.0, -20.0, 520.0, 20.0),
                (-20.0, 480.0, 20.0, 520.0)}
    assert {tuple(np.round(b, 9)) for b in by_component.values()} == expected


def test_buffering_expands_area_by_the_analytic_amount(blob_with_boundaries):
    """Buffering a 40x40 square by d gives 40^2 + 4*40*d + pi*d^2 (Minkowski sum)."""
    _, boundaries = blob_with_boundaries
    d = 10.0
    buffered = niche.buffer_niche_boundaries(
        boundaries, component_key="tumour_component", expand_by=d
    )

    expected = 40.0**2 + 4 * 40.0 * d + np.pi * d**2
    for geom in buffered["expanded_geometry"]:
        # Shapely approximates the rounded corners with line segments, so the
        # result is a slight under-estimate of the exact Minkowski area.
        assert geom.area == pytest.approx(expected, rel=1e-3)

    # The original geometry and its area column are left untouched by design.
    np.testing.assert_allclose(buffered["area"].to_numpy(), 1600.0, rtol=1e-12)


def test_buffering_is_monotone_in_the_expansion_distance(blob_with_boundaries):
    _, boundaries = blob_with_boundaries
    areas = []
    for d in (0.0, 5.0, 20.0):
        out = niche.buffer_niche_boundaries(
            boundaries, component_key="tumour_component", expand_by=d
        )
        areas.append(out["expanded_geometry"].iloc[0].area)
    assert areas[0] < areas[1] < areas[2]


def test_small_components_are_dropped_by_min_cluster_size(three_blobs):
    out = niche.cluster_spatial_components(
        three_blobs, label_key="label", label_value="tumour", radius=100
    )
    kept = niche.build_niche_boundaries(
        out, component_key="tumour_component", min_cluster_size=26, method="convex_hull"
    )
    assert len(kept) == 0, "no component has 26 cells, so none should survive"


# --------------------------------------------------------------------------- #
# Region assignment and composition
# --------------------------------------------------------------------------- #


def test_every_cell_of_a_component_is_assigned_to_its_own_region(blob_with_boundaries):
    out, boundaries = blob_with_boundaries
    assignments = niche.assign_cells_to_niche_regions(
        out, boundaries, component_key="tumour_component"
    )

    assert len(assignments) == 75
    assert set(assignments["region"]) <= {"core", "boundary", "surround", "outside"}
    # Each assigned cell belongs to the component it was clustered into.
    merged = assignments.merge(
        out.obs.reset_index()[["index", "tumour_component"]],
        left_on="cell_id", right_on="index", suffixes=("", "_orig"),
    )
    assert (merged["tumour_component"] == merged["tumour_component_orig"]).all()


def test_composition_proportions_are_exact_and_sum_to_one():
    """13 duct + 12 immune in one blob -> proportions 0.52 / 0.48."""
    coords = _square_block(0, 0)
    adata = _adata_from(coords, label="tumour", phenotype=["duct"] * 13 + ["immune"] * 12)

    out = niche.cluster_spatial_components(
        adata, label_key="label", label_value="tumour", radius=100
    )
    boundaries = niche.build_niche_boundaries(
        out, component_key="tumour_component", min_cluster_size=5, method="convex_hull"
    )
    assignments = niche.assign_cells_to_niche_regions(
        out, boundaries, component_key="tumour_component"
    )
    comp = niche.summarize_niche_composition(
        out, assignments, component_key="tumour_component", phenotype_key="phenotype"
    )

    counts = comp.set_index("phenotype")["count"].to_dict()
    assert counts == {"duct": 13, "immune": 12}

    props = comp.set_index("phenotype")["proportion"]
    assert props["duct"] == pytest.approx(13 / 25)
    assert props["immune"] == pytest.approx(12 / 25)
    assert props.sum() == pytest.approx(1.0)


def test_composition_counts_are_conserved_across_components(three_blobs):
    """Summing composition counts must recover the total number of cells."""
    phen = (["duct"] * 13 + ["immune"] * 12) * 3
    adata = _adata_from(
        np.vstack([_square_block(0, 0), _square_block(500, 0), _square_block(0, 500)]),
        label="tumour",
        phenotype=phen,
    )
    out = niche.cluster_spatial_components(
        adata, label_key="label", label_value="tumour", radius=100
    )
    boundaries = niche.build_niche_boundaries(
        out, component_key="tumour_component", min_cluster_size=5, method="convex_hull"
    )
    assignments = niche.assign_cells_to_niche_regions(
        out, boundaries, component_key="tumour_component"
    )
    comp = niche.summarize_niche_composition(
        out, assignments, component_key="tumour_component", phenotype_key="phenotype"
    )

    assert comp["count"].sum() == 75
    assert comp.groupby("phenotype")["count"].sum().to_dict() == {"duct": 39, "immune": 36}
