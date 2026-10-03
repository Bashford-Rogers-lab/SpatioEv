"""Tumour / envelope / stroma regions and group comparison."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatioev.tl.compare import aggregate_to_units, compare_groups
from spatioev.tl.niche import (
    REGION_CODES,
    define_tissue_regions,
    rasterize_tissue_regions,
    read_region_boundaries,
    summarize_region_composition,
    write_region_boundaries,
)


def nest_tissue(n=6000, seed=0, envelope_myeloid=True) -> ad.AnnData:
    """Two dense circular tumour nests in a uniform stroma (pixel units)."""
    rng = np.random.default_rng(seed)
    side = 1600.0
    xy = rng.uniform(0, side, size=(n, 2))
    centres = np.array([[450, 450], [1150, 1100]])
    radius = 220.0
    d = np.min(np.linalg.norm(xy[:, None] - centres[None], axis=2), axis=1)
    phenotype = np.where(d < radius, "Tumour", rng.choice(["Fibroblast", "Myeloid", "T cell"], n, p=[0.5, 0.3, 0.2]))
    if envelope_myeloid:
        ring = (d >= radius) & (d < radius + 60)
        phenotype[ring & (rng.random(n) < 0.6)] = "Myeloid"
    obs = pd.DataFrame(
        {"imageid": "img1", "X_centroid": xy[:, 0], "Y_centroid": xy[:, 1], "phenotype": phenotype},
        index=[f"c{i}" for i in range(n)],
    )
    return ad.AnnData(obs=obs)


def test_regions_follow_nests_and_envelope():
    adata = nest_tissue()
    regions, boundaries = define_tissue_regions(adata, "phenotype", "Tumour", envelope_width=60)
    assert len(boundaries) == 2
    obs = regions.obs
    assert set(obs["tissue_region"].astype(str)) == {"tumour", "envelope", "stroma"}
    tumour_cells = obs[obs["phenotype"] == "Tumour"]
    assert (tumour_cells["tissue_region"] == "tumour").mean() > 0.95
    # The planted envelope enrichment of myeloid cells is recovered.
    composition = summarize_region_composition(regions, "phenotype")
    myeloid = composition[composition["phenotype"] == "Myeloid"].set_index("tissue_region")["proportion"]
    assert myeloid["envelope"] > 1.5 * myeloid["stroma"]
    totals = composition.groupby(["imageid", "tissue_region"])["proportion"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0)
    assert composition.loc[composition["tissue_region"] == "all", "region_total"].iloc[0] == adata.n_obs


def test_no_tumour_means_everything_is_stroma():
    adata = nest_tissue()
    adata.obs["phenotype"] = adata.obs["phenotype"].replace("Tumour", "Fibroblast")
    regions, boundaries = define_tissue_regions(adata, "phenotype", "Tumour", envelope_width=60)
    assert boundaries.empty
    assert (regions.obs["tissue_region"] == "stroma").all()


def test_raster_and_geojson_round_trip(tmp_path):
    adata = nest_tissue()
    regions, boundaries = define_tissue_regions(adata, "phenotype", "Tumour", envelope_width=60)
    raster = rasterize_tissue_regions(boundaries, regions.obs, (1600, 1600), tissue_radius=80, downsample=4)
    assert raster.shape == (400, 400)
    assert set(np.unique(raster)) <= set(REGION_CODES) | {0}
    # A pixel at a nest centre is tumour; one just outside the nest is envelope.
    assert raster[450 // 4, 450 // 4] == 1
    path = write_region_boundaries(boundaries, tmp_path / "nests.geojson")
    back = read_region_boundaries(path)
    assert len(back) == 2
    assert sorted(g.area for g in back["geometry"]) == pytest.approx(sorted(g.area for g in boundaries["geometry"]))
    assert sorted(g.area for g in back["expanded_geometry"]) == pytest.approx(sorted(g.area for g in boundaries["expanded_geometry"]))


def long_table(values_a, values_b, feature="f1"):
    rows = [{"unit": f"a{i}", "group": "A", "feature": feature, "value": v} for i, v in enumerate(values_a)]
    rows += [{"unit": f"b{i}", "group": "B", "feature": feature, "value": v} for i, v in enumerate(values_b)]
    return pd.DataFrame(rows)


def test_compare_groups_separated_groups_hit_the_minimum_p():
    table = long_table([1, 2, 3], [10, 11, 12])
    result = compare_groups(table, unit_key="unit", groups=["A", "B"]).iloc[0]
    assert result["test"] == "mann_whitney_u"
    assert result["p_value"] == pytest.approx(0.1)
    assert result["min_achievable_p"] == pytest.approx(0.1)
    assert result["median_difference"] == pytest.approx(9)


def test_compare_groups_averages_within_units_first():
    table = long_table([1, 1, 1, 1], [5, 5, 5, 5])
    table["unit"] = ["p1", "p1", "p2", "p2", "q1", "q1", "q2", "q2"]
    result = compare_groups(table, unit_key="unit", groups=["A", "B"], min_units_per_group=2).iloc[0]
    assert result["n_A"] == 2 and result["n_B"] == 2


def test_compare_groups_rejects_a_unit_in_two_groups_and_adjusts_p():
    table = long_table([1, 2], [3, 4])
    table.loc[0, "unit"] = "b0"
    with pytest.raises(ValueError, match="more than one group"):
        aggregate_to_units(table, "unit")
    rng = np.random.default_rng(0)
    many = pd.concat([long_table(rng.normal(0, 1, 8), rng.normal(0, 1, 8), f"f{i}") for i in range(20)])
    result = compare_groups(many, unit_key="unit")
    assert (result["q_value"] >= result["p_value"] - 1e-12).all()
    assert result["q_value"].max() <= 1
