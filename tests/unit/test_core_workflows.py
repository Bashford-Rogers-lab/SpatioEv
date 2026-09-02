from __future__ import annotations

import numpy as np
import pandas as pd

from spatioev.pp import (
    QCConfig,
    add_obs_from_var,
    add_zscore_obs_features,
    compute_tissue_areas,
    detect_edge_cells,
    generate_qc_summary,
    run_segmentation_qc,
    validate_spatial_coordinates,
)
from spatioev.tl import (
    add_local_morans_i,
    annotate_from_csv,
    annotate_interactive,
    assign_pseudotime_bins,
    assign_tiles,
    build_feature_matrix,
    compute_epithelial_centered_interaction_dynamics,
    compute_general_density,
    compute_local_density_all_cells,
    compute_phenotype_density,
    compute_radius_density,
    cross_morans_i,
    cross_ripley_local_counts,
    cross_ripleys_k_by_phenotype,
    morans_i,
    phenotype_density_correlation,
    phenotype_interaction_density,
    ripleys_k_by_phenotype,
    run_scimap_prior_knowledge_phenotyping,
    summarize_epithelial_interaction_dynamics,
    summarize_target_features_around_source_cells,
)


def test_qc_preprocessing_and_ml_features(toy_adata):
    validate_spatial_coordinates(toy_adata)

    qc = run_segmentation_qc(toy_adata.copy(), QCConfig(pixel_size=0.5))
    summary = generate_qc_summary(qc, groupby="imageid")
    assert set(summary["imageid"]) == {"img1", "img2"}
    assert "area_um2" in qc.obs

    adata = add_obs_from_var(toy_adata.copy(), ["CD8", "Ki67"])
    adata = add_zscore_obs_features(adata, ["area"])
    assert "CD8_expr_z" in adata.obs
    assert "area_z" in adata.obs

    X = build_feature_matrix(toy_adata, markers=["CD8", "Ki67"], morph_weight=0.25)
    assert X.shape[0] == toy_adata.n_obs
    assert X.shape[1] > 2
    assert np.isfinite(X).all()


def test_interactive_annotation_helpers(toy_adata, monkeypatch, tmp_path):
    adata = toy_adata.copy()
    adata.obs["leiden"] = ["0", "1", "10"] * 4

    labels = iter(["ductal", "", "stromal"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(labels))

    annotated, mapping = annotate_interactive(
        adata,
        cluster_key="leiden",
        new_key="annotation",
    )

    assert list(mapping) == ["0", "1", "10"]
    assert mapping == {"0": "ductal", "1": "1", "10": "stromal"}
    assert set(annotated.obs["annotation"]) == {"ductal", "1", "stromal"}

    mapping_path = tmp_path / "cluster_annotations.csv"
    mapping_path.write_text(
        "cluster,annotation\n0,epithelial\n1,immune\n10,stroma\n",
        encoding="utf-8",
    )
    from_csv = annotate_from_csv(
        adata.copy(),
        mapping_path,
        cluster_key="leiden",
        new_key="annotation_from_csv",
    )
    assert set(from_csv.obs["annotation_from_csv"]) == {"epithelial", "immune", "stroma"}


def test_scimap_prior_knowledge_wrapper_uses_rescale_then_phenotype(toy_adata, monkeypatch):
    calls = []

    def fake_rescale(adata, gate=None, **kwargs):
        calls.append(("rescale", gate.copy(), kwargs))
        adata = adata.copy()
        adata.layers["scaled"] = adata.X.copy()
        return adata

    def fake_phenotype_cells(adata, phenotype=None, label="phenotype", **kwargs):
        calls.append(("phenotype_cells", phenotype.copy(), label, kwargs))
        adata = adata.copy()
        adata.obs[label] = "mock_phenotype"
        return adata

    # The wrappers import from the vendored copy (spatioev._vendor.scimap),
    # not from an installed scimap, so patch there.
    import spatioev._vendor.scimap as vendored

    monkeypatch.setattr(vendored, "rescale", fake_rescale)
    monkeypatch.setattr(vendored, "phenotype_cells", fake_phenotype_cells)

    manual_gates = pd.DataFrame({"markers": ["CD8"], "img1": [1.0]})
    phenotype = pd.DataFrame({"parent": ["all"], "phenotype": ["mock"], "CD8": ["pos"]})

    out = run_scimap_prior_knowledge_phenotyping(
        toy_adata.copy(),
        manual_gates=manual_gates,
        phenotype_workflow=phenotype,
        label="scimap_phenotype",
        rescale_kwargs={"gmm_components": 2},
        phenotype_kwargs={"subset": None},
    )

    assert calls[0][0] == "rescale"
    assert calls[0][2] == {"gmm_components": 2}
    assert calls[1][0] == "phenotype_cells"
    assert calls[1][2] == "scimap_phenotype"
    assert calls[1][3] == {"subset": None}
    assert set(out.obs["scimap_phenotype"]) == {"mock_phenotype"}


def test_density_and_interaction_workflow(toy_adata):
    tiled = assign_tiles(toy_adata, tile_size=20)
    density = compute_general_density(tiled, tile_size=20)
    phenotype_density = compute_phenotype_density(tiled, phenotype_key="phenotype", tile_size=20)
    corr = phenotype_density_correlation(phenotype_density, phenotype_key="phenotype")

    assert not density.empty
    assert not phenotype_density.empty
    assert set(corr.index) <= set(toy_adata.obs["phenotype"])

    adata = compute_local_density_all_cells(toy_adata.copy(), k_neighbors=2)
    adata = compute_radius_density(adata, radius=25)
    adata = phenotype_interaction_density(
        adata,
        phenotype_key="phenotype",
        source_pheno="duct",
        target_pheno="immune",
        radius=30,
    )
    assert adata.obs["density_all"].notna().any()
    assert adata.obs["radius_density"].notna().any()
    assert adata.obs.filter(like="interaction_density").notna().any().any()


def test_spatial_statistics_and_pseudotime(toy_adata):
    tissue = compute_tissue_areas(toy_adata)
    assert tissue["tissue_area"].notna().all()

    edge_adata = detect_edge_cells(toy_adata.copy(), radius=5)
    assert "edge_cell" in edge_adata.obs

    ripley = ripleys_k_by_phenotype(toy_adata, phenotype_key="phenotype", radius=30)
    cross_ripley = cross_ripleys_k_by_phenotype(
        toy_adata,
        phenotype_key="phenotype",
        source_phenotype="duct",
        target_phenotype="immune",
        radius=30,
    )
    local_cross = cross_ripley_local_counts(
        toy_adata,
        phenotype_key="phenotype",
        source_phenotype="duct",
        target_phenotype="immune",
        radius=30,
    )
    assert not ripley.empty
    assert not cross_ripley.empty
    assert not local_cross.empty

    coords = toy_adata.obs[["X_centroid", "Y_centroid"]].to_numpy()
    assert np.isfinite(morans_i(coords, toy_adata.obs["feature_a"], k=2))
    assert np.isfinite(
        cross_morans_i(coords, toy_adata.obs["feature_a"], toy_adata.obs["feature_b"], k=2)
    )

    with_local = add_local_morans_i(toy_adata.copy(), value_key="feature_a", k=2)
    assert with_local.obs["local_morans_i__feature_a"].notna().any()

    neighbor_summary = summarize_target_features_around_source_cells(
        toy_adata,
        phenotype_key="phenotype",
        source_phenotype="duct",
        target_phenotype="immune",
        target_feature_keys=["feature_b"],
        radius=40,
    )
    assert "neighbor_mean__feature_b" in neighbor_summary

    bins, bin_summary = assign_pseudotime_bins(toy_adata.obs["pseudotime"], n_bins=4)
    assert bins.notna().all()
    assert len(bin_summary) == 4

    dynamics = compute_epithelial_centered_interaction_dynamics(
        toy_adata,
        pseudotime_key="pseudotime",
        phenotype_key="phenotype",
        source_phenotype="duct",
        target_phenotypes=["immune"],
        radius=40,
        pseudotime_bin_count=3,
    )
    summary = summarize_epithelial_interaction_dynamics(
        dynamics,
        pseudotime_key="pseudotime",
    )
    assert not summary.empty
