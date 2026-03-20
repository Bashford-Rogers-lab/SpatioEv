import numpy as np
from sklearn.preprocessing import StandardScaler


def build_marker_features(
    adata,
    markers
):
    """
    Extract and z-normalize marker expression matrix.
    """

    X = adata[:, markers].X

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X


def build_morphology_features(
    adata,
    morph_weight=0.4
):
    """
    Transform and scale morphology features.
    """

    obs = adata.obs.copy()

    # log transform size-like features
    log_features = [
        "area",
        "convex_area",
        "perimeter",
        "major_axis_length",
        "minor_axis_length",
        "feret_diameter_max",
        "equivalent_diameter",
        "num_concavities",
        "centroid_dif",
    ]

    obs[log_features] = np.log1p(obs[log_features])


    morph_features = log_features + [
        "eccentricity",
        "solidity",
        "major_minor_axis_ratio",
        "perim_square_over_area",
        "major_axis_equiv_diam_ratio",
        "convex_hull_resid",
        "circularity",
        "fractal_dimension",
        "boundary_irregularity",
        "nc_ratio",
    ]

    X = obs[morph_features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # apply morphology weight
    X = X * morph_weight

    return X, morph_features


def build_feature_matrix(
    adata,
    markers,
    morph_weight=0.4
):
    """
    Combine marker and morphology features.
    """

    X_marker = build_marker_features(adata, markers)

    X_morph, morph_features = build_morphology_features(
        adata,
        morph_weight=morph_weight
    )

    X = np.concatenate([X_marker, X_morph], axis=1)

    return X