"""Machine-learning helpers: feature construction, SVM training, prediction, and inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if TYPE_CHECKING:
    import anndata as ad


# ============================================================
# Section 1: Feature construction  (from archive/ml/features.py)
# ============================================================


def build_marker_features(
    adata: ad.AnnData,
    markers: list[str],
) -> np.ndarray:
    """Extract and z-normalise marker expression matrix.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with markers as variables.
    markers : list of str
        Variable names in ``adata.var_names`` to extract.

    Returns
    -------
    numpy.ndarray
        Z-score-normalised (cells × markers) feature matrix.
    """
    X = adata[:, markers].X
    if sparse.issparse(X):
        X = X.toarray()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X


def build_morphology_features(
    adata: ad.AnnData,
    morph_weight: float = 0.4,
) -> tuple[np.ndarray, list[str]]:
    """Transform and scale morphology features.

    Log-transforms size-related columns and z-scores all morphology
    features, then scales the result by *morph_weight*.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with morphology columns in ``adata.obs``.
    morph_weight : float
        Scalar multiplied into the morphology feature matrix to balance
        its contribution against marker features.

    Returns
    -------
    numpy.ndarray
        Scaled morphology feature matrix (cells × morphology_features).
    list of str
        Column names corresponding to the feature matrix.

    Raises
    ------
    ValueError
        If any expected morphology column is absent from ``adata.obs``.
    """
    obs = adata.obs.copy()

    if "fractal_dimension" not in obs.columns and "fractual_dimension" in obs.columns:
        obs["fractal_dimension"] = obs["fractual_dimension"]

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

    missing = [feature for feature in morph_features if feature not in obs.columns]
    if missing:
        raise ValueError(f"Missing morphology features in adata.obs: {missing}")

    X = obs[morph_features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # apply morphology weight
    X = X * morph_weight

    return X, morph_features


def build_feature_matrix(
    adata: ad.AnnData,
    markers: list[str],
    morph_weight: float = 0.4,
) -> np.ndarray:
    """Combine marker and morphology features into a single matrix.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with markers as variables and morphology columns
        in ``adata.obs``.
    markers : list of str
        Marker names to include as expression features.
    morph_weight : float
        Weight applied to morphology features before concatenation.

    Returns
    -------
    numpy.ndarray
        Combined feature matrix (cells × (markers + morphology)).
    """
    X_marker = build_marker_features(adata, markers)

    X_morph, _morph_features = build_morphology_features(
        adata,
        morph_weight=morph_weight,
    )

    X = np.concatenate([X_marker, X_morph], axis=1)

    return X


# ============================================================
# Section 2: SVM training  (from archive/ml/training.py)
# ============================================================


def train_svm_classifier(
    X: np.ndarray,
    y: np.ndarray | pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[SVC, dict]:
    """Train a balanced RBF-SVM classifier and evaluate on a held-out split.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix (samples × features).
    y : array-like
        Class labels aligned with *X*.
    test_size : float
        Fraction of samples reserved for the held-out test set.
    random_state : int
        Random seed for the stratified train/test split.

    Returns
    -------
    sklearn.svm.SVC
        Fitted SVM model.
    dict
        Classification report dictionary (from
        ``sklearn.metrics.classification_report`` with
        ``output_dict=True``).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
    )

    return model, report


# ============================================================
# Section 3: SVM prediction  (from archive/ml/prediction.py)
# ============================================================


def predict_svm(
    model: SVC,
    X: np.ndarray,
    classes: list[str] | np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predict phenotypes and per-class probabilities for a feature matrix.

    Parameters
    ----------
    model : sklearn.svm.SVC
        Fitted SVM model with ``probability=True``.
    X : numpy.ndarray
        Feature matrix (samples × features).
    classes : list of str or array-like
        Class labels corresponding to ``model.classes_``.

    Returns
    -------
    numpy.ndarray
        Predicted class label for each sample.
    pandas.DataFrame
        Probability columns named ``svm_prob_<class>``, one row per sample.
    """
    probs = model.predict_proba(X)

    pred = model.predict(X)

    prob_df = pd.DataFrame(
        probs,
        columns=[f"svm_prob_{c}" for c in classes],
    )

    return pred, prob_df


# ============================================================
# Section 4: SVM pipeline  (from archive/ml/svm.py)
# ============================================================


def run_svm_phenotyping(
    adata: ad.AnnData,
    markers: list[str],
    label_key: str,
    morph_weight: float = 0.4,
) -> tuple[ad.AnnData, SVC, dict]:
    """Train an SVM classifier and predict phenotypes for all cells.

    Cells whose *label_key* value is ``"unknown"`` are excluded from
    training but receive a prediction.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with markers as variables and morphology columns in
        ``adata.obs``.
    markers : list of str
        Marker features used for training.
    label_key : str
        Column in ``adata.obs`` containing phenotype labels.
    morph_weight : float
        Weight applied to morphology features.

    Returns
    -------
    anndata.AnnData
        *adata* updated in-place with ``"svm_prediction"`` and
        ``"svm_prob_<class>"`` columns in ``adata.obs``.
    sklearn.svm.SVC
        Fitted SVM model.
    dict
        Classification report dictionary.
    """
    # build feature matrix
    X = build_feature_matrix(
        adata,
        markers,
        morph_weight=morph_weight,
    )

    # training cells (exclude unknown)
    train_mask = adata.obs[label_key] != "unknown"

    X_train = X[train_mask]
    y_train = adata.obs[label_key][train_mask]

    # train model
    model, report = train_svm_classifier(
        X_train,
        y_train,
    )

    # predict all cells
    pred, prob_df = predict_svm(
        model,
        X,
        model.classes_,
    )

    # write predictions into AnnData
    adata.obs["svm_prediction"] = pred

    for col in prob_df.columns:
        adata.obs[col] = prob_df[col].values

    return adata, model, report


# ============================================================
# Section 5: SVM inspection  (from archive/ml/svm_inspection.py)
# ============================================================

def _require_scimap():
    try:
        import scimap as sm
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "SVM inspection viewers require optional scimap support. "
            "Install SpatioEv with `pip install -e '.[viewer]'` or install "
            "`scimap[napari]`."
        ) from exc

    return sm


def inspect_reassigned_cells(
    adata: ad.AnnData,
    image_path: str,
    original_label: str = "annotated_clusters_update3",
    predicted_label: str = "svm_prediction",
    original_value: str = "Unknown",
    point_size: int = 6,
) -> None:
    """Visualise cells reassigned by the SVM in an image viewer.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with both annotation and SVM prediction columns.
    image_path : str
        Path to the tissue image file.
    original_label : str
        Column containing manual annotations.
    predicted_label : str
        Column containing SVM predictions.
    original_value : str
        Manual annotation value to select cells from (e.g. ``"Unknown"``).
    point_size : int
        Point size used in the Napari overlay.

    Returns
    -------
    None
        Opens an interactive Napari viewer.
    """
    sm = _require_scimap()

    mask = adata.obs[original_label] == original_value

    print(f"Cells originally '{original_value}': {mask.sum()}")

    subset = adata[mask].copy()

    sm.pl.image_viewer(
        image_path=image_path,
        adata=subset,
        overlay=predicted_label,
        point_size=point_size,
        point_color="white",
    )


def inspect_disagreements(
    adata: ad.AnnData,
    image_path: str,
    original_label: str = "annotated_clusters_update3",
    predicted_label: str = "svm_prediction",
    point_size: int = 6,
) -> None:
    """Visualise cells where manual annotation and SVM prediction disagree.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with both annotation and SVM prediction columns.
    image_path : str
        Path to the tissue image file.
    original_label : str
        Column containing manual annotations.
    predicted_label : str
        Column containing SVM predictions.
    point_size : int
        Point size used in the Napari overlay.

    Returns
    -------
    None
        Opens an interactive Napari viewer showing disagreement cells.
    """
    sm = _require_scimap()

    mask = adata.obs[original_label] != adata.obs[predicted_label]

    print(f"Disagreement cells: {mask.sum()}")

    subset = adata[mask].copy()

    sm.pl.image_viewer(
        image_path=image_path,
        adata=subset,
        overlay=predicted_label,
        point_size=point_size,
        point_color="white",
    )


__all__ = [
    "build_feature_matrix",
    "build_marker_features",
    "build_morphology_features",
    "train_svm_classifier",
    "predict_svm",
    "run_svm_phenotyping",
    "inspect_reassigned_cells",
    "inspect_disagreements",
]
