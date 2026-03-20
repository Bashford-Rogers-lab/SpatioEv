from .features import build_feature_matrix
from .training import train_svm_classifier
from .prediction import predict_svm


def run_svm_phenotyping(
    adata,
    markers,
    label_key,
    morph_weight=0.4
):
    """
    Train SVM classifier and predict phenotypes.

    Parameters
    ----------
    adata : AnnData
    markers : list
        Marker features for training
    label_key : str
        Column containing phenotype labels
    morph_weight : float
        Weight applied to morphology features
    """

    # build feature matrix
    X = build_feature_matrix(
        adata,
        markers,
        morph_weight=morph_weight
    )

    # training cells (exclude unknown)
    train_mask = adata.obs[label_key] != "unknown"

    X_train = X[train_mask]
    y_train = adata.obs[label_key][train_mask]

    # train model
    model, report = train_svm_classifier(
        X_train,
        y_train
    )

    # predict all cells
    pred, prob_df = predict_svm(
        model,
        X,
        model.classes_
    )

    # write predictions into AnnData
    adata.obs["svm_prediction"] = pred

    for col in prob_df.columns:
        adata.obs[col] = prob_df[col].values

    return adata, model, report