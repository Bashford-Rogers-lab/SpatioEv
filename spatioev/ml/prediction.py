import pandas as pd


def predict_svm(
    model,
    X,
    classes
):
    """
    Predict phenotypes and probabilities.
    """

    probs = model.predict_proba(X)

    pred = model.predict(X)

    prob_df = pd.DataFrame(
        probs,
        columns=[f"svm_prob_{c}" for c in classes]
    )

    return pred, prob_df