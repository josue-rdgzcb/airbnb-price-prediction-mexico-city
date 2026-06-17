from sklearn.base import RegressorMixin
import pandas as pd


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model: RegressorMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RegressorMixin:
    """
    Train a regression model using the training dataset.

    Parameters
    ----------
    model : RegressorMixin
        Scikit-learn compatible regression model.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    Returns
    -------
    RegressorMixin
        Fitted model.
    """

    # Fit model on training data
    model.fit(
        X_train,
        y_train
    )

    return model