from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import RegressorMixin

import pandas as pd


def tune_model(
    model: RegressorMixin,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
    scoring: str = "r2",
    random_state: int = 42,
    n_jobs: int = -1
) -> RandomizedSearchCV:
    """
    Perform hyperparameter optimization using RandomizedSearchCV.

    Parameters
    ----------
    model : RegressorMixin
        Base estimator.

    param_distributions : dict
        Hyperparameter search space.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    n_iter : int, default=50
        Number of parameter combinations sampled.

    cv : int, default=5
        Number of cross-validation folds.

    scoring : str, default="r2"
        Evaluation metric.

    random_state : int, default=42
        Random seed.

    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    RandomizedSearchCV
        Fitted RandomizedSearchCV object.
    """

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )

    search.fit(X_train, y_train)

    return search