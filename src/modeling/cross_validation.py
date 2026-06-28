from sklearn.model_selection import cross_val_score
from sklearn.base import RegressorMixin

import pandas as pd
import numpy as np


# ==========================================================
# CROSS VALIDATION
# ==========================================================

def cross_validate_model(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "r2",
    verbose: bool = True
) -> dict:
    """
    Perform k-fold cross-validation.

    Parameters
    ----------
    model : RegressorMixin
        Scikit-learn compatible regression model.

    X : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.

    cv : int, default=5
        Number of folds.

    scoring : str, default="r2"
        Evaluation metric.

    verbose : bool, default=True
        Whether to print results.

    Returns
    -------
    dict
        Cross-validation results and summary statistics.
    """

    # Execute cross validation
    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results = {
        "scores": scores,
        "mean": np.mean(scores),
        "std": np.std(scores),
        "min": np.min(scores),
        "max": np.max(scores)
    }

    if verbose:

        print("\n" + "=" * 50)
        print("FOLD CROSS VALIDATION RESULTS")
        print("=" * 50)

        for i, score in enumerate(scores, start=1):
            print(
                f"Fold {i}: {score:.4f}"
            )

        metric_name = scoring.upper()

        print()
        print(
            f"Mean {metric_name}: {results['mean']:.4f}"
        )
        print(
            f"Std {metric_name}:  {results['std']:.4f}"
        )

    return results