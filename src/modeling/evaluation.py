from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from sklearn.base import RegressorMixin

import pandas as pd

from typing import Callable


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(
    model: RegressorMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
    target_transform: Callable | None = None,
    verbose: bool = True
) -> dict:
    """
    Evaluate model performance on training and validation sets.

    Metrics:
    - R²
    - RMSE
    - MAE

    Optionally computes metrics in the original target scale
    if a reverse transformation function is provided.

    Parameters
    ----------
    model : RegressorMixin
        Trained model.

    X_train : pd.DataFrame
        Training features.

    y_train : pd.Series
        Training target.

    X_val : pd.DataFrame
        Validation features.

    y_val : pd.Series
        Validation target.

    model_name : str
        Model name for reporting.

    target_transform : callable, optional
        Function used to reverse target transformation
        (e.g. np.expm1).

    verbose : bool, default=True
        Whether to print formatted results.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """

    # Generate predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    results = {

        "train_r2": r2_score(y_train, train_pred),
        "train_rmse": root_mean_squared_error(y_train, train_pred),
        "train_mae": mean_absolute_error(y_train, train_pred),

        "val_r2": r2_score(y_val, val_pred),
        "val_rmse": root_mean_squared_error(y_val, val_pred),
        "val_mae": mean_absolute_error(y_val, val_pred)
    }

    # Metrics in original target scale
    if target_transform is not None:

        y_val_real = target_transform(y_val)
        val_pred_real = target_transform(val_pred)

        results["val_rmse_real"] = root_mean_squared_error(
            y_val_real,
            val_pred_real
        )

        results["val_mae_real"] = mean_absolute_error(
            y_val_real,
            val_pred_real
        )

    # Print report
    if verbose:

        print("\n" + "=" * 50)
        print(f"{model_name.upper()} PERFORMANCE")
        print("=" * 50)

        print("\nTRAINING SET")
        print(f"R²:   {results['train_r2']:.4f}")
        print(f"RMSE: {results['train_rmse']:.4f}")
        print(f"MAE:  {results['train_mae']:.4f}")

        print("\nVALIDATION SET")
        print(f"R²:   {results['val_r2']:.4f}")
        print(f"RMSE: {results['val_rmse']:.4f}")
        print(f"MAE:  {results['val_mae']:.4f}")

        if target_transform is not None:

            print("\nREAL PRICE METRICS (VALIDATION)")
            print(
                f"RMSE ($): ${results['val_rmse_real']:,.2f}"
            )
            print(
                f"MAE ($):  ${results['val_mae_real']:,.2f}"
            )

    return results