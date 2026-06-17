from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split a dataset into training, validation, and test sets.

    The split is performed in two stages:
    1. Separate the training set from a temporary holdout set.
    2. Split the temporary holdout set into validation and test sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    test_size : float, default=0.15
        Proportion of observations assigned to the test set.
    validation_size : float, default=0.15
        Proportion of observations assigned to the validation set.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : bool, default False
        If True, data is split in a stratified fashion, using y as class labels.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    # Validate split proportions
    total_holdout = test_size + validation_size

    if total_holdout >= 1:
        raise ValueError(
            "The sum of test_size and validation_size must be less than 1."
        )

    # Determine stratification target
    stratify_y = y if stratify else None

    # First split: Separate training data from holdout data
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=total_holdout,
        random_state=random_state,
        stratify=stratify_y,
    )

    # Compute relative test size inside the holdout set
    relative_test_size = test_size / total_holdout
    stratify_holdout = y_holdout if stratify else None

    # Second split: Divide holdout data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=stratify_holdout,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test