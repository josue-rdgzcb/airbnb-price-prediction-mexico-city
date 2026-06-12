from sklearn.impute import SimpleImputer
import pandas as pd

from src.settings.preprocess_settings import (
    MEAN_IMPUTE_FEATURES,
    MEDIAN_IMPUTE_FEATURES,
    MOST_FREQUENT_NUMERIC_FEATURES,
    MOST_FREQUENT_CATEGORIC_FEATURES,
    MISSING_CATEGORY_IMPUTE_FEATURES
)

# ==========================================================
# FIT IMPUTERS
# ==========================================================

# Fit imputers (training step): learn statistics from training data
# Store strategy and fitted imputers for each feature group
def fit_imputers(df: pd.DataFrame) -> dict:
    """
    Fit all imputers using the training dataset.

    This function learns the imputation statistics
    (mean, median, mode, constant values) from the
    training data only.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    dict
        Dictionary containing fitted imputers and
        their associated feature groups.
    """

    imputers = {

        "mean": {
            "features": MEAN_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="mean")
        },

        "median": {
            "features": MEDIAN_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="median")
        },

        "most_frequent_numeric": {
            "features": MOST_FREQUENT_NUMERIC_FEATURES,
            "imputer": SimpleImputer(strategy="most_frequent")
        },

        "most_frequent_categoric": {
            "features": MOST_FREQUENT_CATEGORIC_FEATURES,
            "imputer": SimpleImputer(strategy="most_frequent")
        },

        "missing_category": {
            "features": MISSING_CATEGORY_IMPUTE_FEATURES,
            "imputer": SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )
        }
    }

    # Fit each imputer on training data
    for config in imputers.values():

        features = config["features"]

        if not features:
            continue

        config["imputer"].fit(df[features])

    return imputers

# ==========================================================
# TRANSFORM IMPUTERS
# ==========================================================

# Apply imputers (inference step): transform dataset using learned statistics
# Replace missing values in each feature group with fitted imputers
def transform_imputers(
    df: pd.DataFrame,
    imputers: dict
) -> pd.DataFrame:
    """
    Apply previously fitted imputers to a dataset.

    This function uses imputation statistics learned
    from the training data and applies them to any
    dataset (train, validation, test, or inference).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to transform.

    imputers : dict
        Dictionary containing fitted imputers.

    Returns
    -------
    pandas.DataFrame
        Imputed dataframe.
    """

    df = df.copy()

    # Apply each fitted imputer
    for config in imputers.values():

        features = config["features"]

        if not features:
            continue

        df[features] = config["imputer"].transform(
            df[features]
        )

    return df


