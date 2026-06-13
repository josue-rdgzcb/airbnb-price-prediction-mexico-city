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
# Store fitted imputers and associated feature groups
def fit_imputers(df: pd.DataFrame) -> dict:
    """
    Fit all imputers using the training dataset.

    This function learns the imputation statistics
    (mean, median, mode, constant values) from the
    training data only.

    Only features present in the input dataframe are 
    fitted and registered.

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

    imputers = {}

    # ======== Mean Imputation ========

    mean_features = [
        col
        for col in MEAN_IMPUTE_FEATURES
        if col in df.columns
    ]

    if mean_features:

        mean_imputer = SimpleImputer(
            strategy="mean"
        )

        mean_imputer.fit(
            df[mean_features]
        )

        imputers["mean"] = {
            "features": mean_features,
            "imputer": mean_imputer
        }

    # ======== Median Imputation ========

    median_features = [
        col
        for col in MEDIAN_IMPUTE_FEATURES
        if col in df.columns
    ]

    if median_features:

        median_imputer = SimpleImputer(
            strategy="median"
        )

        median_imputer.fit(
            df[median_features]
        )

        imputers["median"] = {
            "features": median_features,
            "imputer": median_imputer
        }

    # ======== Most Frequent (Numeric) ========

    numeric_features = [
        col
        for col in MOST_FREQUENT_NUMERIC_FEATURES
        if col in df.columns
    ]

    if numeric_features:

        numeric_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        numeric_imputer.fit(
            df[numeric_features]
        )

        imputers["most_frequent_numeric"] = {
            "features": numeric_features,
            "imputer": numeric_imputer
        }

    # ======== Most Frequent (Categorical) ========

    categoric_features = [
        col
        for col in MOST_FREQUENT_CATEGORIC_FEATURES
        if col in df.columns
    ]

    if categoric_features:

        categoric_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        categoric_imputer.fit(
            df[categoric_features]
        )

        imputers["most_frequent_categoric"] = {
            "features": categoric_features,
            "imputer": categoric_imputer
        }

    # ======== Missing Category ========

    missing_category_features = [
        col
        for col in MISSING_CATEGORY_IMPUTE_FEATURES
        if col in df.columns
    ]

    if missing_category_features:

        missing_category_imputer = SimpleImputer(
            strategy="constant",
            fill_value="missing"
        )

        missing_category_imputer.fit(
            df[missing_category_features]
        )

        imputers["missing_category"] = {
            "features": missing_category_features,
            "imputer": missing_category_imputer
        }

    return imputers


# ==========================================================
# TRANSFORM IMPUTERS
# ==========================================================

# Apply imputers (inference step): transform dataset using learned statistics
# Replace missing values using fitted imputers
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
        Dictionary returned by fit_imputers().

    Returns
    -------
    pandas.DataFrame
        Imputed dataframe.
    """

    df = df.copy()

    # Apply each fitted imputer
    for config in imputers.values():

        features = config["features"]

        imputer = config["imputer"]

        df[features] = imputer.transform(
            df[features]
        )

    return df


