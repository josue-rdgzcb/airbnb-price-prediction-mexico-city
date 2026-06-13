from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from src.settings.preprocess_settings import (
    STANDARD_SCALE_FEATURES,
    MINMAX_SCALE_FEATURES,
    ROBUST_SCALE_FEATURES
)

import pandas as pd

# ==========================================================
# FIT SCALERS
# ==========================================================

# Fit scalers (training step): learn scaling parameters from training data
# Store fitted scaler objects and associated feature groups
def fit_scalers(df: pd.DataFrame) -> dict:
    """
    Fit all scalers using training data only.

    Only features present in the input dataframe are 
    fitted and registered.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    dict
        Dictionary containing fitted scalers and
        their associated feature groups.
    """

    scalers = {}

    # ======== Standard Scaler ========

    standard_features = [
        col
        for col in STANDARD_SCALE_FEATURES
        if col in df.columns
    ]

    if standard_features:

        standard_scaler = StandardScaler()

        standard_scaler.fit(
            df[standard_features]
        )

        scalers["standard"] = {
            "features": standard_features,
            "scaler": standard_scaler
        }

    # ======== MinMax Scaler ========

    minmax_features = [
        col
        for col in MINMAX_SCALE_FEATURES
        if col in df.columns
    ]

    if minmax_features:

        minmax_scaler = MinMaxScaler()

        minmax_scaler.fit(
            df[minmax_features]
        )

        scalers["minmax"] = {
            "features": minmax_features,
            "scaler": minmax_scaler
        }

    # ======== Robust Scaler ========

    robust_features = [
        col
        for col in ROBUST_SCALE_FEATURES
        if col in df.columns
    ]

    if robust_features:

        robust_scaler = RobustScaler()

        robust_scaler.fit(
            df[robust_features]
        )

        scalers["robust"] = {
            "features": robust_features,
            "scaler": robust_scaler
        }

    return scalers


# ==========================================================
# TRANSFORM SCALERS
# ==========================================================

# Apply scalers (inference step): transform dataset using learned scaling parameters
# Replace original values with scaled versions for each feature group
def transform_scalers(
    df: pd.DataFrame,
    scalers: dict,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Apply previously fitted scalers.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to transform.

    scalers : dict
        Dictionary returned by fit_scalers().

    drop_original : bool, default=True
        If True, replaces original columns with scaled values.
        If False, keeps original columns and creates new scaled columns.

    Returns
    -------
    pandas.DataFrame
        Dataframe with scaled features.
    """

    df = df.copy()

    # ======== Standard Scaling ========

    if "standard" in scalers:

        features = scalers["standard"]["features"]

        scaler = scalers["standard"]["scaler"]

        scaled_values = scaler.transform(
            df[features]
        )

        if drop_original:

            df[features] = scaled_values

        else:

            new_features = [
                f"{col}_std"
                for col in features
            ]

            df[new_features] = scaled_values

    # ======== MinMax Scaling ========

    if "minmax" in scalers:

        features = scalers["minmax"]["features"]

        scaler = scalers["minmax"]["scaler"]

        scaled_values = scaler.transform(
            df[features]
        )

        if drop_original:

            df[features] = scaled_values

        else:

            new_features = [
                f"{col}_minmax"
                for col in features
            ]

            df[new_features] = scaled_values

    # ======== Robust Scaling ========

    if "robust" in scalers:

        features = scalers["robust"]["features"]

        scaler = scalers["robust"]["scaler"]

        scaled_values = scaler.transform(
            df[features]
        )

        if drop_original:

            df[features] = scaled_values

        else:

            new_features = [
                f"{col}_robust"
                for col in features
            ]

            df[new_features] = scaled_values

    return df
