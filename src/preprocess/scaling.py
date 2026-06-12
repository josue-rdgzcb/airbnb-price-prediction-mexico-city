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
# Store strategy and fitted scaler objects for each feature group
def fit_scalers(df: pd.DataFrame) -> dict:
    """
    Fit all scalers using training data only.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Dictionary containing fitted scalers.
    """

    scalers = {}

    # ======== Standard Scaler ========

    if STANDARD_SCALE_FEATURES:

        standard_scaler = StandardScaler()

        standard_scaler.fit(
            df[STANDARD_SCALE_FEATURES]
        )

        scalers["standard"] = standard_scaler

    # ======== MinMax Scaler ========

    if MINMAX_SCALE_FEATURES:

        minmax_scaler = MinMaxScaler()

        minmax_scaler.fit(
            df[MINMAX_SCALE_FEATURES]
        )

        scalers["minmax"] = minmax_scaler

    # ======== Robust Scaler ========

    if ROBUST_SCALE_FEATURES:

        robust_scaler = RobustScaler()

        robust_scaler.fit(
            df[ROBUST_SCALE_FEATURES]
        )

        scalers["robust"] = robust_scaler

    return scalers

# ==========================================================
# TRANSFORM SCALERS
# ==========================================================

# Apply scalers (inference step): transform dataset using learned scaling parameters
# Replace original values with scaled versions for each feature group
import pandas as pd


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

    drop_original : bool, default True
        If True, replaces the original columns with the scaled values.
        If False, keeps original columns and creates new ones with specific suffixes.

    Returns
    -------
    pandas.DataFrame
        Dataframe with scaled features.
    """

    df = df.copy()

    # ======== Standard Scaling ========

    if STANDARD_SCALE_FEATURES:
        scaled_values = scalers["standard"].transform(
            df[STANDARD_SCALE_FEATURES]
        )

        if drop_original:
            df[STANDARD_SCALE_FEATURES] = scaled_values
        else:
            new_features = [f"{col}_std" for col in STANDARD_SCALE_FEATURES]
            df[new_features] = scaled_values

    # ======== MinMax Scaling ========

    if MINMAX_SCALE_FEATURES:
        scaled_values = scalers["minmax"].transform(df[MINMAX_SCALE_FEATURES])

        if drop_original:
            df[MINMAX_SCALE_FEATURES] = scaled_values
        else:
            new_features = [
                f"{col}_minmax" for col in MINMAX_SCALE_FEATURES
            ]
            df[new_features] = scaled_values

    # ======== Robust Scaling ========

    if ROBUST_SCALE_FEATURES:
        scaled_values = scalers["robust"].transform(df[ROBUST_SCALE_FEATURES])

        if drop_original:
            df[ROBUST_SCALE_FEATURES] = scaled_values
        else:
            new_features = [
                f"{col}_robust" for col in ROBUST_SCALE_FEATURES
            ]
            df[new_features] = scaled_values

    return df
