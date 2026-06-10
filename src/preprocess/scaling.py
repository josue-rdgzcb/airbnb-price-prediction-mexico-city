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
def transform_scalers(
    df: pd.DataFrame,
    scalers: dict
) -> pd.DataFrame:
    """
    Apply previously fitted scalers.

    Parameters
    ----------
    df : pandas.DataFrame

    scalers : dict
        Dictionary returned by fit_scalers().

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    # ======== Standard Scaling ========

    if STANDARD_SCALE_FEATURES:

        df[STANDARD_SCALE_FEATURES] = (
            scalers["standard"]
            .transform(
                df[STANDARD_SCALE_FEATURES]
            )
        )

    # ======== MinMax Scaling ========

    if MINMAX_SCALE_FEATURES:

        df[MINMAX_SCALE_FEATURES] = (
            scalers["minmax"]
            .transform(
                df[MINMAX_SCALE_FEATURES]
            )
        )

    # ======== Robust Scaling ========

    if ROBUST_SCALE_FEATURES:

        df[ROBUST_SCALE_FEATURES] = (
            scalers["robust"]
            .transform(
                df[ROBUST_SCALE_FEATURES]
            )
        )

    return df