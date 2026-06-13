 
import numpy as np
import pandas as pd

from src.settings.preprocess_settings import (
    LOG_FEATURES,
    BINNING_FEATURES
)

# ==========================================================
# LOG TRANSFORMATION
# ==========================================================

def log_transformation(
    df: pd.DataFrame,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Apply log1p transformation to selected numeric features.

    Only features present in the input dataframe are
    transformed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    drop_original : bool, default=True
        If True, original features are removed after
        transformation.
        If False, original features are preserved and
        transformed features are appended with the "_log"
        suffix.

    Returns
    -------
    pandas.DataFrame
        Dataframe with log-transformed features.
    """

    df = df.copy()

    available_features = [
        col
        for col in LOG_FEATURES
        if col in df.columns
    ]

    if not available_features:
        return df

    # Create transformed features
    for col in available_features:

        df[f"{col}_log"] = np.log1p(
            df[col]
        )

    # Remove original features
    if drop_original:

        df = df.drop(
            columns=available_features
        )

    return df


# ==========================================================
# BINNING
# ==========================================================

# Learn bin edges (training step) from dataset
# Store method and learned bin edges for each feature
def fit_binners(df: pd.DataFrame) -> dict:
    """
    Learn bin edges from training data.

    Only features present in the input dataframe are
    fitted and registered.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    dict
        Dictionary containing fitted binning
        configurations for each feature.
    """

    binners = {}

    for col, config in BINNING_FEATURES.items():

        # Skip features that are not present
        if col not in df.columns:
            continue

        method = config.get("method", "qcut")
        bins = config.get("bins", 4)

        # ======== Quantile Binning ========

        if method == "qcut":

            _, bin_edges = pd.qcut(
                df[col],
                q=bins,
                retbins=True,
                duplicates="drop"
            )

        # ======== Equal-Width Binning ========

        elif method == "cut":

            _, bin_edges = pd.cut(
                df[col],
                bins=bins,
                retbins=True
            )

        else:

            raise ValueError(
                f"Invalid method '{method}' "
                f"for column '{col}'"
            )

        binners[col] = {
            "features": [col],
            "method": method,
            "bin_edges": bin_edges
        }

    return binners


# ==========================================================
# TRANSFORM BINNERS
# ==========================================================

# Apply previously learned bin edges to dataset
# Create new binned columns and optionally drop originals
def transform_binners(
    df: pd.DataFrame,
    binners: dict,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Apply previously learned bin edges.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to transform.

    binners : dict
        Dictionary returned by fit_binners().

    drop_original : bool, default=True
        Whether to drop original columns after binning.

    Returns
    -------
    pandas.DataFrame
        Dataset with binned features.
    """

    df = df.copy()

    features_to_drop = []

    # Apply fitted binning configuration
    for config in binners.values():

        col = config["features"][0]

        new_col = f"{col}_binned"

        df[new_col] = pd.cut(
            df[col],
            bins=config["bin_edges"],
            include_lowest=True
        )

        features_to_drop.extend(
            config["features"]
        )

    # Remove original features if requested
    if drop_original:

        df = df.drop(
            columns=features_to_drop
        )

    return df


