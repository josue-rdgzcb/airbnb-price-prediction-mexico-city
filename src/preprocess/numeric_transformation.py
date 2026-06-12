 
import numpy as np
import pandas as pd

from src.settings.preprocess_settings import (
    LOG_FEATURES,
    BINNING_FEATURES
)

# =============== LOG TRANSFORMATION ===============

def log_transformation(
    df: pd.DataFrame,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Apply log1p transformation to selected columns.

    Parameters
    ----------
    df : pandas.DataFrame

    suffix : str, default="_log"
        Suffix for transformed columns.

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    if not LOG_FEATURES:
            return df

    for col in LOG_FEATURES:
        # Create transformed feature
        df[f"{col}_log"] = np.log1p(df[col])

    # Drop original feature
    if drop_original:
        df = df.drop(columns=LOG_FEATURES)
    
    return df


# =============== BINNING ===============

# Learn bin edges (training step) from dataset
# Store method and edges for each feature in dictionary
def fit_binners(df: pd.DataFrame) -> dict:
    """
    Learn bin edges from training data.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    dict
        Dictionary containing learned bin edges
        for each feature.
    """

    binners = {}

    for col, config in BINNING_FEATURES.items():

        method = config.get("method", "qcut")
        bins = config.get("bins", 4)

        # Quantile binning
        if method == "qcut":

            _, bin_edges = pd.qcut(
                df[col],
                q=bins,
                retbins=True,
                duplicates="drop"
            )

        # Equal-width binning
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
            "method": method,
            "bin_edges": bin_edges
        }

    return binners

# Apply previously learned bin edges to dataset (inference step)
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
        Dictionary containing fitted bin edges.

    suffix : str, default="_binned"
        Suffix appended to new binned columns.

    drop_original : bool, default=True
        Whether to drop original columns after binning.

    Returns
    -------
    pandas.DataFrame
        Transformed dataset with binned features.
    """

    df = df.copy()

    # Iterate over binners and apply binning to each feature
    for col, config in binners.items():
        new_col = f"{col}_binned"
        df[new_col] = pd.cut(
            df[col],
            bins=config["bin_edges"],
            include_lowest=True
        )
        
    # Drop original feature
    if drop_original:
        df = df.drop(columns=list(binners.keys()))

    return df


