 
import numpy as np

from src.settings.transform_config import LOG_FEATURES


def apply_log_transformations(
    df,
    suffix="_log",
    drop_original=True
):
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
        df[f"{col}{suffix}"] = np.log1p(df[col])

    # Drop original feature
    if drop_original:
        df = df.drop(columns=LOG_FEATURES)
    
    return df