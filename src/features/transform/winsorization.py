import pandas as pd

from src.settings.transform_config import (WINSOR_FEATURES)


def apply_winsorization(
    df,
    suffix="_winsor",
    drop_original = True
):
    """
    Apply upper-tail winsorization.

    Parameters
    ----------
    df : pandas.DataFrame

    suffix : str, default="_winsor"

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    for col, config in WINSOR_FEATURES.items():

        if not WINSOR_FEATURES:
            return df

        lower_q = config.get("lower", 0.00)
        upper_q = config.get("upper", 0.99)

        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)

        df[f"{col}{suffix}"] = df[col].clip(
            lower,
            upper
        )

    # Drop original feature
    if drop_original:
        df = df.drop(columns=WINSOR_FEATURES)

    return df