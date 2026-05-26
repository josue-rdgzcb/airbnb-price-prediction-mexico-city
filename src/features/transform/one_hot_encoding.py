import pandas as pd
from src.settings.transform_config import ONEHOT_FEATURES

def apply_onehot_encoding(
    df,
    drop_original=True
):
    """
    Apply one-hot encoding to categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame

    drop_original : bool, default=True
        Whether to drop original columns.

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()
    
    if not ONEHOT_FEATURES:
        return df

    encoded = pd.get_dummies(
        df[ONEHOT_FEATURES],
        prefix=ONEHOT_FEATURES,
        drop_first=False,
        dtype=int
    )

    df = pd.concat(
        [df, encoded],
        axis=1
    )

    if drop_original:
        df = df.drop(columns=ONEHOT_FEATURES)

    return df