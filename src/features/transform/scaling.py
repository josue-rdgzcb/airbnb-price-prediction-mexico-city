import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from src.settings.transform_config import (
    ROBUST_SCALE_FEATURES,
    MINMAX_SCALE_FEATURES,
    STANDARD_SCALE_FEATURES
)


def apply_scaling(df):
    """
    Apply feature scaling to selected numerical features.

    This function applies different scaling strategies
    based on feature groups defined in the settings module.

    Scaling methods supported:
    - RobustScaler
    - MinMaxScaler
    - StandardScaler

    Features not included in scaling groups
    remain unchanged.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled features.
    """

    df = df.copy()

    scaler = ColumnTransformer(

        transformers=[

            (
                "robust",
                RobustScaler(),
                ROBUST_SCALE_FEATURES
            ),

            (
                "minmax",
                MinMaxScaler(),
                MINMAX_SCALE_FEATURES
            ),

            (
                "standard",
                StandardScaler(),
                STANDARD_SCALE_FEATURES
            )

        ],

        remainder="passthrough",
        verbose_feature_names_out=False
    )

    scaler.set_output(transform="pandas")

    df = scaler.fit_transform(df)

    return df