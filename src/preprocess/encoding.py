import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.settings.preprocess_settings import (
    FREQUENCY_ENCODING_FEATURES,
    ONE_HOT_ENCODING_FEATURES,
    ORDINAL_ENCODING_FEATURES
)

# ==========================================================
# FIT ENCODERS
# ==========================================================

def fit_encoders(df: pd.DataFrame) -> dict:
    """
    Fit all encoders using training data only.

    This function learns:
    - Ordinal encoding static mappings (registered for serialization consistency)
    - Frequency encoding mappings
    - One-hot encoder categories

    Only features present in the input dataframe are 
    fitted and registered.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.

    Returns
    -------
    dict
        Dictionary containing fitted encoders and mappings.
    """

    encoders = {}

    # ======== Ordinal Encoding ========

    ordinal_features = {
        col: mapping
        for col, mapping in ORDINAL_ENCODING_FEATURES.items()
        if col in df.columns
    }

    encoders["ordinal"] = {
        "features": list(ordinal_features.keys()),
        "mapping": ordinal_features
    }

    # ======== Frequency Encoding ========

    frequency_features = [
        col
        for col in FREQUENCY_ENCODING_FEATURES
        if col in df.columns
    ]

    frequency_maps = {}

    for col in frequency_features:

        frequency_maps[col] = (
            df[col]
            .value_counts(normalize=True)
            .to_dict()
        )

    encoders["frequency"] = {
        "features": frequency_features,
        "mapping": frequency_maps
    }

    # ======== One Hot Encoding ========

    onehot_features = [
        col
        for col in ONE_HOT_ENCODING_FEATURES
        if col in df.columns
    ]

    if onehot_features:

        onehot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32
        )

        onehot_encoder.fit(
            df[onehot_features]
        )

        encoders["onehot"] = {
            "features": onehot_features,
            "encoder": onehot_encoder
        }

    return encoders


# ==========================================================
# TRANSFORM ENCODERS
# ==========================================================

def transform_encoders(
    df: pd.DataFrame,
    encoders: dict,
    drop_original: bool = True
) -> pd.DataFrame:
    """
    Apply all encodings using previously fitted encoders.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to transform.
    encoders : dict
        Dictionary returned by fit_encoders().
    drop_original : bool, default=True
        If True, drop original categorical columns after encoding.

    Returns
    -------
    pandas.DataFrame
        Dataframe with all encodings applied.
    """

    df = df.copy()

    # ======== Ordinal Encoding ========

    ordinal_config = encoders.get("ordinal", {})

    ordinal_maps = ordinal_config.get(
        "mapping",
        {}
    )

    for col, mapping in ordinal_maps.items():

        new_col = f"{col}_ord"

        df[new_col] = df[col].map(mapping)

        if drop_original:

            df.drop(
                columns=[col],
                inplace=True
            )

    # ======== Frequency Encoding ========

    frequency_config = encoders.get(
        "frequency",
        {}
    )

    frequency_maps = frequency_config.get(
        "mapping",
        {}
    )

    for col, mapping in frequency_maps.items():

        is_na = df[col].isna()

        new_col = f"{col}_freq"

        df[new_col] = df[col].map(mapping)

        # Unseen categories → frequency 0
        df.loc[
            ~is_na & df[new_col].isna(),
            new_col
        ] = 0.0

        if drop_original:

            df.drop(
                columns=[col],
                inplace=True
            )

    # ======== One Hot Encoding ========

    onehot_config = encoders.get(
        "onehot"
    )

    if onehot_config:

        features = onehot_config["features"]

        encoder = onehot_config["encoder"]

        encoded_array = encoder.transform(
            df[features]
        )

        feature_names = (
            encoder.get_feature_names_out(
                features
            )
        )

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=df.index
        )

        if drop_original:

            df = pd.concat(
                [
                    df.drop(columns=features),
                    encoded_df
                ],
                axis=1
            )

        else:

            df = pd.concat(
                [
                    df,
                    encoded_df
                ],
                axis=1
            )

    return df
