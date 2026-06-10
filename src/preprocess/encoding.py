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

    # Register the static mapping so that the encoder dictionary is 
    # self-contained and sufficient for production.
    encoders["ordinal"] = ORDINAL_ENCODING_FEATURES or {}

    # ======== Frequency Encoding ======== 

    frequency_maps = {}

    if FREQUENCY_ENCODING_FEATURES:
        for col in FREQUENCY_ENCODING_FEATURES:
            frequency_maps[col] = (
                df[col]
                .value_counts(normalize=True)
                .to_dict()
            )

    encoders["frequency"] = frequency_maps

    # ======== One Hot Encoding ========

    if ONE_HOT_ENCODING_FEATURES:
        onehot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32
        )

        onehot_encoder.fit(df[ONE_HOT_ENCODING_FEATURES])

        encoders["onehot"] = onehot_encoder

    else:
        encoders["onehot"] = None

    return encoders


# ==========================================================
# TRANSFORM ENCODERS
# ==========================================================

def transform_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Apply all encodings using previously fitted encoders.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to transform.
    encoders : dict
        Dictionary returned by fit_encoders().

    Returns
    -------
    pandas.DataFrame
        Dataframe with all encodings applied.
    """
    df = df.copy()

    # ======== Ordinal Encoding ======== 
    
    ordinal_maps = encoders.get("ordinal", {})

    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            new_col = f"{col}_ord"
            df[new_col] = df[col].map(mapping)

    # ======== Frequency Encoding ======== 

    frequency_maps = encoders.get("frequency", {})

    for col, mapping in frequency_maps.items():
        if col in df.columns:
            # Preserve NaNs, unknown categories → 0.0
            is_na = df[col].isna()
            new_col = f"{col}_freq"
            df[new_col] = df[col].map(mapping)
            df.loc[~is_na & df[new_col].isna(), new_col] = 0.0

    # ======== One Hot Encoding ======== 
    
    onehot_encoder = encoders.get("onehot")

    if ONE_HOT_ENCODING_FEATURES and onehot_encoder is not None:
        encoded_array = onehot_encoder.transform(df[ONE_HOT_ENCODING_FEATURES])

        # Generate the coded column names
        feature_names = onehot_encoder.get_feature_names_out(ONE_HOT_ENCODING_FEATURES)

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=df.index
        )

        # Delete the original columns
        df = pd.concat(
            [df.drop(columns=ONE_HOT_ENCODING_FEATURES), encoded_df],
            axis=1
        )

    return df
