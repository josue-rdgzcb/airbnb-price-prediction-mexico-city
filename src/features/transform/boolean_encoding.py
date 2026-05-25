
from src.settings.transform_config import BOOLEAN_FEATURES

def apply_boolean_encoding(df):
    """
    Convert boolean columns to integer encoding.

    True  -> 1
    False -> 0
    """

    for col in BOOLEAN_FEATURES:

        df[col] = df[col].astype(int)

    return df