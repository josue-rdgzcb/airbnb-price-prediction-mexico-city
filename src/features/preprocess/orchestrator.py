
from src.features.preprocess.preprocess_utils import (
    normalize_percentage_columns,
    convert_boolean_columns
)


def preprocess_features(df):

    df = df.copy()

    df = normalize_percentage_columns(df)

    df = convert_boolean_columns(df)

    return df