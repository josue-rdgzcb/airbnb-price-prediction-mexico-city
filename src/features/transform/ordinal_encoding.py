from src.settings.transform_config import (
    ORDINAL_FEATURES,
    ORDINAL_MAPPINGS
)

def apply_ordinal_encoding(
    df,
    suffix="_ordinal",
    drop_original=True
):
    """
    Apply ordinal encoding to categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame

    suffix : str, default="_ordinal"

    drop_original : bool, default=True

    Returns
    -------
    pandas.DataFrame
    """

    df = df.copy()

    if not ORDINAL_FEATURES:
        return df

    for col in ORDINAL_FEATURES:

        mapping = ORDINAL_MAPPINGS[col]

        df[f"{col}{suffix}"] = (
            df[col]
            .map(mapping)
            .astype("int64")
        )

    if drop_original:
        df = df.drop(columns=ORDINAL_FEATURES)

    return df