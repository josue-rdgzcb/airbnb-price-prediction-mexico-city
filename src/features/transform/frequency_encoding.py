from src.settings.transform_config import FREQUENCY_FEATURES

def apply_frequency_encoding(
    df,
    suffix="_freq",
    drop_original=True
):
    """
    Apply frequency encoding to categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame

    suffix : str, default="_freq"

    drop_original : bool, default=True

    Returns
    -------
    pandas.DataFrame
    """
    
    if not FREQUENCY_FEATURES:
        return df

    for col in FREQUENCY_FEATURES:

        freq_map = (
            df[col]
            .value_counts(normalize=True)
        )

        df[f"{col}{suffix}"] = (
            df[col]
            .map(freq_map)
        )

    if drop_original:
        df = df.drop(columns=FREQUENCY_FEATURES)

    return df