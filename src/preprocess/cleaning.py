import ast
import pandas as pd

from src.settings.preprocess_settings import (
    PERCENTAGE_COLUMNS,
    BINARY_COLUMNS,
    PARSE_COLUMNS,
    STRING_COLUMNS

)


# ==========================================================
# CLEANING FUNCTIONS
# ==========================================================

# Normalize percentage columns (e.g., "85%") into numeric values between 0 and 1
def normalize_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert percentage string columns into numeric values.

    Only features present in the input dataframe are
    processed.

    Example
    -------
    "87%" -> 0.87

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized percentage columns.
    """

    df = df.copy()

    available_features = [
        col
        for col in PERCENTAGE_COLUMNS
        if col in df.columns
    ]

    if not available_features:
        return df

    for col in available_features:

        df[col] = (
            pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False),
                errors="coerce"
            )
            / 100
        )

    return df

# Convert string-based indicators ("t"/"f") into numeric binary values (1.0/0.0)
def convert_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert indicators ("t"/"f") into numeric binary values.

    Only features present in the input dataframe are
    processed.

    Example
    -------
    "t" -> 1.0
    "f" -> 0.0
    NaN -> NaN

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with converted binary columns.
    """

    df = df.copy()

    available_features = [
        col
        for col in BINARY_COLUMNS
        if col in df.columns
    ]

    if not available_features:
        return df

    binary_mapping = {
        "t": 1.0,
        "f": 0.0
    }

    for col in available_features:

        df[col] = df[col].map(
            binary_mapping
        )

    return df


# Normalize categorical string values (lowercase + replace spaces with underscores)
def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string values by converting them to lowercase
    and replacing spaces with underscores.

    Only features present in the input dataframe are
    processed.

    Example
    -------
    "Entire Home" -> "entire_home"

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized string columns.
    """

    df = df.copy()

    available_features = [
        col
        for col in STRING_COLUMNS
        if col in df.columns
    ]

    if not available_features:
        return df

    for col in available_features:

        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(
                " ",
                "_",
                regex=False
            )
        )

    return df

# Parse stringified lists into Python lists
def parse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string representations of Python lists
    into actual Python lists.

    Only features present in the input dataframe are
    processed.

    Example
    -------
    "['email', 'phone']"
        ->
    ['email', 'phone']

    Invalid or missing values return an empty list.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with parsed list columns.
    """

    df = df.copy()

    available_features = [
        col
        for col in PARSE_COLUMNS
        if col in df.columns
    ]

    if not available_features:
        return df

    for col in available_features:

        def safe_parse(value):

            if pd.isna(value):
                return []

            if not isinstance(value, str):
                return []

            try:
                return ast.literal_eval(value)

            except (
                ValueError,
                SyntaxError
            ):
                return []

        df[col] = df[col].apply(
            safe_parse
        )

    return df


# ==========================================================
# CLEANING ORCHESTRATOR
# ==========================================================

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all cleaning steps required before
    feature engineering.

    Steps
    -----
    1. Normalize percentage columns.
    2. Convert boolean columns.
    3. Normalize string columns.
    4. Parse columns

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    df = normalize_percentage_columns(df)
    df = convert_binary_columns(df)
    df = normalize_string_columns(df)
    df = parse_columns(df)

    return df