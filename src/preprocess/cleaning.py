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

    if not PERCENTAGE_COLUMNS:
        return df

    df = df.copy()

    for col in PERCENTAGE_COLUMNS:

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
    This function maps text indicators into 1.0 and 0.0.

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
        Dataframe with converted numeric binary columns.
    """

    if not BINARY_COLUMNS:
        return df

    df = df.copy()

    # Mapeo numérico que preserva la compatibilidad con NaN
    binary_mapping = {
        "t": 1.0,
        "f": 0.0
    }

    for col in BINARY_COLUMNS:
        df[col] = df[col].map(binary_mapping)

    return df


# Normalize categorical string values (lowercase + replace spaces with underscores)
def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string values by converting to lowercase
    and replacing spaces with underscores.

    Example
    -------
    "Entire Home" -> "entire_home"

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.Series
    """

    if not STRING_COLUMNS:
        return df

    df = df.copy()
    for col in STRING_COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
        )
    return df

# Parse column into Python lists
def parse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a string representation of a Python list
    into an actual Python list.

    Example
    -------
    "['email', 'phone']"
        ->
    ['email', 'phone']

    Invalid or missing values return an empty list.

    Parameters
    ----------
    value : Any

    Returns
    -------
    list
    """
    if not PARSE_COLUMNS:
        return df

    df = df.copy()
    for col in PARSE_COLUMNS:
        df[col] = df[col].apply(
            lambda val: [] if pd.isna(val) else (
                ast.literal_eval(val) if isinstance(val, str) else []
            )
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