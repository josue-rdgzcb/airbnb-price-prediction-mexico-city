import pandas as pd 
import ast


# =========== PIPELINE FUNCTIONS =======================

from src.settings.preprocess_settings import (
    PERCENTAGE_COLUMNS,
    BOOLEAN_COLUMNS
)


# Normalize percentage columns (e.g., "85%") into numeric values between 0 and 1
def normalize_percentage_columns(df):
    """
    Convert percentage string columns to float.

    Example
    -------
    "87%" -> 0.87

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """
    for col in PERCENTAGE_COLUMNS:

        df[col] = (
            df[col]
            .str.replace("%", "", regex=False)
            .astype(float)
            / 100
        )

    return df


# Convert string-based boolean columns ("t"/"f") into proper boolean dtype
def convert_boolean_columns(df):
    """
    Convert string boolean columns to bool dtype.

    Example
    -------
    "t" -> True
    "f" -> False

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    bool_mapping = {
        "t": True,
        "f": False
    }

    for col in BOOLEAN_COLUMNS:

        df[col] = df[col].map(bool_mapping).astype(bool)

    return df


# =========== HELPER FUNCTIONS =======================

# Normalize categorical string values (lowercase + replace spaces with underscores)
def normalize_string_column(series):
    """
    Normalize a pandas Series of strings:
    - Convert to lowercase
    - Replace spaces with underscores
    """
    return series.str.lower().str.replace(" ", "_")

# Parse column into Python lists
def parse_column(x):
    """
    Parse a stringified host verifications value into a Python list.
    - Return [] if value is NaN or cannot be parsed.
    """
    if pd.isna(x):
        return []
    
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []
    

    