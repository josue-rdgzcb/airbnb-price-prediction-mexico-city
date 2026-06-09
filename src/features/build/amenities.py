import re
import pandas as pd

# Import amenity parameters from settings
from src.settings.features_params import (
    AMENITIES_SCORE,
    AMENITIES_WEIGHTS
)

# Clean individual amenity string
def _clean_amenity(a: str) -> str:
    """
    Clean individual amenity string:
    - Lowercase
    - Remove non-alphanumeric characters except spaces and hyphens
    - Strip leading/trailing spaces
    """
    return re.sub(r'[^a-z0-9\s+\-]', '', a.lower()).strip()


# Feature: count number of amenities per listing
def add_amenity_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a feature column 'amenity_count' with the number of amenities per listing.
    """
    df = df.copy()
    df["amenity_count"] = df["amenities"].apply(len)
    return df


# Normalize amenities into a set for keyword matching
def _amenities_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'amenities_set' column with cleaned amenities as a set.
    """
    df = df.copy()
    df["amenities_set"] = df["amenities"].apply(
        lambda lst: {_clean_amenity(a) for a in lst}
    )
    return df


# Keyword matching with exclusions
def _has_keyword(s, include, exclude=None):
    """
    Check if any keyword in 'include' is present in set 's',
    excluding matches with keywords in 'exclude'.
    """
    exclude = exclude or []
    return any(
        (inc in a) and not any(exc in a for exc in exclude)
        for a in s for inc in include
    )


# Adding special amenities features
def add_special_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary features for special amenities:
    - Washer (excluding dishwasher)
    - Pool (excluding pool table, whirlpool variants)
    - Streaming platforms (Netflix, HBO, etc.)
    """
    df = _amenities_set(df)

    # Washer (excluding dishwasher)
    df["has_washer"] = df["amenities_set"].apply(
        lambda s: _has_keyword(s, ["washer"], ["dishwasher"])
    ).astype(float)

    # Pool (excluding noise)
    df["has_pool"] = df["amenities_set"].apply(
        lambda s: _has_keyword(
            s,
            ["pool"],
            ["pool table", "whirlpool", "wirpool", "whirpool"]
        )
    ).astype(float)

    # Streaming
    streaming_keywords = ['netflix','hbo','prime video','disney+','apple tv','hulu']
    df["has_streaming_platform"] = df["amenities_set"].apply(
        lambda s: any(any(k in a for k in streaming_keywords) for a in s)
    ).astype(float)

    return df


# Adding simple amenities features
def add_simple_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary features for a predefined list of simple amenities.
    """
    df = _amenities_set(df)

    simple_amenities = [
        'wifi','kitchen','hot water','essentials','bed linens','microwave',
        'refrigerator','air conditioning','heating','cooking basics',
        'dishes and silverware','iron','hair dryer','dedicated workspace',
        'dining table','dishwasher','freezer','coffee maker','blender',
        'self check-in','private entrance','elevator','free parking',
        'pets allowed','cleaning available during stay','tv','pool table',
        'piano','game console','ping pong table','patio or balcony',
        'backyard','sauna','city skyline view','outdoor furniture',
        'outdoor dining area','smoke alarm','carbon monoxide alarm',
        'fire extinguisher','exterior security cameras on property',
        'first aid kit'
    ]

    for amenity in simple_amenities:
        col = "has_" + amenity.replace(" ", "_").replace("+", "plus").replace("-", "_")
        df[col] = df["amenities_set"].apply(
            lambda s: any(amenity in a for a in s)
        ).astype(float)

    return df


# Feature: orchestrator -> combine special and simple amenities
def add_has_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrator: add both special and simple amenity features.
    """
    df = add_special_amenities(df)
    df = add_simple_amenities(df)

    # Drop intermediate column not needed in final dataset
    if "amenities_set" in df.columns:
        df = df.drop(columns=["amenities_set"])
    
    return df


# Feature: compute weighted amenity score
def add_amenity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted amenity score based on SELECTED_AMENITIES and AMENITIES_WEIGHTS.
    """
    df = df.copy()
    df["amenity_score"] = 0.0
    for col in AMENITIES_SCORE:
        if col in df.columns:
            df["amenity_score"] += df[col] * AMENITIES_WEIGHTS[col]
    return df

