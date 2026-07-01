"""
Utilities for generating complementary market insights.
"""

import pandas as pd

from app.services.artifacts import REFERENCE_DATA


# ==========================================================
# COMPARABLE LISTINGS
# ==========================================================

def _find_comparables(
    listing: pd.DataFrame,
    min_comparables: int = 20
) -> pd.DataFrame:
    """
    Find comparable Airbnb listings using progressively
    relaxed filtering rules.
    """

    row = listing.iloc[0]

    # ------------------------------------------------------
    # LEVEL 1 (STRICT)
    # ------------------------------------------------------

    comparables = REFERENCE_DATA[
        (REFERENCE_DATA["neighbourhood_cleansed"] == row["neighbourhood_cleansed"]) &
        (REFERENCE_DATA["room_type"] == row["room_type"]) &
        (REFERENCE_DATA["accommodates"].between(
            row["accommodates"] - 1,
            row["accommodates"] + 1
        )) &
        (REFERENCE_DATA["bedrooms"].between(
            row["bedrooms"] - 1,
            row["bedrooms"] + 1
        )) &
        (REFERENCE_DATA["bathrooms"].between(
            row["bathrooms"] - 1,
            row["bathrooms"] + 1
        ))
    ]

    # ------------------------------------------------------
    # LEVEL 2 (RELAXED)
    # ------------------------------------------------------

    if len(comparables) < min_comparables:

        comparables = REFERENCE_DATA[
            (REFERENCE_DATA["neighbourhood_cleansed"] == row["neighbourhood_cleansed"]) &
            (REFERENCE_DATA["room_type"] == row["room_type"]) &
            (REFERENCE_DATA["accommodates"].between(
                row["accommodates"] - 2,
                row["accommodates"] + 2
            )) &
            (REFERENCE_DATA["bedrooms"].between(
                row["bedrooms"] - 2,
                row["bedrooms"] + 2
            )) &
            (REFERENCE_DATA["bathrooms"].between(
                row["bathrooms"] - 1,
                row["bathrooms"] + 1
            ))
        ]

    # ------------------------------------------------------
    # LEVEL 3 (FALLBACK)
    # ------------------------------------------------------

    if len(comparables) < min_comparables:

        comparables = REFERENCE_DATA[
            (REFERENCE_DATA["neighbourhood_cleansed"] == row["neighbourhood_cleansed"]) &
            (REFERENCE_DATA["room_type"] == row["room_type"])
        ]

    return comparables


# ==========================================================
# MARKET INSIGHTS
# ==========================================================

def generate_market_insights(
    listing: pd.DataFrame,
    predicted_price: float,
    min_comparables: int = 20
) -> dict:
    """
    Generate complementary market insights for a listing.

    Parameters
    ----------
    listing : pd.DataFrame
        Raw Airbnb listing.

    predicted_price : float
        Predicted nightly price (MXN).

    min_comparables : int, default=20
        Minimum desired number of comparable listings.

    Returns
    -------
    dict
        Dictionary containing market insights.
    """

    comparables = _find_comparables(
        listing=listing,
        min_comparables=min_comparables
    )

    # ------------------------------------------------------
    # NO COMPARABLES FOUND
    # ------------------------------------------------------

    if comparables.empty:

        return {

            "market_average": None,

            "market_price_min": None,

            "market_price_max": None,

            "listing_position": "Unknown",

            "confidence": "Low",

            "num_comparables": 0

        }

    # ------------------------------------------------------
    # MARKET STATISTICS
    # ------------------------------------------------------

    market_average = comparables["clean_price"].mean()

    market_price_min = comparables["clean_price"].quantile(0.25)

    market_price_max = comparables["clean_price"].quantile(0.75)

    # ------------------------------------------------------
    # LISTING POSITION
    # ------------------------------------------------------

    if predicted_price < market_price_min:

        listing_position = "Below Market"

    elif predicted_price > market_price_max:

        listing_position = "Above Market"

    else:

        listing_position = "Fairly Priced"

    # ------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------

    n = len(comparables)

    if n >= 50:

        confidence = "High"

    elif n >= 20:

        confidence = "Medium"

    else:

        confidence = "Low"

    # ------------------------------------------------------
    # RETURN RESULTS
    # ------------------------------------------------------

    return {

        "market_average": round(float(market_average), 2),

        "market_price_min": round(float(market_price_min), 2),

        "market_price_max": round(float(market_price_max), 2),

        "listing_position": listing_position,

        "confidence": confidence,

        "num_comparables": int(n)

    }