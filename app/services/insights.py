"""
Utilities for generating complementary market insights.
"""

import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from app.services.artifacts import REFERENCE_DATA


# ==========================================================
# COMPARABLE LISTINGS
# ==========================================================

# CONSTANTS
STRICT_RADIUS_KM = 0.5
RELAXED_RADIUS_KM = 1.0
FALLBACK_RADIUS_KM = 2.0


# GEO UTILITIES
def _get_listings_within_radius(
    latitude: float,
    longitude: float,
    radius_km: float
) -> pd.DataFrame:
    """
    Return all listings located within the specified radius.
    """

    coords = REFERENCE_DATA[["latitude", "longitude"]]

    coords_rad = np.radians(coords.values)

    tree = BallTree(
        coords_rad,
        metric="haversine"
    )

    query_point = np.radians(
        [[latitude, longitude]]
    )

    earth_radius_km = 6371

    radius_rad = radius_km / earth_radius_km

    indices = tree.query_radius(
        query_point,
        r=radius_rad
    )[0]

    return REFERENCE_DATA.iloc[indices].copy()



# FIND COMPARABLES
def _find_comparables(
    listing: pd.DataFrame,
    min_comparables: int = 20
) -> dict:
    """
    Find comparable Airbnb listings using progressively
    relaxed geographic filters.

    Returns
    -------
    dict
        Dictionary containing:
        - comparables: Comparable listings.
        - search_radius_km: Radius used to retrieve them.
    """

    row = listing.iloc[0]

    # LEVEL 1

    comparables = _get_listings_within_radius(
        latitude=row["latitude"],
        longitude=row["longitude"],
        radius_km=STRICT_RADIUS_KM
    )

    comparables = comparables[
        (comparables["room_type"] == row["room_type"]) &
        (comparables["accommodates"].between(
            row["accommodates"] - 1,
            row["accommodates"] + 1
        )) &
        (comparables["bedrooms"].between(
            row["bedrooms"] - 1,
            row["bedrooms"] + 1
        )) &
        (comparables["bathrooms"].between(
            row["bathrooms"] - 1,
            row["bathrooms"] + 1
        ))
    ]

    if len(comparables) >= min_comparables:

        return {
            "comparables": comparables,
            "search_radius_km": STRICT_RADIUS_KM
        }

    # LEVEL 2

    comparables = _get_listings_within_radius(
        latitude=row["latitude"],
        longitude=row["longitude"],
        radius_km=RELAXED_RADIUS_KM
    )

    comparables = comparables[
        (comparables["room_type"] == row["room_type"]) &
        (comparables["accommodates"].between(
            row["accommodates"] - 2,
            row["accommodates"] + 2
        )) &
        (comparables["bedrooms"].between(
            row["bedrooms"] - 2,
            row["bedrooms"] + 2
        )) &
        (comparables["bathrooms"].between(
            row["bathrooms"] - 1,
            row["bathrooms"] + 1
        ))
    ]

    if len(comparables) >= min_comparables:

        return {
            "comparables": comparables,
            "search_radius_km": RELAXED_RADIUS_KM
        }

    # LEVEL 3

    comparables = _get_listings_within_radius(
        latitude=row["latitude"],
        longitude=row["longitude"],
        radius_km=FALLBACK_RADIUS_KM
    )

    comparables = comparables[
        comparables["room_type"] == row["room_type"]
    ]

    return {
        "comparables": comparables,
        "search_radius_km": FALLBACK_RADIUS_KM
    }


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

    search_results = _find_comparables(
        listing=listing,
        min_comparables=min_comparables
    )

    comparables = search_results["comparables"]

    search_radius_km = search_results["search_radius_km"]

    # ------------------------------------------------------
    # NO COMPARABLES FOUND
    # ------------------------------------------------------

    if comparables.empty:

        return {

            "market_average": None,

            "market_price_lower": None,

            "market_price_upper": None,

            "listing_position": "Unknown",

            "confidence": "Low",

            "num_comparables": 0

        }

    # ------------------------------------------------------
    # MARKET STATISTICS
    # ------------------------------------------------------

    market_median = comparables["clean_price"].median()

    market_price_lower = comparables["clean_price"].quantile(0.25)

    market_price_upper = comparables["clean_price"].quantile(0.75)

    # ------------------------------------------------------
    # LISTING POSITION
    # ------------------------------------------------------

    if predicted_price < market_price_lower:

        listing_position = "Below Market"

    elif predicted_price > market_price_upper:

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

        "market_median": round(float(market_median), 2),

        "market_price_lower": round(float(market_price_lower), 2),

        "market_price_upper": round(float(market_price_upper), 2),

        "listing_position": listing_position,

        "confidence": confidence,

        "num_comparables": int(n),

        "search_radius_km": search_radius_km

    }