"""
Geospatial utilities.

Resolve latitude/longitude coordinates into the corresponding
Mexico City neighbourhood.
"""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point


# ==========================================================
# LOAD NEIGHBOURHOOD POLYGONS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

GEOJSON_PATH = (
    ROOT_DIR
    / "data"
    / "external"
    / "neighbourhoods.geojson"
)

NEIGHBOURHOODS = gpd.read_file(GEOJSON_PATH)

# Ensure WGS84 coordinates
NEIGHBOURHOODS = NEIGHBOURHOODS.to_crs("EPSG:4326")


# ==========================================================
# PUBLIC FUNCTION
# ==========================================================

def get_neighbourhood(
    latitude: float,
    longitude: float
) -> str | None:
    """
    Return the Mexico City neighbourhood that contains
    the given geographic coordinates.

    Parameters
    ----------
    latitude : float
        Listing latitude.

    longitude : float
        Listing longitude.

    Returns
    -------
    str | None

        Borough name if the point belongs to CDMX.
        Otherwise None.
    """

    point = Point(
        longitude,
        latitude
    )

    match = NEIGHBOURHOODS[
        NEIGHBOURHOODS.contains(point)
    ]

    if match.empty:

        return None

    return str(
        match.iloc[0]["neighbourhood"]
    )