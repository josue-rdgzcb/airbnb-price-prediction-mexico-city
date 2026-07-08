"""
Request and response schemas for the Airbnb Smart Pricing API.
"""

from pydantic import BaseModel, Field


# ==========================================================
# REQUEST SCHEMA
# ==========================================================

class PredictionRequest(BaseModel):
    """
    Input schema for a new Airbnb listing.
    """

    # -------------------------
    # Location
    # -------------------------

    latitude: float = Field(
        ...,
        description="Latitude of the listing."
    )

    longitude: float = Field(
        ...,
        description="Longitude of the listing."
    )

    neighbourhood_cleansed: str = Field(
        ...,
        description="Neighborhood where the listing is located."
    )

    # -------------------------
    # Property
    # -------------------------

    room_type: str = Field(
        ...,
        description="Room type."
    )

    property_type: str = Field(
        ...,
        description="Property type."
    )

    accommodates: int = Field(
        ...,
        description="Maximum guest capacity."
    )

    bedrooms: float = Field(
        ...,
        description="Number of bedrooms."
    )

    bathrooms: float = Field(
        ...,
        description="Number of bathrooms."
    )

    beds: float = Field(
        ...,
        description="Number of beds."
    )

    minimum_nights: int = Field(
        ...,
        description="Minimum number of nights."
    )

    # -------------------------
    # Host
    # -------------------------

    host_is_superhost: str = Field(
        ...,
        description="Whether the host is a Superhost."
    )

    host_verifications: str = Field(
        ...,
        description="Host verification methods."
    )

    host_total_listings_count: int = Field(
        ...,
        description="Total listings managed by the host."
    )

    calculated_host_listings_count_entire_homes: int = Field(
        ...,
        description="Entire homes managed by the host."
    )

    calculated_host_listings_count_private_rooms: int = Field(
        ...,
        description="Private rooms managed by the host."
    )

    # -------------------------
    # Booking
    # -------------------------

    instant_bookable: str = Field(
        ...,
        description="Whether instant booking is enabled."
    )

    # -------------------------
    # Reviews
    # -------------------------

    review_scores_rating: float | None = Field(
        None,
        description="Overall review score."
    )

    review_scores_accuracy: float | None = None
    review_scores_cleanliness: float | None = None
    review_scores_checkin: float | None = None
    review_scores_communication: float | None = None
    review_scores_location: float | None = None
    review_scores_value: float | None = None

    # -------------------------
    # Amenities
    # -------------------------

    amenities: str = Field(
        ...,
        description="Raw amenities string."
    )


# ==========================================================
# RESPONSE SCHEMA
# ==========================================================

class PredictionResponse(BaseModel):
    """
    Output schema returned by the prediction endpoint.
    """

    estimated_price: float

    typical_market_price: float

    market_price_lower: float

    market_price_upper: float

    listing_position: str

    comparable_listings: int

    confidence: str

    search_radius_km: float