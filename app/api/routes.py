"""
API endpoints for the Airbnb Smart Pricing service.
"""

import pandas as pd

from fastapi import APIRouter

from app.api.schemas import (
    PredictionRequest,
    PredictionResponse
)

from app.services.inference import predict_listing
from app.services.insights import generate_market_insights


# ==========================================================
# ROUTER
# ==========================================================

router = APIRouter()


# ==========================================================
# PREDICTION ENDPOINT
# ==========================================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(
    request: PredictionRequest
) -> PredictionResponse:
    """
    Predict the recommended nightly price of an Airbnb listing
    and generate complementary market insights.

    Parameters
    ----------
    request : PredictionRequest
        Airbnb listing features.

    Returns
    -------
    PredictionResponse
        Predicted price and market insights.
    """

    # ------------------------------------------------------
    # Convert request to DataFrame
    # ------------------------------------------------------

    listing = pd.DataFrame(
        [request.model_dump()]
    )

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    estimated_price = predict_listing(
        listing
    )

    # ------------------------------------------------------
    # Market insights
    # ------------------------------------------------------

    insights = generate_market_insights(
        listing,
        estimated_price
    )

    # ------------------------------------------------------
    # Build response
    # ------------------------------------------------------

    return PredictionResponse(

        estimated_price=round(
            estimated_price,
            2
        ),

        typical_market_price=round(
            insights["market_median"],
            2
        ),

        market_price_lower=round(
            insights["market_price_lower"],
            2
        ),

        market_price_upper=round(
            insights["market_price_upper"],
            2
        ),

        listing_position=insights[
            "listing_position"
        ],

        comparable_listings=insights[
            "num_comparables"
        ],

        confidence=insights[
            "confidence"
        ],

        search_radius_km=insights[
            "search_radius_km"
        ]

    )