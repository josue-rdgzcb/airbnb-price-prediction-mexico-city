"""
Application entry point.

This module initializes the FastAPI application and
registers all API routes.
"""

from fastapi import FastAPI

from app.api.routes import router


# ==========================================================
# CREATE APPLICATION
# ==========================================================

app = FastAPI(
    title="Airbnb Smart Pricing API",
    description=(
        "REST API for predicting Airbnb listing prices "
        "and generating market insights."
    ),
    version="1.0.0"
)


# ==========================================================
# REGISTER ROUTES
# ==========================================================

app.include_router(router)


# ==========================================================
# ROOT ENDPOINT
# ==========================================================

@app.get("/")
def root():
    """
    Root endpoint.

    Returns
    -------
    dict
        Simple status message.
    """

    return {
        "message": "Airbnb Smart Pricing API is running."
    }