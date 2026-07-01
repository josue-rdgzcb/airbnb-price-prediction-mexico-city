"""
Utilities for generating predictions with the production model.
"""

import numpy as np
import pandas as pd

from app.services.artifacts import (
    MODEL,
    PREPROCESSOR,
    FEATURE_LIST
)

from app.services.features import prepare_features

from src.preprocess.preprocess_features import transform_preprocessing_pipeline


# ==========================================================
# PREDICTION
# ==========================================================

def predict_listing(
    listing: pd.DataFrame
) -> float:
    """
    Predict the nightly price of an Airbnb listing.

    Parameters
    ----------
    listing : pd.DataFrame
        Raw Airbnb listing.

    Returns
    -------
    float
        Predicted nightly price (MXN).
    """

    # Prepare features
    prepared_listing = prepare_features(
        listing
    )

    # Apply preprocessing pipeline
    processed_listing = transform_preprocessing_pipeline(
        X=prepared_listing,
        preprocessors=PREPROCESSOR
    )

    # Validate feature order
    assert list(processed_listing.columns) == FEATURE_LIST

    # Predict (log space)
    prediction_log = MODEL.predict(
        processed_listing
    )[0]

    # Convert back to MXN
    prediction = np.expm1(
        prediction_log
    )

    return float(prediction)