"""
Prepare raw Airbnb listings for model inference.
"""

import pandas as pd

from src.cleaning.cleaning_utils import clean_features
from src.features.build.orchestrator import build_features
from src.settings.model_settings import MODEL_FEATURES


# ==========================================================
# FEATURE PREPARATION PIPELINE
# ==========================================================

def prepare_features(
    listing: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare a raw Airbnb listing for model inference.

    The same feature preparation pipeline used during
    model training is executed to guarantee consistency
    between training and production.

    Steps
    -----
    1. Clean raw features.
    2. Build engineered features.
    3. Select model features.

    Parameters
    ----------
    listing : pd.DataFrame
        Raw Airbnb listing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the model features.
    """

    # Clean raw variables
    listing = clean_features(listing)

    # Create engineered features
    listing = build_features(listing)

    # Keep only model features
    listing = listing[MODEL_FEATURES]

    return listing