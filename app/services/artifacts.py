"""
Utilities for loading production artifacts.

This module centralizes the loading of all production artifacts
required during inference. All artifacts are loaded once when the
application starts and reused throughout its lifetime.
"""

from pathlib import Path
import json

import joblib
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Production artifacts directory
ARTIFACTS_DIR = BASE_DIR / "models" / "production"

# Processed data directory
DATA_DIR = BASE_DIR / "data" / "processed"


# ==========================================================
# LOADERS
# ==========================================================

def load_preprocessing_pipeline():
    """
    Load preprocessing pipeline.

    Returns
    -------
    dict
        Dictionary containing all fitted preprocessing components.
    """
    return joblib.load(
        ARTIFACTS_DIR / "preprocess_pipeline.joblib"
    )


def load_model():
    """
    Load trained machine learning model.

    Returns
    -------
    RegressorMixin
        Trained regression model.
    """
    return joblib.load(
        ARTIFACTS_DIR / "final_model.joblib"
    )


def load_feature_list():
    """
    Load ordered feature names.

    Returns
    -------
    list
        Ordered list of feature names expected by the model.
    """
    with open(
        ARTIFACTS_DIR / "feature_list.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_metrics():
    """
    Load evaluation metrics.

    Returns
    -------
    dict
        Model evaluation metrics.
    """
    with open(
        ARTIFACTS_DIR / "metrics.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_config():
    """
    Load model configuration.

    Returns
    -------
    dict
        Model metadata and configuration.
    """
    with open(
        ARTIFACTS_DIR / "config.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_reference_data():
    """
    Load reference Airbnb listings used to generate
    complementary market insights.

    Returns
    -------
    pd.DataFrame
        Reference dataset containing listing features and prices.
    """
    return pd.read_csv(
        DATA_DIR / "df_features.csv"
    )


# ==========================================================
# LOAD ARTIFACTS ON STARTUP
# ==========================================================

PREPROCESSOR = load_preprocessing_pipeline()

MODEL = load_model()

FEATURE_LIST = load_feature_list()

METRICS = load_metrics()

CONFIG = load_config()

REFERENCE_DATA = load_reference_data()