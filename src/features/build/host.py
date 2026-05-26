# Import utility to parse stringified lists into Python lists
from src.features.preprocess.preprocess_utils import parse_column

import pandas as pd

from src.settings.features_params import (
    HOST_TOTAL_LISTINGS_BINS
)

# Create host_verifications_list column
def host_verifications_list(df):
    """
    Parse the 'host_verifications' column into Python lists and
    add a new column 'host_verifications_list' if not already present.
    """
    if "host_verifications_list" not in df.columns:
        df["host_verifications_list"] = df["host_verifications"].apply(parse_column)
    return df


# Group host verifications into categories
def group_host_verifications(verif_list):
    """
    Categorize a list of host verifications into groups:
    - unknown: not a list
    - none: empty list
    - extended: contains 'work_email'
    - basic: contains only 'email' and 'phone'
    - low: single verification
    - other: all other cases
    """
    if not isinstance(verif_list, list):
        return "unknown"

    if len(verif_list) == 0:
        return "none"

    if "work_email" in verif_list:
        return "extended"

    if set(verif_list) == {"email", "phone"}:
        return "basic"

    if len(verif_list) == 1:
        return "low"

    return "other"


# Feature: add grouped host verifications
def add_host_verifications_grouped(df):
    """
    Create a new column 'host_verifications_grouped' by applying group_host_verifications.
    """
    df = df.copy()

    if "host_verifications_list" not in df.columns:
        df = host_verifications_list(df)

    df["host_verifications_grouped"] = df["host_verifications_list"].apply(
        group_host_verifications
    )

    # Drop intermediate columns not needed in final dataset
    drop_cols = ["host_verifications_list"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

# Feature: segment hosts by portfolio size
def add_host_total_listings_segment(df):
    """
    Segment hosts by total listings count using
    fixed bins from settings.HOST_TOTAL_LISTINGS_BINS.

    Labels are defined directly in the function.
    """

    df = df.copy()

    df["host_total_listings_segment"] = pd.cut(
        df["host_total_listings_count"],
        bins=HOST_TOTAL_LISTINGS_BINS,
        labels=[
            "small_host",
            "medium_host",
            "large_host",
            "professional_host"
        ],
        include_lowest=True
    )

    return df
