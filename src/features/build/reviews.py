import pandas as pd

from src.settings.features_params import (
    REVIEW_SCORE_COLUMNS,
    REVIEW_SCORES_MEAN_BINS
)

# Feature: compute mean of review score columns
def add_review_scores_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean of all review score columns defined in REVIEW_SCORE_COLUMNS
    and add a new column 'review_scores_mean' to the DataFrame.
    """
    df = df.copy()
    df["review_scores_mean"] = df[REVIEW_SCORE_COLUMNS].mean(axis=1)
    return df


# Feature: add numeric binary indicator if listing has any review
def add_has_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary feature 'has_review' to the DataFrame.
    The column is 1.0 if any of the review score columns
    defined in REVIEW_SCORE_COLUMNS are non-null, otherwise 0.0.
    """
    df = df.copy()

    df["has_review"] = df[REVIEW_SCORE_COLUMNS].notna().any(axis=1).astype(float)
    
    return df

# Feature: segment listings by review score quality
def add_review_scores_mean_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment listings by review score quality using
    fixed bins from settings.REVIEW_SCORES_MEAN_BINS.

    Labels are defined directly in the function.
    """

    df = df.copy()

    df["review_scores_mean_segment"] = pd.cut(
        df["review_scores_mean"],
        bins=REVIEW_SCORES_MEAN_BINS,
        labels=[
            "low_review",
            "medium_review",
            "high_review"
        ],
        include_lowest=True
    ).astype("object")

    return df





