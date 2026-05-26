from sklearn.impute import SimpleImputer

from src.settings.transform_config import (
    MEAN_IMPUTE_FEATURES,
    MEDIAN_IMPUTE_FEATURES,
    MOST_FREQUENT_IMPUTE_FEATURES,
    MISSING_CATEGORY_IMPUTE_FEATURES
)


def apply_imputation(df):
    """
    Apply missing value imputation to selected features.

    This function applies different imputation strategies
    based on feature groups defined in the settings module.

    Strategies supported:
    - Mean imputation
    - Median imputation
    - Most frequent imputation
    - Missing category imputation

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with imputed features.
    """

    df = df.copy()

    imputers = {

        "mean": {
            "features": MEAN_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="mean")
        },

        "median": {
            "features": MEDIAN_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="median")
        },

        "most_frequent": {
            "features": MOST_FREQUENT_IMPUTE_FEATURES,
            "imputer": SimpleImputer(strategy="most_frequent")
        },

        "missing_category": {
            "features": MISSING_CATEGORY_IMPUTE_FEATURES,
            "imputer": SimpleImputer(
                strategy="constant",
                fill_value="missing"
            )
        }
    }

    for _, config in imputers.items():

        features = config["features"]

        imputer = config["imputer"]

        if not features:
            continue

        df[features] = imputer.fit_transform(df[features])

    return df

