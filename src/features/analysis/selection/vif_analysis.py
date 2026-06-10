
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ================= VIF ANALYSIS =================

def vif_analysis(df):
    """
    Compute Variance Inflation Factor (VIF)
    for numerical model features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing only
        numerical predictor variables.

    Returns
    -------
    pandas.DataFrame
        VIF table sorted descending.
    """

    # Copy
    X = df.copy()

    # Ensure numeric only
    X = X.select_dtypes(include=["number"])

    vif_data = pd.DataFrame()

    vif_data["feature"] = X.columns

    vif_data["VIF"] = [

        variance_inflation_factor(
            X.values,
            i
        )

        for i in range(X.shape[1])
    ]

    vif_data = vif_data.sort_values(
        by="VIF",
        ascending=False
    ).reset_index(drop=True)

    return vif_data