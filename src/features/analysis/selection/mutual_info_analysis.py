
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# ================= MUTUAL INFORMATION ANALYSIS =================

def mutual_information_analysis(
    df,
    target="log_price",
    random_state=42
):
    """
    Compute Mutual Information scores
    between features and target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    target : str, default="log_price"
        Target variable.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Mutual Information scores sorted descending.
    """

    # Features and target
    X = df.drop(columns=[target])

    y = df[target]

    # Keep numeric only
    X = X.select_dtypes(include=["number"])

    # Compute MI
    mi_scores = mutual_info_regression(
        X,
        y,
        random_state=random_state
    )

    # Results table
    mi_df = pd.DataFrame({

        "feature": X.columns,
        "mutual_information": mi_scores

    })

    mi_df = mi_df.sort_values(
        by="mutual_information",
        ascending=False
    ).reset_index(drop=True)

    return mi_df