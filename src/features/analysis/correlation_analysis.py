import pandas as pd
import numpy as np

# ================= CORRELATION ANALYSIS =================

def feature_correlation_analysis(
    df,
    target="log_price",
    threshold=0.85
):
    """
    Analyze feature redundancy using correlation.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    target : str, default="log_price"
        Target variable.

    threshold : float, default=0.85
        Correlation threshold to flag redundancy.

    Returns
    -------
    corr_matrix : pandas.DataFrame
        Full correlation matrix.

    high_corr_pairs : pandas.DataFrame
        Feature pairs with high correlation.
    """

    # Numeric-only correlation
    corr_matrix = df.corr(numeric_only=True)

    # Remove target from redundancy analysis
    features_corr = corr_matrix.drop(
        index=target,
        columns=target
    )

    high_corr_pairs = []

    cols = features_corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            col_1 = cols[i]
            col_2 = cols[j]

            corr_value = features_corr.loc[col_1, col_2]

            if abs(corr_value) >= threshold:

                high_corr_pairs.append({
                    "feature_1": col_1,
                    "feature_2": col_2,
                    "correlation": corr_value
                })

    high_corr_pairs = pd.DataFrame(
        high_corr_pairs
    ).sort_values(
        by="correlation",
        ascending=False,
        key=np.abs
    )

    return corr_matrix, high_corr_pairs