
import pandas as pd
import numpy as np

# ===========================================================
# =================== NUMERIC DIAGNOSTICS ===================
# ===========================================================
def numeric_diagnostics(df, target="log_price"):
    """
    Perform numeric diagnostics on DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric features and target.
    target : str, default="log_price"
        Name of the target variable used for correlation, excluded from the final diagnostics.

    Returns
    -------
    pandas.DataFrame
        Table with diagnostics for each numeric feature, including:
        - Percentage of nulls
        - Skewness and kurtosis
        - Standard deviation, min, max
        - Outlier percentage (IQR method)
        - Correlation with target
    """
    results = []

    # Select numeric columns, excluding the target
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target)

    for col in num_cols:
        series = df[col].dropna()

        # Outlier detection using IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
        outliers_pct = outliers / len(series)

        # Collect diagnostics for each feature
        results.append({
            "feature": col,
            "nulls_%": df[col].isna().mean(),
            "skew": series.skew(),
            "kurtosis": series.kurt(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "outliers_%": outliers_pct,
            "corr_with_target": df[col].corr(df[target])
        })

    return pd.DataFrame(results).set_index("feature").reset_index()


# =================== HEURISTICS: NULL TREATMENT ===================
def suggest_null_treatment(row):
    """
    Suggest null value treatment based on feature statistics.

    Parameters
    ----------
    row : pandas.Series
        Row from diagnostics table containing feature statistics.

    Returns
    -------
    str
        Suggested null treatment strategy:
        - "no_impute" if no nulls
        - "median" if skewed or with outliers
        - "mean" otherwise
    """
    nulls = row["nulls_%"]
    skew = row["skew"]
    outliers = row["outliers_%"]

    if nulls == 0:
        return "no_impute"

    if outliers > 0.05 or abs(skew) > 1:
        return "median"
    else:
        return "mean"


# =================== HEURISTICS: TRANSFORMATION ===================
def suggest_transform_treatment(row):
    """
    Suggest a transformation treatment for a numeric feature based on skewness and outlier percentage.

    Parameters
    ----------
    row : pandas.Series
        A row from the diagnostics DataFrame containing at least:
        - "skew": skewness of the feature
        - "outliers_%": percentage of outliers detected

    Returns
    -------
    str
        Recommended transformation:
        - "log" for strong skewness (> 2)
        - "sqrt" for moderate skewness (> 1)
        - "winsor_strong" for heavy outliers (> 10%)
        - "winsor" for moderate outliers (> 5%)
        - "no_transform" if no adjustment is needed
    """

    skew = abs(row["skew"])
    outliers = row["outliers_%"]

    # Priority: strong skewness → log transform
    if skew > 2:
        return "log"

    # Moderate skewness → square root transform
    if skew > 1:
        return "sqrt"

    # Heavy outliers (without strong skew) → strong winsorization
    if outliers > 0.10:
        return "winsor_strong"

    # Moderate outliers → winsorization
    if outliers > 0.05:
        return "winsor"

    # Default: no transformation needed
    return "no_transform"


# =================== HEURISTICS: BINNING ===================
def suggest_binning(row):
    """
    Suggest whether a feature could benefit from binning.

    Parameters
    ----------
    row : pandas.Series
        Row from diagnostics table containing feature statistics.

    Returns
    -------
    str
        Binning suggestion:
        - "binning_candidate" with reason
        - "no_binning" otherwise
    """
    skew = abs(row["skew"])
    outliers = row["outliers_%"]
    corr = abs(row["corr_with_target"])
    std = row["std"]

    # Candidates due to non-linearity (low correlation + skew)
    if corr < 0.1 and skew > 1:
        return "binning_candidate (non_linear)"

    # Candidates due to extreme values
    if outliers > 0.10:
        return "binning_candidate (extreme_values)"

    # Candidates due to low variance / discreteness
    if std < 1:
        return "binning_candidate (low_variance/discrete)"

    return "no_binning"

# =================== HEURISTICS: SCALING ===================
def suggest_scaling(row):
    """
    Suggest a scaling method based on outlier percentage and skewness.

    Parameters
    ----------
    row : pandas.Series
        Must contain 'outliers_%' and 'skew'.

    Returns
    -------
    str
        - "robust" if many outliers (> 5%)
        - "standard" if distribution is near normal (skew < 0.5)
        - "minmax" otherwise
    """

    outliers = row["outliers_%"]
    skew = abs(row["skew"])

    # Many outliers → robust scaling
    if outliers > 0.05:
        return "robust"

    # Distribution close to normal → standard scaling
    if skew < 0.5:
        return "standard"

    # Default → min-max scaling
    return "minmax"

# =================== HEURISTICS: SIGNAL ===================
def suggest_signal(row):
    """
    Suggest signal strength based on correlation with target.

    Parameters
    ----------
    row : pandas.Series
        Must contain 'corr_with_target'.

    Returns
    -------
    str
        - "low_signal" if correlation < 0.05
        - "weak_signal" if correlation < 0.15
        - "useful_signal" otherwise
    """

    corr = abs(row["corr_with_target"])

    # Very low correlation → low signal
    if corr < 0.05:
        return "low_signal"

    # Weak correlation → weak signal
    elif corr < 0.15:
        return "weak_signal"

    # Otherwise → useful signal
    else:
        return "useful_signal"



# =================== FINAL PIPELINE ===================
def build_numeric_diagnostics(df, target="log_price"):
    """
    Build complete numeric diagnostics pipeline.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric features and target.
    target : str, default="log_price"
        Name of the target variable used for correlation, excluded from the final diagnostics.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table with:
        - Null treatment suggestion
        - Transformation suggestion
        - Binning suggestion
    """
    diag = numeric_diagnostics(df, target)

    diag["signal_suggestion"] = diag.apply(suggest_signal, axis=1)
    diag["null_treatment_suggestion"] = diag.apply(suggest_null_treatment, axis=1)
    diag["transform_suggestion"] = diag.apply(suggest_transform_treatment, axis=1)
    diag["binning_suggestion"] = diag.apply(suggest_binning, axis=1)
    diag["scaling_suggestion"] = diag.apply(suggest_scaling, axis=1)

    return diag

# ===========================================================
# =================== CATEGORICAL DIAGNOSTICS ===============
# ===========================================================

import numpy as np
import pandas as pd


# =================== 1. CATEGORICAL DIAGNOSTICS ===================
def categorical_diagnostics(df, target="log_price"):
    """
    Perform categorical diagnostics on DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing categorical features and target.

    target : str, default="log_price"
        Name of the target variable.

    Returns
    -------
    pandas.DataFrame
        Table with diagnostics for each categorical feature.
    """

    results = []

    # Select categorical columns
    cat_cols = df.select_dtypes(
        include=["object", "category", "str"]
    ).columns

    for col in cat_cols:

        series = df[col]

        # Basic stats
        nulls_pct = series.isna().mean()
        unique = series.nunique(dropna=True)

        # Frequency distribution
        value_counts = series.value_counts(
            normalize=True,
            dropna=True
        )

        top_category_pct = (
            value_counts.iloc[0]
            if len(value_counts) > 0
            else np.nan
        )

        # Rare categories (<1%)
        rare_pct = (
            value_counts[value_counts < 0.01].sum()
            if len(value_counts) > 0
            else 0
        )

        # Signal estimation:
        # median target variance across categories
        target_variance = (
            df.groupby(col)[target]
            .median()
            .var()
        )

        results.append({
            "feature": col,
            "nulls_%": nulls_pct,
            "unique_categories": unique,
            "top_category_pct": top_category_pct,
            "rare_categories_pct": rare_pct,
            "target_median_variance": target_variance
        })

    return pd.DataFrame(results)


# =================== 2. HEURISTICS: SIGNAL ===================
def suggest_categorical_signal(row):
    """
    Suggest signal strength for categorical features.
    """

    variance = row["target_median_variance"]

    if variance < 0.01:
        return "low_signal"

    elif variance < 0.05:
        return "weak_signal"

    return "useful_signal"


# =================== 3. HEURISTICS: NULL TREATMENT ===================
def suggest_categorical_null_treatment(row):
    """
    Suggest null treatment strategy.
    """

    nulls = row["nulls_%"]

    if nulls == 0:
        return "no_impute"

    if nulls < 0.05:
        return "mode"

    return "missing_category"


# =================== 4. HEURISTICS: ENCODING ===================
def suggest_encoding(row):
    """
    Suggest encoding strategy based on cardinality and rarity.
    """

    unique = row["unique_categories"]
    rare_pct = row["rare_categories_pct"]

    # Binary variables
    if unique == 2:
        return "binary"

    # Low-cardinality
    if unique <= 10:

        if rare_pct > 0.30:
            return "group_rare + onehot"

        return "onehot"

    # Medium-cardinality
    if unique <= 30:
        return "frequency"

    # High-cardinality
    return "frequency/group_rare"


# =================== 5. FINAL PIPELINE ===================
def build_categorical_diagnostics(
    df,
    target="log_price"
):
    """
    Build complete categorical diagnostics pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing categorical features.

    target : str, default="log_price"
        Target variable.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table with:
        - signal suggestion
        - null treatment suggestion
        - encoding suggestion
    """

    diag = categorical_diagnostics(df, target)

    diag["signal_suggestion"] = diag.apply(
        suggest_categorical_signal,
        axis=1
    )

    diag["null_treatment_suggestion"] = diag.apply(
        suggest_categorical_null_treatment,
        axis=1
    )

    diag["encoding_suggestion"] = diag.apply(
        suggest_encoding,
        axis=1
    )

    return diag


# =================== BOOLEAN DIAGNOSTICS ===================

def boolean_diagnostics(df, target="log_price"):
    """
    Perform diagnostics for boolean features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    target : str, default="log_price"
        Target variable.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table for boolean variables.
    """

    results = []

    # Select boolean columns
    bool_cols = df.select_dtypes(include=["bool", "str", "category"]).columns

    for col in bool_cols:

        series = df[col]

        true_pct = series.mean()

        target_true = df.loc[series == True, target].median()
        target_false = df.loc[series == False, target].median()

        target_diff = abs(target_true - target_false)

        corr = series.astype(int).corr(df[target])

        results.append({
            "feature": col,
            "nulls_%": series.isna().mean(),
            "true_pct": true_pct,
            "target_median_diff": target_diff,
            "corr_with_target": corr
        })

    return pd.DataFrame(results)