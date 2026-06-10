
import pandas as pd
import numpy as np

# ===========================================================
# =================== NUMERIC DIAGNOSTICS ===================
# ===========================================================

# Perform diagnostics for numeric features using a provided list of columns
def numeric_diagnostics(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target: str = "log_price"
) -> pd.DataFrame:
    """
    Perform numeric diagnostics on selected DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric features and target.
    numeric_columns : list of str
        List of numeric columns to evaluate.
    target : str, default="log_price"
        Name of the target variable used for correlation.

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

    for col in numeric_columns:
        series = df[col].dropna()

        # Outlier detection using IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = ((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum()
        outliers_pct = outliers / len(series)

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


# Suggest null imputation strategy for numeric features
def suggest_null_treatment(row: pd.Series) -> str:
    """
    Suggest null value treatment based on feature statistics.
    """
    nulls = row["nulls_%"]
    skew = row["skew"]
    outliers = row["outliers_%"]

    if nulls == 0:
        return "no_impute"
    if outliers > 0.05 or abs(skew) > 1:
        return "median"
    return "mean"


# Suggest transformation method for numeric features (log, sqrt, winsorization)
def suggest_transform_treatment(row: pd.Series) -> str:
    """
    Suggest a transformation treatment for a numeric feature based on skewness and outlier percentage.
    """
    skew = abs(row["skew"])
    outliers = row["outliers_%"]

    if skew > 2:
        return "log"
    if skew > 1:
        return "sqrt"
    if outliers > 0.10:
        return "winsor_strong"
    if outliers > 0.05:
        return "winsor"
    return "no_transform"


# Suggest binning strategy for numeric features (non-linear, extreme values, low variance)
def suggest_binning(row: pd.Series) -> str:
    """
    Suggest whether a feature could benefit from binning.
    """
    skew = abs(row["skew"])
    outliers = row["outliers_%"]
    corr = abs(row["corr_with_target"])
    std = row["std"]

    if corr < 0.1 and skew > 1:
        return "binning_candidate (non_linear)"
    if outliers > 0.10:
        return "binning_candidate (extreme_values)"
    if std < 1:
        return "binning_candidate (low_variance/discrete)"
    return "no_binning"


# Suggest scaling method for numeric features (robust, standard, minmax)
def suggest_scaling(row: pd.Series) -> str:
    """
    Suggest a scaling method based on outlier percentage and skewness.
    """
    outliers = row["outliers_%"]
    skew = abs(row["skew"])

    if outliers > 0.05:
        return "robust"
    if skew < 0.5:
        return "standard"
    return "minmax"


# Suggest signal strength category for numeric features based on correlation with target
def suggest_signal(row: pd.Series) -> str:
    """
    Suggest signal strength based on correlation with target.
    """
    corr = abs(row["corr_with_target"])

    if corr < 0.05:
        return "low_signal"
    elif corr < 0.15:
        return "weak_signal"
    return "useful_signal"


# Build complete numeric diagnostics pipeline with all suggestions
def build_numeric_diagnostics(
    df: pd.DataFrame,
    numeric_columns: list[str],
    target: str = "log_price"
) -> pd.DataFrame:
    """
    Build complete numeric diagnostics pipeline.
    """
    diag = numeric_diagnostics(df, numeric_columns, target)

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


## Perform diagnostics for categorical features using a provided list of columns
def categorical_diagnostics(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target: str = "log_price"
) -> pd.DataFrame:
    """
    Perform categorical diagnostics on selected DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing categorical features and target.
    categorical_columns : list of str
        List of categorical columns to evaluate.
    target : str, default="log_price"
        Name of the target variable.

    Returns
    -------
    pandas.DataFrame
        Table with diagnostics for each categorical feature, including:
        - Percentage of nulls
        - Number of unique categories
        - Top category percentage
        - Rare categories percentage (<1%)
        - Variance of target median across categories
    """
    results = []

    for col in categorical_columns:
        series = df[col]

        # Basic stats
        nulls_pct = series.isna().mean()
        unique = series.nunique(dropna=True)

        # Frequency distribution
        value_counts = series.value_counts(normalize=True, dropna=True)

        top_category_pct = value_counts.iloc[0] if len(value_counts) > 0 else np.nan

        # Rare categories (<1%)
        rare_pct = value_counts[value_counts < 0.01].sum() if len(value_counts) > 0 else 0

        # Signal estimation: variance of target median across categories
        target_variance = df.groupby(col)[target].median().var()

        results.append({
            "feature": col,
            "nulls_%": nulls_pct,
            "unique_categories": unique,
            "top_category_pct": top_category_pct,
            "rare_categories_pct": rare_pct,
            "target_median_variance": target_variance
        })

    return pd.DataFrame(results)


# Suggest signal strength for categorical features based on target median variance
def suggest_categorical_signal(row: pd.Series) -> str:
    """
    Suggest signal strength for categorical features.
    """
    variance = row["target_median_variance"]

    if variance < 0.01:
        return "low_signal"
    elif variance < 0.05:
        return "weak_signal"
    return "useful_signal"


# Suggest null treatment strategy for categorical features
def suggest_categorical_null_treatment(row: pd.Series) -> str:
    """
    Suggest null treatment strategy.
    """
    nulls = row["nulls_%"]

    if nulls == 0:
        return "no_impute"
    if nulls < 0.05:
        return "mode"
    return "missing_category"


# Suggest encoding strategy for categorical features based on cardinality and rarity
def suggest_encoding(row: pd.Series) -> str:
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


# Build complete categorical diagnostics pipeline with signal, null, and encoding suggestions
def build_categorical_diagnostics(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target: str = "log_price"
) -> pd.DataFrame:
    """
    Build complete categorical diagnostics pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing categorical features.
    categorical_columns : list of str
        List of categorical columns to evaluate.
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
    diag = categorical_diagnostics(df, categorical_columns, target)

    diag["signal_suggestion"] = diag.apply(suggest_categorical_signal, axis=1)
    diag["null_treatment_suggestion"] = diag.apply(suggest_categorical_null_treatment, axis=1)
    diag["encoding_suggestion"] = diag.apply(suggest_encoding, axis=1)

    return diag


# ===========================================================
# =================== BOOLEAN DIAGNOSTICS ===================
# ===========================================================

# Perform diagnostics for boolean-like features (floats 0/1) using a provided list of columns
def boolean_diagnostics(
    df: pd.DataFrame,
    boolean_columns: list[str],
    target: str = "log_price"
) -> pd.DataFrame:
    """
    Perform diagnostics for boolean features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    boolean_columns : list of str
        List of columns to treat as boolean features.

    target : str, default="log_price"
        Target variable.

    Returns
    -------
    pandas.DataFrame
        Diagnostics table for boolean variables.
    """

    results = []

    for col in boolean_columns:
        series = df[col]

        # Interpret 1.0 as True and 0.0 as False
        true_pct = series.mean()

        target_true = df.loc[series == 1.0, target].median()
        target_false = df.loc[series == 0.0, target].median()

        target_diff = abs(target_true - target_false)

        corr = series.corr(df[target])

        results.append({
            "feature": col,
            "nulls_%": series.isna().mean(),
            "true_pct": true_pct,
            "target_median_diff": target_diff,
            "corr_with_target": corr
        })

    return pd.DataFrame(results)



# Suggest signal strength category for a boolean feature based on target difference and correlation
def suggest_boolean_signal(row: pd.Series) -> str:
    """
    Suggest signal strength for boolean feature.
    """
    diff = row["target_median_diff"]
    corr = abs(row["corr_with_target"])

    if diff >= 0.30 or corr >= 0.20:
        return "strong_signal"
    elif diff >= 0.15 or corr >= 0.10:
        return "useful_signal"
    elif diff >= 0.05 or corr >= 0.03:
        return "weak_signal"
    else:
        return "low_signal"


# Suggest appropriate null treatment strategy for a boolean feature
def suggest_boolean_null_treatment(row: pd.Series) -> str:
    """
    Suggest null treatment for boolean feature.
    """
    nulls = row["nulls_%"]

    if nulls == 0:
        return "no_impute"
    elif nulls < 0.05:
        return "mode"
    else:
        return "evaluate_missingness"


# Suggest whether to keep or drop a boolean feature based on prevalence and signal strength
def suggest_boolean_keep(row: pd.Series) -> str:
    """
    Suggest keep/drop decision for boolean feature.
    """
    true_pct = row["true_pct"]
    diff = row["target_median_diff"]
    corr = abs(row["corr_with_target"])

    # Too common or too rare
    if true_pct < 0.03 or true_pct > 0.95:
        return "drop_candidate"

    # Strong individual signal
    if diff >= 0.30 and corr >= 0.15:
        return "keep_candidate"

    # Moderate candidate
    if diff >= 0.20 and corr >= 0.10:
        return "evaluate"

    return "drop_candidate"


# Build complete diagnostics table for boolean features including signal, null treatment, and keep suggestions
def build_boolean_diagnostics(df: pd.DataFrame, boolean_columns: list[str],  target: str = "log_price") -> pd.DataFrame:
    """
    Build complete boolean diagnostics.
    """
    diag = boolean_diagnostics(df, boolean_columns, target)

    diag["signal_suggestion"] = diag.apply(suggest_boolean_signal, axis=1)
    diag["null_treatment_suggestion"] = diag.apply(suggest_boolean_null_treatment, axis=1)
    diag["keep_suggestion"] = diag.apply(suggest_boolean_keep, axis=1)

    return diag
