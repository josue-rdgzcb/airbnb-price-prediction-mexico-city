from src.preprocess.imputation import (
    fit_imputers,
    transform_imputers
)

from src.preprocess.encoding import (
    fit_encoders,
    transform_encoders
)

from src.preprocess.scaling import (
    fit_scalers,
    transform_scalers
)

from src.preprocess.numeric_transformation import (
    log_transformation
)


def fit_preprocessing_pipeline(X):
    """
    Fit all preprocessing components using the training data.

    This function learns any parameters required by the
    preprocessing steps, such as:

    - Imputation statistics
    - Category encodings
    - Scaling parameters

    Deterministic transformations such as log transformations
    are not fitted because they do not learn parameters from data.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix.

    Returns
    -------
    dict
        Dictionary containing all fitted preprocessing artifacts.
    """

    # Fit imputers
    imputers = fit_imputers(X)

    # Apply imputation before fitting downstream components
    X_processed = transform_imputers(X, imputers)

    # Apply deterministic numeric transformations
    X_processed = log_transformation(X_processed)

    # Fit encoders
    encoders = fit_encoders(X_processed)

    # Apply encoding before fitting scalers
    X_processed = transform_encoders(
        X_processed,
        encoders
    )

    # Fit scalers
    scalers = fit_scalers(X_processed)

    return {
        "imputers": imputers,
        "encoders": encoders,
        "scalers": scalers
    }


def transform_preprocessing_pipeline(
    X,
    preprocessors
):
    """
    Apply the complete preprocessing pipeline.

    This function applies all previously fitted preprocessing
    components in the correct order.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataset to transform.

    preprocessors : dict
        Dictionary returned by
        fit_preprocessing_pipeline().

    Returns
    -------
    pandas.DataFrame
        Fully transformed dataset.
    """

    X_processed = X.copy()

    # Apply imputations
    X_processed = transform_imputers(
        X_processed,
        preprocessors["imputers"]
    )

    # Apply deterministic numeric transformations
    X_processed = log_transformation(
        X_processed
    )

    # Apply encodings
    X_processed = transform_encoders(
        X_processed,
        preprocessors["encoders"]
    )

    # Apply scaling
    X_processed = transform_scalers(
        X_processed,
        preprocessors["scalers"]
    )

    return X_processed