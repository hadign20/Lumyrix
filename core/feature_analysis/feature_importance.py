# core/feature_analysis/feature_importance.py

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.inspection import permutation_importance
import shap

logger = logging.getLogger(__name__)


def calculate_tree_importance(
        df: pd.DataFrame,
        outcome_column: str,
        exclude_columns: Optional[List[str]] = None,
        n_estimators: int = 100,
        cv_folds: int = 5,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.

    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        exclude_columns: Columns to exclude
        n_estimators: Number of trees in Random Forest
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility

    Returns:
        DataFrame with features and their importance scores
    """
    exclude_columns = exclude_columns or []

    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")

    # Get features and outcome
    X_columns = [col for col in df.columns if col not in exclude_columns + [outcome_column]]
    X = df[X_columns].values
    y = df[outcome_column].values

    # Determine if classification or regression
    unique_values = np.unique(y)
    is_classification = len(unique_values) <= 10  # Assume classification if ≤10 unique values

    # Cross-validation split
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        forest = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    # Calculate importance with cross-validation
    importance_values = np.zeros(len(X_columns))

    for train_idx, _ in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]

        # Train model
        forest.fit(X_train, y_train)

        # Accumulate feature importance
        importance_values += forest.feature_importances_

    # Average importance across folds
    importance_values /= cv_folds

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_columns,
        'Importance': importance_values
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    logger.info(f"Calculated tree importance for {len(X_columns)} features using {cv_folds}-fold CV")

    return importance_df


def calculate_permutation_importance(
        df: pd.DataFrame,
        outcome_column: str,
        exclude_columns: Optional[List[str]] = None,
        model_type: str = "random_forest",
        n_repeats: int = 10,
        cv_folds: int = 5,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation feature importance.

    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        exclude_columns: Columns to exclude
        model_type: Type of model to use ("random_forest", "svm", etc.)
        n_repeats: Number of times to permute each feature
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility

    Returns:
        DataFrame with features and their importance scores
    """
    exclude_columns = exclude_columns or []

    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")

    # Get features and outcome
    X_columns = [col for col in df.columns if col not in exclude_columns + [outcome_column]]
    X = df[X_columns].values
    y = df[outcome_column].values

    # Determine if classification or regression
    unique_values = np.unique(y)
    is_classification = len(unique_values) <= 10  # Assume classification if ≤10 unique values

    # Get model based on type
    if model_type == "random_forest":
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == "svm":
        if is_classification:
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=random_state)
        else:
            from sklearn.svm import SVR
            model = SVR()
    elif model_type == "linear":
        if is_classification:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=random_state)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Cross-validation split
    if is_classification:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'roc_auc'
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = 'r2'

    # Calculate permutation importance with cross-validation
    result = permutation_importance(model, X, y,
                                    n_repeats=n_repeats,
                                    random_state=random_state,
                                    n_jobs=-1,
                                    scoring=scoring)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_columns,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    logger.info(f"Calculated permutation importance for {len(X_columns)} features using {model_type}")

    return importance_df


def calculate_shap_importance(
        df: pd.DataFrame,
        outcome_column: str,
        exclude_columns: Optional[List[str]] = None,
        model_type: str = "tree",
        n_samples: int = 100,
        random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate SHAP feature importance values.

    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        exclude_columns: Columns to exclude
        model_type: Type of model to use ("tree", "linear", etc.)
        n_samples: Number of samples for SHAP explanation
        random_state: Random state for reproducibility

    Returns:
        DataFrame with features and their SHAP importance scores
    """
    exclude_columns = exclude_columns or []

    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")

    # Get features and outcome
    X_columns = [col for col in df.columns if col not in exclude_columns + [outcome_column]]
    X = df[X_columns]
    y = df[outcome_column]

    # Determine if classification or regression
    unique_values = np.unique(y)
    is_classification = len(unique_values) <= 10  # Assume classification if ≤10 unique values

    # Get model based on type
    if model_type == "tree":
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == "linear":
        if is_classification:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=random_state)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
    elif model_type == "xgboost":
        try:
            if is_classification:
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=random_state)
            else:
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=random_state)
        except ImportError:
            logger.warning("XGBoost not available, falling back to Random Forest")
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    model.fit(X, y)

    # Calculate SHAP values
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)

    # Limit to a reasonable number of samples for efficiency
    X_sample = X.sample(min(n_samples, len(X)), random_state=random_state)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        # For multi-class classification, use class 1 (positive class)
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_columns,
        'SHAP_MeanAbs': mean_abs_shap
    })

    # Sort by importance (descending)
    importance_df = importance_df.sort_values(by='SHAP_MeanAbs', ascending=False)

    logger.info(f"Calculated SHAP importance for {len(X_columns)} features using {model_type}")

    return importance_df


def calculate_auc_values(
        df: pd.DataFrame,
        outcome_column: str,
        categorical_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        cv_folds: int = 5
) -> pd.DataFrame:
    """
    Calculate AUC values for each feature with respect to the outcome variable.

    Uses cross-validated logistic regression to handle potential overfitting.

    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        categorical_columns: List of categorical feature names
        exclude_columns: List of columns to exclude
        cv_folds: Number of cross-validation folds

    Returns:
        DataFrame with features and corresponding AUC values
    """
    categorical_columns = categorical_columns or []
    exclude_columns = exclude_columns or []

    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")

    # Get outcome values
    y = df[outcome_column].values

    # Initialize scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Initialize results
    results = []

    # Process each feature
    for column in df.columns:
        # Skip excluded columns and outcome
        if column in exclude_columns or column == outcome_column:
            continue

        try:
            # Handle categorical features (convert to numeric codes)
            if column in categorical_columns:
                X = df[column].astype('category').cat.codes.values.reshape(-1, 1)
            else:
                X = df[column].values.reshape(-1, 1)

            # Skip if feature has all same values
            if np.unique(X).size <= 1:
                logger.warning(f"Feature '{column}' has only one unique value, skipping AUC calculation")
                results.append({
                    'Feature': column,
                    'AUC': 0.5
                })
                continue

            # Scale feature
            X = scaler.fit_transform(X)

            # Set up cross-validation
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Use logistic regression with cross-validation
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

            # Calculate mean AUC
            mean_auc = np.mean(auc_scores)

            # Ensure AUC >= 0.5 (flip if necessary)
            if mean_auc < 0.5:
                mean_auc = 1 - mean_auc

            # Add to results
            results.append({
                'Feature': column,
                'AUC': mean_auc
            })

        except Exception as e:
            logger.warning(f"Error calculating AUC for feature '{column}': {str(e)}")
            results.append({
                'Feature': column,
                'AUC': 0.5
            })

    # Create DataFrame and sort by AUC (descending)
    auc_values_df = pd.DataFrame(results)
    auc_values_df = auc_values_df.sort_values(by='AUC', ascending=False)

    logger.info(f"Calculated AUC values for {len(results)} features")

    return auc_values_df