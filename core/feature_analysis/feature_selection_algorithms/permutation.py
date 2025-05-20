import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import List

def run(
    df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: List[str],
    task_type: str,
    config: dict
) -> pd.DataFrame:
    """
    Computes permutation feature importance using a random forest model.
    """
    print("[Permutation Importance] Running permutation-based importance...")

    df = df.dropna()
    X = df.drop(columns=exclude_columns + [outcome_column])
    y = df[outcome_column]

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            random_state=42
        )
    elif task_type == "regression":
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            random_state=42
        )
    else:
        raise ValueError("Invalid task_type. Must be 'classification' or 'regression'.")

    model.fit(X_train, y_train)
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Permutation_Importance": result.importances_mean
    }).sort_values(by="Permutation_Importance", ascending=False).reset_index(drop=True)

    return importance_df

def run_mutual_info(
    df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: List[str],
    categorical_columns: List[str],
    task_type: str
) -> pd.DataFrame:
    """
    Computes mutual information scores for each feature against the target.
    """
    print("[Mutual Information] Computing mutual information scores...")

    df = df.dropna()
    X = df.drop(columns=exclude_columns + [outcome_column])
    y = df[outcome_column]

    discrete_features = [X.columns.get_loc(col) for col in categorical_columns if col in X.columns]

    if task_type == "classification":
        mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=42)
    elif task_type == "regression":
        mi = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
    else:
        raise ValueError("Invalid task_type. Must be 'classification' or 'regression'.")

    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "Mutual_Info": mi
    }).sort_values(by="Mutual_Info", ascending=False).reset_index(drop=True)

    return mi_df
