import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List


def run(
    df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: List[str],
    task_type: str,
    config: dict
) -> pd.DataFrame:
    """
    Computes feature importance based on a Random Forest model.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        outcome_column (str): Target variable.
        exclude_columns (List[str]): Columns to exclude.
        task_type (str): 'classification' or 'regression'.
        config (dict): Contains config options like n_estimators, max_depth, etc.

    Returns:
        pd.DataFrame: Sorted dataframe of feature importances.
    """
    print("[Tree-Based Importance] Computing feature importances using Random Forest...")

    df = df.dropna()
    X = df.drop(columns=exclude_columns + [outcome_column])
    y = df[outcome_column]

    # Normalize features
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

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

    model.fit(X_scaled, y)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Tree_Importance": importances
    }).sort_values(by="Tree_Importance", ascending=False).reset_index(drop=True)

    return importance_df
