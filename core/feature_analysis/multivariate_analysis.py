__all__ = ["run"]

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Optional
from sklearn.preprocessing import StandardScaler


def run_multivariate_analysis(
    df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    task_type: str = "classification"
) -> pd.DataFrame:
    """
    Run multivariate analysis (logistic or linear regression) on all selected features.

    Args:
        df (pd.DataFrame): The input dataframe with all features and outcome.
        outcome_column (str): The name of the outcome column.
        exclude_columns (List[str], optional): List of columns to exclude.
        categorical_columns (List[str], optional): Columns to one-hot encode.
        task_type (str): Either 'classification' or 'regression'.

    Returns:
        pd.DataFrame: Summary of multivariate analysis with coefficient and p-value.
    """
    if exclude_columns is None:
        exclude_columns = []
    if categorical_columns is None:
        categorical_columns = []

    df = df.dropna()
    df = df.copy()
    features = [col for col in df.columns if col not in exclude_columns + [outcome_column]]

    X = df[features]

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=[col for col in categorical_columns if col in X.columns], drop_first=True)

    # Add constant
    X = sm.add_constant(X)

    # Standardize continuous variables
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    y = df[outcome_column]

    if task_type == "classification":
        model = sm.Logit(y, X)
    elif task_type == "regression":
        model = sm.OLS(y, X)
    else:
        raise ValueError("Invalid task_type. Must be 'classification' or 'regression'.")

    try:
        result = model.fit(disp=0)
        summary_df = pd.DataFrame({
            "Feature": result.params.index,
            "Coefficient": result.params.values,
            "P_Value": result.pvalues.values
        }).sort_values(by="P_Value")
    except Exception as e:
        print(f"[Multivariate Analysis] Error fitting model: {e}")
        summary_df = pd.DataFrame()

    return summary_df


# ✅ Top-level entrypoint required for pipeline compatibility
def run(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    return run_multivariate_analysis(
        df=df,
        outcome_column=config["setup"]["outcome_column"],
        exclude_columns=config["setup"].get("exclude_columns", []),
        categorical_columns=config["setup"].get("categorical_columns", []),
        task_type=config["setup"]["task_type"]
    )
