__all__ = ["run"]

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from core.visualization.plotting import plot_barplot
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def run(
    df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: List[str],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Lasso-based feature selection for regression or classification.

    Args:
        df (pd.DataFrame): Input dataframe with features and outcome.
        outcome_column (str): The outcome/target column name.
        exclude_columns (List[str]): List of columns to exclude.
        config (Dict): Lasso-specific config dictionary.

    Returns:
        pd.DataFrame: Features with non-zero coefficients from Lasso.
    """
    task_type = config["setup"]["task_type"]
    output_path = config["paths"]["output_dir"]
    cv_folds = config.get("cv_folds", 10)

    df = df.dropna()
    X = df.drop(columns=[outcome_column] + exclude_columns)
    y = df[outcome_column].copy()

    # Encode classification labels
    if task_type == "classification":
        if y.dtype == "object" or y.dtype.name == "category":
            y = LabelEncoder().fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Fit model
    if task_type == "regression":
        model = LassoCV(cv=cv_folds, random_state=42).fit(X_scaled, y)
        coefs = model.coef_
    elif task_type == "classification":
        model = LogisticRegressionCV(cv=cv_folds, penalty="l1", solver="liblinear", random_state=42).fit(X_scaled, y)
        coefs = model.coef_[0]
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coefs
    })

    selected_df = coef_df[coef_df["Coefficient"] != 0].copy()
    selected_df = selected_df.sort_values(by="Coefficient", key=np.abs, ascending=False)

    # Save
    selected_df.to_excel(os.path.join(output_path, "lasso_selected_features.xlsx"), index=False)

    # Plot
    plot_barplot(
        selected_df,
        x_col="Coefficient",
        y_col="Feature",
        title="Lasso Feature Coefficients",
        xlabel="Coefficient (Magnitude)",
        ylabel="Features",
        save_path=os.path.join(output_path, "lasso_feature_ranking.png"),
        orientation="horizontal"
    )

    return selected_df
