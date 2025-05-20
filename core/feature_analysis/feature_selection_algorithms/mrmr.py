__all__ = ["run"]

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from mrmr import mrmr_classif
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from core.visualization.plotting import plot_barplot


def run(
    df: pd.DataFrame,
    outcome_column: str,
    categorical_columns: List[str],
    exclude_columns: List[str],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Perform MRMR feature ranking using cross-validation.
    Only supports classification.

    Args:
        df (pd.DataFrame): Data with features and outcome.
        outcome_column (str): Target column name.
        categorical_columns (List[str]): List of categorical features.
        exclude_columns (List[str]): Columns to exclude from analysis.
        config (Dict): Dictionary of config params like K, folds, and output path.

    Returns:
        pd.DataFrame: Ranked features based on MRMR counts.
    """
    if config["setup"]["task_type"] != "classification":
        print("[MRMR] Skipped: Only supports classification tasks.")
        return pd.DataFrame(columns=["Feature", "MRMR_Count"])

    df = df.dropna()
    X = df.drop(columns=[outcome_column] + exclude_columns)
    y = df[outcome_column].copy()

    # Encode labels if not numeric
    if y.dtype == "object" or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y)

    num_features = config.get("num_features", 10)
    cv_folds = config.get("cv_folds", 10)
    output_path = config["paths"]["output_dir"]

    print(f"[MRMR] Running {cv_folds}-fold CV with top {num_features} features")

    selected_feature_count = defaultdict(int)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fold, (train_idx, _) in enumerate(kf.split(X)):
        x_train = X.iloc[train_idx]
        y_train = y[train_idx]
        try:
            selected = mrmr_classif(X=x_train, y=y_train, K=num_features)
            for f in selected:
                selected_feature_count[f] += 1
        except Exception as e:
            print(f"[MRMR] Fold {fold+1}: Error - {e}")

    result_df = pd.DataFrame(list(selected_feature_count.items()), columns=["Feature", "MRMR_Count"])
    result_df = result_df.sort_values(by="MRMR_Count", ascending=False)

    # Save table and barplot
    result_df.to_excel(os.path.join(output_path, "mrmr_feature_ranking.xlsx"), index=False)
    plot_barplot(
        result_df,
        x_col="MRMR_Count",
        y_col="Feature",
        title="MRMR Feature Ranking",
        xlabel="Selection Count (across CV folds)",
        ylabel="Features",
        save_path=os.path.join(output_path, "mrmr_feature_ranking.png"),
        orientation="horizontal"
    )

    return result_df
