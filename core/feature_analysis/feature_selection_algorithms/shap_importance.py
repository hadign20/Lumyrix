import os
import numpy as np
import pandas as pd
import shap
from typing import List
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from core.visualization.plotting import plot_feature_importance_barplot


def run(df: pd.DataFrame, outcome_column: str, exclude_columns: List[str], task_type: str, config: dict) -> pd.DataFrame:
    """
    Compute SHAP values using a tree-based model and return mean absolute SHAP values per feature.
    """
    model_type = config.get("model", "random_forest")
    max_features = config.get("max_features", 20)
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    features = [col for col in df.columns if col not in exclude_columns + [outcome_column]]
    X = df[features].dropna()
    y = df[outcome_column].loc[X.index]

    # Normalize for regression
    if task_type == 'regression':
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    model = RandomForestClassifier(n_estimators=100, random_state=42) if task_type == 'classification' \
        else RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = np.abs(shap_values[1] if task_type == 'classification' else shap_values)
    else:
        shap_values = np.abs(shap_values)

    mean_abs_shap = np.mean(shap_values, axis=0)
    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP_Importance': mean_abs_shap
    }).sort_values(by='SHAP_Importance', ascending=False)

    shap_df = shap_df.head(max_features)
    plot_feature_importance_barplot(shap_df, score_column='SHAP_Importance',
                                    title="Top SHAP Feature Importances",
                                    output_path=os.path.join(output_dir, "shap_feature_importance.png"))

    return shap_df.reset_index(drop=True)

