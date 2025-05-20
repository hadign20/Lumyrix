
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    precision_recall_curve, roc_curve
)
from core.visualization.plotting import plot_roc_curve
from core.classical_ml.metrics import classification_metrics, regression_metrics

def evaluate_models(models: dict, X_val, y_val, task_type: str, output_dir: str) -> pd.DataFrame:
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_val)

        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_val)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            else:
                y_prob = None

            metrics = classification_metrics(y_val, y_pred, y_prob)
            if y_prob is not None:
                plot_roc_curve(y_val, y_prob, name, output_dir)
        else:
            metrics = regression_metrics(y_val, y_pred)

        results.append({"Model": name, **metrics})

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_dir, "validation_scores.xlsx"), index=False)
    return results_df

def get_classification_curves(y_true, y_scores) -> Dict[str, Any]:
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        return {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
        }
    except ValueError:
        return {
            "fpr": [],
            "tpr": [],
            "roc_thresholds": [],
            "precision": [],
            "recall": [],
            "pr_thresholds": [],
        }

def get_confusion_matrix(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])

def summarize_metrics(metric_list: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(metric_list)
    summary = df.agg(["mean", "std"])
    return summary.reset_index().rename(columns={"index": "Metric"})
