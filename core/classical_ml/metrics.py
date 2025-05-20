
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
    classification_report
)

def classification_metrics(y_true, y_pred, y_prob=None, average='binary'):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    return metrics

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2_Score": r2_score(y_true, y_pred)
    }

def get_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)

def full_classification_report(y_true, y_pred, labels=None):
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report_dict).transpose()

def summarize_metrics(metric_df_list):
    summary = pd.concat(metric_df_list, axis=1)
    mean = summary.mean(axis=1)
    std = summary.std(axis=1)
    formatted = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
    return pd.DataFrame({"Mean±SD": formatted})
