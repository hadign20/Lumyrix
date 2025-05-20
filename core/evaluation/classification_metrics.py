# core/evaluation/classification_metrics.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

logger = logging.getLogger(__name__)

def calculate_classification_metrics(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_proba: Optional[Union[List, np.ndarray, pd.Series]] = None,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Probability predictions (for AUC)
        threshold: Probability threshold for binary classification
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        
        # Apply threshold if provided
        if threshold is not None:
            y_pred = (y_proba >= threshold).astype(int)
    
    # Get confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # Handle non-binary case or empty predictions
        logger.warning("Could not calculate confusion matrix, falling back to individual metrics")
        tp, tn, fp, fn = 0, 0, 0, 0
    
    # Calculate metrics
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # AUC
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except:
            metrics["roc_auc"] = 0.5
    
    # Precision, recall, F1
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity and specificity
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # PPV and NPV
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # core/evaluation/classification_metrics.py (continued)
    # Store confusion matrix components
    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    
    # Store threshold if provided
    if threshold is not None:
        metrics["threshold"] = threshold
    
    return metrics

def compute_confidence_interval(
    y_true: Union[List, np.ndarray, pd.Series],
    y_pred: Union[List, np.ndarray, pd.Series],
    y_proba: Optional[Union[List, np.ndarray, pd.Series]] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for classification metrics using bootstrap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Probability predictions (for AUC)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 95%)
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of metric names and confidence intervals
    """
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
    
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Calculate alpha for confidence interval
    alpha = (1 - confidence_level) / 2
    
    # Initialize bootstrap distributions
    bootstrap_metrics = {
        "accuracy": [],
        "roc_auc": [],
        "sensitivity": [],
        "specificity": [],
        "ppv": [],
        "npv": [],
        "f1_score": []
    }
    
    # Perform bootstrap
    for _ in range(n_bootstrap):
        # Sample indices with replacement
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        
        # Get bootstrap sample
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if y_proba is not None:
            y_proba_boot = y_proba[indices]
        else:
            y_proba_boot = None
        
        # Calculate metrics for bootstrap sample
        metrics = calculate_classification_metrics(y_true_boot, y_pred_boot, y_proba_boot)
        
        # Store metrics
        for metric in bootstrap_metrics.keys():
            if metric in metrics:
                bootstrap_metrics[metric].append(metrics[metric])
    
    # Calculate confidence intervals
    confidence_intervals = {}
    
    for metric, values in bootstrap_metrics.items():
        if values:
            lower = np.percentile(values, 100 * alpha)
            upper = np.percentile(values, 100 * (1 - alpha))
            confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals
    