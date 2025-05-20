# core/visualization/plotting.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Union, Dict
import logging
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)

from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(
        corr_matrix: pd.DataFrame,
        title: str = 'Correlation Matrix',
        cmap: str = 'coolwarm',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        mask_upper: bool = False,
        annot: bool = True
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.

    Args:
        corr_matrix: Correlation matrix as DataFrame
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        save_path: Path to save the figure
        mask_upper: Whether to mask the upper triangle
        annot: Whether to annotate cells

    Returns:
        Matplotlib figure
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Create mask for upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=annot,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .7},
        fmt=".2f"
    )

    plt.title(title)
    plt.tight_layout()

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


# Update this function in core/visualization/plotting.py

def plot_feature_importance(
        df: pd.DataFrame,
        x_col: str = 'Importance',
        y_col: str = 'Feature',
        title: str = 'Feature Importance',
        color: str = 'skyblue',
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        df: DataFrame with feature names and importance scores
        x_col: Column name for importance values
        y_col: Column name for feature names
        title: Plot title
        color: Bar color
        top_n: Number of top features to show (None for all)
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if y_col not in df.columns or x_col not in df.columns:
        logger.warning(f"Required columns '{y_col}' or '{x_col}' not found in dataframe")
        return None

    # Remove rows with NaN values in importance column
    valid_df = df.dropna(subset=[x_col])

    if len(valid_df) == 0:
        logger.warning(f"No valid importance values to plot (all NaN)")
        # Create an empty plot with a message
        plt.figure(figsize=figsize)
        plt.text(0.5, 0.5, "No valid importance values available",
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.title(title)

        # Save empty figure if filepath provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        fig = plt.gcf()
        plt.close()

        return fig

    # Sort by importance
    sorted_df = valid_df.sort_values(by=x_col, ascending=False)

    # Limit to top N features if specified
    if top_n is not None:
        sorted_df = sorted_df.head(top_n)

    # Create plot
    plt.figure(figsize=figsize)
    plt.barh(sorted_df[y_col], sorted_df[x_col], color=color)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)

    # Add values on bars
    for i, v in enumerate(sorted_df[x_col]):
        plt.text(v + 0.01, i, f"{v:.3f}", va='center')

    # Adjust layout
    plt.tight_layout()

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


# Update this function in core/visualization/plotting.py

def plot_pvalue_distribution(
        df: pd.DataFrame,
        p_col: str = 'P_Value',
        title: str = 'P-Value Distribution',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot p-value distribution.

    Args:
        df: DataFrame with p-values
        p_col: Column name for p-values
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if p_col not in df.columns:
        logger.warning(f"P-value column '{p_col}' not found in dataframe")
        return None

    # Create figure
    plt.figure(figsize=figsize)

    # Filter out NaN values
    valid_values = df[p_col].dropna()

    if len(valid_values) == 0:
        logger.warning(f"No valid p-values to plot (all NaN)")
        # Create an empty plot with a message
        plt.text(0.5, 0.5, "No valid p-values available",
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.title(title)

        # Save empty figure if filepath provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        fig = plt.gcf()
        plt.close()

        return fig

    # Plot histogram
    plt.hist(valid_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Add reference line at p=0.05
    plt.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')

    # Add formatting
    plt.xlabel('P-Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    # Add annotation with significant feature count
    sig_count = sum(valid_values <= 0.05)
    plt.annotate(f"Significant features (p≤0.05): {sig_count}/{len(valid_values)}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
                 verticalalignment='top')

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


# Update this function in core/visualization/plotting.py

def plot_auc_distribution(
        df: pd.DataFrame,
        auc_col: str = 'AUC',
        title: str = 'AUC Distribution',
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot AUC distribution.

    Args:
        df: DataFrame with AUC values
        auc_col: Column name for AUC values
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if auc_col not in df.columns:
        logger.warning(f"AUC column '{auc_col}' not found in dataframe")
        return None

    # Create figure
    plt.figure(figsize=figsize)

    # Filter out NaN values
    valid_values = df[auc_col].dropna()

    if len(valid_values) == 0:
        logger.warning(f"No valid AUC values to plot (all NaN)")
        # Create an empty plot with a message
        plt.text(0.5, 0.5, "No valid AUC values available",
                 ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
        plt.title(title)

        # Save empty figure if filepath provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        fig = plt.gcf()
        plt.close()

        return fig

    # Plot histogram
    plt.hist(valid_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Add reference line at AUC=0.5 (random chance)
    plt.axvline(x=0.5, color='red', linestyle='--', label='AUC=0.5 (random)')

    # Add reference line at AUC=0.7 (good performance)
    plt.axvline(x=0.7, color='green', linestyle='--', label='AUC=0.7 (good)')

    # Add formatting
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    # Add annotation with good feature count
    good_count = sum(valid_values >= 0.7)
    plt.annotate(f"Good features (AUC≥0.7): {good_count}/{len(valid_values)}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
                 verticalalignment='top')

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_pvalue_heatmap(
        p_values: pd.DataFrame,
        title: str = 'P-Value Heatmap',
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot p-values as a heatmap.

    Args:
        p_values: DataFrame with p-values (features in rows, methods in columns)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Create colormap
    cmap = plt.cm.YlOrRd_r

    # Create heatmap
    sns.heatmap(
        -np.log10(p_values),
        annot=True,
        cmap=cmap,
        linewidths=.5,
        fmt=".2f",
        cbar_kws={"label": "-log10(p-value)"}
    )

    plt.title(title)
    plt.tight_layout()

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_feature_distribution_comparison(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of feature distributions between training and validation sets.

    Args:
        df_train: Training dataframe
        df_val: Validation dataframe
        feature: Feature name to compare
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    if feature not in df_train.columns or feature not in df_val.columns:
        logger.warning(f"Feature '{feature}' not found in both datasets")
        return None

    plt.figure(figsize=figsize)

    # Get data
    train_values = df_train[feature].dropna()
    val_values = df_val[feature].dropna()

    # Calculate statistics for annotation
    from scipy.stats import ks_2samp
    ks_stat, ks_pval = ks_2samp(train_values, val_values)

    train_mean = train_values.mean()
    train_std = train_values.std()
    val_mean = val_values.mean()
    val_std = val_values.std()

    # Create histogram
    plt.hist(train_values, bins=30, alpha=0.6, label=f'Train (n={len(train_values)})', color='blue')
    plt.hist(val_values, bins=30, alpha=0.6, label=f'Validation (n={len(val_values)})', color='red')

    # Add vertical lines for means
    plt.axvline(train_mean, color='blue', linestyle='dashed', linewidth=2, label=f'Train mean: {train_mean:.3f}')
    plt.axvline(val_mean, color='red', linestyle='dashed', linewidth=2, label=f'Val mean: {val_mean:.3f}')

    # Add title and annotation with statistics
    if title is None:
        title = f"Distribution Comparison for {feature}"

    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    annotation = (f"KS test p-value: {ks_pval:.4f}\n"
                  f"Train mean: {train_mean:.3f}, std: {train_std:.3f}\n"
                  f"Val mean: {val_mean:.3f}, std: {val_std:.3f}\n"
                  f"Mean diff: {abs(train_mean - val_mean):.3f}")

    plt.annotate(annotation, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
                 verticalalignment='top')

    plt.legend()
    plt.grid(alpha=0.3)

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'ROC Curve',
        figsize: Tuple[int, int] = (8, 6),
        filepath: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve with confidence interval.

    Args:
        y_true: True labels
        y_proba: Probability predictions
        title: Plot title
        figsize: Figure size
        filepath: Path to save the figure

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate confidence intervals
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_aucs = []

    # Bootstrap sampling
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # Skip if bootstrap sample contains only one class
            continue

        fpr, tpr, _ = roc_curve(y_true[indices], y_proba[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))

    # Calculate 95% confidence interval
    auc_ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    auc_ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f}, 95% CI: [{auc_ci_lower:.3f}-{auc_ci_upper:.3f}])')

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # Add formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save figure if filepath provided
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_calibration_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Calibration Curve',
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 6),
        filepath: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve to show reliability of probability estimates.

    Args:
        y_true: True labels
        y_proba: Probability predictions
        title: Plot title
        n_bins: Number of bins for calibration
        figsize: Figure size
        filepath: Path to save the figure

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    # Plot calibration curve
    plt.plot(prob_pred, prob_true, 's-', color='blue', lw=2, label='Calibration curve')

    # Plot perfect calibration reference line
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Perfectly calibrated')

    # Add formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save figure if filepath provided
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = 'Confusion Matrix',
        labels: List[str] = ['Negative', 'Positive'],
        normalize: bool = False,
        figsize: Tuple[int, int] = (8, 6),
        filepath: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        labels: Class labels
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
        filepath: Path to save the figure

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    # Add class labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Save figure if filepath provided
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Precision-Recall Curve',
        figsize: Tuple[int, int] = (8, 6),
        filepath: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_proba: Probability predictions
        title: Plot title
        figsize: Figure size
        filepath: Path to save the figure

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    # Plot precision-recall curve
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')

    # Add formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    # Save figure if filepath provided
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


def plot_decision_curve_analysis(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = 'Decision Curve Analysis',
        figsize: Tuple[int, int] = (8, 6),
        filepath: Optional[str] = None
) -> plt.Figure:
    """
    Plot decision curve analysis showing net benefit across threshold probabilities.

    Args:
        y_true: True labels
        y_proba: Probability predictions
        title: Plot title
        figsize: Figure size
        filepath: Path to save the figure

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Generate threshold probabilities
    threshold_probabilities = np.linspace(0.01, 0.99, 99)

    # Calculate net benefit for model
    net_benefit_model = []

    for threshold in threshold_probabilities:
        # Calculate number of true positives and false positives at this threshold
        y_pred_thresh = (y_proba >= threshold).astype(int)

        TP = np.sum((y_pred_thresh == 1) & (y_true == 1))
        FP = np.sum((y_pred_thresh == 1) & (y_true == 0))
        n = len(y_true)

        # Calculate net benefit
        if TP + FP == 0:
            net_benefit = 0
        else:
            net_benefit = (TP / n) - (FP / n) * (threshold / (1 - threshold))

        net_benefit_model.append(net_benefit)

    # Calculate net benefit for "treat all" strategy
    net_benefit_all = []
    prevalence = np.mean(y_true)

    for threshold in threshold_probabilities:
        net_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        net_benefit_all.append(max(net_benefit, 0))

    # Plot decision curves
    plt.plot(threshold_probabilities, net_benefit_model, color='blue', lw=2, label='Model')
    plt.plot(threshold_probabilities, net_benefit_all, color='red', lw=2, label='Treat all')
    plt.plot([0, 1], [0, 0], color='gray', lw=1, linestyle='--', label='Treat none')

    # Add formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, max(max(net_benefit_model), max(net_benefit_all)) + 0.05])
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    # Save figure if filepath provided
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig


# Add this function to core/visualization/plotting.py

def plot_barplot(
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        color: str = 'skyblue',
        figsize: Tuple[int, int] = (10, 6),
        horizontal: bool = False,
        save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar plot from DataFrame columns.

    Args:
        data: Input DataFrame
        x: Column name for x-axis
        y: Column name for y-axis
        title: Plot title
        xlabel: X-axis label (defaults to x if None)
        ylabel: Y-axis label (defaults to y if None)
        color: Bar color
        figsize: Figure size
        horizontal: Whether to create a horizontal bar plot
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)

    # Set default labels if not provided
    if xlabel is None:
        xlabel = x
    if ylabel is None:
        ylabel = y

    # Create plot
    if horizontal:
        # Horizontal bar plot
        bars = plt.barh(data[x], data[y], color=color)
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
    else:
        # Vertical bar plot
        bars = plt.bar(data[x], data[y], color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    # Add title
    if title:
        plt.title(title)

    # Add grid
    plt.grid(axis='both' if horizontal else 'y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        if horizontal:
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height() / 2
            plt.text(width + 0.01, y_pos, f"{width:.2f}",
                     ha='left', va='center')
        else:
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            plt.text(x_pos, height + 0.01, f"{height:.2f}",
                     ha='center', va='bottom')

    # Adjust layout
    plt.tight_layout()

    # Save figure if filepath provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    fig = plt.gcf()
    plt.close()

    return fig