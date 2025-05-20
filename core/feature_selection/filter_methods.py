# core/feature_selection/filter_methods.py

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def MRMR_feature_selection(
        df: pd.DataFrame,
        outcome_column: str,
        categorical_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        n_features: int = 15,
        cv_folds: int = 5,
        chunk_size: int = 1000  # Add parameter for large datasets
) -> pd.DataFrame:
    """
    Minimum Redundancy Maximum Relevance (MRMR) feature selection.

    Optimized version for large datasets.

    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        categorical_columns: List of categorical features
        exclude_columns: List of columns to exclude
        n_features: Number of features to select
        cv_folds: Number of cross-validation folds
        chunk_size: Size of feature chunks to process at once

    Returns:
        DataFrame with features and their selection counts
    """
    categorical_columns = categorical_columns or []
    exclude_columns = exclude_columns or []

    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")

    # Get outcome and feature columns
    y = df[outcome_column].values
    feature_columns = [col for col in df.columns if col not in exclude_columns + [outcome_column]]

    # For large datasets, pre-filter features using simple univariate methods
    if len(feature_columns) > chunk_size:
        logger.info(
            f"Large feature set detected ({len(feature_columns)} features). Pre-filtering using univariate methods.")

        # Calculate mutual information for all features
        X_df = df[feature_columns].copy()
        for col in categorical_columns:
            if col in X_df.columns:
                X_df[col] = X_df[col].astype('category').cat.codes

        X_df = X_df.fillna(X_df.mean())

        # Use mutual information to pre-filter
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_df, y, random_state=42)
        mi_features = pd.DataFrame({'Feature': feature_columns, 'MI_Score': mi_scores})
        mi_features = mi_features.sort_values(by='MI_Score', ascending=False)

        # Keep top chunk_size features for MRMR
        prefiltered_features = mi_features.head(chunk_size)['Feature'].tolist()
        logger.info(f"Pre-filtered to top {len(prefiltered_features)} features using mutual information")
        feature_columns = prefiltered_features

    # Convert categorical columns to numeric
    X_df = df[feature_columns].copy()
    for col in categorical_columns:
        if col in X_df.columns:
            X_df[col] = X_df[col].astype('category').cat.codes

    # Handle missing values
    X_df = X_df.fillna(X_df.mean())

    # Cross-validate to avoid overfitting
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Track feature selection count across folds
    feature_counts = {feature: 0 for feature in feature_columns}

    # MRMR feature selection on each fold
    for fold, (train_idx, _) in enumerate(cv.split(X_df, y)):
        X_train = X_df.iloc[train_idx].values
        y_train = y[train_idx]

        # Calculate mutual information (relevance) for each feature
        if np.unique(y_train).size <= 5:  # Classification
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        else:  # Regression
            mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

        relevance = dict(zip(feature_columns, mi_scores))

        # Initialize selected features with the most relevant one
        selected = [max(relevance.items(), key=lambda x: x[1])[0]]

        # Calculate feature correlation matrix
        correlation_matrix = np.corrcoef(X_train.T)
        feature_idx = {f: i for i, f in enumerate(feature_columns)}

        # Select remaining features
        remaining = [f for f in feature_columns if f not in selected]

        # Use lazy evaluation to improve performance
        batch_size = 50  # Process in batches for large feature sets
        while len(selected) < min(n_features, len(feature_columns)) and remaining:
            if len(remaining) > batch_size:
                # Process in batches
                batch_remaining = remaining[:batch_size]

                # Calculate MRMR score for batch
                mrmr_scores = {}
                for feature in batch_remaining:
                    # Relevance term
                    relevance_score = relevance[feature]

                    # Redundancy term (mean correlation with already selected features)
                    redundancy_score = 0
                    if selected:
                        for sel_feat in selected:
                            idx1 = feature_idx[feature]
                            idx2 = feature_idx[sel_feat]
                            redundancy_score += abs(correlation_matrix[idx1, idx2])
                        redundancy_score /= len(selected)

                    # MRMR score: relevance - redundancy
                    mrmr_scores[feature] = relevance_score - redundancy_score

                # Select feature with highest MRMR score in this batch
                next_feature = max(mrmr_scores.items(), key=lambda x: x[1])[0]
                selected.append(next_feature)
                remaining.remove(next_feature)

                # Remove processed batch features from remaining
                for feat in batch_remaining:
                    if feat in remaining:
                        remaining.remove(feat)
            else:
                # Process all remaining features at once for small sets
                mrmr_scores = {}
                for feature in remaining:
                    # Relevance term
                    relevance_score = relevance[feature]

                    # Redundancy term
                    redundancy_score = 0
                    if selected:
                        for sel_feat in selected:
                            idx1 = feature_idx[feature]
                            idx2 = feature_idx[sel_feat]
                            redundancy_score += abs(correlation_matrix[idx1, idx2])
                        redundancy_score /= len(selected)

                    mrmr_scores[feature] = relevance_score - redundancy_score

                # Select feature with highest MRMR score
                next_feature = max(mrmr_scores.items(), key=lambda x: x[1])[0]
                selected.append(next_feature)
                remaining.remove(next_feature)

        # Increment count for selected features
        for feature in selected:
            feature_counts[feature] += 1

        logger.debug(f"Fold {fold + 1}/{cv_folds}: Selected {len(selected)} features")

    # Create ranking DataFrame
    ranking = [{"Feature": feature, "Count": count} for feature, count in feature_counts.items()]
    ranking_df = pd.DataFrame(ranking)
    ranking_df = ranking_df.sort_values(by="Count", ascending=False)

    logger.info(f"Completed MRMR feature selection across {cv_folds} folds")

    return ranking_df