# core/feature_analysis/feature_correlation.py

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
        df: pd.DataFrame,
        method: str = 'spearman',
        exclude_columns: Optional[List[str]] = None,
        max_features: int = 1000  # Add threshold for large datasets
) -> pd.DataFrame:
    """
    Compute correlation matrix for features in the dataframe.

    Args:
        df: Input dataframe
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        exclude_columns: Columns to exclude from correlation calculation
        max_features: Maximum number of features to process at once

    Returns:
        Correlation matrix as DataFrame
    """
    exclude_columns = exclude_columns or []

    # Filter columns
    columns = [col for col in df.columns if col not in exclude_columns]

    if len(columns) <= 1:
        logger.warning("Not enough features to compute correlation matrix")
        return pd.DataFrame()

    # Get numeric columns
    numeric_df = df[columns].select_dtypes(include=['number'])

    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation calculation")
        return pd.DataFrame()

    # Handle large datasets by chunking
    n_features = len(numeric_df.columns)
    logger.info(f"Computing correlation matrix for {n_features} features")

    if n_features > max_features:
        logger.warning(f"Large feature set detected ({n_features} features). Using chunked correlation calculation.")

        # Split into manageable chunks
        chunks = [numeric_df.columns[i:i + max_features] for i in range(0, n_features, max_features)]
        corr_matrix = pd.DataFrame(index=numeric_df.columns, columns=numeric_df.columns)

        # Compute correlation in chunks
        for i, chunk1 in enumerate(chunks):
            # Diagonal blocks
            chunk_corr = numeric_df[chunk1].corr(method=method)
            corr_matrix.loc[chunk1, chunk1] = chunk_corr

            # Off-diagonal blocks
            for j, chunk2 in enumerate(chunks[i + 1:], i + 1):
                logger.info(f"Computing correlation chunk {i + 1}/{len(chunks)} with chunk {j + 1}/{len(chunks)}")
                chunk_corr = numeric_df[chunk1].corrwith(numeric_df[chunk2], method=method, axis=0).unstack()
                corr_matrix.loc[chunk1, chunk2] = chunk_corr
                corr_matrix.loc[chunk2, chunk1] = chunk_corr.T

        # Fill diagonal with 1.0
        np.fill_diagonal(corr_matrix.values, 1.0)
        return corr_matrix

    # For smaller datasets, use standard method
    try:
        corr_matrix = numeric_df.corr(method=method)
        logger.info(f"Computed {method} correlation matrix with shape {corr_matrix.shape}")
        return corr_matrix
    except Exception as e:
        logger.error(f"Error computing correlation matrix: {str(e)}")
        return pd.DataFrame()

def find_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.8,
    method: str = 'spearman',
    exclude_columns: Optional[List[str]] = None
) -> List[Tuple[str, str, float]]:
    """
    Find pairs of highly correlated features.
    
    Args:
        df: Input dataframe
        threshold: Correlation threshold
        method: Correlation method
        exclude_columns: Columns to exclude
        
    Returns:
        List of tuples (feature1, feature2, correlation)
    """
    corr_matrix = compute_correlation_matrix(df, method, exclude_columns)
    
    if corr_matrix.empty:
        return []
    
    # Find highly correlated pairs
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            
            if abs(corr) >= threshold:
                pairs.append((col1, col2, corr))
    
    # Sort by absolute correlation (descending)
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return pairs

def remove_collinear_features(
    df: pd.DataFrame,
    threshold: float = 0.8,
    method: str = 'spearman',
    exclude_columns: Optional[List[str]] = None,
    favor_higher_auc: bool = False,
    auc_values: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Remove highly correlated features from the dataframe.
    
    Args:
        df: Input dataframe
        threshold: Correlation threshold for removal
        method: Correlation method
        exclude_columns: Columns to exclude from correlation calculation and removal
        favor_higher_auc: Whether to favor features with higher AUC when removing
        auc_values: Dictionary mapping feature names to AUC values
        
    Returns:
        Dataframe with correlated features removed
    """
    exclude_columns = exclude_columns or []
    result_df = df.copy()
    
    # Get columns for correlation analysis
    corr_columns = [col for col in df.columns if col not in exclude_columns]
    
    if len(corr_columns) <= 1:
        logger.info("Not enough features to check for collinearity")
        return result_df
    
    # Calculate correlation matrix
    corr_matrix = compute_correlation_matrix(df[corr_columns], method)
    
    if corr_matrix.empty:
        return result_df
    
    # Find features to drop
    to_drop = set()
    seen_pairs = set()
    
    # Iterate over pairs of features
    for i in range(len(corr_columns)):
        for j in range(i+1, len(corr_columns)):
            col1 = corr_columns[i]
            col2 = corr_columns[j]
            
            # Skip if either column already marked for dropping
            if col1 in to_drop or col2 in to_drop:
                continue
            
            # Check if correlation exceeds threshold
            if abs(corr_matrix.loc[col1, col2]) > threshold:
                # Determine which feature to drop
                if favor_higher_auc and auc_values:
                    # Drop feature with lower AUC
                    auc1 = auc_values.get(col1, 0)
                    auc2 = auc_values.get(col2, 0)
                    drop_col = col1 if auc1 < auc2 else col2
                else:
                    # Default: drop feature with higher mean absolute correlation
                    mean_corr1 = corr_matrix[col1].abs().mean()
                    mean_corr2 = corr_matrix[col2].abs().mean()
                    drop_col = col1 if mean_corr1 > mean_corr2 else col2
                
                to_drop.add(drop_col)
                seen_pairs.add((col1, col2))
                logger.debug(f"Marking {drop_col} for removal (correlated with {col1 if drop_col == col2 else col2}, ρ={corr_matrix.loc[col1, col2]:.3f})")
    
    # Drop selected columns
    if to_drop:
        to_drop_list = list(to_drop)
        result_df = result_df.drop(columns=to_drop_list)
        logger.info(f"Removed {len(to_drop_list)} collinear features: {', '.join(to_drop_list)}")
    else:
        logger.info("No collinear features found above threshold")
    
    return result_df

def remove_collinear_features_with_priority(
    df: pd.DataFrame, 
    threshold: float = 0.8, 
    method: str = 'spearman',
    auc_values: Optional[Dict[str, float]] = None,
    p_values: Optional[Dict[str, float]] = None,
    priority_features: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Remove highly correlated features with priority-based selection.
    
    When two features are highly correlated, the one with higher AUC, lower p-value,
    or in priority_features will be kept.
    
    Args:
        df: Input dataframe
        threshold: Correlation threshold for removal
        method: Correlation method
        auc_values: Dictionary mapping feature names to AUC values
        p_values: Dictionary mapping feature names to p-values
        priority_features: List of features to prioritize (not remove if possible)
        exclude_columns: Columns to exclude from correlation calculation and removal
        
    Returns:
        Dataframe with correlated features removed
    """
    exclude_columns = exclude_columns or []
    priority_features = priority_features or []
    auc_values = auc_values or {}
    p_values = p_values or {}
    
    result_df = df.copy()
    
    # Get columns for correlation analysis
    corr_columns = [col for col in df.columns if col not in exclude_columns]
    
    if len(corr_columns) <= 1:
        logger.info("Not enough features to check for collinearity")
        return result_df
    
    # Calculate correlation matrix
    corr_matrix = compute_correlation_matrix(df[corr_columns], method)
    
    if corr_matrix.empty:
        return result_df
    
    # Find features to drop
    to_drop = set()
    
    # Iterate over pairs of features
    for i in range(len(corr_columns)):
        for j in range(i+1, len(corr_columns)):
            col1 = corr_columns[i]
            col2 = corr_columns[j]
            
            # Skip if either column already marked for dropping
            if col1 in to_drop or col2 in to_drop:
                continue
            
            # Check if correlation exceeds threshold
            if abs(corr_matrix.loc[col1, col2]) > threshold:
                # Determine which feature to drop based on priorities
                
                # Priority 1: Keep features in priority_features list
                if col1 in priority_features and col2 not in priority_features:
                    drop_col = col2
                elif col2 in priority_features and col1 not in priority_features:
                    drop_col = col1
                # Priority 2: Keep features with higher AUC
                elif col1 in auc_values and col2 in auc_values:
                    drop_col = col1 if auc_values[col1] < auc_values[col2] else col2
                # Priority 3: Keep features with lower p-value
                elif col1 in p_values and col2 in p_values:
                    drop_col = col1 if p_values[col1] > p_values[col2] else col2
                # Priority 4: Drop feature with higher mean correlation
                else:
                    mean_corr1 = corr_matrix[col1].abs().mean()
                    mean_corr2 = corr_matrix[col2].abs().mean()
                    drop_col = col1 if mean_corr1 > mean_corr2 else col2
                
                to_drop.add(drop_col)
                logger.debug(f"Marking {drop_col} for removal (correlated with {col1 if drop_col == col2 else col2}, ρ={corr_matrix.loc[col1, col2]:.3f})")
    
    # Drop selected columns
    if to_drop:
        to_drop_list = list(to_drop)
        result_df = result_df.drop(columns=to_drop_list)
        logger.info(f"Removed {len(to_drop_list)} collinear features with priority rules: {', '.join(to_drop_list)}")
    else:
        logger.info("No collinear features found above threshold")
    
    return result_df