# core/feature_analysis/univariate_analysis.py

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
import logging
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def calculate_p_values(
    df: pd.DataFrame,
    outcome_column: str,
    categorical_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate p-values for each feature with respect to the outcome variable.
    
    For categorical features, uses chi-square test or Fisher's exact test.
    For continuous features, uses Mann-Whitney U test (non-parametric).
    
    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        categorical_columns: List of categorical feature names
        exclude_columns: List of columns to exclude
        
    Returns:
        DataFrame with features and corresponding p-values
    """
    categorical_columns = categorical_columns or []
    exclude_columns = exclude_columns or []
    
    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")
    
    # Initialize results
    results = []
    
    # Process each feature
    for column in df.columns:
        # Skip excluded columns and outcome
        if column in exclude_columns or column == outcome_column:
            continue
        
        try:
            # Handle categorical features
            if column in categorical_columns:
                # Create contingency table
                contingency_table = pd.crosstab(df[column], df[outcome_column])
                
                # Use Fisher's exact test for 2x2 tables, chi-square otherwise
                if contingency_table.shape == (2, 2):
                    _, p_value = fisher_exact(contingency_table)
                else:
                    _, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Handle continuous features
            else:
                # Get values for each outcome group
                pos_group = df[df[outcome_column] == 1][column].dropna()
                neg_group = df[df[outcome_column] == 0][column].dropna()
                
                # Use Mann-Whitney U test
                _, p_value = mannwhitneyu(pos_group, neg_group, alternative='two-sided')
            
            # Add to results
            results.append({
                'Feature': column,
                'P_Value': p_value
            })
        
        except Exception as e:
            logger.warning(f"Error calculating p-value for feature '{column}': {str(e)}")
            results.append({
                'Feature': column,
                'P_Value': 1.0
            })
    
    # Create DataFrame and sort by p-value
    p_values_df = pd.DataFrame(results)
    p_values_df = p_values_df.sort_values(by='P_Value')
    
    logger.info(f"Calculated p-values for {len(results)} features")
    
    return p_values_df

def calculate_auc_values(
    df: pd.DataFrame,
    outcome_column: str,
    categorical_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Calculate AUC values for each feature with respect to the outcome variable.
    
    Uses cross-validated logistic regression to handle potential overfitting.
    
    Args:
        df: Input dataframe
        outcome_column: Name of outcome column
        categorical_columns: List of categorical feature names
        exclude_columns: List of columns to exclude
        cv_folds: Number of cross-validation folds
        
    Returns:
        DataFrame with features and corresponding AUC values
    """
    categorical_columns = categorical_columns or []
    exclude_columns = exclude_columns or []
    
    # Check if outcome column exists
    if outcome_column not in df.columns:
        raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")
    
    # Get outcome values
    y = df[outcome_column].values
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Initialize results
    results = []
    
    # Process each feature
    for column in df.columns:
        # Skip excluded columns and outcome
        if column in exclude_columns or column == outcome_column:
            continue
        
        try:
            # Handle categorical features (convert to numeric codes)
            if column in categorical_columns:
                X = df[column].astype('category').cat.codes.values.reshape(-1, 1)
            else:
                X = df[column].values.reshape(-1, 1)
            
            # Skip if feature has all same values
            if np.unique(X).size <= 1:
                logger.warning(f"Feature '{column}' has only one unique value, skipping AUC calculation")
                results.append({
                    'Feature': column,
                    'AUC': 0.5
                })
                continue
            
            # Scale feature
            X = scaler.fit_transform(X)
            
            # Set up cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Use logistic regression with cross-validation
            model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            # Calculate mean AUC
            mean_auc = np.mean(auc_scores)
            
            # Ensure AUC >= 0.5 (flip if necessary)
            if mean_auc < 0.5:
                mean_auc = 1 - mean_auc
            
            # Add to results
            results.append({
                'Feature': column,
                'AUC': mean_auc
            })
        
        except Exception as e:
            logger.warning(f"Error calculating AUC for feature '{column}': {str(e)}")
            results.append({
                'Feature': column,
                'AUC': 0.5
            })
    
    # Create DataFrame and sort by AUC (descending)
    auc_values_df = pd.DataFrame(results)
    auc_values_df = auc_values_df.sort_values(by='AUC', ascending=False)
    
    logger.info(f"Calculated AUC values for {len(results)} features")
    
    return auc_values_df