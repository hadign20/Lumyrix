# core/feature_analysis/composite_importance.py

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
import logging

from core.visualization.plotting import plot_barplot

logger = logging.getLogger(__name__)


def calculate_composite_score(
        feature_analysis_results: Dict[str, pd.DataFrame],
        config: Dict,
        output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate composite feature importance scores from multiple analysis results.

    Args:
        feature_analysis_results: Dictionary of feature analysis DataFrames
        config: Configuration dictionary
        output_dir: Output directory for saving results

    Returns:
        DataFrame with composite scores
    """
    # Get weights from config
    weights = config.get("weights", {})
    p_value_weight = weights.get("p_value", 1.0)
    auc_weight = weights.get("auc", 1.0)
    mrmr_weight = weights.get("mrmr", 1.0)
    tree_weight = weights.get("tree_importance", 0.0)

    # Extract feature rankings from various methods
    all_features = set()
    feature_scores = {}

    # Process p-values (convert to score: 1 - p_value)
    if 'p_values' in feature_analysis_results and not feature_analysis_results['p_values'].empty:
        p_values_df = feature_analysis_results['p_values']
        for idx, row in p_values_df.iterrows():
            feature = row['Feature']
            all_features.add(feature)
            if feature not in feature_scores:
                feature_scores[feature] = {}
            feature_scores[feature]['p_value'] = 1.0 - row['P_Value']

    # Process AUC values
    if 'auc_values' in feature_analysis_results and not feature_analysis_results['auc_values'].empty:
        auc_df = feature_analysis_results['auc_values']
        for idx, row in auc_df.iterrows():
            feature = row['Feature']
            all_features.add(feature)
            if feature not in feature_scores:
                feature_scores[feature] = {}
            feature_scores[feature]['auc'] = row['AUC']

    # Process MRMR rankings
    if 'mrmr' in feature_analysis_results and not feature_analysis_results['mrmr'].empty:
        mrmr_df = feature_analysis_results['mrmr']

        # Determine MRMR score column
        mrmr_score_col = None
        for col in ['Count', 'MRMR_Score']:
            if col in mrmr_df.columns:
                mrmr_score_col = col
                break

        if mrmr_score_col:
            # Normalize MRMR scores to 0-1 range
            max_mrmr = mrmr_df[mrmr_score_col].max()
            if max_mrmr > 0:
                for idx, row in mrmr_df.iterrows():
                    feature = row['Feature']
                    all_features.add(feature)
                    if feature not in feature_scores:
                        feature_scores[feature] = {}
                    feature_scores[feature]['mrmr'] = row[mrmr_score_col] / max_mrmr

    # Process tree importance
    if 'tree_importance' in feature_analysis_results and not feature_analysis_results['tree_importance'].empty:
        tree_df = feature_analysis_results['tree_importance']

        # Normalize tree importance to 0-1 range
        max_importance = tree_df['Importance'].max()
        if max_importance > 0:
            for idx, row in tree_df.iterrows():
                feature = row['Feature']
                all_features.add(feature)
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['tree'] = row['Importance'] / max_importance

    # Calculate composite scores
    results = []

    for feature in all_features:
        scores = feature_scores.get(feature, {})

        # Get individual scores or default to 0
        p_value_score = scores.get('p_value', 0.0)
        auc_score = scores.get('auc', 0.5)  # Default AUC is 0.5 (random chance)
        mrmr_score = scores.get('mrmr', 0.0)
        tree_score = scores.get('tree', 0.0)

        # Normalize AUC to 0-1 range (0.5-1.0 -> 0.0-1.0)
        auc_normalized = (auc_score - 0.5) * 2 if auc_score > 0.5 else 0.0

        # Calculate weighted composite score
        total_weight = p_value_weight + auc_weight + mrmr_weight + tree_weight

        if total_weight > 0:
            composite_score = (
                                      p_value_weight * p_value_score +
                                      auc_weight * auc_normalized +
                                      mrmr_weight * mrmr_score +
                                      tree_weight * tree_score
                              ) / total_weight
        else:
            composite_score = 0.0

        # Create result entry
        result = {
            'Feature': feature,
            'P_Value_Score': p_value_score,
            'AUC_Score': auc_normalized,
            'MRMR_Score': mrmr_score,
            'Tree_Score': tree_score,
            'Composite_Score': composite_score
        }

        results.append(result)

    # Create DataFrame and sort by composite score
    composite_df = pd.DataFrame(results)
    composite_df = composite_df.sort_values(by='Composite_Score', ascending=False)

    # Save results if output directory provided
    if output_dir:
        # Save to CSV
        output_file = os.path.join(output_dir, "composite_feature_scores.xlsx")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        composite_df.to_excel(output_file, index=False)

        # Create visualization
        top_n = min(20, len(composite_df))
        top_features = composite_df.head(top_n)

        plot_barplot(
            data=top_features,
            x='Feature',
            y='Composite_Score',
            title=f"Top {top_n} Features by Composite Score",
            horizontal=True,
            figsize=(10, 8),
            save_path=os.path.join(output_dir, "plots", "composite_scores.png")
        )

    logger.info(f"Calculated composite scores for {len(composite_df)} features")

    return composite_df