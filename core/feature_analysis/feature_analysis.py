# core/feature_analysis/feature_analysis.py

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import logging


from core.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class FeatureAnalysis:
    """
    Comprehensive feature analysis module for radiomics and other features.
    Performs statistical analysis, importance ranking, and feature comparison.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the feature analysis module.
        
        Args:
            config_path: Path to the feature analysis configuration
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        self.setup_config = self.config.get("setup", {})
        self.output_dir = self.config.get("paths", {}).get("output_dir", "./results/feature_analysis")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized feature analysis with output directory: {self.output_dir}")
    
    def analyze_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run complete feature analysis pipeline.
        
        Args:
            df: DataFrame with features and outcome column
            
        Returns:
            Dictionary of analysis results (DataFrames)
        """
        self.logger.info("Starting feature analysis pipeline")
        
        # Setup parameters
        outcome_column = self.setup_config.get("outcome_column")
        task_type = self.setup_config.get("task_type", "classification")
        categorical_columns = self.setup_config.get("categorical_columns", [])
        exclude_columns = self.setup_config.get("exclude_columns", [])
        
        # Validation
        if outcome_column not in df.columns:
            raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")
        
        # Initialize results dictionary
        results = {}
        
        # Step 1: Univariate Analysis
        self.logger.info("Step 1: Univariate analysis")
        from core.feature_analysis.univariate_analysis import calculate_p_values, calculate_auc_values
        
        results['p_values'] = calculate_p_values(
            df, outcome_column, categorical_columns, exclude_columns
        )
        
        results['auc_values'] = calculate_auc_values(
            df, outcome_column, categorical_columns, exclude_columns,
            cv_folds=self.config.get("univariate", {}).get("cv_folds", 5)
        )
        
        # Step 2: Correlation Analysis
        self.logger.info("Step 2: Correlation analysis")
        from core.feature_analysis.feature_correlation import compute_correlation_matrix
        
        results['correlation'] = compute_correlation_matrix(
            df,
            exclude_columns=exclude_columns + [outcome_column]
        )
        
        # Step 3: MRMR Feature Selection
        self.logger.info("Step 3: MRMR feature selection")
        from core.feature_selection.filter_methods import MRMR_feature_selection
        
        results['mrmr'] = MRMR_feature_selection(
            df, outcome_column, categorical_columns, exclude_columns,
            n_features=self.config.get("mrmr", {}).get("n_features", 15),
            cv_folds=self.config.get("mrmr", {}).get("cv_folds", 5)
        )
        
        # Step 4: Random Forest Feature Importance
        self.logger.info("Step 4: Random forest importance")
        from core.feature_analysis.feature_importance import calculate_tree_importance
        
        if task_type == "classification":
            results['tree_importance'] = calculate_tree_importance(
                df, outcome_column, exclude_columns,
                n_estimators=self.config.get("tree_importance", {}).get("n_estimators", 100),
                cv_folds=self.config.get("tree_importance", {}).get("cv_folds", 5)
            )
        
        # Step 5: SHAP Importance (if configured)
        if self.config.get("shap_importance", {}).get("enabled", False):
            self.logger.info("Step 5: SHAP importance")
            from core.feature_analysis.feature_importance import calculate_shap_importance
            
            results['shap'] = calculate_shap_importance(
                df, outcome_column, exclude_columns,
                model_type=self.config.get("shap_importance", {}).get("model_type", "tree"),
                n_samples=self.config.get("shap_importance", {}).get("n_samples", 100)
            )
        
        # Step 6: Composite Feature Importance
        self.logger.info("Step 6: Composite feature importance")
        from core.feature_analysis.composite_importance import calculate_composite_score
        
        results['composite'] = self._calculate_composite_score(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def compare_feature_distributions(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> pd.DataFrame:
        """
        Compare feature distributions between training and validation datasets.
        
        Args:
            df_train: Training dataframe
            df_val: Validation dataframe
            
        Returns:
            DataFrame with comparison statistics
        """
        self.logger.info("Comparing feature distributions between training and validation sets")
        
        outcome_column = self.setup_config.get("outcome_column")
        exclude_columns = self.setup_config.get("exclude_columns", [])
        
        # Validation
        if outcome_column not in df_train.columns or outcome_column not in df_val.columns:
            raise ValueError(f"Outcome column '{outcome_column}' not found in both datasets")
        
        # Get common features
        train_features = [col for col in df_train.columns 
                          if col not in exclude_columns + [outcome_column]]
        val_features = [col for col in df_val.columns 
                        if col not in exclude_columns + [outcome_column]]
        common_features = list(set(train_features).intersection(set(val_features)))
        
        if not common_features:
            self.logger.warning("No common features found between training and validation sets")
            return pd.DataFrame()
        
        from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu
        from sklearn.metrics import roc_auc_score
        
        # Compute comparison metrics
        comparison_results = []
        
        for feature in common_features:
            train_values = df_train[feature].dropna()
            val_values = df_val[feature].dropna()
            
            try:
                # Statistical tests
                ks_stat, ks_pval = ks_2samp(train_values, val_values)
                t_stat, t_pval = ttest_ind(train_values, val_values, equal_var=False)
                mw_stat, mw_pval = mannwhitneyu(train_values, val_values, alternative='two-sided')
                
                # AUC in each dataset
                try:
                    train_auc = roc_auc_score(df_train[outcome_column], df_train[feature])
                except:
                    train_auc = np.nan
                    
                try:
                    val_auc = roc_auc_score(df_val[outcome_column], df_val[feature])
                except:
                    val_auc = np.nan
                
                # Descriptive statistics
                train_mean = train_values.mean()
                train_std = train_values.std()
                val_mean = val_values.mean()
                val_std = val_values.std()
                
                comparison_results.append({
                    'Feature': feature,
                    'KS_p_value': ks_pval,
                    'TTest_p_value': t_pval,
                    'MannWhitney_p_value': mw_pval,
                    'Train_AUC': train_auc,
                    'Val_AUC': val_auc,
                    'Train_Mean': train_mean,
                    'Train_Std': train_std,
                    'Val_Mean': val_mean,
                    'Val_Std': val_std,
                    'Mean_Diff': abs(train_mean - val_mean),
                    'AUC_Diff': abs(train_auc - val_auc) if not np.isnan(train_auc) and not np.isnan(val_auc) else np.nan
                })
            except Exception as e:
                self.logger.warning(f"Error comparing feature '{feature}': {str(e)}")
                continue
        
        # Create and save DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            # Sort by statistical significance
            comparison_df = comparison_df.sort_values(by='KS_p_value')
            
            # Save to file
            output_file = os.path.join(self.output_dir, "feature_distribution_comparison.xlsx")
            comparison_df.to_excel(output_file, index=False)
            self.logger.info(f"Saved feature distribution comparison to {output_file}")
            
            # Create summary of most different features
            self._plot_feature_distribution_comparison(df_train, df_val, comparison_df)
        
        return comparison_df
    
    def _calculate_composite_score(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate composite importance score from multiple feature ranking methods.
        
        Args:
            results: Dictionary of analysis results
            
        Returns:
            DataFrame with composite scores
        """
        # Get weights from config
        weights = self.config.get("composite", {}).get("weights", {})
        p_value_weight = weights.get("p_value", 1.0)
        auc_weight = weights.get("auc", 1.0)
        mrmr_weight = weights.get("mrmr", 1.0)
        tree_weight = weights.get("tree_importance", 1.0)
        shap_weight = weights.get("shap", 0.0)
        
        # Start with p-values (invert so higher=better)
        if 'p_values' in results:
            feature_df = results['p_values'].copy()
            feature_df['P_Value_Score'] = 1 - feature_df['P_Value']
        else:
            self.logger.warning("P-values not found in results, using AUC values as base")
            feature_df = results['auc_values'].copy()
            feature_df['P_Value_Score'] = 0.0
        
        # Add AUC scores
        if 'auc_values' in results:
            auc_df = results['auc_values']
            feature_df = pd.merge(feature_df, auc_df, on='Feature', how='outer')
        else:
            feature_df['AUC'] = 0.5
        
        # Add MRMR scores (normalize to 0-1)
        if 'mrmr' in results:
            mrmr_df = results['mrmr']
            if 'Count' in mrmr_df.columns and mrmr_df['Count'].max() > 0:
                mrmr_df['MRMR_Score'] = mrmr_df['Count'] / mrmr_df['Count'].max()
            else:
                mrmr_df['MRMR_Score'] = mrmr_df['MRMR_Score'] if 'MRMR_Score' in mrmr_df.columns else 0
            
            feature_df = pd.merge(feature_df, mrmr_df[['Feature', 'MRMR_Score']], on='Feature', how='outer')
        else:
            feature_df['MRMR_Score'] = 0.0
        
        # Add tree importance (normalize to 0-1)
        if 'tree_importance' in results:
            tree_df = results['tree_importance']
            if 'Importance' in tree_df.columns and tree_df['Importance'].max() > 0:
                tree_df['Tree_Score'] = tree_df['Importance'] / tree_df['Importance'].max()
            else:
                tree_df['Tree_Score'] = tree_df['Tree_Score'] if 'Tree_Score' in tree_df.columns else 0
            
            feature_df = pd.merge(feature_df, tree_df[['Feature', 'Tree_Score']], on='Feature', how='outer')
        else:
            feature_df['Tree_Score'] = 0.0
        
        # Add SHAP values (normalize to 0-1)
        if 'shap' in results:
            shap_df = results['shap']
            if 'SHAP_MeanAbs' in shap_df.columns and shap_df['SHAP_MeanAbs'].max() > 0:
                shap_df['SHAP_Score'] = shap_df['SHAP_MeanAbs'] / shap_df['SHAP_MeanAbs'].max()
            else:
                shap_df['SHAP_Score'] = shap_df['SHAP_Score'] if 'SHAP_Score' in shap_df.columns else 0
            
            feature_df = pd.merge(feature_df, shap_df[['Feature', 'SHAP_Score']], on='Feature', how='outer')
        else:
            feature_df['SHAP_Score'] = 0.0
        
        # Fill NaN values
        feature_df = feature_df.fillna({
            'P_Value_Score': 0.0,
            'AUC': 0.5,
            'MRMR_Score': 0.0,
            'Tree_Score': 0.0,
            'SHAP_Score': 0.0
        })
        
        # Calculate weighted composite score
        feature_df['Composite_Score'] = (
            p_value_weight * feature_df['P_Value_Score'] +
            auc_weight * feature_df['AUC'] +
            mrmr_weight * feature_df['MRMR_Score'] +
            tree_weight * feature_df['Tree_Score'] +
            shap_weight * feature_df['SHAP_Score']
        ) / (p_value_weight + auc_weight + mrmr_weight + tree_weight + shap_weight)
        
        # Sort by composite score
        feature_df = feature_df.sort_values(by='Composite_Score', ascending=False)
        
        return feature_df
    
    def _save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """
        Save all analysis results to Excel files.
        
        Args:
            results: Dictionary of analysis results
        """
        # Summary file
        summary_file = os.path.join(self.output_dir, "feature_analysis_summary.xlsx")
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            for name, df_result in results.items():
                if isinstance(df_result, pd.DataFrame):
                    # Limit sheet name length to 31 characters (Excel limitation)
                    sheet_name = name[:31]
                    df_result.to_excel(writer, sheet_name=sheet_name, index=False)
        
        self.logger.info(f"Saved analysis summary to {summary_file}")
        
        # Individual files for each analysis
        for name, df_result in results.items():
            if isinstance(df_result, pd.DataFrame):
                output_file = os.path.join(self.output_dir, f"{name}.xlsx")
                df_result.to_excel(output_file, index=False if name != 'correlation' else True)
                self.logger.info(f"Saved {name} results to {output_file}")
        
        # Generate plots
        self._generate_analysis_plots(results)
    
    def _generate_analysis_plots(self, results: Dict[str, pd.DataFrame]) -> None:
        """
        Generate visualization plots for feature analysis results.
        
        Args:
            results: Dictionary of analysis results
        """
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Import visualization functions
        from core.visualization.plotting import (
            plot_correlation_heatmap,
            plot_feature_importance,
            plot_pvalue_distribution,
            plot_auc_distribution
        )
        
        # Correlation heatmap
        if 'correlation' in results:
            corr_matrix = results['correlation']
            plot_correlation_heatmap(
                corr_matrix, 
                title="Feature Correlation Matrix",
                figsize=(12, 10),
                save_path=os.path.join(plots_dir, "correlation_heatmap.png")
            )
        
        # Feature importance plot
        if 'composite' in results:
            composite_df = results['composite']
            top_n = min(20, len(composite_df))
            plot_feature_importance(
                composite_df.head(top_n), 
                x_col='Composite_Score',
                y_col='Feature',
                title=f"Top {top_n} Features by Composite Score",
                figsize=(10, 8),
                save_path=os.path.join(plots_dir, "composite_importance.png")
            )
        
        # P-value distribution
        if 'p_values' in results:
            p_values_df = results['p_values']
            plot_pvalue_distribution(
                p_values_df, 
                p_col='P_Value',
                title="Feature P-Value Distribution",
                figsize=(10, 6),
                save_path=os.path.join(plots_dir, "pvalue_distribution.png")
            )
        
        # AUC distribution
        if 'auc_values' in results:
            auc_df = results['auc_values']
            plot_auc_distribution(
                auc_df, 
                auc_col='AUC',
                title="Feature AUC Distribution",
                figsize=(10, 6),
                save_path=os.path.join(plots_dir, "auc_distribution.png")
            )
    
    def _plot_feature_distribution_comparison(self, df_train: pd.DataFrame, df_val: pd.DataFrame, 
                                             comparison_df: pd.DataFrame) -> None:
        """
        Plot distribution comparisons for top different features.
        
        Args:
            df_train: Training dataframe
            df_val: Validation dataframe
            comparison_df: Feature comparison results
        """
        plots_dir = os.path.join(self.output_dir, "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        from core.visualization.plotting import plot_feature_distribution_comparison
        
        # Get top different features
        top_diff_features = comparison_df.sort_values(by='KS_p_value').head(10)['Feature'].tolist()
        
        for feature in top_diff_features:
            if feature in df_train.columns and feature in df_val.columns:
                plot_feature_distribution_comparison(
                    df_train, 
                    df_val, 
                    feature, 
                    title=f"Distribution Comparison: {feature}",
                    save_path=os.path.join(plots_dir, f"compare_{feature}.png")
                )

