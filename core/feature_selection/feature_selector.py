# core/feature_selection/feature_selector.py

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Class for selecting optimal feature subsets using various feature selection techniques.
    
    Supports filtering by statistical significance, correlation, 
    and ranking by importance methods.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the feature selector.
        
        Args:
            config_path: Path to the feature selection configuration
        """
        from core.utils.config_loader import ConfigLoader
        
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        self.setup_config = self.config.get("setup", {})
        self.output_dir = self.config.get("paths", {}).get("output_dir", "./results/feature_selection")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized feature selector with output directory: {self.output_dir}")
    
    def run_feature_selection(
        self, 
        df: pd.DataFrame, 
        analysis_results: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Execute feature selection pipeline.
        
        Args:
            df: Input dataframe with features and outcome
            analysis_results: Dictionary of feature analysis results (optional)
            
        Returns:
            Tuple of (selected feature names, feature ranking dataframe)
        """
        outcome_column = self.setup_config.get("outcome_column")
        exclude_columns = self.setup_config.get("exclude_columns", [])
        categorical_columns = self.setup_config.get("categorical_columns", [])
        required_features = self.setup_config.get("required_features", [])
        
        # Validation
        if outcome_column not in df.columns:
            raise ValueError(f"Outcome column '{outcome_column}' not found in dataframe")
        
        self.logger.info("Starting feature selection pipeline")
        
        # Run feature analysis if not provided
        if analysis_results is None:
            self.logger.info("Feature analysis results not provided, running analysis")
            analysis_results = self._run_feature_analysis(df)
        
        # Get feature selection steps from config
        selection_steps = self.config.get("selection_pipeline", ["pvalue", "correlation", "mrmr", "rank"])
        
        # Initialize with all features
        all_features = [col for col in df.columns if col not in exclude_columns + [outcome_column]]
        current_features = all_features.copy()
        
        # Track step-by-step results
        step_results = {
            "initial": current_features
        }
        
        # Execute each selection step
        for step in selection_steps:
            step_name = step.lower()
            self.logger.info(f"Executing selection step: {step_name}")
            
            if step_name == "pvalue":
                # Filter by statistical significance
                p_threshold = self.config.get("pvalue", {}).get("threshold", 0.05)
                current_features = self._filter_by_pvalue(
                    current_features, 
                    analysis_results.get("p_values"),
                    p_threshold
                )
                step_results["pvalue"] = current_features
            
            elif step_name == "auc":
                # Filter by AUC
                auc_threshold = self.config.get("auc", {}).get("threshold", 0.6)
                current_features = self._filter_by_auc(
                    current_features, 
                    analysis_results.get("auc_values"),
                    auc_threshold
                )
                step_results["auc"] = current_features
            
            elif step_name == "correlation":
                # Remove correlated features
                corr_threshold = self.config.get("correlation", {}).get("threshold", 0.8)
                favor_higher_auc = self.config.get("correlation", {}).get("favor_higher_auc", True)
                
                # Get AUC values if available for correlation removal prioritization
                auc_values = None
                if favor_higher_auc and "auc_values" in analysis_results:
                    auc_df = analysis_results["auc_values"]
                    auc_values = dict(zip(auc_df["Feature"], auc_df["AUC"]))
                
                # Remove correlated features
                from core.feature_analysis.feature_correlation import remove_collinear_features
                df_filtered = remove_collinear_features(
                    df[current_features + [outcome_column]],
                    threshold=corr_threshold,
                    exclude_columns=[outcome_column],
                    favor_higher_auc=favor_higher_auc,
                    auc_values=auc_values
                )
                
                current_features = [col for col in df_filtered.columns if col != outcome_column]
                step_results["correlation"] = current_features
            
            elif step_name == "mrmr":
                # Select top features by MRMR
                n_features = self.config.get("mrmr", {}).get("n_features")
                if n_features is None:
                    n_features = len(current_features)
                else:
                    n_features = min(n_features, len(current_features))
                
                if "mrmr" in analysis_results:
                    current_features = self._rank_by_mrmr(
                        current_features, 
                        analysis_results["mrmr"],
                        n_features
                    )
                else:
                    self.logger.warning("MRMR results not available, skipping step")
                
                step_results["mrmr"] = current_features
            
            elif step_name == "composite":
                # Select top features by composite score
                n_features = self.config.get("composite", {}).get("n_features")
                if n_features is None:
                    n_features = len(current_features)
                else:
                    n_features = min(n_features, len(current_features))
                
                if "composite" in analysis_results:
                    current_features = self._rank_by_composite(
                        current_features, 
                        analysis_results["composite"],
                        n_features
                    )
                else:
                    self.logger.warning("Composite results not available, skipping step")
                
                step_results["composite"] = current_features
            
            elif step_name == "tree":
                # Select top features by tree importance
                n_features = self.config.get("tree", {}).get("n_features")
                if n_features is None:
                    n_features = len(current_features)
                else:
                    n_features = min(n_features, len(current_features))
                
                if "tree_importance" in analysis_results:
                    current_features = self._rank_by_tree_importance(
                        current_features, 
                        analysis_results["tree_importance"],
                        n_features
                    )
                else:
                    self.logger.warning("Tree importance results not available, skipping step")
                
                step_results["tree"] = current_features
            
            else:
                self.logger.warning(f"Unknown selection step: {step_name}")
        
        # Ensure required features are included
        for feature in required_features:
            if feature not in current_features and feature in df.columns:
                self.logger.info(f"Adding required feature: {feature}")
                current_features.append(feature)
        
        # Save selection results
        ranking_df = self._create_feature_ranking(all_features, current_features, analysis_results)
        self._save_selection_results(current_features, step_results, ranking_df)
        
        return current_features, ranking_df
    
    def _run_feature_analysis(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run feature analysis to get feature statistics.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of analysis results
        """
        from core.feature_analysis.feature_analysis import FeatureAnalysis
        
        # Create an analysis config from the feature selection config
        analysis_config_path = self.config.get("analysis_config_path")
        
        if analysis_config_path:
            analyzer = FeatureAnalysis(analysis_config_path)
        else:
            # Use default analysis if no specific config provided
            analyzer = FeatureAnalysis(os.path.dirname(self.config_loader.config_path))
        
        # Run analysis
        results = analyzer.analyze_features(df)
        
        return results
    
    def _filter_by_pvalue(
        self, 
        features: List[str], 
        p_values_df: Optional[pd.DataFrame], 
        threshold: float = 0.05
    ) -> List[str]:
        """
        Filter features by p-value.
        
        Args:
            features: List of feature names
            p_values_df: DataFrame with p-values
            threshold: P-value threshold
            
        Returns:
            Filtered feature list
        """
        if p_values_df is None or len(p_values_df) == 0:
            self.logger.warning("No p-values provided, skipping p-value filtering")
            return features
        
        # Filter features by p-value
        significant = p_values_df[
            (p_values_df["Feature"].isin(features)) & 
            (p_values_df["P_Value"] <= threshold)
        ]["Feature"].tolist()
        
        self.logger.info(f"Filtered {len(features)} features to {len(significant)} by p-value <= {threshold}")
        
        return significant
    
    def _filter_by_auc(
        self, 
        features: List[str], 
        auc_values_df: Optional[pd.DataFrame], 
        threshold: float = 0.6
    ) -> List[str]:
        """
        Filter features by AUC.
        
        Args:
            features: List of feature names
            auc_values_df: DataFrame with AUC values
            threshold: AUC threshold
            
        Returns:
            Filtered feature list
        """
        if auc_values_df is None or len(auc_values_df) == 0:
            self.logger.warning("No AUC values provided, skipping AUC filtering")
            return features
        
        # Filter features by AUC
        significant = auc_values_df[
            (auc_values_df["Feature"].isin(features)) & 
            (auc_values_df["AUC"] >= threshold)
        ]["Feature"].tolist()
        
        self.logger.info(f"Filtered {len(features)} features to {len(significant)} by AUC >= {threshold}")
        
        return significant
    
    def _rank_by_mrmr(
        self, 
        features: List[str], 
        mrmr_df: pd.DataFrame, 
        n_features: int
    ) -> List[str]:
        """
        Rank features by MRMR score.
        
        Args:
            features: List of feature names
            mrmr_df: DataFrame with MRMR results
            n_features: Number of top features to select
            
        Returns:
            Top ranked features
        """
        # Filter to features in the current set
        filtered_df = mrmr_df[mrmr_df["Feature"].isin(features)]
        
        # Sort by MRMR metric
        if "Count" in filtered_df.columns:
            sorted_df = filtered_df.sort_values(by="Count", ascending=False)
        elif "MRMR_Score" in filtered_df.columns:
            sorted_df = filtered_df.sort_values(by="MRMR_Score", ascending=False)
        else:
            self.logger.warning("No ranking column found in MRMR results")
            return features[:n_features]
        
        # Select top features
        top_features = sorted_df.head(n_features)["Feature"].tolist()
        
        self.logger.info(f"Selected top {len(top_features)} features by MRMR ranking")
        
        return top_features
    
    def _rank_by_composite(
        self, 
        features: List[str], 
        composite_df: pd.DataFrame, 
        n_features: int
    ) -> List[str]:
        """
        Rank features by composite score.
        
        Args:
            features: List of feature names
            composite_df: DataFrame with composite scores
            n_features: Number of top features to select
            
        Returns:
            Top ranked features
        """
        # Filter to features in the current set
        filtered_df = composite_df[composite_df["Feature"].isin(features)]
        
        # Sort by composite score
        if "Composite_Score" in filtered_df.columns:
            sorted_df = filtered_df.sort_values(by="Composite_Score", ascending=False)
        else:
            self.logger.warning("No Composite_Score column found in results")
            return features[:n_features]
        
        # Select top features
        top_features = sorted_df.head(n_features)["Feature"].tolist()
        
        self.logger.info(f"Selected top {len(top_features)} features by composite ranking")
        
        return top_features
    
    # core/feature_selection/feature_selector.py (continued)
    def _rank_by_tree_importance(
        self, 
        features: List[str], 
        tree_df: pd.DataFrame, 
        n_features: int
    ) -> List[str]:
        """
        Rank features by tree importance.
        
        Args:
            features: List of feature names
            tree_df: DataFrame with tree importance
            n_features: Number of top features to select
            
        Returns:
            Top ranked features
        """
        # Filter to features in the current set
        filtered_df = tree_df[tree_df["Feature"].isin(features)]
        
        # Sort by importance
        if "Importance" in filtered_df.columns:
            sorted_df = filtered_df.sort_values(by="Importance", ascending=False)
        else:
            self.logger.warning("No Importance column found in tree results")
            return features[:n_features]
        
        # Select top features
        top_features = sorted_df.head(n_features)["Feature"].tolist()
        
        self.logger.info(f"Selected top {len(top_features)} features by tree importance")
        
        return top_features
    
    def _create_feature_ranking(
        self, 
        all_features: List[str], 
        selected_features: List[str],
        results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Create a comprehensive ranking of all features.
        
        Args:
            all_features: List of all feature names
            selected_features: List of selected feature names
            results: Dictionary of analysis results
            
        Returns:
            DataFrame with feature rankings
        """
        # Start with all features
        ranking_data = []
        
        for feature in all_features:
            feature_data = {"Feature": feature, "Selected": feature in selected_features}
            
            # Add p-values if available
            if "p_values" in results and not results["p_values"].empty:
                p_val_row = results["p_values"][results["p_values"]["Feature"] == feature]
                feature_data["P_Value"] = p_val_row["P_Value"].values[0] if not p_val_row.empty else np.nan
            
            # Add AUC values if available
            if "auc_values" in results and not results["auc_values"].empty:
                auc_row = results["auc_values"][results["auc_values"]["Feature"] == feature]
                feature_data["AUC"] = auc_row["AUC"].values[0] if not auc_row.empty else np.nan
            
            # Add MRMR scores if available
            if "mrmr" in results and not results["mrmr"].empty:
                mrmr_row = results["mrmr"][results["mrmr"]["Feature"] == feature]
                # Try different column names for MRMR
                for col_name in ["Count", "MRMR_Score"]:
                    if col_name in results["mrmr"].columns:
                        feature_data["MRMR"] = mrmr_row[col_name].values[0] if not mrmr_row.empty else np.nan
                        break
            
            # Add tree importance if available
            if "tree_importance" in results and not results["tree_importance"].empty:
                tree_row = results["tree_importance"][results["tree_importance"]["Feature"] == feature]
                feature_data["Tree_Importance"] = tree_row["Importance"].values[0] if not tree_row.empty else np.nan
            
            # Add composite score if available
            if "composite" in results and not results["composite"].empty:
                comp_row = results["composite"][results["composite"]["Feature"] == feature]
                feature_data["Composite_Score"] = comp_row["Composite_Score"].values[0] if not comp_row.empty else np.nan
            
            ranking_data.append(feature_data)
        
        # Create DataFrame
        ranking_df = pd.DataFrame(ranking_data)
        
        # Sort by composite score if available, otherwise by p-value
        if "Composite_Score" in ranking_df.columns:
            ranking_df = ranking_df.sort_values(by="Composite_Score", ascending=False)
        elif "P_Value" in ranking_df.columns:
            ranking_df = ranking_df.sort_values(by="P_Value")
        
        return ranking_df
    
    def _save_selection_results(
        self, 
        selected_features: List[str], 
        step_results: Dict[str, List[str]],
        ranking_df: pd.DataFrame
    ) -> None:
        """
        Save feature selection results.
        
        Args:
            selected_features: List of selected feature names
            step_results: Dictionary with step-by-step feature lists
            ranking_df: DataFrame with feature rankings
        """
        # Save selected features list
        selected_file = os.path.join(self.output_dir, "selected_features.txt")
        with open(selected_file, "w") as f:
            f.write(f"Selected Features ({len(selected_features)}):\n")
            for feature in selected_features:
                f.write(f"- {feature}\n")
        
        # Save feature ranking
        ranking_file = os.path.join(self.output_dir, "feature_ranking.xlsx")
        ranking_df.to_excel(ranking_file, index=False)
        
        # Save step-by-step results
        steps_file = os.path.join(self.output_dir, "selection_steps.xlsx")
        with pd.ExcelWriter(steps_file, engine='openpyxl') as writer:
            for step, features in step_results.items():
                step_df = pd.DataFrame({
                    "Feature": features
                })
                step_df.to_excel(writer, sheet_name=step[:31], index=False)
        
        self.logger.info(f"Saved selection results to {self.output_dir}")