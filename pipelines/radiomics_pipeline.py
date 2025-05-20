# pipelines/radiomics_pipeline.py

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

from core.utils.config_loader import ConfigLoader
from core.feature_analysis.feature_analysis import FeatureAnalysis
from core.feature_selection.feature_selector import FeatureSelector
from core.classical_ml.model_selection import ModelSelector

logger = logging.getLogger(__name__)

class RadiomicsPipeline:
    """
    Pipeline for radiomics-based model development and evaluation.
    
    This pipeline handles:
    1. Data loading and preprocessing
    2. Feature analysis and selection
    3. Model training and evaluation
    4. External validation
    5. Result visualization and reporting
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the radiomics pipeline.
        
        Args:
            config_path: Path to the pipeline configuration directory
        """
        self.start_time = time.time()
        
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.project_config = self.config_loader.load_project_config()
        self.feature_config = self.config_loader.load_feature_config()
        self.model_config = self.config_loader.load_model_config()
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing radiomics pipeline")
        
        # Initialize paths
        self.data_path = self.project_config.get("paths", {}).get("data_path", "data")
        self.result_path = self.project_config.get("paths", {}).get("result_path", "results")
        
        # Create results directory
        os.makedirs(self.result_path, exist_ok=True)
    
    def run(self):
        """
        Execute the complete radiomics pipeline.
        """
        self.logger.info("Starting radiomics pipeline")
        
        # Step 1: Load and preprocess data
        train_data, validation_data = self._load_data()
        
        # Step 2: Feature analysis
        feature_analysis_results = self._analyze_features(train_data)
        
        # Step 3: Feature selection
        selected_features = self._select_features(train_data, feature_analysis_results)
        
        # Step 4: Model training and evaluation
        model_results = self._train_and_evaluate_models(train_data, selected_features)
        
        # Step 5: External validation (if enabled)
        if validation_data and self.model_config.get("external_validation", {}).get("enabled", False):
            validation_results = self._validate_models(validation_data, selected_features)
        
        # Log execution time
        execution_time = time.time() - self.start_time
        self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        
        return {
            "feature_analysis": feature_analysis_results,
            "selected_features": selected_features,
            "model_results": model_results
        }
    
    def _load_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load training and validation data from files.
        
        Returns:
            Tuple of (training data dict, validation data dict)
        """
        self.logger.info("Loading data")
        
        # Get file paths from config
        train_files = self.project_config.get("files", {}).get("train_files", [])
        validation_files = self.project_config.get("files", {}).get("validation_files", [])
        train_sheets = self.project_config.get("sheets", {}).get("train_sheets", [])
        validation_sheets = self.project_config.get("sheets", {}).get("validation_sheets", [])
        
        # Load training data
        train_data = {}
        for file in train_files:
            file_path = os.path.join(self.data_path, f"{file}.xlsx")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"Training file not found: {file_path}")
                continue
            
            # Load Excel file
            xls = pd.ExcelFile(file_path)
            
            # Determine sheets to use
            sheets = xls.sheet_names if not train_sheets else train_sheets
            
            for sheet in sheets:
                # Skip if sheet doesn't exist
                if sheet not in xls.sheet_names:
                    self.logger.warning(f"Sheet '{sheet}' not found in {file_path}")
                    continue
                
                # Load sheet data
                df = pd.read_excel(file_path, sheet_name=sheet)
                
                # Preprocess data
                df = self._preprocess_data(df)
                
                # Store data
                train_data[f"{file}_{sheet}"] = df
                self.logger.info(f"Loaded training data from {file_path}, sheet '{sheet}', shape: {df.shape}")
        
        # Load validation data
        validation_data = {}
        if validation_files:
            for file in validation_files:
                file_path = os.path.join(self.data_path, f"{file}.xlsx")
                
                if not os.path.exists(file_path):
                    self.logger.warning(f"Validation file not found: {file_path}")
                    continue
                
                # Load Excel file
                xls = pd.ExcelFile(file_path)
                
                # Determine sheets to use
                sheets = xls.sheet_names if not validation_sheets else validation_sheets
                
                for sheet in sheets:
                    # Skip if sheet doesn't exist
                    if sheet not in xls.sheet_names:
                        self.logger.warning(f"Sheet '{sheet}' not found in {file_path}")
                        continue
                    
                    # Load sheet data
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    
                    # Preprocess data
                    df = self._preprocess_data(df)
                    
                    # Store data
                    validation_data[f"{file}_{sheet}"] = df
                    self.logger.info(f"Loaded validation data from {file_path}, sheet '{sheet}', shape: {df.shape}")
        
        return train_data, validation_data

    # Update in pipelines/radiomics_pipeline.py

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data: handle missing values, normalize, etc.

        Args:
            df: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        # Get settings from config
        outcome_column = self.project_config.get("columns", {}).get("outcome_column")
        exclude_columns = self.project_config.get("columns", {}).get("exclude_columns", [])

        # Check if outcome column exists
        if outcome_column not in df.columns:
            self.logger.warning(f"Outcome column '{outcome_column}' not found in dataframe")
            return df

        # Check for non-numeric columns that aren't in exclude_columns
        non_numeric_cols = []
        for col in df.columns:
            if col != outcome_column and col not in exclude_columns and df[col].dtype.kind not in 'iuf':
                non_numeric_cols.append(col)

        if non_numeric_cols:
            self.logger.warning(f"Found non-numeric columns that aren't excluded: {non_numeric_cols}")
            # Add them to exclude_columns
            exclude_columns = exclude_columns + non_numeric_cols

        # Drop rows with missing outcome
        df = df.dropna(subset=[outcome_column])

        # Fill other missing values with 0 (common for radiomics)
        df = df.fillna(0)

        # Apply normalization if configured
        if self.feature_config.get("normalization", {}).get("enabled", False):
            df = self._normalize_dataframe(df)

        return df
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features in the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Normalized dataframe
        """
        from sklearn.preprocessing import StandardScaler
        
        # Get settings from config
        outcome_column = self.project_config.get("columns", {}).get("outcome_column")
        exclude_columns = self.project_config.get("columns", {}).get("exclude_columns", [])
        
        # Determine columns to normalize
        excluded_columns = exclude_columns + [outcome_column]
        cols_to_scale = [col for col in df.columns if col not in excluded_columns]
        
        if not cols_to_scale:
            return df
        
        # Apply StandardScaler
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[cols_to_scale])
        scaled_df = pd.DataFrame(scaled_values, columns=cols_to_scale, index=df.index)
        
        # Reconstruct dataframe with excluded columns
        result_df = pd.concat([
            df[exclude_columns], 
            scaled_df, 
            df[[outcome_column]]
        ], axis=1)
        
        return result_df
    
    def _analyze_features(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Analyze features in training data.
        
        Args:
            train_data: Dictionary of training dataframes
            
        Returns:
            Dictionary of analysis results
        """
        self.logger.info("Analyzing features")
        
        # Create feature analysis instance
        feature_analysis = FeatureAnalysis(os.path.join(self.config_loader.config_dir, "feature_config.yaml"))
        
        # Analyze each training dataset
        results = {}
        
        for dataset_name, df in train_data.items():
            self.logger.info(f"Analyzing features in dataset: {dataset_name}")
            
            # Create output directory for this dataset
            output_dir = os.path.join(self.result_path, dataset_name, "feature_analysis")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run analysis
            dataset_results = feature_analysis.analyze_features(df)
            
            # Save results
            results[dataset_name] = dataset_results
        
        return results
    
    def _select_features(
        self, 
        train_data: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Select optimal features for model building.
        
        Args:
            train_data: Dictionary of training dataframes
            analysis_results: Dictionary of feature analysis results
            
        Returns:
            Dictionary of selected features
        """
        self.logger.info("Selecting features")
        
        # Create feature selector instance
        feature_selector = FeatureSelector(os.path.join(self.config_loader.config_dir, "feature_config.yaml"))
        
        # Select features for each training dataset
        results = {}
        
        for dataset_name, df in train_data.items():
            self.logger.info(f"Selecting features in dataset: {dataset_name}")
            
            # Create output directory for this dataset
            output_dir = os.path.join(self.result_path, dataset_name, "feature_selection")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run feature selection
            selected_features, ranking_df = feature_selector.run_feature_selection(
                df, analysis_results.get(dataset_name, {})
            )
            
            # Save results
            results[dataset_name] = {
                "features": selected_features,
                "ranking": ranking_df
            }
            
            # Save to file
            ranking_file = os.path.join(output_dir, "feature_ranking.xlsx")
            ranking_df.to_excel(ranking_file, index=False)
            
            self.logger.info(f"Selected {len(selected_features)} features for dataset: {dataset_name}")
        
        return results

    # Update in pipelines/radiomics_pipeline.py

    def _train_and_evaluate_models(
            self,
            train_data: Dict[str, pd.DataFrame],
            selection_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Train and evaluate machine learning models.

        Args:
            train_data: Dictionary of training dataframes
            selection_results: Dictionary of feature selection results

        Returns:
            Dictionary of model evaluation results
        """
        self.logger.info("Training and evaluating models")

        # Get model configuration
        model_building_enabled = self.model_config.get("model_building", {}).get("enabled", True)
        if not model_building_enabled:
            self.logger.info("Model building is disabled in configuration, skipping")
            return {}

        # Get evaluation settings
        evaluation_method = self.model_config.get("evaluation", {}).get("method", "train_test_split")
        test_size = self.model_config.get("evaluation", {}).get("test_size", 0.3)
        cv_folds = self.model_config.get("evaluation", {}).get("cv_folds", 5)
        hyperparameter_tuning = self.model_config.get("evaluation", {}).get("hyperparameter_tuning", True)

        # Get resampling settings
        resampling_enabled = self.model_config.get("resampling", {}).get("enabled", False)
        resampling_method = self.model_config.get("resampling", {}).get("method")

        # Set up resampling method if enabled
        resampling_obj = None
        if resampling_enabled and resampling_method:
            if resampling_method == "RandomOverSampler":
                from imblearn.over_sampling import RandomOverSampler
                resampling_obj = RandomOverSampler(random_state=42)
            elif resampling_method == "SMOTEENN":
                from imblearn.combine import SMOTEENN
                resampling_obj = SMOTEENN(random_state=42)
            elif resampling_method == "SMOTE":
                from imblearn.over_sampling import SMOTE
                resampling_obj = SMOTE(random_state=42)

        # Set up model selector
        model_selector = ModelSelector(hyperparameter_tuning=hyperparameter_tuning)

        # Train and evaluate models for each dataset
        results = {}

        for dataset_name, df in train_data.items():
            self.logger.info(f"Training models for dataset: {dataset_name}")

            # Get selected features for this dataset
            if dataset_name not in selection_results:
                self.logger.warning(f"No feature selection results for dataset: {dataset_name}")
                continue

            selected_features = selection_results[dataset_name]["features"]

            # Get outcome and excluded columns
            outcome_column = self.project_config.get("columns", {}).get("outcome_column")
            exclude_columns = self.project_config.get("columns", {}).get("exclude_columns", [])

            # Create output directory for this dataset
            output_dir = os.path.join(self.result_path, dataset_name, "model_evaluation")
            os.makedirs(output_dir, exist_ok=True)

            # Get feature range for model building
            min_features = self.feature_config.get("feature_selection", {}).get("min_features", 10)
            max_features = self.feature_config.get("feature_selection", {}).get("max_features", 10)

            # Create range of feature counts to evaluate
            feature_counts = range(min_features, max_features + 1)

            # Initialize dataset results
            dataset_results = {}

            # Check if external validation is enabled and data is available
            external_validation_enabled = self.model_config.get("external_validation", {}).get("enabled", False)

            for num_features in feature_counts:
                self.logger.info(f"Training with {num_features} features")

                # Limit features to top N
                top_features = selected_features[:num_features]

                # Ensure all features exist in the dataframe
                valid_features = [f for f in top_features if f in df.columns]
                missing_features = [f for f in top_features if f not in df.columns]

                if missing_features:
                    self.logger.warning(
                        f"Missing features: {missing_features}. Proceeding with {len(valid_features)} valid features.")

                # Check for non-numeric features
                non_numeric_features = [f for f in valid_features if df[f].dtype.kind not in 'iuf']
                if non_numeric_features:
                    self.logger.warning(
                        f"Non-numeric features found: {non_numeric_features}. These will be excluded during model training.")
                    valid_features = [f for f in valid_features if f not in non_numeric_features]

                # Include exclude_columns only if they are numeric
                valid_exclude_columns = [c for c in exclude_columns if c in df.columns and df[c].dtype.kind in 'iuf']

                # Create feature directory for this feature count
                feature_output_dir = os.path.join(output_dir, f"{num_features}_features")
                os.makedirs(feature_output_dir, exist_ok=True)

                # If external validation is enabled and validation data is available
                if external_validation_enabled and hasattr(self, '_validation_data') and self._validation_data:
                    # Get validation data for this dataset
                    val_datasets = self._get_matching_validation_datasets(dataset_name)

                    if val_datasets:
                        # For each validation dataset
                        for val_dataset_name, val_df in val_datasets.items():
                            self.logger.info(f"Using external validation with dataset: {val_dataset_name}")

                            # Create X and y for training
                            X_train = df[valid_features + valid_exclude_columns]
                            y_train = df[outcome_column]

                            # Check if we have any numeric features
                            X_train_numeric = X_train.select_dtypes(include=['number'])
                            if X_train_numeric.empty:
                                self.logger.error(
                                    f"No numeric features available for training in dataset: {dataset_name}")
                                # Create an empty directory to store results, even if there are no valid features
                                os.makedirs(feature_output_dir, exist_ok=True)

                                # Create a dummy results file
                                dummy_results = {"error": "No numeric features available for training"}
                                with open(os.path.join(feature_output_dir, "error.txt"), 'w') as f:
                                    f.write("Error: No numeric features available for training in this dataset.")

                                # Return empty results for this feature count
                                model_results = {}
                            else:
                                # Continue with normal training and evaluation
                                X_train = X_train_numeric
                                y_train = df[outcome_column]






                            # Create X and y for validation (ensure same features)
                            X_val = val_df[[c for c in valid_features + valid_exclude_columns if c in val_df.columns]]
                            if len(X_val.columns) < len(valid_features):
                                missing_val_features = [f for f in valid_features if f not in X_val.columns]
                                self.logger.warning(f"Missing features in validation data: {missing_val_features}")

                            if outcome_column not in val_df.columns:
                                self.logger.error(f"Outcome column '{outcome_column}' not found in validation data")
                                continue

                            y_val = val_df[outcome_column]

                            # Train and evaluate with external validation
                            model_results = model_selector.evaluate_models_external_validation(
                                X_train, y_train,
                                X_val, y_val,
                                cv_folds=cv_folds,
                                resampling_method=resampling_obj,
                                result_path=feature_output_dir,
                                num_features=num_features
                            )

                            # Save validation results in a separate directory
                            val_output_dir = os.path.join(feature_output_dir, "validation", val_dataset_name)
                            os.makedirs(val_output_dir, exist_ok=True)

                            # Create visualizations for validation results
                            self._create_validation_visualizations(
                                model_results,
                                y_val,
                                val_output_dir,
                                num_features
                            )
                    else:
                        self.logger.warning("External validation enabled but no matching validation datasets found")
                        # Fall back to train-test split
                        X = df[valid_features + valid_exclude_columns]
                        y = df[outcome_column]

                        model_results = model_selector.evaluate_models_train_test_split(
                            X, y,
                            test_size=test_size,
                            resampling_method=resampling_obj,
                            result_path=feature_output_dir,
                            num_features=num_features
                        )
                else:
                    # Use train-test split with cross-validation
                    X = df[valid_features + valid_exclude_columns]
                    y = df[outcome_column]

                    model_results = model_selector.evaluate_models_train_test_split(
                        X, y,
                        test_size=test_size,
                        resampling_method=resampling_obj,
                        result_path=feature_output_dir,
                        num_features=num_features
                    )

                # Save results file
                results_file = os.path.join(feature_output_dir, "model_evaluation_results.xlsx")
                model_selector.save_classification_results(
                    model_results,
                    results_file,
                    num_features,
                    method=evaluation_method
                )

                # Store results
                dataset_results[num_features] = model_results

            # Save summary results
            self._save_model_summary_results(dataset_results, dataset_name, output_dir)

            # Store dataset results
            results[dataset_name] = dataset_results

        return results

    def _get_matching_validation_datasets(self, train_dataset_name: str) -> Dict[str, pd.DataFrame]:
        """
        Get validation datasets that match a training dataset.

        Args:
            train_dataset_name: Name of the training dataset

        Returns:
            Dictionary of matching validation datasets
        """
        # If no validation data is loaded, return empty dict
        if not hasattr(self, '_validation_data') or not self._validation_data:
            return {}

        # Extract parts of the training dataset name to match
        parts = train_dataset_name.split('_')

        # Find validation datasets with similar naming pattern
        matching_datasets = {}

        for val_name, val_df in self._validation_data.items():
            if any(part in val_name for part in parts):
                matching_datasets[val_name] = val_df

        return matching_datasets

    # pipelines/radiomics_pipeline.py (continued)
    def _create_validation_visualizations(
            self,
            model_results: Dict,
            y_val: pd.Series,
            output_dir: str,
            num_features: int
    ) -> None:
        """
        Create visualization plots for validation results.

        Args:
            model_results: Model evaluation results
            y_val: Validation target values
            output_dir: Output directory
            num_features: Number of features used
        """
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Import visualization functions
        from core.visualization.plotting import (
            plot_roc_curve,
            plot_calibration_curve,
            plot_confusion_matrix,
            plot_precision_recall_curve,
            plot_decision_curve_analysis
        )

        # For each model
        if "validation" in model_results:
            for model_name, model_result in model_results["validation"].items():
                y_pred = model_result["predictions"]
                y_proba = model_result["probabilities"]

                # ROC curve
                plot_roc_curve(
                    y_val,
                    y_proba,
                    title=f"ROC Curve - {model_name} ({num_features} features)",
                    filepath=os.path.join(plots_dir, f"{model_name}_roc_curve.png")
                )

                # Calibration curve
                plot_calibration_curve(
                    y_val,
                    y_proba,
                    title=f"Calibration Curve - {model_name} ({num_features} features)",
                    filepath=os.path.join(plots_dir, f"{model_name}_calibration_curve.png")
                )

                # Confusion matrix
                plot_confusion_matrix(
                    y_val,
                    y_pred,
                    title=f"Confusion Matrix - {model_name} ({num_features} features)",
                    filepath=os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
                )

                # Precision-recall curve
                plot_precision_recall_curve(
                    y_val,
                    y_proba,
                    title=f"Precision-Recall Curve - {model_name} ({num_features} features)",
                    filepath=os.path.join(plots_dir, f"{model_name}_pr_curve.png")
                )

                # Decision curve analysis
                plot_decision_curve_analysis(
                    y_val,
                    y_proba,
                    title=f"Decision Curve Analysis - {model_name} ({num_features} features)",
                    filepath=os.path.join(plots_dir, f"{model_name}_dca_curve.png")
                )

    def _save_model_summary_results(
            self,
            results: Dict,
            dataset_name: str,
            output_dir: str
    ) -> None:
        """
        Save summary of model evaluation results.

        Args:
            results: Model evaluation results
            dataset_name: Name of the dataset
            output_dir: Output directory
        """
        summary_rows = []

        for num_features, feature_results in results.items():
            # Process test/validation results
            result_type = "validation" if "validation" in feature_results else "test"

            for classifier, model_result in feature_results.get(result_type, {}).items():
                metrics = model_result["metrics"]

                summary_rows.append({
                    'Dataset': dataset_name,
                    'Num_Features': num_features,
                    'Classifier': classifier,
                    'Set': result_type.capitalize(),
                    'AUC': metrics.get('roc_auc', 'N/A'),
                    'Accuracy': metrics.get('accuracy', 'N/A'),
                    'Sensitivity': metrics.get('sensitivity', 'N/A'),
                    'Specificity': metrics.get('specificity', 'N/A'),
                    'PPV': metrics.get('ppv', 'N/A'),
                    'NPV': metrics.get('npv', 'N/A'),
                    'F1_Score': metrics.get('f1_score', 'N/A')
                })

        # Create summary dataframe
        summary_df = pd.DataFrame(summary_rows)

        if summary_df.empty:
            self.logger.warning(f"No summary results to save for dataset: {dataset_name}")
            return

        # Save to file
        summary_file = os.path.join(output_dir, "summary_results.xlsx")

        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            # Save full summary
            summary_df.to_excel(writer, sheet_name="All_Results", index=False)

            # Save best results by AUC
            best_df = summary_df.sort_values(by='AUC', ascending=False)
            best_df.to_excel(writer, sheet_name="Best_Results", index=False)

            # Group by feature count
            for num_features in summary_df['Num_Features'].unique():
                feat_df = summary_df[summary_df['Num_Features'] == num_features]
                feat_df.to_excel(writer, sheet_name=f"Features_{num_features}", index=False)

        self.logger.info(f"Saved model summary results to {summary_file}")

    def run(self):
        """
        Execute the complete radiomics pipeline.
        """
        self.logger.info("Starting radiomics pipeline")

        # Step 1: Load and preprocess data
        train_data, validation_data = self._load_data()
        self._validation_data = validation_data  # Store for later use

        # Step 2: Feature analysis
        feature_analysis_results = self._analyze_features(train_data)

        # Step 3: Feature selection
        selected_features = self._select_features(train_data, feature_analysis_results)

        # Step 4: Model training and evaluation
        model_results = self._train_and_evaluate_models(train_data, selected_features)

        # Log execution time
        execution_time = time.time() - self.start_time
        self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")

        return {
            "feature_analysis": feature_analysis_results,
            "selected_features": selected_features,
            "model_results": model_results
        }
    
    def _validate_models(
        self,
        validation_data: Dict[str, pd.DataFrame],
        selection_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Validate trained models on external data.
        
        Args:
            validation_data: Dictionary of validation dataframes
            selection_results: Dictionary of feature selection results
            
        Returns:
            Dictionary of validation results
        """
        self.logger.info("Validating models on external data")
        
        # Get validation settings
        normalize_val = self.model_config.get("external_validation", {}).get("normalize", True)
        remove_outliers_val = self.model_config.get("external_validation", {}).get("remove_outliers", False)
        
        # Initialize results
        results = {}
        
        # Iterate through each training dataset
        for train_dataset, train_selection in selection_results.items():
            train_features = train_selection["features"]
            
            # Create result directory
            train_dir = os.path.join(self.result_path, train_dataset)
            
            # Get feature counts with models
            model_dir = os.path.join(train_dir, "model_evaluation")
            feature_counts = []
            
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    if os.path.isdir(os.path.join(model_dir, item)) and item.endswith("_features"):
                        try:
                            count = int(item.split("_")[0])
                            feature_counts.append(count)
                        except:
                            continue
            
            if not feature_counts:
                self.logger.warning(f"No models found for dataset: {train_dataset}")
                continue
            
            # Initialize dataset results
            dataset_results = {}
            
            # Validate on each validation dataset
            for val_dataset, val_df in validation_data.items():
                self.logger.info(f"Validating models from {train_dataset} on {val_dataset}")
                
                # Create output directory
                val_dir = os.path.join(train_dir, "validation", val_dataset)
                os.makedirs(val_dir, exist_ok=True)
                
                # Initialize validation results
                val_results = {}
                
                # Get outcome column
                outcome_column = self.project_config.get("columns", {}).get("outcome_column")
                
                # Validate for each feature count
                for num_features in feature_counts:
                    # Get model directory
                    models_dir = os.path.join(model_dir, f"{num_features}_features", "Saved_Models")
                    
                    if not os.path.exists(models_dir):
                        self.logger.warning(f"No saved models found for {train_dataset}/{num_features} features")
                        continue
                    
                    # Create output directory for this feature count
                    feature_val_dir = os.path.join(val_dir, f"{num_features}_features")
                    os.makedirs(feature_val_dir, exist_ok=True)
                    
                    # Get model files
                    model_files = [f for f in os.listdir(models_dir) if f.endswith(f'_{num_features}_features.pkl')]
                    
                    if not model_files:
                        self.logger.warning(f"No model files found in {models_dir}")
                        continue
                    
                    # Initialize feature results
                    feature_results = {}
                    
                    # Validate each model
                    for model_file in model_files:
                        model_name = model_file.split('_')[0]
                        self.logger.info(f"Validating {model_name} on {val_dataset}")
                        
                        # Load model
                        model_path = os.path.join(models_dir, model_file)
                        try:
                            import joblib
                            model = joblib.load(model_path)
                        except Exception as e:
                            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
                            continue
                        
                        # Get feature names used for training
                        if hasattr(model, 'feature_names'):
                            feature_names = model.feature_names
                        else:
                            # Use top N features if not stored in model
                            feature_names = train_features[:num_features]
                        
                        # Check if all features exist in validation data
                        missing_features = [f for f in feature_names if f not in val_df.columns]
                        if missing_features:
                            self.logger.warning(f"Features missing in validation data: {missing_features}")
                            continue
                        
                        # Prepare validation data
                        X_val = val_df[feature_names]
                        y_val = val_df[outcome_column]
                        
                        # Apply preprocessing if configured
                        if normalize_val:
                            from sklearn.preprocessing import StandardScaler
                            X_val = pd.DataFrame(
                                StandardScaler().fit_transform(X_val),
                                columns=X_val.columns,
                                index=X_val.index
                            )
                        
                        if remove_outliers_val:
                            X_val = self._remove_outliers(X_val)
                        
                        # Generate predictions
                        try:
                            y_pred = model.predict(X_val)
                            y_proba = model.predict_proba(X_val)[:, 1]
                            
                            # Get optimal threshold from model if available
                            threshold = getattr(model, 'selected_thresh', 0.5)
                            
                            # Calculate metrics
                            from core.evaluation.classification_metrics import (
                                calculate_classification_metrics,
                                compute_confidence_interval
                            )
                            
                            metrics = calculate_classification_metrics(y_val, y_pred, y_proba, threshold)
                            ci = compute_confidence_interval(y_val, y_pred, y_proba)
                            
                            # Create visualization plots
                            self._create_validation_plots(
                                y_val, y_pred, y_proba, 
                                model_name, num_features,
                                feature_val_dir
                            )
                            
                            # Save predictions
                            self._save_predictions(
                                val_df, y_pred, y_proba,
                                model_name, num_features,
                                feature_val_dir
                            )
                            
                            # Store results
                            feature_results[model_name] = {
                                "metrics": metrics,
                                "confidence_intervals": ci
                            }
                            
                        except Exception as e:
                            self.logger.error(f"Error validating model {model_name}: {str(e)}")
                            continue
                    
                    # Store feature results
                    val_results[num_features] = feature_results
                
                # Store validation results
                dataset_results[val_dataset] = val_results
            
            # Store dataset results
            results[train_dataset] = dataset_results
            
            # Save validation summary
            self._save_validation_summary(dataset_results, train_dataset)
        
        return results
    
    # pipelines/radiomics_pipeline.py (continued)
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with outliers removed/capped
        """
        out_df = df.copy()
        
        for col in df.columns:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            # Cap values outside bounds (Winsorizing)
            out_df[col] = np.where(out_df[col] < lower, lower, out_df[col])
            out_df[col] = np.where(out_df[col] > upper, upper, out_df[col])
        
        return out_df
    
    def _create_validation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        num_features: int,
        output_dir: str
    ) -> None:
        """
        Create validation visualization plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Probability predictions
            model_name: Name of the model
            num_features: Number of features
            output_dir: Output directory
        """
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Import visualization functions
        from core.visualization.performance_plots import (
            plot_roc_curve,
            plot_calibration_curve,
            plot_confusion_matrix,
            plot_precision_recall_curve,
            plot_decision_curve_analysis
        )
        
        # ROC curve
        plot_roc_curve(
            y_true, 
            y_proba, 
            title=f"ROC Curve - {model_name} ({num_features} features)",
            filepath=os.path.join(plots_dir, f"{model_name}_roc_curve.png")
        )
        
        # Calibration curve
        plot_calibration_curve(
            y_true, 
            y_proba, 
            title=f"Calibration Curve - {model_name} ({num_features} features)",
            filepath=os.path.join(plots_dir, f"{model_name}_calibration_curve.png")
        )
        
        # Confusion matrix
        plot_confusion_matrix(
            y_true, 
            y_pred, 
            title=f"Confusion Matrix - {model_name} ({num_features} features)",
            filepath=os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
        )
        
        # Precision-recall curve
        plot_precision_recall_curve(
            y_true, 
            y_proba, 
            title=f"Precision-Recall Curve - {model_name} ({num_features} features)",
            filepath=os.path.join(plots_dir, f"{model_name}_pr_curve.png")
        )
        
        # Decision curve analysis
        plot_decision_curve_analysis(
            y_true, 
            y_proba, 
            title=f"Decision Curve Analysis - {model_name} ({num_features} features)",
            filepath=os.path.join(plots_dir, f"{model_name}_dca_curve.png")
        )
    
    def _save_predictions(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        num_features: int,
        output_dir: str
    ) -> None:
        """
        Save model predictions to file.
        
        Args:
            df: Original dataframe
            y_pred: Predicted labels
            y_proba: Probability predictions
            model_name: Name of the model
            num_features: Number of features
            output_dir: Output directory
        """
        # Get outcome and case ID columns
        outcome_column = self.project_config.get("columns", {}).get("outcome_column")
        case_column = self.project_config.get("columns", {}).get("case_column", "Case")
        
        # Check if case column exists
        if case_column not in df.columns:
            # Use index as case ID
            case_ids = df.index
        else:
            case_ids = df[case_column]
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'Case_ID': case_ids,
            'Actual_Outcome': df[outcome_column],
            'Predicted_Outcome': y_pred,
            'Probability_Score': y_proba
        })
        
        # Save to file
        output_file = os.path.join(output_dir, f"{model_name}_predictions.xlsx")
        predictions_df.to_excel(output_file, index=False)
    
    def _save_summary_results(
        self,
        results: Dict,
        dataset_name: str,
        output_dir: str,
        evaluation_method: str
    ) -> None:
        """
        Save summary of model evaluation results.
        
        Args:
            results: Model evaluation results
            dataset_name: Name of the dataset
            output_dir: Output directory
            evaluation_method: Evaluation method used
        """
        # Initialize summary rows
        summary_rows = []
        
        # Process results based on evaluation method
        for num_features, feature_results in results.items():
            if evaluation_method == "cross_validation":
                # Process CV results
                for classifier, model_result in feature_results.items():
                    metrics = model_result["metrics"]
                    
                    summary_rows.append({
                        'Dataset': dataset_name,
                        'Num_Features': num_features,
                        'Classifier': classifier,
                        'AUC': metrics.get('roc_auc', 'N/A'),
                        'Accuracy': metrics.get('accuracy', 'N/A'),
                        'Sensitivity': metrics.get('sensitivity', 'N/A'),
                        'Specificity': metrics.get('specificity', 'N/A'),
                        'PPV': metrics.get('ppv', 'N/A'),
                        'NPV': metrics.get('npv', 'N/A'),
                        'F1_Score': metrics.get('f1_score', 'N/A')
                    })
            
            elif evaluation_method == "train_test_split":
                # Process train/test results (use test metrics)
                if "test" in feature_results:
                    for classifier, model_result in feature_results["test"].items():
                        metrics = model_result["metrics"]
                        
                        summary_rows.append({
                            'Dataset': dataset_name,
                            'Num_Features': num_features,
                            'Classifier': classifier,
                            'AUC': metrics.get('roc_auc', 'N/A'),
                            'Accuracy': metrics.get('accuracy', 'N/A'),
                            'Sensitivity': metrics.get('sensitivity', 'N/A'),
                            'Specificity': metrics.get('specificity', 'N/A'),
                            'PPV': metrics.get('ppv', 'N/A'),
                            'NPV': metrics.get('npv', 'N/A'),
                            'F1_Score': metrics.get('f1_score', 'N/A')
                        })
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_rows)
        
        if summary_df.empty:
            self.logger.warning(f"No results to summarize for dataset: {dataset_name}")
            return
        
        # Save to file
        summary_file = os.path.join(output_dir, "summary_results.xlsx")
        
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            # Save full summary
            summary_df.to_excel(writer, sheet_name="All_Results", index=False)
            
            # Save best results by AUC
            best_df = summary_df.sort_values(by='AUC', ascending=False)
            best_df.to_excel(writer, sheet_name="Best_Results", index=False)
            
            # Group by feature count
            for num_features in summary_df['Num_Features'].unique():
                feat_df = summary_df[summary_df['Num_Features'] == num_features]
                feat_df.to_excel(writer, sheet_name=f"Features_{num_features}", index=False)
    
    def _save_validation_summary(
        self,
        results: Dict,
        train_dataset: str
    ) -> None:
        """
        Save summary of validation results.
        
        Args:
            results: Validation results
            train_dataset: Name of the training dataset
        """
        # Initialize summary rows
        summary_rows = []
        
        # Process validation results
        for val_dataset, val_results in results.items():
            for num_features, feature_results in val_results.items():
                for model_name, model_result in feature_results.items():
                    metrics = model_result["metrics"]
                    ci = model_result.get("confidence_intervals", {})
                    
                    # Create row with metrics and confidence intervals
                    row = {
                        'Train_Dataset': train_dataset,
                        'Val_Dataset': val_dataset,
                        'Num_Features': num_features,
                        'Model': model_name,
                        'AUC': metrics.get('roc_auc', 'N/A'),
                        'Accuracy': metrics.get('accuracy', 'N/A'),
                        'Sensitivity': metrics.get('sensitivity', 'N/A'),
                        'Specificity': metrics.get('specificity', 'N/A'),
                        'PPV': metrics.get('ppv', 'N/A'),
                        'NPV': metrics.get('npv', 'N/A'),
                        'F1_Score': metrics.get('f1_score', 'N/A')
                    }
                    
                    # Add confidence intervals if available
                    for metric, interval in ci.items():
                        if interval and len(interval) == 2:
                            row[f'{metric}_CI'] = f"({interval[0]:.3f}, {interval[1]:.3f})"
                    
                    summary_rows.append(row)
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_rows)
        
        if summary_df.empty:
            self.logger.warning(f"No validation results to summarize for dataset: {train_dataset}")
            return
        
        # Save to file
        output_dir = os.path.join(self.result_path, train_dataset, "validation")
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, "validation_summary.xlsx")
        
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            # Save full summary
            summary_df.to_excel(writer, sheet_name="All_Results", index=False)
            
            # Save best results by AUC
            best_df = summary_df.sort_values(by='AUC', ascending=False)
            best_df.to_excel(writer, sheet_name="Best_Results", index=False)
            
            # Group by validation dataset
            for val_dataset in summary_df['Val_Dataset'].unique():
                val_df = summary_df[summary_df['Val_Dataset'] == val_dataset]
                sheet_name = val_dataset.replace(':', '_')[:31]  # Excel sheet name limits
                val_df.to_excel(writer, sheet_name=sheet_name, index=False)