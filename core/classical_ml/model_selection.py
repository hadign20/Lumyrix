# core/classical_ml/model_selection.py

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score
import joblib

logger = logging.getLogger(__name__)

class ModelSelector:
    """
    Class for selecting and evaluating machine learning models.
    """
    
    def __init__(self, hyperparameter_tuning: bool = True):
        """
        Initialize the model selector.
        
        Args:
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        """
        self.hyperparameter_tuning = hyperparameter_tuning
        
        # Define model configurations
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        
        self.models = {
            "LogisticRegression": {
                "model": LogisticRegression(random_state=42, max_iter=10000),
                "params": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "penalty": ["l2"],
                    "solver": ["liblinear", "lbfgs"]
                }
            },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "SVM": {
                "model": SVC(random_state=42, probability=True),
                "params": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto", 0.1, 0.01]
                }
            },
            "NaiveBayes": {
                "model": GaussianNB(),
                "params": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            "MLP": {
                "model": MLPClassifier(random_state=42, max_iter=1000),
                "params": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"]
                }
            }
        }

    # Update in core/classical_ml/model_selection.py

    def train_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            model_name: str,
            tuning: bool = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model with optional hyperparameter tuning.

        Args:
            X: Feature dataframe
            y: Target series
            model_name: Name of the model to train
            tuning: Whether to perform hyperparameter tuning

        Returns:
            Tuple of (trained model, best parameters)
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        if tuning is None:
            tuning = self.hyperparameter_tuning

        # Get model and parameters
        model = self.models[model_name]["model"]
        params = self.models[model_name]["params"]

        # Determine appropriate number of folds based on class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        cv_folds = 5  # Default

        # Adjust folds if necessary
        adjusted_cv_folds = min(cv_folds, min_class_count)
        if adjusted_cv_folds < cv_folds:
            logger.warning(
                f"Reducing GridSearchCV folds from {cv_folds} to {adjusted_cv_folds} due to limited samples in smallest class (only {min_class_count} samples)")
            cv_folds = adjusted_cv_folds

        # Hyperparameter tuning if requested and possible
        if tuning and cv_folds >= 2:
            # Create appropriate CV object
            if len(unique_classes) > 1:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv = cv_folds  # Integer for KFold with specified n_splits

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                return_train_score=True
            )

            try:
                grid_search.fit(X, y)

                # Get best model and parameters
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                logger.info(f"{model_name} best parameters: {best_params}")
                logger.info(f"{model_name} best CV score: {grid_search.best_score_:.4f}")
            except Exception as e:
                logger.warning(f"GridSearchCV failed for {model_name}: {str(e)}. Using default parameters.")
                # Train with default parameters
                model.fit(X, y)
                best_model = model
                best_params = {}
        else:
            # Train with default parameters
            model.fit(X, y)
            best_model = model
            best_params = {}

        return best_model, best_params

    # Update in core/classical_ml/model_selection.py

    def evaluate_models_train_test_split(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            test_size: float = 0.3,
            random_state: int = 42,
            resampling_method: Optional[Any] = None,
            result_path: Optional[str] = None,
            num_features: Optional[int] = None
    ) -> Dict:
        """
        Evaluate models by splitting data into train and test sets.

        This method uses cross-validation on the training set for model development,
        then evaluates final performance on the held-out test set.

        Args:
            X: Feature dataframe
            y: Target series
            test_size: Test size for train-test split
            random_state: Random state for reproducibility
            resampling_method: Method for resampling imbalanced data
            result_path: Path to save model files
            num_features: Number of features used

        Returns:
            Dictionary of evaluation results
        """
        from core.evaluation.classification_metrics import calculate_classification_metrics

        # Ensure X contains only numeric data
        X_numeric = X.select_dtypes(include=['number'])

        # Check for dropped columns
        dropped_columns = list(set(X.columns) - set(X_numeric.columns))
        if dropped_columns:
            logger.warning(f"Dropped non-numeric columns: {dropped_columns}")

        if X_numeric.empty:
            logger.error("No numeric features available for modeling")
            return {}

        # Use only numeric features for modeling
        X = X_numeric

        # Feature names for saving models
        feature_names = X.columns.tolist()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Initialize results
        results = {"train": {}, "test": {}}

        # Number of folds for cross-validation on training data
        cv_folds = 5

        # Determine appropriate number of folds based on class distribution in training set
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = np.min(class_counts)

        # Adjust folds if necessary
        adjusted_cv_folds = min(cv_folds, min_class_count)
        if adjusted_cv_folds < cv_folds:
            logger.warning(
                f"Reducing cross-validation folds from {cv_folds} to {adjusted_cv_folds} due to limited samples in smallest class (only {min_class_count} samples)")
            cv_folds = adjusted_cv_folds

        # Ensure at least 2 folds
        if cv_folds < 2:
            logger.warning(
                f"Cannot perform cross-validation with only {min_class_count} samples in smallest class. Using simple train/test evaluation.")
            cv_folds = None

        # Process each model
        for model_name in self.models.keys():
            logger.info(f"Training {model_name} using train-test split with CV on training set")

            # If we can do cross-validation on the training set
            if cv_folds and cv_folds >= 2:
                # Create cross-validation folds on training data
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                # Initialize arrays for CV predictions on training set
                y_train_true = np.array([])
                y_train_pred = np.array([])
                y_train_proba = np.array([])

                # Track CV metrics on training set
                fold_metrics = []

                # Perform cross-validation on training set
                for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    logger.info(f"  Fold {fold + 1}/{cv_folds}")

                    # Split training data into CV train and validation sets
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    # Apply resampling if provided (only on CV training data)
                    if resampling_method is not None:
                        X_cv_train_resampled, y_cv_train_resampled = resampling_method.fit_resample(X_cv_train,
                                                                                                    y_cv_train)
                    else:
                        X_cv_train_resampled, y_cv_train_resampled = X_cv_train, y_cv_train

                    # Train model on CV training data
                    cv_model, _ = self.train_model(X_cv_train_resampled, y_cv_train_resampled, model_name, tuning=False)

                    # Make predictions on CV validation data
                    fold_y_pred = cv_model.predict(X_cv_val)
                    fold_y_proba = cv_model.predict_proba(X_cv_val)[:, 1]

                    # Append predictions and targets
                    y_train_true = np.append(y_train_true, y_cv_val)
                    y_train_pred = np.append(y_train_pred, fold_y_pred)
                    y_train_proba = np.append(y_train_proba, fold_y_proba)

                    # Calculate fold metrics
                    fold_metric = calculate_classification_metrics(y_cv_val, fold_y_pred, fold_y_proba)
                    fold_metrics.append(fold_metric)

                # Calculate overall metrics on full training set (using CV predictions)
                train_metrics = calculate_classification_metrics(y_train_true, y_train_pred, y_train_proba)

            else:
                # Simple evaluation on training set without CV
                logger.warning(f"Using simple training evaluation without CV for {model_name}")

                # Apply resampling to full training set if provided
                if resampling_method is not None:
                    X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train

                # Train model on full training set
                train_model, _ = self.train_model(X_train_resampled, y_train_resampled, model_name, tuning=False)

                # Make predictions on training set
                y_train_pred = train_model.predict(X_train)
                y_train_proba = train_model.predict_proba(X_train)[:, 1]

                # Calculate metrics on training set
                train_metrics = calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
                fold_metrics = [train_metrics]  # Just one "fold"

            # Save training results
            results["train"][model_name] = {
                "metrics": train_metrics,
                "fold_metrics": fold_metrics,
                "predictions": y_train_pred,
                "probabilities": y_train_proba
            }

            # Now train the final model on the entire training set with tuning
            if resampling_method is not None:
                X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            # Train final model with hyperparameter tuning
            final_model, best_params = self.train_model(X_train_resampled, y_train_resampled, model_name, tuning=True)

            # Find optimal threshold on training data
            y_train_proba_final = final_model.predict_proba(X_train)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, y_train_proba_final)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            # Make predictions on test set
            y_test_pred = final_model.predict(X_test)
            y_test_proba = final_model.predict_proba(X_test)[:, 1]

            # Calculate metrics on test set
            test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)

            # Save test results
            results["test"][model_name] = {
                "metrics": test_metrics,
                "predictions": y_test_pred,
                "probabilities": y_test_proba,
                "parameters": best_params
            }

            # Save model if path provided
            if result_path is not None and num_features is not None:
                save_dir = os.path.join(result_path, "Saved_Models")
                os.makedirs(save_dir, exist_ok=True)

                # Add metadata to model for later use
                final_model.feature_names = feature_names
                final_model.selected_thresh = optimal_threshold
                final_model.model_name = model_name

                # Save model
                model_file = os.path.join(save_dir, f"{model_name}_{num_features}_features.pkl")
                joblib.dump(final_model, model_file)
                logger.info(f"Model saved to {model_file}")

        return results

    def evaluate_models_external_validation(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            cv_folds: int = 5,
            resampling_method: Optional[Any] = None,
            result_path: Optional[str] = None,
            num_features: Optional[int] = None
    ) -> Dict:
        """
        Evaluate models with training and external validation datasets.

        This method uses cross-validation on the training set for model development,
        then evaluates final performance on the external validation set.

        Args:
            X_train: Training feature dataframe
            y_train: Training target series
            X_val: Validation feature dataframe
            y_val: Validation target series
            cv_folds: Number of cross-validation folds for training
            resampling_method: Method for resampling imbalanced data
            result_path: Path to save model files
            num_features: Number of features used

        Returns:
            Dictionary of evaluation results
        """
        from core.evaluation.classification_metrics import calculate_classification_metrics

        # Ensure X contains only numeric data
        X_train_numeric = X_train.select_dtypes(include=['number'])
        X_val_numeric = X_val.select_dtypes(include=['number'])

        # Check for dropped columns
        dropped_train_columns = list(set(X_train.columns) - set(X_train_numeric.columns))
        if dropped_train_columns:
            logger.warning(f"Dropped non-numeric training columns: {dropped_train_columns}")

        if X_train_numeric.empty:
            logger.error("No numeric features available for training")
            return {}

        # Ensure same features in validation set
        common_features = list(set(X_train_numeric.columns).intersection(set(X_val_numeric.columns)))
        if len(common_features) != len(X_train_numeric.columns):
            missing_features = list(set(X_train_numeric.columns) - set(common_features))
            logger.warning(f"Missing features in validation set: {missing_features}")

        if not common_features:
            logger.error("No common features between training and validation datasets")
            return {}

        # Use only common numeric features
        X_train = X_train_numeric[common_features]
        X_val = X_val_numeric[common_features]

        # Feature names for saving models
        feature_names = common_features

        # Initialize results
        results = {"train": {}, "validation": {}}

        # Determine appropriate number of folds based on class distribution
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = np.min(class_counts)

        # Adjust folds if necessary
        adjusted_cv_folds = min(cv_folds, min_class_count)
        if adjusted_cv_folds < cv_folds:
            logger.warning(
                f"Reducing cross-validation folds from {cv_folds} to {adjusted_cv_folds} due to limited samples in smallest class (only {min_class_count} samples)")
            cv_folds = adjusted_cv_folds

        # Ensure at least 2 folds
        if cv_folds < 2:
            logger.warning(
                f"Cannot perform cross-validation with only {min_class_count} samples in smallest class. Using simple training evaluation.")
            cv_folds = None

        # Process each model
        for model_name in self.models.keys():
            logger.info(f"Training {model_name} with external validation")

            # If we can do cross-validation on the training set
            if cv_folds and cv_folds >= 2:
                # Create cross-validation folds on training data
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                # Initialize arrays for CV predictions on training set
                y_train_true = np.array([])
                y_train_pred = np.array([])
                y_train_proba = np.array([])

                # Track CV metrics on training set
                fold_metrics = []

                # Perform cross-validation on training set
                for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    logger.info(f"  Fold {fold + 1}/{cv_folds}")

                    # Split training data into CV train and validation sets
                    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                    # Apply resampling if provided (only on CV training data)
                    if resampling_method is not None:
                        X_cv_train_resampled, y_cv_train_resampled = resampling_method.fit_resample(X_cv_train,
                                                                                                    y_cv_train)
                    else:
                        X_cv_train_resampled, y_cv_train_resampled = X_cv_train, y_cv_train

                    # Train model on CV training data
                    cv_model, _ = self.train_model(X_cv_train_resampled, y_cv_train_resampled, model_name, tuning=False)

                    # Make predictions on CV validation data
                    fold_y_pred = cv_model.predict(X_cv_val)
                    fold_y_proba = cv_model.predict_proba(X_cv_val)[:, 1]

                    # Append predictions and targets
                    y_train_true = np.append(y_train_true, y_cv_val)
                    y_train_pred = np.append(y_train_pred, fold_y_pred)
                    y_train_proba = np.append(y_train_proba, fold_y_proba)

                    # Calculate fold metrics
                    fold_metric = calculate_classification_metrics(y_cv_val, fold_y_pred, fold_y_proba)
                    fold_metrics.append(fold_metric)

                # Calculate overall metrics on full training set (using CV predictions)
                train_metrics = calculate_classification_metrics(y_train_true, y_train_pred, y_train_proba)

            else:
                # Simple evaluation on training set without CV
                logger.warning(f"Using simple training evaluation without CV for {model_name}")

                # Apply resampling to full training set if provided
                if resampling_method is not None:
                    X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train

                # Train model on full training set
                train_model, _ = self.train_model(X_train_resampled, y_train_resampled, model_name, tuning=False)

                # Make predictions on training set
                y_train_pred = train_model.predict(X_train)
                y_train_proba = train_model.predict_proba(X_train)[:, 1]

                # Calculate metrics on training set
                train_metrics = calculate_classification_metrics(y_train, y_train_pred, y_train_proba)
                fold_metrics = [train_metrics]  # Just one "fold"

            # Save training results
            results["train"][model_name] = {
                "metrics": train_metrics,
                "fold_metrics": fold_metrics,
                "predictions": y_train_pred,
                "probabilities": y_train_proba
            }

            # Now train the final model on the entire training set with tuning
            if resampling_method is not None:
                X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            # Train final model with hyperparameter tuning
            final_model, best_params = self.train_model(X_train_resampled, y_train_resampled, model_name, tuning=True)

            # Find optimal threshold on training data
            y_train_proba_final = final_model.predict_proba(X_train)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, y_train_proba_final)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            # Make predictions on validation set
            y_val_pred = final_model.predict(X_val)
            y_val_proba = final_model.predict_proba(X_val)[:, 1]

            # Calculate metrics on validation set
            val_metrics = calculate_classification_metrics(y_val, y_val_pred, y_val_proba)

            # Save validation results
            results["validation"][model_name] = {
                "metrics": val_metrics,
                "predictions": y_val_pred,
                "probabilities": y_val_proba,
                "parameters": best_params
            }

            # Save model if path provided
            if result_path is not None and num_features is not None:
                save_dir = os.path.join(result_path, "Saved_Models")
                os.makedirs(save_dir, exist_ok=True)

                # Add metadata to model for later use
                final_model.feature_names = feature_names
                final_model.selected_thresh = optimal_threshold
                final_model.model_name = model_name

                # Save model
                model_file = os.path.join(save_dir, f"{model_name}_{num_features}_features.pkl")
                joblib.dump(final_model, model_file)
                logger.info(f"Model saved to {model_file}")

        return results





    def evaluate_models_cross_validation(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            cv_folds: int = 5,
            resampling_method: Optional[Any] = None,
            result_path: Optional[str] = None,
            num_features: Optional[int] = None
    ) -> Dict:
        """
        Evaluate models using cross-validation.

        Args:
            X: Feature dataframe
            y: Target series
            cv_folds: Number of cross-validation folds
            resampling_method: Method for resampling imbalanced data
            result_path: Path to save model files
            num_features: Number of features used

        Returns:
            Dictionary of evaluation results
        """
        from core.evaluation.classification_metrics import calculate_classification_metrics

        # Ensure X contains only numeric data
        X_numeric = X.select_dtypes(include=['number'])

        # Check for dropped columns
        dropped_columns = list(set(X.columns) - set(X_numeric.columns))
        if dropped_columns:
            logger.warning(f"Dropped non-numeric columns: {dropped_columns}")

        if X_numeric.empty:
            logger.error("No numeric features available for modeling")
            return {}

        # Use only numeric features for modeling
        X = X_numeric

        # Feature names for saving models
        feature_names = X.columns.tolist()

        # Initialize results
        results = {}

        # Determine appropriate number of folds based on class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)

        # Adjust folds if necessary
        adjusted_cv_folds = min(cv_folds, min_class_count)
        if adjusted_cv_folds < cv_folds:
            logger.warning(
                f"Reducing cross-validation folds from {cv_folds} to {adjusted_cv_folds} due to limited samples in smallest class (only {min_class_count} samples)")
            cv_folds = adjusted_cv_folds

        # Ensure at least 2 folds
        if cv_folds < 2:
            logger.warning(
                f"Cannot perform cross-validation with only {min_class_count} samples in smallest class. Switching to train-test split.")
            return self.evaluate_models_train_test_split(
                X, y,
                test_size=0.2,  # Use 20% for test
                resampling_method=resampling_method,
                result_path=result_path,
                num_features=num_features
            )

        # Create cross-validation folds
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for model_name in self.models.keys():
            logger.info(f"Training {model_name} with cross-validation...")

            # Initialize arrays for predictions and targets
            y_true = np.array([])
            y_pred = np.array([])
            y_proba = np.array([])

            # Track metrics across folds
            fold_metrics = []

            # Train and evaluate on each fold
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                logger.info(f"  Fold {fold + 1}/{cv_folds}")

                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Apply resampling if provided
                if resampling_method is not None:
                    X_train_resampled, y_train_resampled = resampling_method.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train

                # Train model
                model, _ = self.train_model(X_train_resampled, y_train_resampled, model_name)

                # Make predictions
                fold_y_pred = model.predict(X_test)
                fold_y_proba = model.predict_proba(X_test)[:, 1]

                # Append predictions and targets
                y_true = np.append(y_true, y_test)
                y_pred = np.append(y_pred, fold_y_pred)
                y_proba = np.append(y_proba, fold_y_proba)

                # Calculate fold metrics
                fold_metric = calculate_classification_metrics(y_test, fold_y_pred, fold_y_proba)
                fold_metrics.append(fold_metric)

            # Calculate overall metrics
            overall_metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

            # Train final model on all data
            if resampling_method is not None:
                X_resampled, y_resampled = resampling_method.fit_resample(X, y)
            else:
                X_resampled, y_resampled = X, y

            final_model, best_params = self.train_model(X_resampled, y_resampled, model_name)

            # Find optimal threshold
            y_proba_all = final_model.predict_proba(X)[:, 1]
            fpr, tpr, thresholds = roc_curve(y, y_proba_all)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            # Save results
            results[model_name] = {
                "metrics": overall_metrics,
                "fold_metrics": fold_metrics,
                "parameters": best_params
            }

            # Save model if path provided
            if result_path is not None and num_features is not None:
                save_dir = os.path.join(result_path, "Saved_Models")
                os.makedirs(save_dir, exist_ok=True)

                # Add metadata to model for later use
                final_model.feature_names = feature_names
                final_model.selected_thresh = optimal_threshold
                final_model.model_name = model_name

                # Save model
                model_file = os.path.join(save_dir, f"{model_name}_{num_features}_features.pkl")
                joblib.dump(final_model, model_file)
                logger.info(f"Model saved to {model_file}")

        return results

    # Update in core/classical_ml/model_selection.py

    def save_classification_results(
            self,
            results: Dict,
            output_file: str,
            num_features: int,
            method: str = "train_test_split"
    ) -> None:
        """
        Save classification results to Excel.

        Args:
            results: Results dictionary
            output_file: Output file path
            num_features: Number of features used
            method: Evaluation method ('train_test_split', 'cross_validation', or 'external_validation')
        """
        # Check if there are any results to save
        if not results or (
                ("train" not in results or not results["train"]) and
                ("test" not in results or not results["test"]) and
                ("validation" not in results or not results["validation"])
        ):
            logger.warning(f"No results to save for {num_features} features")

            # Create a dummy dataframe with a message
            dummy_df = pd.DataFrame([["No results available. Check log for errors."]],
                                    columns=["Message"])

            # Save the dummy dataframe
            dummy_df.to_excel(output_file, sheet_name="No_Results", index=False)
            return

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save results for training set
            if "train" in results and results["train"]:
                rows = []

                for classifier, result in results["train"].items():
                    metrics = result["metrics"]
                    params = result.get("parameters", {})

                    # Format parameters
                    params_str = ', '.join([f"{k}={v}" for k, v in params.items()])

                    row = [
                        classifier,
                        metrics.get("accuracy", "N/A"),
                        metrics.get("roc_auc", "N/A"),
                        metrics.get("sensitivity", "N/A"),
                        metrics.get("specificity", "N/A"),
                        metrics.get("ppv", "N/A"),
                        metrics.get("npv", "N/A"),
                        metrics.get("f1_score", "N/A"),
                        params_str
                    ]
                    rows.append(row)

                if rows:
                    df = pd.DataFrame(
                        rows,
                        columns=["Classifier", "Accuracy", "AUC", "Sensitivity", "Specificity",
                                 "PPV", "NPV", "F1 Score", "Parameters"]
                    )

                    df.to_excel(writer, sheet_name=f"Train_{num_features}_features", index=False)
                else:
                    # Add a dummy sheet if no rows
                    pd.DataFrame(["No training results"]).to_excel(
                        writer, sheet_name=f"Train_{num_features}_features", index=False, header=False)

            # Save results for test or validation set
            result_type = "validation" if "validation" in results else "test"

            if result_type in results and results[result_type]:
                rows = []

                for classifier, result in results[result_type].items():
                    metrics = result["metrics"]
                    params = result.get("parameters", {})

                    # Format parameters
                    params_str = ', '.join([f"{k}={v}" for k, v in params.items()])

                    row = [
                        classifier,
                        metrics.get("accuracy", "N/A"),
                        metrics.get("roc_auc", "N/A"),
                        metrics.get("sensitivity", "N/A"),
                        metrics.get("specificity", "N/A"),
                        metrics.get("ppv", "N/A"),
                        metrics.get("npv", "N/A"),
                        metrics.get("f1_score", "N/A"),
                        params_str
                    ]
                    rows.append(row)

                if rows:
                    df = pd.DataFrame(
                        rows,
                        columns=["Classifier", "Accuracy", "AUC", "Sensitivity", "Specificity",
                                 "PPV", "NPV", "F1 Score", "Parameters"]
                    )

                    df.to_excel(writer, sheet_name=f"{result_type.capitalize()}_{num_features}_features", index=False)
                else:
                    # Add a dummy sheet if no rows
                    pd.DataFrame([f"No {result_type} results"]).to_excel(
                        writer, sheet_name=f"{result_type.capitalize()}_{num_features}_features", index=False,
                        header=False)

            # Save per-fold metrics for training
            if "train" in results and results["train"]:
                for classifier, result in results["train"].items():
                    if "fold_metrics" in result and result["fold_metrics"]:
                        fold_rows = []
                        fold_metrics = result["fold_metrics"]

                        for fold_idx, fold_metric in enumerate(fold_metrics):
                            fold_rows.append([
                                fold_idx + 1,
                                fold_metric.get("accuracy", "N/A"),
                                fold_metric.get("roc_auc", "N/A"),
                                fold_metric.get("sensitivity", "N/A"),
                                fold_metric.get("specificity", "N/A"),
                                fold_metric.get("ppv", "N/A"),
                                fold_metric.get("npv", "N/A"),
                                fold_metric.get("f1_score", "N/A")
                            ])

                        if fold_rows:
                            fold_df = pd.DataFrame(
                                fold_rows,
                                columns=["Fold", "Accuracy", "AUC", "Sensitivity", "Specificity",
                                         "PPV", "NPV", "F1 Score"]
                            )

                            sheet_name = f"{classifier}_Folds_{num_features}"
                            # Ensure sheet name is not too long (Excel limit is 31 chars)
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:31]

                            fold_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Sort by AUC and save best sheet for test/validation results
            if result_type in results and results[result_type]:
                best_rows = []

                for classifier, result in results[result_type].items():
                    metrics = result["metrics"]

                    best_rows.append([
                        classifier,
                        metrics.get("accuracy", "N/A"),
                        metrics.get("roc_auc", "N/A"),
                        metrics.get("sensitivity", "N/A"),
                        metrics.get("specificity", "N/A"),
                        metrics.get("ppv", "N/A"),
                        metrics.get("npv", "N/A"),
                        metrics.get("f1_score", "N/A")
                    ])

                if best_rows:
                    best_df = pd.DataFrame(
                        best_rows,
                        columns=["Classifier", "Accuracy", "AUC", "Sensitivity", "Specificity",
                                 "PPV", "NPV", "F1 Score"]
                    )

                    # Sort by AUC (descending)
                    best_df = best_df.sort_values(by="AUC", ascending=False)

                    sheet_name = f"Best_{result_type}_{num_features}"
                    # Ensure sheet name is not too long
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]

                    best_df.to_excel(writer, sheet_name=sheet_name, index=False)