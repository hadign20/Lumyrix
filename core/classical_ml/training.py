# core/model_training/training.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from core.utils.config_loader import load_yaml
from core.classical_ml.model_utils import get_model_from_config
from sklearn.base import clone

def train_models_cv(X, y, config: dict, task_type: str, global_config: dict) -> dict:
    """
    Train models with optional tuning, CV, and resampling.
    Returns a dictionary of best-trained models.
    """
    models_config_path = config.get("model_config", "config/classical_ml_config/classical_ml_models.yaml")
    param_grid_path = config.get("grid_search_config", "config/classical_ml_config/grid_search_configs.yaml")

    model_defs = load_yaml(models_config_path)[task_type]
    grid_defs = load_yaml(param_grid_path).get(task_type, {})

    cv_folds = config.get("cv_folds", 5)
    scoring = config.get("scoring", "roc_auc" if task_type == "classification" else "neg_mean_squared_error")
    tuning = config.get("tuning", False)
    resampling = config.get("resampling", False)
    resampling_method = config.get("resampling_method", None)

    trained_models = {}
    for model_name in config["classifiers"]:
        print(f"\n🔍 Training {model_name} with {'GridSearchCV' if tuning else 'default config'}")

        model = get_model_from_config(model_defs[model_name], model_name)
        param_grid = grid_defs.get(model_name, {}) if tuning else {}

        # Wrap in pipeline if resampling
        steps = []
        if resampling:
            sampler = {
                "SMOTE": SMOTE(random_state=42),
                "RandomOverSampler": RandomOverSampler(random_state=42),
                "RandomUnderSampler": RandomUnderSampler(random_state=42),
                "SMOTEENN": SMOTEENN(random_state=42),
            }.get(resampling_method)
            if sampler:
                steps.append(("sampler", sampler))
        steps.append(("model", model))
        pipeline = Pipeline(steps)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if task_type == "classification" else \
             KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        if tuning and param_grid:
            grid = GridSearchCV(pipeline, param_grid={f"model__{k}": v for k, v in param_grid.items()},
                                scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
            grid.fit(X, y)
            trained_models[model_name] = grid.best_estimator_
        else:
            pipeline.fit(X, y)
            trained_models[model_name] = pipeline

    return trained_models
