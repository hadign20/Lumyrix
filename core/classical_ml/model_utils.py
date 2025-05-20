# core/model_training/model_utils.py

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def get_model(model_name: str, task_type: str, **kwargs):
    """
    Return an initialized model object based on model name and task type.

    Args:
        model_name (str): Name of the model.
        task_type (str): 'classification' or 'regression'.
        kwargs: Optional keyword arguments for the model.

    Returns:
        A scikit-learn or compatible estimator.
    """
    model_name = model_name.lower()

    if task_type == "classification":
        if model_name == "logisticregression":
            return LogisticRegression(**kwargs)
        elif model_name == "randomforest":
            return RandomForestClassifier(**kwargs)
        elif model_name == "xgboost":
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
        elif model_name == "lightgbm":
            return LGBMClassifier(**kwargs)
        elif model_name == "svm":
            return SVC(probability=True, **kwargs)
        elif model_name == "naivebayes":
            return GaussianNB(**kwargs)
        elif model_name == "decisiontree":
            return DecisionTreeClassifier(**kwargs)
        elif model_name == "knn":
            return KNeighborsClassifier(**kwargs)
        elif model_name == "gradientboosting":
            return GradientBoostingClassifier(**kwargs)
        else:
            raise ValueError(f"Unsupported classification model: {model_name}")

    elif task_type == "regression":
        if model_name == "lasso":
            return Lasso(**kwargs)
        elif model_name == "ridge":
            return Ridge(**kwargs)
        elif model_name == "randomforest":
            return RandomForestRegressor(**kwargs)
        elif model_name == "xgboost":
            return XGBRegressor(**kwargs)
        elif model_name == "lightgbm":
            return LGBMRegressor(**kwargs)
        elif model_name == "svr":
            return SVR(**kwargs)
        elif model_name == "decisiontree":
            return DecisionTreeRegressor(**kwargs)
        elif model_name == "knn":
            return KNeighborsRegressor(**kwargs)
        elif model_name == "gradientboosting":
            return GradientBoostingRegressor(**kwargs)
        else:
            raise ValueError(f"Unsupported regression model: {model_name}")

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")



# core/model_training/model_utils.py

import importlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def get_model_from_config(model_dict, model_name):
    module_path = model_dict["module"]
    class_name = model_dict["class"]
    params = model_dict.get("params", {})

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model = cls(**params)

    # Optional: Wrap with scaler if needed
    if model_name == "LogisticRegression":
        return Pipeline([("scaler", StandardScaler()), ("lr", model)])
    if model_name == "SVM":
        return Pipeline([("scaler", StandardScaler()), ("svc", model)])

    return model
