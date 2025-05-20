import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.multiclass import type_of_target


def detect_task_type(y: pd.Series) -> str:
    """Detect task type based on outcome variable."""
    unique_vals = y.dropna().unique()
    if len(unique_vals) <= 10 and y.dtype.kind in 'biu':
        return "classification"
    return "regression"


def mrmr_feature_selection(df: pd.DataFrame,
                           outcome_column: str,
                           exclude_columns: List[str] = [],
                           num_features: int = 10,
                           cv_folds: int = 5) -> pd.DataFrame:
    """
    Perform MRMR feature selection across CV folds (classification only).
    """
    print("Performing MRMR feature selection via cross-validation.")
    X = df.drop(columns=exclude_columns + [outcome_column])
    y = df[outcome_column]
    selected_feature_count = {}

    for feature in X.columns:
        selected_feature_count[feature] = 0

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for train_idx, _ in kf.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        try:
            selected = mrmr_classif(X_train, y_train, K=num_features)
            for f in selected:
                selected_feature_count[f] += 1
        except:
            continue

    df_result = pd.DataFrame(list(selected_feature_count.items()), columns=['Feature', 'MRMR_Count'])
    return df_result.sort_values(by="MRMR_Count", ascending=False)


def lasso_feature_selection(df: pd.DataFrame,
                            outcome_column: str,
                            exclude_columns: List[str] = [],
                            cv_folds: int = 5) -> pd.DataFrame:
    """
    Perform LassoCV for both regression and classification tasks.
    """
    X = df.drop(columns=exclude_columns + [outcome_column])
    y = df[outcome_column]
    task_type = detect_task_type(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if task_type == 'regression':
        lasso = LassoCV(cv=cv_folds, random_state=42).fit(X_scaled, y)
    else:
        # Use logistic regression with L1 penalty for classification
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=5000, C=1.0)
        lasso.fit(X_scaled, y)

    coef = lasso.coef_[0] if hasattr(lasso, 'coef_') else lasso.coef_
    features = X.columns[coef != 0]
    coef_values = coef[coef != 0]

    df_result = pd.DataFrame({'Feature': features, 'Lasso_Coefficient': coef_values})
    return df_result.sort_values(by='Lasso_Coefficient', key=np.abs, ascending=False)


def pca_feature_ranking(df: pd.DataFrame,
                        outcome_column: str,
                        exclude_columns: List[str] = [],
                        n_components: int = 2) -> pd.DataFrame:
    """
    Rank features based on their contribution to the first N PCA components.
    """
    X = df.drop(columns=exclude_columns + [outcome_column])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    feature_weights = np.mean(np.abs(pca.components_), axis=0)

    df_result = pd.DataFrame({
        'Feature': X.columns,
        'PCA_Weight': feature_weights
    })

    return df_result.sort_values(by='PCA_Weight', ascending=False)


def tree_based_feature_importance(df: pd.DataFrame,
                                   outcome_column: str,
                                   exclude_columns: List[str] = [],
                                   model_type: str = 'random_forest',
                                   num_features: int = 20) -> pd.DataFrame:
    """
    Use Random Forest to rank features by importance.
    Supports both classification and regression.
    """
    X = df.drop(columns=exclude_columns + [outcome_column])
    y = df[outcome_column]
    task_type = detect_task_type(y)

    if model_type == 'random_forest':
        if task_type == 'regression':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Currently only supports 'random_forest'.")

    model.fit(X, y)
    importances = model.feature_importances_

    df_result = pd.DataFrame({
        'Feature': X.columns,
        'Tree_Importance': importances
    })

    return df_result.sort_values(by='Tree_Importance', ascending=False).head(num_features)
