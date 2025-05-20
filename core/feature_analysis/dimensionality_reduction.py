__all__ = ["run"]

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy.stats import (
    ttest_ind, mannwhitneyu, fisher_exact, chi2_contingency,
    spearmanr, pearsonr, f_oneway
)
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict





def run_univariate_analysis(df: pd.DataFrame, config: dict, cv_folds: int = 5) -> pd.DataFrame:
    task_type = config["setup"]["task_type"]
    outcome_col = config["setup"]["outcome_column"]
    categorical_cols = config["setup"].get("categorical_columns", [])
    exclude_cols = config["setup"].get("exclude_columns", [])

    if task_type == "classification":
        pval_df = _p_values_classification_cv(df, outcome_col, categorical_cols, exclude_cols, cv_folds)
        auc_df = _auc_values_classification_cv(df, outcome_col, categorical_cols, exclude_cols, cv_folds)
        return pval_df.merge(auc_df, on="Feature", how="outer")

    elif task_type == "regression":
        pval_df = _p_values_regression(df, outcome_col, categorical_cols, exclude_cols)
        r2_df = _r2_scores_regression_cv(df, outcome_col, categorical_cols, exclude_cols, cv_folds)
        return pval_df.merge(r2_df, on="Feature", how="outer")

    else:
        raise ValueError("Unsupported task_type. Must be 'classification' or 'regression'.")


# ────────────────────────────────────────────────
# CLASSIFICATION P-VALUES + AUC
# ────────────────────────────────────────────────
def _p_values_classification_cv(df, outcome_col, categorical_cols, exclude_cols, cv_folds):
    kf = StratifiedKFold(n_splits=cv_folds)
    feature_pvalue_avg = defaultdict(float)
    count = defaultdict(int)

    for train_idx, _ in kf.split(df, df[outcome_col]):
        train_fold = df.iloc[train_idx]
        for col in df.columns:
            if col in exclude_cols or col == outcome_col:
                continue
            try:
                if col in categorical_cols:
                    table = pd.crosstab(train_fold[col], train_fold[outcome_col])
                    p = fisher_exact(table)[1] if table.shape == (2, 2) else chi2_contingency(table)[1]
                else:
                    g0 = train_fold[train_fold[outcome_col] == 0][col].dropna()
                    g1 = train_fold[train_fold[outcome_col] == 1][col].dropna()
                    p = mannwhitneyu(g0, g1)[1]
            except Exception:
                p = np.nan
            feature_pvalue_avg[col] += p
            count[col] += 1

    result = {k: feature_pvalue_avg[k] / count[k] for k in feature_pvalue_avg}
    return pd.DataFrame(result.items(), columns=["Feature", "P_Value"]).sort_values(by="P_Value")


def _auc_values_classification_cv(df, outcome_col, categorical_cols, exclude_cols, cv_folds):
    aucs = {}
    y = df[outcome_col].values
    scaler = MinMaxScaler()

    for col in df.columns:
        if col == outcome_col or col in exclude_cols:
            continue
        try:
            X = df[col].astype("category").cat.codes if col in categorical_cols else df[[col]].values
            X = scaler.fit_transform(X.reshape(-1, 1)) if X.ndim == 1 else scaler.fit_transform(X)

            model = LogisticRegression(solver='liblinear')
            cv = StratifiedKFold(n_splits=cv_folds)
            auc_score = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
            aucs[col] = auc_score if auc_score >= 0.5 else 1 - auc_score
        except Exception:
            aucs[col] = np.nan

    return pd.DataFrame(aucs.items(), columns=["Feature", "AUC"]).sort_values(by="AUC", ascending=False)


# ────────────────────────────────────────────────
# REGRESSION P-VALUES + R²
# ────────────────────────────────────────────────
def _p_values_regression(df, outcome_col, categorical_cols, exclude_cols):
    pvals = {}
    for col in df.columns:
        if col in exclude_cols or col == outcome_col:
            continue
        try:
            temp_df = df[[col, outcome_col]].dropna()
            X = pd.get_dummies(temp_df[[col]].astype(str), drop_first=True) if col in categorical_cols else temp_df[[col]]
            X = sm.add_constant(X)
            y = temp_df[outcome_col]
            model = sm.OLS(y, X).fit()
            pvals[col] = model.pvalues.drop("const", errors='ignore').min()
        except Exception:
            pvals[col] = np.nan
    return pd.DataFrame(pvals.items(), columns=["Feature", "P_Value"]).sort_values(by="P_Value")


def _r2_scores_regression_cv(df, outcome_col, categorical_cols, exclude_cols, cv_folds):
    r2_scores = {}
    y = df[outcome_col].values
    scaler = MinMaxScaler()

    for col in df.columns:
        if col in exclude_cols or col == outcome_col:
            continue
        try:
            X = pd.factorize(df[col])[0].reshape(-1, 1) if col in categorical_cols else df[[col]].values
            X = scaler.fit_transform(X)
            model = LinearRegression()
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
            r2_scores[col] = r2
        except Exception:
            r2_scores[col] = np.nan

    return pd.DataFrame(r2_scores.items(), columns=["Feature", "R2_Score"]).sort_values(by="R2_Score", ascending=False)



def run(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    return run_univariate_analysis(df, config)