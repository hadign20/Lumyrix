
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score
from core.visualization.plotting import plot_kde_classwise, plot_distribution_grouped, plot_pvalue_heatmap, plot_correlation_heatmap


def run_feature_comparison_analysis(df_train: pd.DataFrame, df_val: pd.DataFrame, selected_features: List[str], config: Dict):
    cfg = config.get("feature_comparison", {})
    output_dir = cfg.get("output_dir", "./results/feature_comparison")
    os.makedirs(output_dir, exist_ok=True)

    outcome_col = config["setup"]["outcome_column"]
    tests = cfg.get("tests", ["ks", "ttest", "mannwhitney"])
    plots = cfg.get("plot_types", ["kde", "hist", "box", "violin"])

    # Ensure outcome is present
    if outcome_col not in df_train.columns or outcome_col not in df_val.columns:
        raise ValueError(f"Outcome column '{outcome_col}' must exist in both training and validation data.")

    # Filter features that exist in both datasets
    selected_features = [f for f in selected_features if f in df_train.columns and f in df_val.columns]
    df_train = df_train[selected_features + [outcome_col]].copy()
    df_val = df_val[selected_features + [outcome_col]].copy()

    # Label cohorts and combine
    df_train["Cohort"] = "Train"
    df_val["Cohort"] = "Validation"
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    results = []
    for feat in selected_features:
        train_vals = df_train[feat].dropna()
        val_vals = df_val[feat].dropna()

        stat = {"Feature": feat}
        if "ks" in tests:
            stat["KS_p_value"] = ks_2samp(train_vals, val_vals).pvalue
        if "ttest" in tests:
            stat["TTest_p_value"] = ttest_ind(train_vals, val_vals, nan_policy='omit').pvalue
        if "mannwhitney" in tests:
            stat["MWU_p_value"] = mannwhitneyu(train_vals, val_vals, alternative='two-sided').pvalue

        try:
            stat["Train_AUC"] = roc_auc_score(df_train[outcome_col], df_train[feat])
            stat["Val_AUC"] = roc_auc_score(df_val[outcome_col], df_val[feat])
        except Exception:
            stat["Train_AUC"], stat["Val_AUC"] = np.nan, np.nan

        results.append(stat)

        if "kde" in plots:
            plot_kde_classwise(df_train, df_val, feat, outcome_col, output_dir)
        if any(p in plots for p in ["hist", "box", "violin"]):
            plot_distribution_grouped(df_all, feat, group_col="Cohort", output_dir=output_dir, plot_types=plots)

    # Save comparison table
    comp_df = pd.DataFrame(results)
    comp_df.to_excel(os.path.join(output_dir, "feature_comparison_table.xlsx"), index=False)

    # P-value heatmap
    if cfg.get("save_table", True):
        pvals = comp_df.set_index("Feature")[["KS_p_value", "TTest_p_value", "MWU_p_value"]]
        plot_pvalue_heatmap(pvals, save_path=os.path.join(output_dir, "pvalue_heatmap.png"))

    # Correlation matrix
    if cfg.get("correlation", True):
        plot_correlation_heatmap(df_train[selected_features], f"Train Correlation", save_path=os.path.join(output_dir, "correlation_train.png"))
        plot_correlation_heatmap(df_val[selected_features], f"Validation Correlation", save_path=os.path.join(output_dir, "correlation_val.png"))

    # Summary stats
    if cfg.get("summary_stats", True):
        summary_train = df_train[selected_features].describe().T
        summary_val = df_val[selected_features].describe().T
        summary_train["Cohort"] = "Train"
        summary_val["Cohort"] = "Validation"
        summary = pd.concat([summary_train, summary_val])
        summary.to_excel(os.path.join(output_dir, "summary_statistics.xlsx"))
