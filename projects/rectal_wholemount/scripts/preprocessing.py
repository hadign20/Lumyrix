# main.py for PNET_Grade_Prediction

import os
import pandas as pd
from core.utils.config_loader import load_yaml
from core.utils.io import save_excel_sheet
from core.feature_analysis.feature_analysis import run_feature_analysis
from core.feature_selection.feature_selector import FeatureSelector
from core.classical_ml.training import train_models_cv
from core.classical_ml.evaluation import evaluate_models
from core.feature_analysis.feature_comparison_analysis import run_feature_comparison_analysis


def main():
    baseline_config_path = "./config/baseline.yaml"
    config = load_yaml(baseline_config_path)

    # Load training data
    train_path = os.path.join(config["tabular_data"]["data_path"], config["tabular_data"]["train_file"])
    df_train = pd.read_excel(train_path)

    print("\n[Step 1] Feature Analysis")
    feature_analysis_results = run_feature_analysis(df_train, baseline_config_path)

    print("\n[Step 2] Feature Selection")
    selector = FeatureSelector(df_train, config, feature_analysis_results)
    selected_features = selector.select_features()
    print(f"Selected Features: {selected_features}")

    print("\n[Step 3] Model Training with Cross-Validation")
    train_models_cv(df_train, selected_features, config)

    if config.get("external_validation", {}).get("enabled", False):
        print("\n[Step 4] External Validation")
        ext_path = config["setup"]["external_validation_file"]
        df_val = pd.read_excel(ext_path)
        from core.classical_ml.model_utils import prepare_data
        from joblib import load

        model_dir = config["model"]["output_dir"]
        best_model_path = os.path.join(model_dir, "best_model.pkl")
        if os.path.exists(best_model_path):
            model = load(best_model_path)
            X_val, y_val = prepare_data(df_val, selected_features, config["setup"]["outcome_column"])
            evaluate_models({"Best_Model": model}, X_val, y_val, config["setup"]["task_type"], model_dir)
        else:
            print(f"⚠️ Could not find best model at: {best_model_path}")

    if config.get("feature_comparison", {}).get("enabled", False):
        print("\n[Step 5] Train vs Test Feature Comparison")
        val_path = config["setup"]["external_validation_file"]
        df_val = pd.read_excel(val_path)
        run_feature_comparison_analysis(df_train, df_val, selected_features, config)

if __name__ == "__main__":
    main()
