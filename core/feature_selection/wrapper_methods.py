# feature_selector.py
# Author: Feature Selection Architect
# Description: Modular and extensible pipeline for feature filtering, ranking, and final subset selection.

import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from core.utils.io import save_excel_sheet


class FeatureSelector:
    def __init__(self, df: pd.DataFrame, config: Dict, analysis_results: Dict[str, pd.DataFrame]):
        self.df = df
        self.config = config
        self.results = analysis_results

        self.outcome = config['setup']['outcome_column']
        self.task_type = config['setup']['task_type']
        self.exclude = config['setup'].get('exclude_columns', [])
        self.output_dir = config['feature_selection'].get('output_dir', './results/feature_selection')
        os.makedirs(self.output_dir, exist_ok=True)

    def select_features(self) -> List[str]:
        steps = self.config['feature_selection'].get('pipeline', ['pvalue', 'correlation', 'mrmr'])
        selected = [c for c in self.df.columns if c not in self.exclude + [self.outcome]]

        for step in steps:
            if step == 'pvalue':
                selected = self._filter_by_pvalue(selected)
            elif step == 'auc':
                selected = self._filter_by_auc(selected)
            elif step == 'r2':
                selected = self._filter_by_r2(selected)
            elif step == 'correlation':
                selected = self._remove_correlated(selected)
            elif step == 'mrmr':
                selected = self._rank_by_mrmr(selected)
            elif step == 'lasso':
                selected = self._rank_by_lasso(selected)
            elif step == 'tree_importance':
                selected = self._rank_by_tree_importance(selected)
            elif step == 'shap':
                selected = self._rank_by_shap(selected)
            elif step == 'composite':
                selected = self._rank_by_composite()

        self._save_summary(selected)
        return selected

    def _filter_by_pvalue(self, features):
        df = self.results['univariate']
        p_thresh = self.config['feature_selection'].get('p_value_threshold', 0.05)
        filtered = df[df['Feature'].isin(features) & (df['P_Value'] <= p_thresh)]['Feature'].tolist()
        return filtered

    def _filter_by_auc(self, features):
        df = self.results['univariate']
        auc_thresh = self.config['feature_selection'].get('auc_threshold', 0.6)
        filtered = df[df['Feature'].isin(features) & (df['AUC'] >= auc_thresh)]['Feature'].tolist()
        return filtered

    def _filter_by_r2(self, features):
        df = self.results['univariate']
        r2_thresh = self.config['feature_selection'].get('r2_threshold', 0.05)
        filtered = df[df['Feature'].isin(features) & (df['R2_Score'] >= r2_thresh)]['Feature'].tolist()
        return filtered

    def _remove_correlated(self, features):
        df = self.results['correlation']
        thresh = self.config['feature_selection'].get('correlation_threshold', 0.8)
        corr_matrix = df.loc[features, features]
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > thresh)]
        return [f for f in features if f not in to_drop]

    def _rank_by_mrmr(self, features):
        df = self.results['mrmr']
        df = df[df['Feature'].isin(features)]
        k = self.config['feature_selection'].get('num_features', 20)
        return df.sort_values(by='MRMR_Count', ascending=False)['Feature'].head(k).tolist()

    def _rank_by_lasso(self, features):
        df = self.results['lasso']
        df = df[df['Feature'].isin(features)]
        return df.sort_values(by='Lasso_Coefficient', key=abs, ascending=False)['Feature'].tolist()

    def _rank_by_tree_importance(self, features):
        df = self.results['tree_importance']
        df = df[df['Feature'].isin(features)]
        return df.sort_values(by='Importance', ascending=False)['Feature'].tolist()

    def _rank_by_shap(self, features):
        df = self.results['shap']
        df = df[df['Feature'].isin(features)]
        return df.sort_values(by='SHAP_MeanAbs', ascending=False)['Feature'].tolist()

    def _rank_by_composite(self):
        df = self.results['composite']
        k = self.config['feature_selection'].get('num_features', 20)
        return df.sort_values(by='Composite_Score', ascending=False)['Feature'].head(k).tolist()

    def _save_summary(self, selected):
        path = os.path.join(self.output_dir, "selected_features.txt")
        with open(path, "w") as f:
            f.write(f"Selected Features ({len(selected)}):\n")
            for feat in selected:
                f.write(f"- {feat}\n")

        pd.DataFrame({'Selected_Features': selected}).to_excel(
            os.path.join(self.output_dir, "selected_features.xlsx"), index=False
        )
