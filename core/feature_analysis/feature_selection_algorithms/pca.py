__all__ = ["run"]

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from core.visualization.plotting import plot_pca_scatter, plot_scree_plot


def run(
    df: pd.DataFrame,
    outcome_column: str,
    exclude_columns: List[str],
    config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Perform PCA on numeric features and return explained variance + 2D projections.

    Args:
        df (pd.DataFrame): Input dataframe.
        outcome_column (str): Column name for outcome.
        exclude_columns (List[str]): List of features to exclude.
        config (Dict[str, Any]): PCA config dictionary.

    Returns:
        Dict[str, pd.DataFrame]: {"loadings": PCA loadings, "projection": PCA projections}
    """
    output_dir = config["paths"]["output_dir"]
    n_components = config.get("n_components", 10)

    os.makedirs(output_dir, exist_ok=True)

    # Clean input
    df = df.dropna()
    X = df.drop(columns=[outcome_column] + exclude_columns)
    y = df[outcome_column]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Scree plot
    plot_scree_plot(
        explained_variance=pca.explained_variance_ratio_,
        save_path=os.path.join(output_dir, "pca_scree_plot.png")
    )

    # 2D projection plot (first 2 components)
    if n_components >= 2:
        projection_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            outcome_column: y.values
        })
        plot_pca_scatter(
            projection_df,
            outcome_column=outcome_column,
            save_path=os.path.join(output_dir, "pca_projection_2d.png")
        )
    else:
        projection_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])

    # Save component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X.columns
    ).reset_index().rename(columns={"index": "Feature"})
    loadings.to_excel(os.path.join(output_dir, "pca_loadings.xlsx"), index=False)

    # Save projection
    projection_df.to_excel(os.path.join(output_dir, "pca_projection.xlsx"), index=False)

    return {
        "loadings": loadings,
        "projection": projection_df
    }
