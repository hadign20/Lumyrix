import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_outliers(df):
    out_df = df.copy()

    # IQR
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        out_df[col] = np.where(out_df[col] < lower, lower, out_df[col])
        out_df[col] = np.where(out_df[col] > upper, upper, out_df[col])

    return out_df

def normalize(X: pd.DataFrame):
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    cols = X.columns
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    return X

def normalize_df(df: pd.DataFrame, outcome_column: str, exclude_columns: list):
    excluded_columns = exclude_columns + [outcome_column]
    cols_to_scale = [col for col in df.columns if col not in excluded_columns]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[cols_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=cols_to_scale, index=df.index)
    result_df = pd.concat([df[exclude_columns], scaled_df, df[[outcome_column]]], axis=1)
    return result_df


def remove_outliers(df):
    out_df = df.copy()

    # IQR
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        out_df[col] = np.where(out_df[col] < lower, lower, out_df[col])
        out_df[col] = np.where(out_df[col] > upper, upper, out_df[col])

    return out_df


def fill_na(X_train: pd.DataFrame,
           X_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    # Engineering missing values in numerical variables
    # numerical_cols = [col for col in X_train.columns if X_train[col].dtypes != '0']
    # numerical_cols = [col for col in X_train.columns if col not in categorical_columns]
    numerical_cols = X_train.columns

    for df1 in [X_train, X_test]:
        for col in numerical_cols:
            col_median = X_train[col].median()
            df1[col].fillna(col_median, inplace=True)

    # # Engineering missing values in categorical variables
    # for df2 in [X_train, X_test]:
    #     for col in categorical_columns:
    #         col_mod = X_train[col].mode()[0]
    #         df2[col].fillna(col_mod, inplace=True)

    return X_train, X_test