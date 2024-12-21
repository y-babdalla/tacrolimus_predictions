import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMeanImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to impute missing values in target columns using group means.
    """

    def __init__(self, group_columns, target_columns):
        self.group_columns = group_columns
        self.target_columns = target_columns

    def fit(self, X, y=None):
        # Calculate the mean for each group and each target column
        self.group_means_ = {}
        self.default_means_ = {}
        for column in self.target_columns:
            self.group_means_[column] = X.groupby(self.group_columns)[column].mean()
            self.default_means_[column] = X[
                column
            ].mean()  # General fallback mean for each column
        return self

    def transform(self, X):
        # Apply the group means to fill missing values
        X = X.copy()
        for column in self.target_columns:
            for index, row in X[X[column].isna()].iterrows():
                group_key = tuple(row[col] for col in self.group_columns)
                # Fill with group mean if available, else fill with general mean
                X.at[index, column] = self.group_means_[column].get(
                    group_key, self.default_means_[column]
                )
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        else:
            return self.group_columns + self.target_columns


class PercentileClipper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clip values in specified columns to a given percentile.
    """

    def __init__(self, columns, percentile=98):
        self.columns = columns
        self.percentile = percentile
        self.clip_values = {}

    def fit(self, X, y=None):
        # Compute the percentile value for each specified column and store it
        for column in self.columns:
            self.clip_values[column] = X[column].quantile(0.98)
        return self

    def transform(self, X):
        # Clip values for each specified column using the stored percentile values
        X_clipped = X.copy()
        for column in self.columns:
            X_clipped[column] = np.clip(
                X_clipped[column], None, self.clip_values[column]
            )
        return X_clipped

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return input_features
        else:
            return self.columns


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply a log transformation to specified columns.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # No fitting necessary for log transformation
        return self

    def transform(self, X):
        # Apply the np.log1p transformation (log(1 + X))
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        # Return the provided feature names or fall back to input_features
        if input_features is not None:
            return input_features
        else:
            return self.feature_names


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns.
    """

    def __init__(self, drop_cols):
        self.drop_cols = drop_cols
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.drop_cols)
        else:
            raise ValueError(
                "DropColumnsTransformer requires input as a pandas DataFrame."
            )

    def get_feature_names_out(self, input_features=None):
        remaining_columns = [
            col for col in self.feature_names_in_ if col not in self.drop_cols
        ]
        return remaining_columns
