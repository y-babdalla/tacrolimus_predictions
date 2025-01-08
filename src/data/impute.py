"""Custom scikit-learn transformers for data processing."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMeanImputer(BaseEstimator, TransformerMixin):
    """Custom transformer to impute missing values using group means.

    This transformer groups the input DataFrame by user-specified columns and imputes
    missing values in the target columns using the mean of each group. If the group
    is not found, it falls back to the overall mean of the column.
    """

    def __init__(self, group_columns: list[str], target_columns: list[str]) -> None:
        self.group_columns = group_columns
        self.target_columns = target_columns
        self.group_means_ = {}
        self.default_means_ = {}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "GroupMeanImputer":
        """Fit the transformer and calculate group means.

        Args:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series | None): Target values (ignored).

        Returns:
            GroupMeanImputer: Fitted transformer instance.
        """
        self.group_means_ = {}
        self.default_means_ = {}
        for column in self.target_columns:
            self.group_means_[column] = x.groupby(self.group_columns)[column].mean()
            self.default_means_[column] = x[column].mean()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by imputing missing values.

        Args:
            x (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with missing values imputed.
        """
        x = x.copy()
        for column in self.target_columns:
            for index, row in x[x[column].isna()].iterrows():
                group_key = tuple(row[col] for col in self.group_columns)
                x.loc[index, column] = self.group_means_[column].get(
                    group_key, self.default_means_[column]
                )
        return x

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the feature names of the transformed DataFrame.

        Args:
            input_features (list[str] | None): Input feature names (ignored).

        Returns:
            list[str]: Output feature names.
        """
        if input_features is not None:
            return input_features
        return self.group_columns + self.target_columns


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Custom transformer to clip values in specified columns to a given percentile."""

    def __init__(self, columns: list[str], percentile: float = 98.0) -> None:
        self.columns = columns
        self.percentile = percentile
        self.clip_values = {}

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "PercentileClipper":
        """Fit the transformer and calculate clip values for specified columns.

        Args:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series | None): Target values (ignored).

        Returns:
            PercentileClipper: Fitted transformer instance.
        """
        for column in self.columns:
            self.clip_values[column] = x[column].quantile(0.98)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by clipping values in specified columns.

        Args:
            x (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with values clipped to the specified percentile.
        """
        x_clipped = x.copy()
        for column in self.columns:
            x_clipped[column] = np.clip(x_clipped[column], None, self.clip_values[column])
        return x_clipped

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the feature names of the transformed DataFrame.

        Args:
            input_features (list[str] | None): Input feature names (ignored).

        Returns:
            list[str]: Output feature names.
        """
        if input_features is not None:
            return input_features
        return self.columns


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to apply a log transformation to specified columns."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | None = None) -> "LogTransformer":
        """Return the transformer instance as log transformation is stateless."""
        return self

    @staticmethod
    def transform(x: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the input DataFrame by applying a log transformation.

        Args:
            x (pd.DataFrame | np.ndarray): Input DataFrame or array.

        Returns:
            pd.DataFrame | np.ndarray: Transformed DataFrame or array with log-transformed values.
        """
        return np.log1p(x)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the feature names of the transformed DataFrame.

        Args:
            input_features (list[str] | None): Input feature names.

        Returns:
            list[str]: Output feature names.
        """
        if input_features is not None:
            return input_features
        return self.feature_names


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified columns."""

    def __init__(self, drop_cols: list[str]) -> None:
        self.drop_cols = drop_cols
        self.feature_names_in_ = None

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "DropColumnsTransformer":
        """Fit the transformer and store input feature names.

        Args:
            x (pd.DataFrame): Input DataFrame.
            y (pd.Series | None): Target values (ignored).

        Returns:
            DropColumnsTransformer: Fitted transformer instance.
        """
        self.feature_names_in_ = x.columns.tolist()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by dropping specified columns.

        Args:
            x (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with specified columns dropped.
        """
        if isinstance(x, pd.DataFrame):
            return x.drop(columns=self.drop_cols)
        raise ValueError("DropColumnsTransformer requires input as a pandas DataFrame.")

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the feature names of the transformed DataFrame.

        Args:
            input_features (list[str] | None): Input feature names.

        Returns:
            list[str]: Output feature names.
        """
        return [col for col in self.feature_names_in_ if col not in self.drop_cols]
