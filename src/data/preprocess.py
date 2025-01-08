"""Data preprocessing functions for tacrolimus data."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype, ndarray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

import src.data.config as data_config
import src.train.config as train_config
from src.data.impute import (
    DropColumnsTransformer,
    GroupMeanImputer,
    LogTransformer,
    PercentileClipper,
)

logger = logging.getLogger(__name__)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the tacrolimus data.

    Args:
        file_path (str):
            The path to the CSV file containing tacrolimus data.

    Returns:
        pd.DataFrame:
            A preprocessed pandas DataFrame containing only relevant numeric, categorical,
            and time columns, with NaN values handled and categorical columns encoded.
    """
    df = load_data(file_path)
    df = df[train_config.NUMERIC_COLS + train_config.CATEGORICAL_COLS + train_config.TIME_COLS]
    df = handle_nan_values(df)
    df = convert_time_columns(df, train_config.TIME_COLS)
    df = df.sort_values(["subject_id", train_config.TIME_ORDERING_COL])
    df = calculate_time_differences(df)
    df = encode_categorical_columns(df, train_config.CATEGORICAL_COLS)
    df = handle_numerical_columns(df, train_config.NUMERIC_COLS)
    final_nan_check(df)

    logger.info(f"Preprocessed data with shape {df.shape}")
    return df


def create_preprocessor(  # noqa: C901, PLR0912
    scale_cols: list[str] | None = None,
    ohe_cols: list[str] | None = None,
    group_impute: dict[str, list[str]] | None = None,
    simple_impute: list[str] | None = None,
    labs: dict[str, list[str]] | None = None,
    ord_cols: list[str] | None = None,
    time_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Create a pipeline for data preprocessing.

    This function returns a ColumnTransformer that can be fitted on training data
    and applied to new data. Its main goal is to combine several transformations:
    percentile clipping, simple imputation, group mean imputation, log transformations,
    scaling, and one-hot or ordinal encodings.

    Args:
        scale_cols (list[str] | None):
            Columns to apply scaling to.
        ohe_cols (list[str] | None):
            Columns to apply one-hot encoding to.
        group_impute (dict[str, list[str]] | None):
            Dictionary containing parameters for GroupMeanImputer.
            Expects the keys 'group_cols', 'tgt_cols', 'keep_cols', and 'scale'.
        simple_impute (list[str] | None):
            Columns to apply SimpleImputer to.
        labs (dict[str, list[str]] | None):
            Dictionary containing keys 'nontransform', 'transform', and 'scale' for lab
            columns that require different transformations (e.g., log transform).
        ord_cols (list[str] | None):
            Columns to apply ordinal encoding to.
        time_cols (list[str] | None):
            Columns to scale (time differences).

    Returns:
        ColumnTransformer:
            A ColumnTransformer instance that applies all specified transformations.
    """
    if scale_cols is None:
        scale_cols = []
    if ohe_cols is None:
        ohe_cols = []
    if group_impute is None:
        group_impute = {"group_cols": [], "tgt_cols": [], "keep_cols": [], "scale": True}
    if simple_impute is None:
        simple_impute = []
    if labs is None:
        labs = {"nontransform": [], "transform": [], "scale": True}
    if ord_cols is None:
        ord_cols = []
    if time_cols is None:
        time_cols = []

    pipelines = []

    # Non-transform lab columns
    if labs["nontransform"]:
        scaler = StandardScaler() if labs["scale"] else None
        pipe_labs_nonskew = Pipeline(
            [
                ("lab_clip", PercentileClipper(columns=labs["nontransform"])),
                ("lab_imp", SimpleImputer(strategy="mean")),
                ("scaler", scaler),
            ]
        )
        pipelines.append(("lab_pipe", pipe_labs_nonskew, labs["nontransform"]))

    # Transform (log) lab columns
    if labs["transform"]:
        scaler = StandardScaler() if labs["scale"] else None
        pipe_labs_skew = Pipeline(
            [
                ("lab_clip", PercentileClipper(columns=labs["transform"])),
                ("lab_imp", SimpleImputer(strategy="mean")),
                ("lab_skew", LogTransformer()),
                ("scaler", scaler),
            ]
        )
        pipelines.append(("lab_skew_pipe", pipe_labs_skew, labs["transform"]))

    # Group mean imputation
    if group_impute["tgt_cols"]:
        group_cols = group_impute["group_cols"]
        tgt_cols = group_impute["tgt_cols"]
        group_imp = GroupMeanImputer(group_columns=group_cols, target_columns=tgt_cols)
        scaler = StandardScaler() if group_impute["scale"] else None
        group_impute_pipe = Pipeline(
            [
                ("group_imputer", group_imp),
                ("drop_cols", DropColumnsTransformer(drop_cols=group_cols)),
                ("scaler", scaler),
            ]
        )
        pipelines.append(("group_impute_pipe", group_impute_pipe, tgt_cols + group_cols))

        if group_impute["keep_cols"]:
            pipelines.append(("passthrough", "passthrough", group_impute["keep_cols"]))

    if simple_impute:
        imputer = SimpleImputer(strategy="mean")
        pipelines.append(("simple_impute", imputer, simple_impute))

    if scale_cols:
        scaler = StandardScaler()
        pipelines.append(("scaler", scaler, scale_cols))

    if ohe_cols:
        formulation_categories = [
            [
                "adoport",
                "prograf",
                "american_health",
                "accord",
                "dr_reddy",
                "envarsus",
                "solution",
                "advagraf",
            ]
        ]
        race_categories = [["white", "black", "latin", "asian", "other"]]

        if "formulation" in ohe_cols:
            formulation_enc = OneHotEncoder(
                categories=formulation_categories, handle_unknown="ignore"
            )
            pipelines.append(("formulation_ohe", formulation_enc, ["formulation"]))

            race_enc = OneHotEncoder(categories=race_categories, handle_unknown="ignore")
            pipelines.append(("race_ohe", race_enc, ["race"]))

            other_cols = [col for col in ohe_cols if col not in ["formulation", "race"]]
            if other_cols:
                other_enc = OneHotEncoder()
                pipelines.append(("other_ohe", other_enc, other_cols))

    if ord_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=6)
        pipelines.append(("ord", enc, ord_cols))

    if time_cols:
        pipelines.append(("time", StandardScaler(), time_cols))

    return ColumnTransformer(pipelines, remainder="passthrough")


def prepare_data(
    data_path: str,
    target: str,
    drop_cols: list[str],
    cols_map: dict[str, dict[str, str]],
    age_group: bool = True,
    followup_imbalance: bool = True,
    num_samples: int = 100,
    split: bool = True,
) -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]
    | tuple[pd.DataFrame, pd.Series, pd.Series]
):
    """Preliminary data preparation and optional split into training and testing sets.

    Args:
        data_path (str):
            Path to the CSV file with raw data.
        target (str):
            Target column name.
        drop_cols (list[str]):
            List of columns to drop.
        cols_map (dict[str, dict[str, str]]):
            Dictionary mapping columns to replacement dictionaries.
        age_group (bool):
            If True, create an age group column.
        followup_imbalance (bool):
            If True, limit maximum follow-ups per patient.
        num_samples (int):
            Number of samples per patient if followup_imbalance is True.
        split (bool):
            If True, return the train/test split. Otherwise return the full prepared data.

    Returns:
        tuple of either:
            (x_train, x_test, y_train, y_test, group_train, group_test)
            or
            (x, y, group)
    """
    df = pd.read_csv(data_path)
    df = data_preparation(df, drop_cols, cols_map, age_group, followup_imbalance, num_samples)

    if split:
        x_train, x_test, y_train, y_test, group_train, group_test = group_split(df, target)
        return x_train, x_test, y_train, y_test, group_train, group_test

    group_col = "subject_id"
    x = df.drop(columns=[target])
    y = df[target]
    group_vals = df[group_col]
    return x, y, group_vals


def data_preparation(
    df: pd.DataFrame,
    drop_cols: list[str],
    cols_map: dict[str, dict[str, str]],
    age_group: bool = True,
    followup_imbalance: bool = True,
    num_samples: int = 100,
) -> pd.DataFrame:
    """Run necessary steps to prepare the DataFrame.

    Includes:
    - Renaming columns
    - Dropping irrelevant columns
    - Removing low frequency formulation
    - Imputing missing dose/level information for first samples
    - Mapping column values according to a provided dictionary
    - (Optionally) creating age groups
    - (Optionally) evenly subsampling follow-ups per patient
    """
    df = df.copy()

    if "bilirubin_total" in df.columns:
        df = df.rename(columns={"bilirubin_total": "bilirubin"})

    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    df = df[df.formulation != "mylan"]
    df.loc[
        df.previous_dose.isna() & df.previous_level.isna(),
        ["previous_dose", "previous_level", "previous_level_timediff"],
    ] = 0

    for col, mapping in cols_map.items():
        df[col] = df[col].replace(mapping)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    if age_group:
        bins = [18, 30, 40, 50, 60, 70, 80, 90]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    if followup_imbalance:
        df["level_time"] = pd.to_datetime(df["level_time"])
        df["dose_time"] = pd.to_datetime(df["dose_time"])

        df = df.sort_values(by=["subject_id", "level_time"])
        df_sampled = df.groupby("subject_id").apply(
            evenly_distributed_rows, num_samples=num_samples
        )
        df = df_sampled.reset_index(drop=True)

    return df


def impute_and_scale_data(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute and scale data using a pre-defined ColumnTransformer.

    Args:
        x_train (pd.DataFrame):
            Training features.
        y_train (pd.Series):
            Training target variable.
        x_test (pd.DataFrame):
            Testing features.
        y_test (pd.Series):
            Testing target variable.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Preprocessed training and testing DataFrames. The target column ('tac_level')
            is appended as the last column in both.
    """
    time_cols = ["previous_level_timediff", "level_dose_timediff", "treatment_days"]
    num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns

    # Filter numeric columns for scaling (excluding known categorical or lab columns)
    num_cols_filter = [
        col
        for col in num_cols
        if (
            col
            not in list(data_config.COLS_MAP.keys())
            + data_config.LAB_NO_SKEW_COLS
            + data_config.LAB_SKEW_COLS
        )
        and col not in ["subject_id", "weight", "height", "previous_level_timediff"]
        and col not in time_cols
        and x_train[col].nunique() > 4
    ]

    cat_cols = ["race", "formulation"]

    preprocessor = create_preprocessor(
        scale_cols=num_cols_filter,
        ohe_cols=cat_cols,
        group_impute={
            "group_cols": data_config.GROUP_COLS,
            "tgt_cols": data_config.TARGET_COLS,
            "keep_cols": data_config.KEEP_COLS,
            "scale": True,
        },
        labs={
            "nontransform": data_config.LAB_NO_SKEW_COLS,
            "transform": data_config.LAB_SKEW_COLS,
            "scale": True,
        },
        time_cols=time_cols,
    )

    x_train_transformed = preprocessor.fit_transform(x_train)
    feature_names = preprocessor.get_feature_names_out()
    x_train_preprocessed = pd.DataFrame(
        x_train_transformed, columns=feature_names, index=x_train.index
    )
    x_train_preprocessed["tac_level"] = y_train

    x_test_transformed = preprocessor.transform(x_test)
    x_test_preprocessed = pd.DataFrame(
        x_test_transformed, columns=feature_names, index=x_test.index
    )
    x_test_preprocessed["tac_level"] = y_test

    return x_train_preprocessed, x_test_preprocessed


def evenly_distributed_rows(group_df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    """Return evenly spaced rows from a DataFrame grouped by subject.

    Args:
        group_df (pd.DataFrame):
            DataFrame segment for a single subject.
        num_samples (int):
            Number of rows to keep.

    Returns:
        pd.DataFrame:
            A subset of `group_df` with rows chosen at evenly spaced indices.
    """
    if len(group_df) <= num_samples:
        return group_df
    indices = np.linspace(0, len(group_df) - 1, num_samples, dtype=int)
    return group_df.iloc[indices]


def group_split(
    df: pd.DataFrame,
    target: str,
    group_col: str = "subject_id",
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Group-aware train/test split using GroupShuffleSplit.

    Args:
        df (pd.DataFrame):
            The input DataFrame.
        target (str):
            Target column name.
        group_col (str):
            Group column name. Default is "subject_id".
        test_size (float):
            Proportion of the data to include in the test split.
        random_state (int):
            Random state for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
            x_train, x_test, y_train, y_test, group_train, group_test
    """
    x = df.drop(columns=[target])
    y = df[target]
    x_train = x_test = y_train = y_test = None

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in gss.split(x, y, groups=x[group_col]):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    group_train = x_train[group_col]
    group_test = x_test[group_col]

    return x_train, x_test, y_train, y_test, group_train, group_test


def create_sequences_for_mlforecast(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Create sequences suitable for MLForecast.

    Flattens rolling windows of size `window_size` into single rows.
    Appends the next measurement's tacrolimus level as the target.

    Args:
        df (pd.DataFrame):
            The input DataFrame, containing 'subject_id' and 'tac_level'.
        window_size (int):
            The length of each sequence.

    Returns:
        pd.DataFrame:
            A new DataFrame where each row represents a flattened sequence of length `window_size`,
            plus the target and identifying fields ('unique_id', 'ds').
    """
    sequences = []
    unique_ids = []
    dates = []
    patient_sequence = None

    for patient_id, patient_data in df.groupby("subject_id"):
        patient_sequence = patient_data.drop(["subject_id", "time_since_last_measurement"], axis=1)

        datetime_columns = patient_sequence.select_dtypes(include=["datetime64"]).columns
        for col in datetime_columns:
            patient_sequence[col] = patient_sequence[col].astype(int) // 10**9

        for i in range(len(patient_sequence) - window_size):
            seq_slice = patient_sequence.iloc[i : i + window_size]
            target_val = patient_sequence.iloc[i + window_size]["tac_level"]

            flattened_seq = seq_slice.to_numpy().flatten()
            row = np.concatenate([flattened_seq, [target_val]])
            sequences.append(row)

            unique_ids.append(patient_id)
            dates.append(patient_sequence.iloc[i + window_size]["level_time"])

    feature_names = [f"{col}_{i}" for i in range(window_size) for col in patient_sequence.columns]
    column_names = [*feature_names, "y"]

    df_sequences = pd.DataFrame(sequences, columns=column_names)
    df_sequences["unique_id"] = unique_ids
    df_sequences["ds"] = pd.to_datetime(dates, unit="s")

    return df_sequences


def create_sequences(
    df: pd.DataFrame, window_size: int, time_diff: bool = False
) -> (
    tuple[
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
    ]
    | tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]
):
    """Create sequences for each patient, optionally returning time differences.

    Args:
        df (pd.DataFrame):
            Input DataFrame with 'remainder__subject_id' and 'remainder__level_time' columns.
        window_size (int):
            Number of time steps in each sequence.
        time_diff (bool):
            If True, returns time difference arrays as well.

    Returns:
        tuple or tuple of tuples:
            If time_diff is False: (sequences, targets, patient_ids)
            If time_diff is True: (sequences, targets, time_diffs, patient_ids)
    """
    sequences, targets, patient_ids = [], [], []
    time_diffs = []
    patient_time_diff = None

    for patient_id, patient_data in df.groupby("remainder__subject_id"):
        patient_data_iter = patient_data.sort_values("remainder__level_time", ascending=True)

        # Prepare the full array for the patient
        patient_sequence = patient_data_iter.drop(
            ["remainder__subject_id", "remainder__level_time", "remainder__dose_time"], axis=1
        ).to_numpy()

        patient_targets = patient_data_iter["tac_level"].to_numpy()

        if time_diff:
            patient_time_diff = patient_data_iter["time__previous_level_timediff"].to_numpy()
            patient_sequence = patient_data_iter.drop(
                [
                    "time__previous_level_timediff",
                    "remainder__subject_id",
                    "remainder__level_time",
                    "remainder__dose_time",
                ],
                axis=1,
            ).to_numpy()

        if len(patient_sequence) < window_size:
            pad_width = window_size - len(patient_sequence)
            patient_sequence = np.pad(patient_sequence, ((pad_width, 0), (0, 0)), mode="constant")

            if time_diff:
                patient_time_diff = np.pad(patient_time_diff, (pad_width, 0), mode="constant")

        # Create rolling windows
        for i in range(len(patient_sequence) - window_size):
            seq_window = patient_sequence[i : i + window_size]
            sequences.append(seq_window)
            targets.append(patient_targets[i + window_size])
            patient_ids.append(patient_id)

            if time_diff:
                time_diff_window = patient_time_diff[i : i + window_size]
                time_diffs.append(time_diff_window)

    logger.info(f"Created sequences with shape {np.array(sequences).shape}")
    logger.info(f"Created targets with shape {np.array(targets).shape}")
    logger.info(f"Created patient IDs with shape {np.array(patient_ids).shape}")

    if time_diff:
        logger.info(f"Created time differences with shape {np.array(time_diffs).shape}")
        return (
            np.array(sequences),
            np.array(targets),
            np.array(time_diffs),
            np.array(patient_ids),
        )

    return np.array(sequences), np.array(targets), np.array(patient_ids)


def split_data_lstm(
    x: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, test_size: float = 0.2
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Split sequences into train and test sets, stratified by patient ID.

    Args:
        x (np.ndarray):
            Input features (sequences).
        y (np.ndarray):
            Target values.
        patient_ids (np.ndarray):
            Array of patient IDs corresponding to each sequence.
        test_size (float):
            Proportion of sequences to use for testing.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
              np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (x_train, x_test, y_train, y_test, patient_ids_train, patient_ids_test,
            train_indices, test_indices)
    """
    unique_patients = np.unique(patient_ids)
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=42
    )

    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    patient_ids_train, patient_ids_test = patient_ids[train_indices], patient_ids[test_indices]

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        patient_ids_train,
        patient_ids_test,
        train_indices,
        test_indices,
    )


def split_data_xgb(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame for XGBoost usage, respecting the time order.

    Sorts the DataFrame by date, then splits by subject ID to ensure no overlap
    between train and test sets. Ensures the test set is strictly after the train set
    in terms of date.

    Args:
        df (pd.DataFrame):
            Input DataFrame with columns 'unique_id' and 'ds'.
        test_size (float):
            Proportion of data to use for testing.
        random_state (int):
            Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            (train_df, test_df)
    """
    df = df.sort_values("ds")
    unique_ids = df["unique_id"].unique()
    logger.info(f"Found {len(unique_ids)} unique subject IDs")

    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )

    train_df = df[df["unique_id"].isin(train_ids)]
    test_df = df[df["unique_id"].isin(test_ids)]

    # Ensure test data is chronologically after train data
    split_date = train_df["ds"].max()
    test_df = test_df[test_df["ds"] > split_date]

    return train_df, test_df


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        file_path (str):
            The path to the CSV file.

    Returns:
        pd.DataFrame:
            Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df


def handle_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check for and handle NaN values.

    Logs a warning and counts of any columns that contain NaNs.

    Args:
        df (pd.DataFrame):
            Input DataFrame.

    Returns:
        pd.DataFrame:
            Same DataFrame (potentially unchanged).
    """
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"NaN values found in columns: {nan_columns}")
        logger.info(f"NaN counts: \n{df[nan_columns].isna().sum()}")
    return df


def convert_time_columns(df: pd.DataFrame, time_columns: list[str]) -> pd.DataFrame:
    """Convert specified columns to datetime and remove rows with invalid values.

    Args:
        df (pd.DataFrame):
            Input DataFrame.
        time_columns (list[str]):
            Columns to convert to datetime.

    Returns:
        pd.DataFrame:
            DataFrame with specified columns converted to datetime.
    """
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().any():
            logger.warning(f"Invalid datetime values found in {col}")
            df = df.dropna(subset=[col])
            logger.info(f"Rows with invalid {col} dropped. New shape: {df.shape}")
    return df


def calculate_time_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time differences between measurements and doses.

    Args:
        df (pd.DataFrame):
            Input DataFrame with 'level_time' and 'dose_time' columns.

    Returns:
        pd.DataFrame:
            DataFrame augmented with 'time_since_last_dose' and
            'time_since_last_measurement' columns.
    """
    # Convert the difference between times to hours
    df["time_since_last_dose"] = (df["level_time"] - df["dose_time"]).dt.total_seconds() / 3600
    df["time_since_last_measurement"] = (
        df.groupby("subject_id")["level_time"].diff().dt.total_seconds() / 3600
    )
    df["time_since_last_measurement"] = df["time_since_last_measurement"].fillna(0)
    return df


def encode_categorical_columns(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """Encode categorical columns using LabelEncoder.

    Args:
        df (pd.DataFrame):
            Input DataFrame.
        categorical_columns (list[str]):
            Columns to be label-encoded.

    Returns:
        pd.DataFrame:
            DataFrame with label-encoded columns.
    """
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def handle_numerical_columns(df: pd.DataFrame, numerical_columns: list[str]) -> pd.DataFrame:
    """Handle NaN and infinite values in numerical columns.

    Args:
        df (pd.DataFrame):
            Input DataFrame.
        numerical_columns (list[str]):
            Numerical columns to handle.

    Returns:
        pd.DataFrame:
            DataFrame with numerical columns cleaned.
    """
    nan_numerical = df[numerical_columns].columns[df[numerical_columns].isna().any()].tolist()
    if nan_numerical:
        logger.warning(f"NaN values found in numerical columns: {nan_numerical}")
        logger.info(f"NaN counts in numerical columns: \n{df[nan_numerical].isna().sum()}")

    for col in nan_numerical:
        fill_value = df[col].mean() if col == "height" else df[col].median()
        df[col] = df[col].fillna(fill_value)

    return df


def final_nan_check(df: pd.DataFrame) -> None:
    """Perform a final check for NaN values.

    Logs a warning if there are still any NaNs in the data.

    Args:
        df (pd.DataFrame):
            Input DataFrame.

    Returns:
        None
    """
    if df.isna().any().any():
        logger.warning(
            "There are still NaN values in the dataset. Consider additional preprocessing or "
            "removing these rows."
        )
        logger.info(f"Final NaN counts: \n{df.isna().sum()}")
    else:
        logger.info("No NaN values remaining in the dataset.")
