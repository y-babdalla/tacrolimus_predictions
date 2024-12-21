import logging
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.train.config import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TIME_COLS,
    TIME_ORDERING_COL,
    TARGET,
)
from src.data.impute import (
    PercentileClipper,
    LogTransformer,
    GroupMeanImputer,
    DropColumnsTransformer,
)

logger = logging.getLogger(__name__)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# Define target
target = "tac_level"
# Define mappings
cols_map = {
    "gender": {"M": 0, "F": 1},
    "route": {"SL": 1, "ORAL": 0},
    "state": {
        "home": 0,
        "hosp": 1,
        "icu": 2,
    },  # negative values are treated as missing values by some models
    "race": {
        "White": "white",
        "Black/African American": "black",
        "Hispanic/Latino": "latin",
        "Asian": "asian",
        "Other": "other",
        "Native American": "other",
        "Multiple": "other",
        "Pacific Islander": "other",
    },
}
# Define columns to drop
drop_cols = [
    "pharmacy_id",
    "previous_route",
    "previous_formulation",
    "previous_dose_timediff",
    "alp",
    "inr",
    "bun",
]


lab_cols = [
    "ast",
    "alt",
    "bilirubin",
    "albumin",
    "creatinine",
    "sodium",
    "potassium",
    "hemoglobin",
    "hematocrit",
]
lab_skew_cols = ["ast", "alt", "bilirubin", "creatinine"]  # + log transform
lab_nonskew_cols = [lab for lab in lab_cols if lab not in lab_skew_cols]
group_cols = ["gender", "age_group"]
tgt_cols = ["weight", "height"]
keep_cols = ["gender"]


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the tacrolimus data."""
    df = load_data(file_path)
    df = df[NUMERIC_COLS + CATEGORICAL_COLS + TIME_COLS]
    df = handle_nan_values(df)
    df = convert_time_columns(df, TIME_COLS)
    df = df.sort_values(["subject_id", TIME_ORDERING_COL])
    df = calculate_time_differences(df)
    df = encode_categorical_columns(df, CATEGORICAL_COLS)
    df = handle_numerical_columns(df, NUMERIC_COLS)
    final_nan_check(df)

    logger.info(f"Preprocessed data with shape {df.shape}")
    return df


def create_preprocessor(
    scale_cols=[],
    ohe_cols=[],
    group_impute={"group_cols": [], "tgt_cols": [], "keep_cols": [], "scale": True},
    simple_impute=[],
    labs={"nontransform": [], "transform": [], "scale": True},
    ord_cols=[],
    time_cols=[]
):
    """
    Create a pipeline for data preprocessing.
    """

    pipelines = []

    # Labs: nonskew
    if len(labs["nontransform"]) > 0:
        scaler = StandardScaler() if labs["scale"] else None
        pipe_labs_nonskew = Pipeline(
            [
                ("lab_clip", PercentileClipper(columns=labs["nontransform"])),
                ("lab_imp", SimpleImputer(strategy="mean")),
                ("scaler", scaler),
            ]
        )
        pipelines.append(("lab_pipe", pipe_labs_nonskew, labs["nontransform"]))

    # Labs: skew
    if len(labs["transform"]) > 0:
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

    # Group imputation
    if len(group_impute["tgt_cols"]) > 0:
        group_cols = group_impute["group_cols"]
        tgt_cols = group_impute["tgt_cols"]
        group_imp = GroupMeanImputer(group_columns=group_cols, target_columns=tgt_cols)
        scaler = StandardScaler() if group_impute["scale"] else None
        group_impute_pipe = Pipeline(
            [
                (
                    "group_imputer",
                    group_imp,
                ),  # Imputation based on gender and age group
                ("drop_cols", DropColumnsTransformer(drop_cols=group_cols)),
                ("scaler", scaler),  # Scaling height and weight after imputation
            ]
        )
        pipelines.append(
            ("group_impute_pipe", group_impute_pipe, tgt_cols + group_cols)
        )

        if len(group_impute["keep_cols"]) > 0:
            pipelines.append(("passthrough", "passthrough", group_impute["keep_cols"]))

    # Simple mean imputation
    if len(simple_impute) > 0:
        imputer = SimpleImputer(strategy="mean")
        pipelines.append(("simple_impute", imputer, simple_impute))

    # Scale
    if len(scale_cols) > 0:
        scaler = StandardScaler()
        pipelines.append(("scaler", scaler, scale_cols))

    # One hot encoding
    if len(ohe_cols) > 0:
        formulation_categories = [
            ['adoport', 'prograf', 'american_health', 'accord', 'dr_reddy', 'envarsus', 'solution', 'advagraf']]
        race_categories = [
            ['white', 'black', 'latin', 'asian', 'other']]

        if 'formulation' in ohe_cols:
            formulation_enc = OneHotEncoder(categories=formulation_categories,  handle_unknown='ignore')
            pipelines.append(('formulation_ohe', formulation_enc, ['formulation']))
            race_enc = OneHotEncoder(categories=race_categories, handle_unknown='ignore')
            pipelines.append(('race_ohe', race_enc, ['race']))

            other_cols = [col for col in ohe_cols if col not in ['formulation', 'race']]
            if other_cols:
                other_enc = OneHotEncoder()
                pipelines.append(('other_ohe', other_enc, other_cols))

    # Ordinal encoding
    if len(ord_cols) > 0:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=6
        )  # TODO: generalise to max unique values of formulation+1
        pipelines.append(("ord", enc, ord_cols))

    if len(time_cols) > 0:
        pipelines.append(("time", StandardScaler(), time_cols))

    # Complete preprocessor
    preprocessor = ColumnTransformer(
        pipelines,
        remainder="passthrough",  # Pass through other columns unchanged if there are any
    )

    return preprocessor


def prepare_data(
    data_path,
    target,
    drop_cols,
    cols_map,
    age_group=True,
    followup_imbalance=True,
    num_samples=100,
    split=True,
):
    """
    Preliminary data preparation and split into training and testing sets
    Args:
        data_path (str): path to CSV file with raw data
        target (str): target column name
        drop_cols (list): list of columns to drop
        cols_map (dict): dictionary with columns and mapping values
        age_group (bool): if True, create age group column
        followup_imbalance (bool): if True, limit max followups per patient, evenly distributed
        num_samples (int): number of samples per patient in case of followup_imbalance
    Returns:
        X_train (pd.DataFrame): training features
        X_test (pd.DataFrame): testing features
        y_train (pd.Series): training target
        y_test (pd.Series): testing target
        group_train (pd.Series): training group
        group_test (pd.Series): testing group
    """

    # Load data
    df = pd.read_csv(data_path)
    # Prepare data
    df = data_preparation(
        df, drop_cols, cols_map, age_group, followup_imbalance, num_samples
    )
    # Split data
    if split:
        X_train, X_test, y_train, y_test, group_train, group_test = group_split(
            df, target
        )
        return X_train, X_test, y_train, y_test, group_train, group_test
    else:
        group = "subject_id"
        X = df.drop(columns=[target])
        y = df[target]
        group = df[group]
        return X, y, group


def data_preparation(
    df, drop_cols, cols_map, age_group=True, followup_imbalance=True, num_samples=100
):
    # Make an explicit copy of the DataFrame
    df = df.copy()

    # Rename columns
    if "bilirubin_total" in df.columns:
        df.rename(columns={"bilirubin_total": "bilirubin"}, inplace=True)
    # Drop irrelevant columns
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df.drop(columns=existing_drop_cols, inplace=True)
    # Remove low frequency formulation: mylan
    df = df[df.formulation != "mylan"]
    # Impute=0 for previous values from first sample per subject
    df.loc[
        df.previous_dose.isna() & df.previous_level.isna(),
        ["previous_dose", "previous_level", "previous_level_timediff"],
    ] = 0
    # Map columns
    for col, mapping in cols_map.items():
        df[col] = df[col].replace(mapping)
    # Adjust data types from object to category
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    # Create temporal age group column
    if age_group:
        bins = [18, 30, 40, 50, 60, 70, 80, 90]
        labels = [0, 1, 2, 3, 4, 5, 6]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    # Limit max followups per patient, evenly distributed
    if followup_imbalance:
        # Adjust data types
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    time_cols = ["previous_level_timediff", "level_dose_timediff", "treatment_days"]
    num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns
    num_cols_filter = [
        col
        for col in num_cols
        if col
        not in list(cols_map.keys())
        + lab_nonskew_cols
        + lab_skew_cols
        and col not in ["subject_id", "weight", "height", "previous_level_timediff" ]
        and col not in time_cols
        and x_train[col].nunique() > 4
    ]  # non categorical
    # Categorical encoding (OHE or Ordinal). Ordinal used when model has native categorical handling
    cat_cols = ["race", "formulation"]

    # ----------------Create preprocessor-----------------
    preprocessor = create_preprocessor(
        scale_cols=num_cols_filter,
        ohe_cols=cat_cols,
        group_impute={
            "group_cols": group_cols,
            "tgt_cols": tgt_cols,
            "keep_cols": keep_cols,
            "scale": True,
        },
        labs={
            "nontransform": lab_nonskew_cols,
            "transform": lab_skew_cols,
            "scale": True,
        },
        time_cols=time_cols
    )

    x_train_transformed = preprocessor.fit_transform(x_train)


    # Convert the transformed data back to a DataFrame
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


def evenly_distributed_rows(group, num_samples):
    if len(group) <= num_samples:
        # If the group size is smaller or equal to sample size, return the entire group
        return group
    else:
        # Get evenly spaced indices
        # TODO: Calculate evenly spaced time points
        indices = np.linspace(0, len(group) - 1, num_samples, dtype=int)
        return group.iloc[indices]


def group_split(df, target, group="subject_id", test_size=0.2, random_state=0):
    X = df.drop(columns=[target])
    y = df[target]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_idx, test_idx in gss.split(X, y, groups=X[group]):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    group_train = X_train[group]
    group_test = X_test[group]

    return X_train, X_test, y_train, y_test, group_train, group_test


def create_one_off_prediction_dataset(df, min_timesteps=3, padding_value=0):
    # Convert level_time to datetime if it's not already
    df["level_time"] = pd.to_datetime(df["level_time"])

    # Sort the DataFrame by subject_id and level_time
    df = df.sort_values(["subject_id", "level_time"])

    # Filter out samples with less than min_timesteps
    sample_counts = df.groupby("subject_id").size()
    valid_samples = sample_counts[sample_counts >= min_timesteps].index
    df = df[df["subject_id"].isin(valid_samples)]

    # Convert level_time to numeric (seconds since the earliest timestamp)
    earliest_time = df["level_time"].min()
    df["time_idx"] = ((df["level_time"] - earliest_time).dt.total_seconds()).astype(int)

    # Determine the features to include (all except subject_id and level_time)
    features = [col for col in df.columns if col not in ["subject_id", "level_time"]]

    # Determine the maximum number of time points
    # TODO do we want to set this to a more reasonable value?
    max_timesteps = df.groupby("subject_id").size().max()
    logger.info(f"Max timesteps: {max_timesteps}")

    # Store original dtypes
    original_dtypes = df[features].dtypes

    # Create time series DataFrame with padding
    time_series_data = []
    for subject_id, group in df.groupby("subject_id"):
        group_data = group[features].values
        pad_length = max_timesteps - len(group)
        padded_data = np.pad(
            group_data,
            ((0, pad_length), (0, 0)),
            mode="constant",
            constant_values=padding_value,
        )
        time_series_data.append(padded_data)

    # Create the padded DataFrame
    time_series_df = pd.DataFrame(
        np.concatenate(time_series_data),
        columns=features,
        index=pd.MultiIndex.from_product(
            [df["subject_id"].unique(), range(max_timesteps)],
            names=["subject_id", "time_idx"],
        ),
    )

    time_series_df = time_series_df.drop(columns=["dose_time"])

    # Convert columns back to their original types
    for col, dtype in original_dtypes.items():
        if col in time_series_df.columns:
            if np.issubdtype(dtype, np.integer):
                time_series_df[col] = (
                    time_series_df[col].fillna(padding_value).astype(dtype)
                )
            elif np.issubdtype(dtype, np.floating):
                time_series_df[col] = time_series_df[col].astype(dtype)
            else:
                time_series_df[col] = (
                    time_series_df[col].fillna(str(padding_value)).astype(dtype)
                )

    # Get the last known value of the target for each subject
    targets_df = df.groupby("subject_id")[TARGET].last().to_frame()

    # Replace 999 with 998 in all numeric columns
    for df in [time_series_df, targets_df]:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        # TODO remove this after figuring out how to configure the padding indicator
        df[numeric_columns] = df[numeric_columns].replace(999, 998)

    # TODO create windows of the data to get more samples?

    # Create OneOffPredictionDataset
    one_off_dataset = dataset.OneOffPredictionDataset(
        time_series=time_series_df, targets=targets_df
    )

    return one_off_dataset


def create_sequences_for_mlforecast(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Create sequences for each patient, suitable for MLForecast."""
    sequences = []
    unique_ids = []
    dates = []

    for patient_id, patient_data in df.groupby("subject_id"):
        patient_sequence = patient_data.drop(
            ["subject_id", "time_since_last_measurement"],
            axis=1,
        )

        # Convert datetime columns to int (timestamp)
        datetime_columns = patient_sequence.select_dtypes(
            include=["datetime64"]
        ).columns
        for col in datetime_columns:
            patient_sequence[col] = (
                patient_sequence[col].astype(int) // 10**9
            )  # Convert to Unix timestamp

        for i in range(len(patient_sequence) - window_size):
            seq = patient_sequence.iloc[i : i + window_size]
            target = patient_sequence.iloc[i + window_size]["tac_level"]

            # Flatten the sequence and add the target
            flat_seq = seq.values.flatten()
            row = np.concatenate([flat_seq, [target]])

            sequences.append(row)
            unique_ids.append(patient_id)
            dates.append(patient_sequence.iloc[i + window_size]["level_time"])

    # Create column names for the flattened sequence
    feature_names = [
        f"{col}_{i}" for i in range(window_size) for col in patient_sequence.columns
    ]
    column_names = feature_names + ["y"]

    df_sequences = pd.DataFrame(sequences, columns=column_names)
    df_sequences["unique_id"] = unique_ids
    df_sequences["ds"] = pd.to_datetime(dates, unit="s")

    return df_sequences


def create_sequences(
        df: pd.DataFrame, window_size: int, time_diff: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create sequences for each patient, back-filling with zeros if needed.
    Optionally returns time differences as a separate tuple if time_diff=True.
    """
    sequences, targets, patient_ids = [], [], []
    time_diffs = []  # List to store time differences if time_diff=True

    for patient_id, patient_data in df.groupby("remainder__subject_id"):
        # Sort by time to ensure proper sequencing
        patient_data = patient_data.sort_values("remainder__level_time", ascending=True)

        # Extract the patient sequence and targets
        # Drop the irrelevant columns and potentially the time_diff column if necessary
        patient_sequence = patient_data.drop(
            ["remainder__subject_id", "remainder__level_time", "remainder__dose_time"],
            axis=1,
        ).values

        patient_targets = patient_data["tac_level"].values

        # Optionally, extract the time differences (scaler__previous_level_timediff) and remove it from the sequence
        if time_diff:
            patient_time_diff = patient_data["time__previous_level_timediff"].values
            patient_sequence = patient_data.drop(
                ["time__previous_level_timediff", "remainder__subject_id", "remainder__level_time", "remainder__dose_time"], axis=1
            ).values

        # Pad the sequence if it's shorter than the window size
        if len(patient_sequence) < window_size:
            pad_width = window_size - len(patient_sequence)
            patient_sequence = np.pad(
                patient_sequence, ((pad_width, 0), (0, 0)), mode="constant"
            )
            if time_diff:
                patient_time_diff = np.pad(patient_time_diff, (pad_width, 0), mode="constant")

        # Create the sequences for each time window
        for i in range(len(patient_sequence) - window_size):
            seq = patient_sequence[i: i + window_size]
            sequences.append(seq)
            targets.append(patient_targets[i + window_size])
            patient_ids.append(patient_id)

            if time_diff:
                # Capture the time difference sequence as well
                time_diff_seq = patient_time_diff[i: i + window_size]
                time_diffs.append(time_diff_seq)

    # Logging the shapes for debugging
    logger.info(f"Created sequences with shape {np.array(sequences).shape}")
    logger.info(f"Created targets with shape {np.array(targets).shape}")
    logger.info(f"Created patient IDs with shape {np.array(patient_ids).shape}")

    # Return the appropriate tuple
    if time_diff:
        logger.info(f"Created time differences with shape {np.array(time_diffs).shape}")
        return (
            np.array(sequences),
            np.array(targets),
            np.array(time_diffs),
            np.array(patient_ids),
        )

    else:
        return (
            np.array(sequences),
            np.array(targets),
            np.array(patient_ids),
        )

def split_data_lstm(
    x: np.ndarray, y: np.ndarray, patient_ids: np.ndarray, test_size: float = 0.2
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split the data into train and test sets, stratified by patient."""
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
    patient_ids_train, patient_ids_test = (
        patient_ids[train_indices],
        patient_ids[test_indices],
    )

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Sort the dataframe by date
    df = df.sort_values("ds")

    # Get unique subject IDs
    unique_ids = df["unique_id"].unique()
    logger.info(f"Found {len(unique_ids)} unique subject IDs")

    # Split subject IDs
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=test_size, random_state=random_state
    )

    # Create train and test dataframes
    train_df = df[df["unique_id"].isin(train_ids)]
    test_df = df[df["unique_id"].isin(test_ids)]

    # Ensure test data is after train data
    split_date = train_df["ds"].max()
    test_df = test_df[test_df["ds"] > split_date]

    return train_df, test_df


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df


def handle_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NaN values in the DataFrame."""
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"NaN values found in columns: {nan_columns}")
        logger.info(f"NaN counts: \n{df[nan_columns].isna().sum()}")
    return df


def convert_time_columns(df: pd.DataFrame, time_columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to datetime and handle invalid values."""
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().any():
            logger.warning(f"Invalid datetime values found in {col}")
            df = df.dropna(subset=[col])
            logger.info(f"Rows with invalid {col} dropped. New shape: {df.shape}")
    return df


def calculate_time_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time differences between measurements and doses."""
    df["time_since_last_dose"] = (
        df["level_time"] - df["dose_time"]
    ).dt.total_seconds() / 3600
    df["time_since_last_measurement"] = (
        df.groupby("subject_id")["level_time"].diff().dt.total_seconds() / 3600
    )
    df["time_since_last_measurement"].fillna(0, inplace=True)
    return df


def encode_categorical_columns(
    df: pd.DataFrame, categorical_columns: List[str]
) -> pd.DataFrame:
    """Encode categorical columns using LabelEncoder."""
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def handle_numerical_columns(
    df: pd.DataFrame, numerical_columns: List[str]
) -> pd.DataFrame:
    """Handle NaN and infinite values in numerical columns."""
    nan_numerical = (
        df[numerical_columns].columns[df[numerical_columns].isna().any()].tolist()
    )

    if nan_numerical:
        logger.warning(f"NaN values found in numerical columns: {nan_numerical}")
        logger.info(
            f"NaN counts in numerical columns: \n{df[nan_numerical].isna().sum()}"
        )

    for col in nan_numerical:
        fill_value = df[col].mean() if col == "height" else df[col].median()
        df[col].fillna(fill_value, inplace=True)

    return df


def final_nan_check(df: pd.DataFrame) -> None:
    """Perform a final check for NaN values."""
    if df.isna().any().any():
        logger.warning(
            "There are still NaN values in the dataset. Consider additional preprocessing or removing these rows."  # noqa: E501
        )
        logger.info(f"Final NaN counts: \n{df.isna().sum()}")
    else:
        logger.info("No NaN values remaining in the dataset.")
