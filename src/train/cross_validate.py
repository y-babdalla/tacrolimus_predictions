"""Utility functions for LSTM cross-validation and training."""
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocess import create_sequences, impute_and_scale_data
from src.train.trainer import train_model

SEED = 42


def cross_validate(
    x: pd.DataFrame,
    y: pd.Series,
    cv: Any,
    group: np.ndarray,
    fit_params: dict[str, float | int],
    lstm_model: str,
) -> dict[str, np.ndarray]:
    """Perform cross-validation for an LSTM model.

    Uses the given CV splitter to split data into training/validation folds, trains the model,
    and collects scores and fit times.

    Args:
        x:
            Features as a pandas DataFrame.
        y:
            Target values as a pandas Series.
        cv:
            Cross-validation splitter (e.g., GroupKFold or similar).
        group:
            Array-like grouping variable (e.g., subject ID) used in cross-validation.
        fit_params (dict[str, float | int]):
            Dictionary of parameters for model fitting (e.g., 'window_size', 'hidden_size', etc.).
        lstm_model (str):
            String identifier of the LSTM model type ('time_aware' or standard).

    Returns:
        dict[str, np.ndarray]:
            A dictionary with keys 'test_score', 'train_score', 'fit_time', containing arrays
            of validation scores, training scores, and fitting times for each fold.
    """
    train_scores = []
    val_scores = []
    fit_times = []

    for _, (train_idx, val_idx) in enumerate(cv.split(x, y, group)):
        x_train_fold = x.iloc[train_idx]
        y_train_fold = y.iloc[train_idx].to_numpy().reshape(-1, 1)
        x_val_fold = x.iloc[val_idx]
        y_val_fold = y.iloc[val_idx].to_numpy().reshape(-1, 1)

        # Train and evaluate model
        rmse, rmse_train, _, _, fit_time = train_one_fold_lstm(
            x_train_fold, y_train_fold, x_val_fold, y_val_fold, fit_params, lstm_model=lstm_model
        )
        fit_times.append(fit_time)
        train_scores.append(rmse_train)
        val_scores.append(rmse)

    return {
        "test_score": np.array(val_scores),
        "train_score": np.array(train_scores),
        "fit_time": np.array(fit_times),
    }


def train_one_fold_lstm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    fit_params: dict[str, float | int],
    lstm_model: str,
    return_pred: bool = False,
    return_ids: bool = False,
    validation: bool = False,
) -> tuple[float, float, float, float, float] | tuple:
    """Train an LSTM model on one fold of data and evaluate its performance.

    Args:
        x_train:
            Training features as a DataFrame.
        y_train:
            Training targets, typically a 2D array of shape (n_samples, 1).
        x_test:
            Validation/testing features as a DataFrame.
        y_test:
            Validation/testing targets, typically a 2D array of shape (n_samples, 1).
        fit_params (dict[str, float | int]):
            Dictionary of parameters for model fitting (e.g., 'window_size', 'hidden_size', etc.).
        lstm_model (str):
            String identifier of the LSTM model type. Could be 'time_aware' or standard LSTM.
        return_pred (bool):
            If True, return predicted values and the ground truth in the output tuple.
        return_ids (bool):
            If True, also return the patient IDs associated with the test set.
        validation (bool):
            If True, training will also prepare validation sets in the trainer for early stopping.

    Returns:
        tuple of various metrics and optionally predictions/IDs:
            (rmse, rmse_train, mae, r2, fit_time) plus optional y_pred, y_test, test_patient_id.
    """
    # Preprocess data
    train_data, test_data = impute_and_scale_data(x_train, y_train, x_test, y_test)

    # Create sequences for standard LSTM
    x_train_arr, y_train_arr, train_patient_id = create_sequences(
        train_data, fit_params["window_size"]
    )
    x_test_arr, y_test_arr, test_patient_id = create_sequences(
        test_data, fit_params["window_size"]
    )

    # Adjust approach if model is time-aware
    if lstm_model == "time_aware":
        x_train_arr, y_train_arr, time_train_arr, train_patient_id = create_sequences(
            train_data, fit_params["window_size"], time_diff=True
        )
        x_test_arr, y_test_arr, time_test_arr, test_patient_id = create_sequences(
            test_data, fit_params["window_size"], time_diff=True
        )

    # Convert to torch.Tensor
    x_train_torch = torch.FloatTensor(x_train_arr.astype(np.float32))
    y_train_torch = torch.FloatTensor(y_train_arr.astype(np.float32))
    x_test_torch = torch.FloatTensor(x_test_arr.astype(np.float32))
    y_test_torch = torch.FloatTensor(y_test_arr.astype(np.float32))

    # Train model
    t_start = time.time()
    if lstm_model == "time_aware":
        time_train_torch = torch.FloatTensor(time_train_arr.astype(np.float32)).unsqueeze(-1)
        time_test_torch = torch.FloatTensor(time_test_arr.astype(np.float32)).unsqueeze(-1)
        model = train_model(
            fit_params,
            x_train_torch,
            y_train_torch,
            time_diff=time_train_torch,
            lstm_model=lstm_model,
            validation=validation,
            groups=train_patient_id,
        )
        t_end = time.time()
        # Predictions
        y_pred = model.predict(x_test_torch, time_test_torch)
        y_pred_train = model.predict(x_train_torch, time_train_torch)
    else:
        model = train_model(
            fit_params,
            x_train_torch,
            y_train_torch,
            lstm_model=lstm_model,
            validation=validation,
            groups=train_patient_id,
        )
        t_end = time.time()
        y_pred = model.predict(x_test_torch)
        y_pred_train = model.predict(x_train_torch)

    # Calculate metrics
    rmse = mean_squared_error(y_test_torch, y_pred, squared=False)
    rmse_train = mean_squared_error(y_train_torch, y_pred_train, squared=False)
    mae = mean_absolute_error(y_test_torch, y_pred)
    r2 = r2_score(y_test_torch, y_pred)

    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"RMSE Train: {rmse_train:.2f}")

    output_values = [rmse, rmse_train, mae, r2, (t_end - t_start)]

    if return_pred:
        output_values.append(y_pred)
        output_values.append(y_test_torch)

    if return_ids:
        output_values.append(test_patient_id)

    return tuple(output_values)


def get_params(trial: Any) -> dict[str, float | int]:
    """Sample hyperparameters from an Optuna trial object.

    Args:
        trial:
            Optuna trial instance used to suggest parameters.

    Returns:
        dict[str, float | int]:
            Dictionary of sampled parameters for model training.
    """
    return {
        "window_size": trial.suggest_int("window_size", 3, 20),
        "hidden_size": trial.suggest_int("hidden_size", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
    }
