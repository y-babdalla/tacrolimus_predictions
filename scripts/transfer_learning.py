"""Script to run the transfer learning study."""

import logging

import joblib
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.data.config import COLS_MAP, DROP_COLS
from src.data.preprocess import create_sequences, impute_and_scale_data, prepare_data
from src.train.config import TARGET
from src.train.trainer import train_tl_model


def main(model_name: str = "time_aware", ensemble: bool = False) -> None:
    """Main function to run the transfer learning study.

    Args:
        model_name (str, optional): Name of the model to use. Defaults to "time_aware".
        ensemble (bool, optional): Whether to use the ensemble model. Defaults to False.
    """
    if model_name == "lstm":
        model_path = "src/models/best_lstm_model.pkl"
        window_size = 16
    elif model_name == "attention":
        model_path = "src/models/best_attention_std.pkl"
        window_size = 20
    elif model_name == "time_aware":
        model_path = "src/models/best_time_aware_std.pkl"
        window_size = 16
        if ensemble:
            model_path = "src/models/best_time_aware_ensemble_std.pkl"
    else:
        raise ValueError("Invalid model name")
    logging.info(f"Running study {STUDY_NAME}")

    wandb.init()
    outer_cv = GroupKFold(n_splits=CV_SPLITS)

    x, y, group = prepare_data(DATA_PATH, TARGET, DROP_COLS, COLS_MAP, age_group=True, split=False)

    for fold_out, (train_out_idx, test_idx) in enumerate(outer_cv.split(x, y, groups=group)):
        x_train_out, x_test = x.iloc[train_out_idx], x.iloc[test_idx]
        y_train_out, y_test = y[train_out_idx], y[test_idx]

        train_data, test_data = impute_and_scale_data(x_train_out, y_train_out, x_test, y_test)

        if model_name != "time_aware":
            x_train, y_train, _ = create_sequences(train_data, window_size)
            x_test, y_test, group_ids = create_sequences(test_data, window_size)

            x_train = torch.FloatTensor(x_train.astype(np.float32))
            y_train = torch.FloatTensor(y_train.astype(np.float32))
            x_test = torch.FloatTensor(x_test.astype(np.float32))
            y_test = torch.FloatTensor(y_test.astype(np.float32))

            model = joblib.load(model_path)

            model = train_tl_model(x_train, y_train, model)

            model.eval()

            with torch.no_grad():
                y_pred = model(x_test)
                y_pred_train = model(x_train)
        else:
            x_train, y_train, time_train, _ = create_sequences(
                train_data, window_size, time_diff=True
            )
            x_test, y_test, time_test, group_ids = create_sequences(
                test_data, window_size, time_diff=True
            )

            x_train = torch.FloatTensor(x_train.astype(np.float32))
            y_train = torch.FloatTensor(y_train.astype(np.float32))
            time_train = torch.FloatTensor(time_train.astype(np.float32))
            x_test = torch.FloatTensor(x_test.astype(np.float32))
            y_test = torch.FloatTensor(y_test.astype(np.float32))
            time_test = torch.FloatTensor(time_test.astype(np.float32))

            model = joblib.load(model_path)

            model = train_tl_model(x_train, y_train, model, time_diff=time_train)

            model.eval()

            with torch.no_grad():
                y_pred = model(x_test, time_test)
                y_pred_train = model(x_train, time_train)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2 = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, y_pred_train)

        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"RMSE Train: {rmse_train:.2f}")

        # Save inference MIMIC
        inference_path = f"{SAVE_INFERENCE}_inference_test_{model_name}_v1_fold{fold_out}.csv"
        inference_test = pd.DataFrame(
            {"group_test": group_ids, "y_test": y_test, "y_pred": y_pred}
        )
        inference_test.to_csv(inference_path, index=False)

        wandb.log(
            {
                "rmse": rmse,
                "train_rmse": rmse_train,
                "mae": mae,
                "train_mae": mae_train,
                "r2": r2,
                "train_r2": r2_train,
                "model_name": model_name,
                "preprocessor_version": "v1",
                "fold_out": fold_out,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    DATA_PATH = "data/mimic/followup_final_v4.csv"
    SAVE_INFERENCE = "models/inference/"

    CV_SPLITS = 5
    SEED = 42

    STUDY_NAME = "LSTM-tranfer_learning"
    MODEL_NAME = "time_aware"  # lstm, attention, time_aware
    ENSEMBLE = False

    main(MODEL_NAME, ENSEMBLE)
