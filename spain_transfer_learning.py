import joblib
import pandas as pd
import torch
import wandb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np
from sklearn.model_selection import GroupKFold

from src.data.preprocess import (
    prepare_data,
    target,
    drop_cols,
    cols_map, impute_and_scale_data, create_sequences,
)

from src.train.trainer import train_tl_model

DATA_PATH = 'data/mimic/followup_final_v4.csv'
SAVE_INFERENCE= 'models/inference/'


CV_SPLITS = 5
SEED = 42

outer_cv = GroupKFold(n_splits=CV_SPLITS)

study_name = f"LSTM-tranfer_learning"
model_name = "time_aware" # lstm, attention, time_aware
print(f"Running study {study_name}")
ensemble =False

if model_name == "lstm":
    MODEL_PATH = "src/models/best_lstm_model.pkl"
    window_size = 16
elif model_name == "attention":
    MODEL_PATH = "src/models/best_attention_std.pkl"
    window_size = 20
elif model_name == "time_aware":
    MODEL_PATH = "src/models/best_time_aware_std.pkl"
    window_size = 16
    if ensemble:
        MODEL_PATH = "src/models/best_time_aware_ensemble_std.pkl"



wandb.init()

# Load and preprocess data
X, y, group = prepare_data(
    DATA_PATH, target, drop_cols, cols_map, age_group=True, split=False
)

for fold_out, (train_out_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups=group)
):
    study_name_fold = f"{study_name}-fold{fold_out}"

    x_train_out, x_test = X.iloc[train_out_idx], X.iloc[test_idx]
    y_train_out, y_test = y[train_out_idx], y[test_idx]
    group_train_out, group_test_out = group[train_out_idx], group[test_idx]

    # wandb configuration
    wandb_kwargs = {
        "project": "tac_dosing_tl_lstm",
        "group": study_name,
        "name": study_name_fold,
        "resume": "allow",
    }

    train_data, test_data = impute_and_scale_data(x_train_out, y_train_out, x_test, y_test)

    if model_name != "time_aware":
        x_train, y_train, _ = create_sequences(train_data, window_size)
        x_test, y_test, group_ids = create_sequences(test_data, window_size)

        x_train = torch.FloatTensor(x_train.astype(np.float32))
        y_train = torch.FloatTensor(y_train.astype(np.float32))
        x_test = torch.FloatTensor(x_test.astype(np.float32))
        y_test = torch.FloatTensor(y_test.astype(np.float32))

        model = joblib.load(MODEL_PATH)

        model = train_tl_model(x_train, y_train, model)

        model.eval()

        with torch.no_grad():
            y_pred = model(x_test)
            y_pred_train = model(x_train)
    else:
        x_train, y_train, time_train, _ = create_sequences(train_data, window_size, time_diff=True)
        x_test, y_test, time_test, group_ids = create_sequences(test_data, window_size, time_diff=True)

        x_train = torch.FloatTensor(x_train.astype(np.float32))
        y_train = torch.FloatTensor(y_train.astype(np.float32))
        time_train = torch.FloatTensor(time_train.astype(np.float32))
        x_test = torch.FloatTensor(x_test.astype(np.float32))
        y_test = torch.FloatTensor(y_test.astype(np.float32))
        time_test = torch.FloatTensor(time_test.astype(np.float32))

        model = joblib.load(MODEL_PATH)

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

    print(f"RMSE: {rmse:.2f}")
    print(f"RMSE Train: {rmse_train:.2f}")

    # Save inference MIMIC
    inference_path = f"{SAVE_INFERENCE}_inference_test_{model_name}_v1_fold{fold_out}.csv"
    inference_test = pd.DataFrame(
        {
            "group_test": group_ids,
            "y_test": y_test,
            "y_pred": y_pred,
        }
    )
    inference_test.to_csv(inference_path, index=False)

    # Log in wandb
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

