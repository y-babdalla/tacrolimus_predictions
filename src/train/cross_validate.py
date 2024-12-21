import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.preprocess import impute_and_scale_data, create_sequences
from src.train.trainer import train_model

SEED = 42


def cross_validate(x, y, cv, group, fit_params, lstm_model):
    train_scores = []
    val_scores = []
    fit_times = []

    for i, (train_idx, val_idx) in enumerate(cv.split(x, y, group)):
        x_train_fold = x.iloc[train_idx]
        y_train_fold = y.iloc[train_idx].values.reshape(-1, 1)
        x_val_fold = x.iloc[val_idx]
        y_val_fold = y.iloc[val_idx].values.reshape(-1, 1)

        # Train and evaluate model
        rmse, rmse_train, _, _, fit_time = train_one_fold_lstm(
            x_train_fold, y_train_fold, x_val_fold, y_val_fold, fit_params, lstm_model=lstm_model
        )
        fit_times.append(fit_time)
        train_scores.append(rmse_train)
        val_scores.append(rmse)

    scores = {
        "test_score": np.array(val_scores),
        "train_score": np.array(train_scores),
        "fit_time": np.array(fit_times),
    }

    return scores


def train_one_fold_lstm(x_train, y_train, x_test, y_test, fit_params, lstm_model, return_pred=False, return_ids= False,  validation=False):
    # Preprocessing step: required for early stopping w/ parameter eval_set
    train_data, test_data = impute_and_scale_data(x_train, y_train, x_test, y_test)

    x_train, y_train, train_patient_id = create_sequences(train_data, fit_params["window_size"])
    x_test, y_test, test_patient_id = create_sequences(test_data, fit_params["window_size"])

    if lstm_model == "time_aware":
        x_train, y_train, time_train, train_patient_id = create_sequences(train_data, fit_params["window_size"], time_diff=True)
        x_test, y_test, time_test, test_patient_id = create_sequences(test_data, fit_params["window_size"], time_diff=True)

    x_train = torch.FloatTensor(x_train.astype(np.float32))
    y_train = torch.FloatTensor(y_train.astype(np.float32))
    x_test = torch.FloatTensor(x_test.astype(np.float32))
    y_test = torch.FloatTensor(y_test.astype(np.float32))

    if lstm_model == "time_aware":
        time_train = torch.FloatTensor(time_train.astype(np.float32)).unsqueeze(-1)
        time_test = torch.FloatTensor(time_test.astype(np.float32)).unsqueeze(-1)
        t1 = time.time()
        model = train_model(fit_params, x_train, y_train, time_diff=time_train, lstm_model=lstm_model, ensemble=ensemble, validation=validation, groups=train_patient_id)
        t2 = time.time()
        y_pred = model.predict(x_test, time_test)
        y_pred_train = model.predict(x_train, time_train)
    else:
         # Fit the model
        t1 = time.time()
        model = train_model(fit_params, x_train, y_train, lstm_model=lstm_model, ensemble=ensemble, validation=validation, groups=train_patient_id)
        t2 = time.time()
        # Evaluate model
        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}")
    print(f"RMSE Train: {rmse_train:.2f}")

    values = [rmse, rmse_train, mae, r2, t2 - t1]

    if return_pred:
        values.append(y_pred)
        values.append(y_test)

    if return_ids:
        values.append(test_patient_id)

    return tuple(values)


def get_params(trial):

    return {
        "window_size": trial.suggest_int("window_size", 3, 20),
        "hidden_size": trial.suggest_int("hidden_size", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
    }
