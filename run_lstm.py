import logging

# from pprint import pprint
import shap

import joblib
import optuna
import pandas as pd
import wandb
import yaml
import numpy as np
import matplotlib.pyplot as plt

from optuna.samplers import TPESampler
from optuna_integration import WeightsAndBiasesCallback
from sklearn.model_selection import GroupKFold

# from tempor import plugin_loader

from src.data.preprocess import (
    # split_data_xgb,
    # create_one_off_prediction_dataset,
    # create_sequences_for_mlforecast,
    prepare_data,
    target,
    drop_cols,
    cols_map,
)

# from src.models.xgb_model import train_xgb_model
from src.train.cross_validate import get_params, cross_validate, train_one_fold_lstm

# from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

CV_SPLITS = 5
BO_TRIALS = 100
POOLING = False
SAVE_DIR = "models/"
SAVE_INFERENCE = 'models/inference/'
SAVE_INFERENCE_POOLED = 'models/inference_pooled/'
SPANISH_DATA_PATH = "/mnt/experiments/brais.mcastro/mtoilt_london/data/exported_2024-09-09/coruña_to_mimic.csv"
MIMIC_DATA_PATH = "data/mimic/followup_final_v4.csv"
lstm_model_name = "attention"

wandb.login(key='c5e9e4bf0be84cbe7aa8a1f813d65c30bd98a930')

def main(pooling: bool = False):
    inner_cv = GroupKFold(n_splits=CV_SPLITS)
    outer_cv = GroupKFold(n_splits=CV_SPLITS)

    if lstm_model_name == "time_aware":
        study_name = f"TimeLSTM"
        model_name = "time_aware"
    elif lstm_model_name == "attention":
        study_name = f"AttentionLSTM"
        model_name = "attention"

    print(f"Running study {study_name}")

    # Load and preprocess data
    X, y, group = prepare_data(
            MIMIC_DATA_PATH, target, drop_cols, cols_map, age_group=True, split=False
    )
    if pooling:
        X_spanish, y_spanish, group_spanish = prepare_data(
            SPANISH_DATA_PATH, target, drop_cols, cols_map, age_group=True, split=False
        )
        spanish_splits_outer = list(
            outer_cv.split(X_spanish, y_spanish, groups=group_spanish)
        )
    for fold_out, (train_out_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups=group)
    ):

        study_name_fold = f"{study_name}-fold{fold_out}"

        x_train_out, x_test = X.iloc[train_out_idx], X.iloc[test_idx]
        y_train_out, y_test = y[train_out_idx], y[test_idx]
        group_train_out, group_test_out = group[train_out_idx], group[test_idx]

        if pooling:
            X_train_out_mimic, X_test_mimic = X.iloc[train_out_idx], X.iloc[test_idx]
            y_train_out_mimic, y_test_mimic = y[train_out_idx], y[test_idx]
            group_train_out_mimic, group_test_mimic = (
                group[train_out_idx],
                group[test_idx],
            )
            # Get spanish split
            train_out_idx_spanish, test_idx_spanish = spanish_splits_outer[fold_out]
            X_train_out_spanish, X_test_spanish = (
                X_spanish.iloc[train_out_idx_spanish],
                X_spanish.iloc[test_idx_spanish],
            )
            y_train_out_spanish, y_test_spanish = (
                y_spanish[train_out_idx_spanish],
                y_spanish[test_idx_spanish],
            )
            group_train_out_spanish, group_test_spanish = (
                group_spanish[train_out_idx_spanish],
                group_spanish[test_idx_spanish],
            )
            # Append training MIMIC and spanish data
            x_train_out = pd.concat([X_train_out_mimic, X_train_out_spanish], axis=0)
            y_train_out = pd.concat([y_train_out_mimic, y_train_out_spanish], axis=0)
            # Check groups dont overlap
            assert (
                len(set(group_train_out_mimic).intersection(set(group_train_out_spanish)))
                == 0
            )
            group_train_out = pd.concat(
                [group_train_out_mimic, group_train_out_spanish], axis=0
            )
        # wandb configuration
        wandb_kwargs = {
            "project": f"nested_cv_dose_pred_{study_name}",
            "group": study_name,
            "name": study_name_fold,
            "resume": "allow",
        }
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=False)

        @wandbc.track_in_wandb()
        def objective(trial):
            # Set parameters to pipeline
            params = get_params(trial)

            try:
                # Train and evaluate model using CV
                scores = cross_validate(
                    x_train_out,
                    y_train_out,
                    cv=inner_cv,
                    group=group_train_out,
                    fit_params=params,
                    lstm_model=model_name)
            except RuntimeError:
                scores = {
                "test_score": np.array([np.inf, np.inf]),
                "train_score": np.array([np.inf, np.inf]),
                "fit_time": np.array([np.inf, np.inf]),
            }


            # Log CV scores
            trial.set_user_attr("mean_fit_time", scores["fit_time"].mean())
            trial.set_user_attr("mean_val_score", float(abs(scores["test_score"].mean())))
            trial.set_user_attr("std_val_score", float(scores["test_score"].std()))
            trial.set_user_attr("mean_train_score", float(abs(scores["train_score"].mean())))
            trial.set_user_attr("std_train_score", float(scores["train_score"].std()))

            return scores["test_score"].mean()


        study = optuna.create_study(
            study_name=study_name_fold,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            storage=f'sqlite:///{study_name_fold}.db',
            load_if_exists=True)


        study.optimize(objective, n_trials=BO_TRIALS, n_jobs=1)

        # Train model with the best hyperparameters from inner CV
        best_params = study.best_params
        model_params = get_params(study.best_trial)

        if pooling:
            (
                rmse_mimic,
                rmse_train_mimic,
                mae_mimic,
                r2_mimic,
                fit_time_pooled,
                y_pred_mimic,
                y_test_mimic,
                mimic_ids
            ) = train_one_fold_lstm(
                x_train_out,
                y_train_out,
                X_test_mimic,
                y_test_mimic,
                model_params,
                return_pred=True,
                return_ids=True,
                lstm_model=model_name
            )
            rmse_spain, rmse_train_spain, mae_spain, r2_spain, fit_time_spain, y_pred_spanish, y_test_spanish, spanish_ids = (
                train_one_fold_lstm(
                    x_train_out,
                    y_train_out,
                    X_test_spanish,
                    y_test_spanish,
                    model_params,
                    return_pred=True,
                    return_ids=True,
                    lstm_model=model_name
                )
            )

            # Save inference MIMIC
            inference_test_path = f"{SAVE_INFERENCE_POOLED}mimic_inference_test_{model_name}_v1_fold{fold_out}.csv"
            inference_test_mimic = pd.DataFrame(
                {
                    "group_test": mimic_ids,
                    "y_test": y_test_mimic,
                    "y_pred": y_pred_mimic,
                }
            )
            inference_test_mimic.to_csv(inference_test_path, index=False)

            # Save inference Spanish
            inference_test_path = f"{SAVE_INFERENCE_POOLED}spanish_inference_test_{model_name}_v1_fold{fold_out}.csv"
            inference_test_spanish = pd.DataFrame(
                {
                    "group_test": spanish_ids,
                    "y_test": y_test_spanish,
                    "y_pred": y_pred_spanish,
                }
            )
            inference_test_spanish.to_csv(inference_test_path, index=False)

            wandb.log(
                {
                    "rmse_mimic": rmse_mimic,
                    "train_rmse_mimic": rmse_train_mimic,
                    "mae_mimic": mae_mimic,
                    "r2_mimic": r2_mimic,
                    "rmse_spain": rmse_spain,
                    "train_rmse_spain": rmse_train_spain,
                    "mae_spain": mae_spain,
                    "r2_spain": r2_spain,
                    "best_params": best_params,
                    "model_name": model_name,
                    "preprocessor_version": "v1",
                    "fold_out": fold_out,
                    "cv_scores": study.best_trial.user_attrs,

                }
            )
        else:
            rmse, rmse_train, mae, r2, fit_time, y_pred_mimic, y_test, group_ids = train_one_fold_lstm(
                x_train_out, y_train_out, x_test, y_test, model_params, return_pred=True, return_ids=True, lstm_model=model_name
            )
            # Save inference MIMIC
            inference_path = f"{SAVE_INFERENCE}_inference_test_{model_name}_v1_fold{fold_out}.csv"
            inference_test = pd.DataFrame(
                {
                    "group_test": group_ids,
                    "y_test": y_test,
                    "y_pred": y_pred_mimic,
                }
            )
            inference_test.to_csv(inference_path, index=False)

            # Log in wandb
            wandb.log(
                {
                    "rmse": rmse,
                    "train_rmse": rmse_train,
                    "mae": mae,
                    "r2": r2,
                    "best_params": best_params,
                    "model_name": model_name,
                    "preprocessor_version": "v1",
                    "fold_out": fold_out,
                    "cv_scores": study.best_trial.user_attrs,

                }
            )

        wandb.finish()

        # Save study
        joblib.dump(study, f"{SAVE_DIR}optuna_{study_name_fold}.pkl")


if __name__ == "__main__":
    main(pooling=POOLING)
