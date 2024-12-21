import joblib
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import numpy as np


from src.data.preprocess import (
    prepare_data,
    target,
    drop_cols,
    cols_map, impute_and_scale_data, create_sequences,
)
from src.models.time_aware_lstm import plot_feature_attention, plot_time_attention_line_graph

model_name = "attention" #time_aware
MIMIC_DATA_PATH = 'data/mimic/followup_final_v4.csv'
SPANISH_DATA_PATH = ...
SAVE_INFERENCE= 'models/inference_pooled/'

# Prepare data
X_train, y_train, train_group = prepare_data(MIMIC_DATA_PATH,target, drop_cols,cols_map,age_group=True, split=False)
X_test, y_test, group = prepare_data(SPANISH_DATA_PATH, target, drop_cols,cols_map,age_group=True, split=False)


mimic_data, spain_data = impute_and_scale_data(X_train, y_train, X_test, y_test)

if model_name == "attention":
    x_spain, y_spain, patient_ids = create_sequences(spain_data, 20)
    x_spain = torch.FloatTensor(x_spain.astype(np.float32))
    y_spain = torch.FloatTensor(y_spain.astype(np.float32))
elif model_name == "time_aware":
    x_spain, y_spain, time, patient_ids = create_sequences(spain_data, 16, time_diff=True)
    x_spain = torch.FloatTensor(x_spain.astype(np.float32))
    y_spain = torch.FloatTensor(y_spain.astype(np.float32))
    time = torch.FloatTensor(time.astype(np.float32))


model = joblib.load(f"src/models/best_{model_name}_lstm.pkl")

if model_name == "attention":
    y_pred_spain =model.predict(x_spain)
elif model_name == "time_aware":
    y_pred_spain =model.predict(x_spain, time)


rmse_spanish = mean_squared_error(y_spain, y_pred_spain, squared=False)
mae_spanish = mean_absolute_error(y_spain, y_pred_spain)
r2_spanish = r2_score(y_spain, y_pred_spain)

# Save inference
inference_spanish = pd.DataFrame({'y_test':y_spain, 'y_pred':y_pred_spain, 'group':patient_ids})
# inference_spanish.to_csv(f'{SAVE_INFERENCE}inference_bestlstm _spanish.csv', index=False)

print(f'RMSE Spanish data: {rmse_spanish:.3f}')
print(f'MAE Spanish data: {mae_spanish:.3f}')
print(f'R2 Spanish data: {r2_spanish:.3f}')

if model_name == "attention":
    train_data_columns = spain_data.drop(["remainder__subject_id", "remainder__level_time", "remainder__dose_time"],
                             axis=1).columns
elif model_name == "time_aware":
    train_data_columns = spain_data.drop(["remainder__subject_id", "remainder__level_time", "remainder__dose_time", "time__previous_level_timediff"],
                             axis=1).columns

if model_name == "attention":
    feature_attn_weights, time_attn_weights = model.extract_attention(X_train)
elif model_name == "time_aware":
    feature_attn_weights, time_attn_weights = model.extract_time_aware(X_train, time)

plot_feature_attention(feature_attn_weights, feature_names=train_data_columns, save=f"src/attention_plots/{model_name}_feature_attn_weights.png")
plot_time_attention_line_graph(time_attn_weights, save=f"src/attention_plots/{model_name}_time_attn_weights.png")