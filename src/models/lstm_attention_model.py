import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import seaborn as sns

from src.models.time_aware_lstm import plot_feature_attention, plot_time_attention_line_graph

#TODO: Combine all LSTMs
class AttentionLSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        window_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        learning_rate: float,
        dropout: float,
        attention: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.attention = attention

        if self.attention:
            self.feature_attention = nn.Linear(window_size, 1)
            self.time_attention = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        # Apply feature attention before the LSTM
        if self.attention:
            attn_scores = self.feature_attention(x.transpose(1, 2))
            feature_attn_weights = F.softmax(attn_scores, dim=1)
            x = x * feature_attn_weights.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Apply time-level attention to LSTM outputs
        if self.attention:
            time_attn_scores = self.time_attention(lstm_out).squeeze(-1)
            time_attn_weights = F.softmax(time_attn_scores, dim=1)
            lstm_out = torch.sum(time_attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        else:
            # Use the last LSTM output if no attention
            lstm_out = lstm_out[:, -1, :]

        # Apply layer normalization and dropout
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        # Output layer
        output = self.fc(lstm_out).squeeze(-1)

        if self.attention and return_attention:
            return output, feature_attn_weights, time_attn_weights
        else:
            return output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)

    def extract_attention(self, x):
        """
        Extract both feature and time attention weights.
        """
        _, feature_attn, time_attn = self.forward(x, return_attention=True)
        return feature_attn, time_attn

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class AttentionLSTMEnsemble(pl.LightningModule):
    def __init__(
            self,
            input_size: int,
            window_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            learning_rate: float,
            dropout: float,
            attention: bool = False,
            num_models: int = 5
    ):
        super().__init__()

        self.save_hyperparameters()
        self.num_models = num_models
        self.attention = attention


        self.models = nn.ModuleList([
            AttentionLSTMModel(
                input_size,
                window_size,
                hidden_size,
                num_layers,
                output_size,
                learning_rate,
                dropout,
                attention
            ) for _ in range(num_models)
        ])

    def forward(self, x, return_attention=False):
        if return_attention and self.attention:
            predictions = []
            feature_attentions = []
            time_attentions = []
            for model in self.models:
                pred, feature_attn, time_attn = model(x, return_attention=True)
                predictions.append(pred)
                feature_attentions.append(feature_attn)
                time_attentions.append(time_attn)
            return (torch.stack(predictions, dim=0),
                    torch.stack(feature_attentions, dim=0),
                    torch.stack(time_attentions, dim=0))
        else:
            predictions = [model(x) for model in self.models]
            return torch.stack(predictions, dim=0)

    def predict(self, x,  return_individual=False, return_attention=False):
        self.eval()
        with torch.no_grad():
            if return_attention and self.attention:
                predictions, feature_attentions, time_attentions = self(x,  return_attention=True)
                mean_prediction = torch.mean(predictions, dim=0).cpu().numpy()
                std = torch.std(predictions, dim=0).cpu().numpy()
                if return_individual:
                    return (predictions.cpu().numpy(), feature_attentions, time_attentions,
                            mean_prediction, std)
                else:
                    return (mean_prediction, std,
                            torch.mean(feature_attentions, dim=0),
                            torch.mean(time_attentions, dim=0))
            else:
                predictions = self(x).cpu()
                mean_prediction = torch.mean(predictions, dim=0).cpu().numpy()
                std = torch.std(predictions, dim=0).cpu().numpy()
                if return_individual:
                    return predictions.cpu().numpy(), mean_prediction, std
                else:
                    return mean_prediction, std

    def training_step(self, batch, batch_idx, optimizer_idx):
        x,  y = batch
        model = self.models[optimizer_idx]
        y_hat = model(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log(f"train_loss_model_{optimizer_idx}", loss)

        # Calculate and log average loss across all models
        all_losses = [nn.MSELoss()(m(x), y) for m in self.models]
        avg_loss = torch.mean(torch.stack(all_losses))
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        ensemble_predictions = self(x)
        losses = [nn.MSELoss()(pred, y) for pred in ensemble_predictions]
        avg_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", avg_loss)

    def configure_optimizers(self):
        # Create an optimizer for each model in the ensemble
        optimizers = [torch.optim.Adam(model.parameters(), lr=self.hparams.learning_rate) for model in self.models]
        return optimizers


if __name__ == "__main__":
    # Random input data (batch_size=32, seq_len=10, input_size=15)
    x = torch.randn(32, 10, 15)
    cols = [f"name {i}" for i in range(15)]

    # Initialize the model
    model = AttentionLSTMModel(
        input_size=x.shape[2],
        hidden_size=128,
        window_size=x.shape[1],
        num_layers=2,
        output_size=1,
        learning_rate=0.001,
        dropout=0.5,
        attention=False,
    )
    model(x)
    exit()
    # Get the model output and attention weights
    feature_attn_weights, time_attn_weights = model.extract_attention(x)

    plot_feature_attention(feature_attn_weights, feature_names=cols)
    plot_time_attention_line_graph(time_attn_weights)
