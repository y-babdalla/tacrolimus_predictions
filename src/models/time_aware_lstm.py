import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import seaborn as sns
#TODO: Combine all LSTMs
class TimeAwareLSTM(pl.LightningModule):
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
    ):
        super().__init__()

        self.save_hyperparameters()
        self.input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.attention = attention

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        if self.attention:
            self.feature_attention = nn.Linear(window_size, 1)
            self.time_attention = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(dropout)
        self.time_decay_layer = nn.Linear(1, 1)


    def forward(self, x, time_deltas=None, return_attention=False):
        # Apply feature attention before the LSTM
        if self.attention:
            attn_scores = self.feature_attention(x.transpose(1, 2))
            feature_attn_weights = F.softmax(attn_scores, dim=1)
            x = x * feature_attn_weights.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        if self.attention and time_deltas is not None:
            # Reshape time_deltas to (batch_size * seq_len, 1) for the linear layer
            time_deltas = time_deltas.view(-1, 1)  # Flatten to (batch_size * seq_len, 1)

            # Compute time decay factors
            time_decay_factors = torch.exp(-self.time_decay_layer(time_deltas)).view(x.size(0), x.size(
                1))  # Shape: (batch_size, seq_len)

            # Compute time attention scores
            time_attn_scores = self.time_attention(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_len)

            # Ensure that time_decay_factors has the shape (batch_size, seq_len)
            if time_decay_factors.dim() == 1:
                time_decay_factors = time_decay_factors.unsqueeze(1)  # Shape: (batch_size, 1)

            # Multiply the attention scores by the decay factors
            time_attn_scores = time_attn_scores * time_decay_factors  # Element-wise multiplication

            # Apply softmax to get attention weights
            time_attn_weights = F.softmax(time_attn_scores, dim=1)

            # Weighted sum of LSTM outputs with decayed attention
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

    def predict(self, x, time_deltas):
        self.eval()
        with torch.no_grad():
            return self(x, time_deltas)

    def extract_attention(self, x, time_deltas):
        """
        Extract both feature and time attention weights.
        """
        _, feature_attn, time_attn = self.forward(x, time_deltas, return_attention=True)
        return feature_attn, time_attn

    def training_step(self, batch, batch_idx):
        x, time_deltas, y = batch
        y_hat = self(x, time_deltas)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, time_deltas, y = batch
        y_hat = self(x, time_deltas)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class TimeAwareLSTMEnsemble(pl.LightningModule):
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
            TimeAwareLSTM(
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

    def forward(self, x, time_deltas=None, return_attention=False):
        if return_attention and self.attention:
            predictions = []
            feature_attentions = []
            time_attentions = []
            for model in self.models:
                pred, feature_attn, time_attn = model(x, time_deltas, return_attention=True)
                predictions.append(pred)
                feature_attentions.append(feature_attn)
                time_attentions.append(time_attn)
            return (torch.stack(predictions, dim=0),
                    torch.stack(feature_attentions, dim=0),
                    torch.stack(time_attentions, dim=0))
        else:
            predictions = [model(x, time_deltas) for model in self.models]
            return torch.stack(predictions, dim=0)

    def predict(self, x, time_deltas, return_individual=False, return_attention=False):
        self.eval()
        with torch.no_grad():
            if return_attention and self.attention:
                predictions, feature_attentions, time_attentions = self(x, time_deltas, return_attention=True)
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
                predictions = self(x, time_deltas)
                mean_prediction = torch.mean(predictions, dim=0).cpu().numpy()
                std = torch.std(predictions, dim=0).cpu().numpy()
                if return_individual:
                    return predictions.cpu().numpy(), mean_prediction, std
                else:
                    return mean_prediction, std

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, time_deltas, y = batch
        model = self.models[optimizer_idx]
        y_hat = model(x, time_deltas)
        loss = nn.MSELoss()(y_hat, y)
        self.log(f"train_loss_model_{optimizer_idx}", loss)

        # Calculate and log average loss across all models
        all_losses = [nn.MSELoss()(m(x, time_deltas), y) for m in self.models]
        avg_loss = torch.mean(torch.stack(all_losses))
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    def validation_step(self, batch, batch_idx):
        x, time_deltas, y = batch
        ensemble_predictions = self(x, time_deltas)
        losses = [nn.MSELoss()(pred, y) for pred in ensemble_predictions]
        avg_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", avg_loss)

    def configure_optimizers(self):
        # Create an optimizer for each model in the ensemble
        optimizers = [torch.optim.Adam(model.parameters(), lr=self.hparams.learning_rate) for model in self.models]
        return optimizers

def plot_feature_attention(feature_attn, feature_names=None, id=None, save=None):
    """
    Plots feature attention weights as a bar plot.

    Parameters:
    - feature_attn (Tensor): Shape [batch_size, features, 1]
    - feature_names (list, optional): Names of the features for x-axis labels.
    - id (int, optional): Specific batch index to visualize. If None, averages across batches.
    """

    feature_attn = feature_attn.detach().numpy()

    # Calculate mean attention weights
    if id is None:
        feature_attn_mean = feature_attn.mean(axis=0)  # Shape: [features, 1]
        feature_attn_std = feature_attn.std(axis=0)  # Shape: [features, 1]
    else:
        feature_attn_mean = feature_attn[id]  # Shape: [features, 1]
        feature_attn_std = np.zeros_like(feature_attn_mean)  # No std for single sample

    # Squeeze to 1D array for easier plotting
    feature_attn_mean = feature_attn_mean.squeeze()
    feature_attn_std = feature_attn_std.squeeze()

    # Create a bar plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(feature_attn_mean)), feature_attn_mean, yerr=feature_attn_std, capsize=5)

    # Add feature names if provided
    if feature_names is not None:
        plt.xticks(range(len(feature_names)), feature_names, rotation='vertical')
    else:
        plt.xticks(range(len(feature_attn_mean)), range(len(feature_attn_mean)))

    plt.tight_layout()

    plt.xlabel('Features')
    plt.ylabel('Attention Weight')

    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()


def plot_time_attention_line_graph(time_attn_weights, seq_idx=None, save=None):
    """
    Plot a line graph with error bars for time-level attention weights.

    Args:
        time_attn_weights (torch.Tensor): Attention weights for time steps. Shape: (batch_size, seq_len)
        seq_idx (int or None): Index of the sequence in the batch to visualize. If None, average across all sequences.
    """
    if seq_idx is None:
        # Compute average and std deviation across all sequences in the batch
        attn_mean = time_attn_weights.mean(dim=0).detach().cpu().numpy()  # Shape: (seq_len,)
        attn_std = time_attn_weights.std(dim=0).detach().cpu().numpy()  # Shape: (seq_len,)
    else:
        # Use the specific sequence, no std needed
        attn_mean = time_attn_weights[seq_idx].detach().cpu().numpy()  # Shape: (seq_len,)
        attn_std = np.zeros_like(attn_mean)  # No standard deviation for a single sequence

    # Create a line graph with error bars
    plt.figure(figsize=(10, 4))
    plt.errorbar(
        range(attn_mean.shape[0]),  # Time steps on x-axis
        attn_mean,  # Mean attention weights
        yerr=attn_std,  # Error bars as standard deviation
        capsize=5,  # Size of the error bar caps
        marker='o',  # Markers for points
        linestyle='-',  # Line style
        label="Attention Weights"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300)

    plt.show()

if __name__ == "__main__":
    # Random input data (batch_size=32, seq_len=10, input_size=15)
    x = torch.randn(32, 10, 16)
    time = torch.randn(32, 10, 1)

    cols = [f"name {i}" for i in range(16)]

    # Initialize the model
    model = TimeAwareLSTMEnsemble(
        input_size=16,
        window_size=10,
        hidden_size=128,
        num_layers=2,
        output_size=1,
        learning_rate=0.001,
        dropout=0.5,
        attention=True,
    )

    # Get the model output and attention weights
    feature_attn_weights, time_attn_weights = model.extract_attention(x, time)

    plot_feature_attention(feature_attn_weights, feature_names=cols)
    plot_time_attention_line_graph(time_attn_weights)
