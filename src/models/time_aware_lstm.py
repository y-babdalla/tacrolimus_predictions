"""Implementation of time-aware LSTM models with optional attention."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt


class TimeAwareLSTM(pl.LightningModule):
    """An LSTM-based model incorporating feature-level attention and time decay attention.

    Args:
        input_size (int):
            Number of features in the input sequence.
        window_size (int):
            Number of time steps in the input sequence.
        hidden_size (int):
            Number of hidden units in the LSTM layers.
        num_layers (int):
            Number of LSTM layers.
        output_size (int):
            Number of output features, e.g., 1 for regression.
        learning_rate (float):
            Learning rate for the optimiser.
        dropout (float):
            Dropout probability in LSTM and subsequent layers.
        attention (bool):
            If True, applies feature-level attention and time-level attention with decay.
    """

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
    ) -> None:
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
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        if self.attention:
            self.feature_attention = nn.Linear(window_size, 1)
            self.time_attention = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.time_decay_layer = nn.Linear(1, 1)  # Used to model exponential time decay

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the LSTM with optional feature and time attention.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).
            time_deltas (torch.Tensor | None):
                Time deltas tensor of shape (batch_size, seq_len, 1), used for time decay.
            return_attention (bool):
                If True, returns feature and time attention weights.

        Returns:
            torch.Tensor or tuple:
                If `return_attention` is False, returns the model output of shape (batch_size,).
                If `return_attention` is True (and attention is enabled), returns:
                (output, feature_attention_weights, time_attention_weights).
        """
        feature_attn_weights = time_attn_weights = None

        if self.attention:
            attn_scores = self.feature_attention(
                x.transpose(1, 2)
            )  # (batch_size, features=window_size, 1)
            feature_attn_weights = f.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
            x = x * feature_attn_weights.transpose(1, 2)  # (batch_size, seq_len, input_size)

        lstm_out, _ = self.lstm(x)

        if self.attention and time_deltas is not None:
            time_deltas = time_deltas.view(-1, 1)
            time_decay_factors = torch.exp(-self.time_decay_layer(time_deltas)).view(
                x.size(0), x.size(1)
            )  # (batch_size, seq_len)

            time_attn_scores = self.time_attention(lstm_out).squeeze(-1)  # (batch_size, seq_len)
            time_attn_scores = time_attn_scores * time_decay_factors

            time_attn_weights = f.softmax(time_attn_scores, dim=1)  # (batch_size, seq_len)
            lstm_out = torch.sum(
                time_attn_weights.unsqueeze(-1) * lstm_out, dim=1
            )  # (batch_size, hidden_size)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)

        output = self.fc(lstm_out).squeeze(-1)

        if self.attention and return_attention:
            return output, feature_attn_weights, time_attn_weights
        return output

    def predict(self, x: torch.Tensor, time_deltas: torch.Tensor) -> torch.Tensor:
        """Inference method.

        Args:
            x (torch.Tensor):
                Input features of shape (batch_size, seq_len, input_size).
            time_deltas (torch.Tensor):
                Time deltas of shape (batch_size, seq_len, 1).

        Returns:
            torch.Tensor:
                Predictions of shape (batch_size,).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, time_deltas)

    def extract_attention(
        self, x: torch.Tensor, time_deltas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract feature and time attention weights.

        Args:
            x (torch.Tensor):
                Input features.
            time_deltas (torch.Tensor):
                Time deltas.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                (feature_attn_weights, time_attn_weights).
        """
        _, feature_attn, time_attn = self.forward(x, time_deltas, return_attention=True)
        return feature_attn, time_attn

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                (x, time_deltas, y).
            batch_idx (int):
                Batch index.

        Returns:
            torch.Tensor:
                Training loss.
        """
        assert isinstance(batch_idx, int)
        x_batch, time_batch, y_batch = batch
        y_hat = self.forward(x_batch, time_batch)
        loss = nn.MSELoss()(y_hat, y_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                (x, time_deltas, y).
            batch_idx (int):
                Batch index.
        """
        assert isinstance(batch_idx, int)
        x_batch, time_batch, y_batch = batch
        y_hat = self.forward(x_batch, time_batch)
        loss = nn.MSELoss()(y_hat, y_batch)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimisers for training.

        Returns:
            list[torch.optim.Optimizer]:
                A list of PyTorch optimisers.
        """
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)]


class TimeAwareLSTMEnsemble(pl.LightningModule):
    """An ensemble of TimeAwareLSTM models.

    Each model is trained with its own optimiser, enabling multiple models to train
    in parallel but share the same initial configuration.

    Args:
        input_size (int):
            Number of features in the input sequence.
        window_size (int):
            Number of time steps in the input sequence.
        hidden_size (int):
            Number of hidden units in each LSTM layer of every model.
        num_layers (int):
            Number of LSTM layers in each model.
        output_size (int):
            Number of output features.
        learning_rate (float):
            Learning rate for each model's optimiser.
        dropout (float):
            Dropout probability for each model.
        attention (bool):
            If True, enables the time and feature attention in each model.
        num_models (int):
            Number of models in the ensemble.
    """

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
        num_models: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_models = num_models
        self.attention = attention

        self.models = nn.ModuleList(
            [
                TimeAwareLSTM(
                    input_size=input_size,
                    window_size=window_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    learning_rate=learning_rate,
                    dropout=dropout,
                    attention=attention,
                )
                for _ in range(num_models)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the ensemble.

        Args:
            x (torch.Tensor):
                Input features.
            time_deltas (torch.Tensor | None):
                Time delta values.
            return_attention (bool):
                If True, returns attention weights from each model.

        Returns:
            torch.Tensor or tuple:
                If `return_attention` is False, returns stacked predictions from all models.
                If `return_attention` is True (and attention is enabled), returns
                (predictions, feature_attentions, time_attentions).
        """
        if return_attention and self.attention:
            predictions = []
            feature_attentions = []
            time_attentions = []
            for model in self.models:
                preds, f_attn, t_attn = model(x, time_deltas, return_attention=True)
                predictions.append(preds)
                feature_attentions.append(f_attn)
                time_attentions.append(t_attn)
            return (
                torch.stack(predictions, dim=0),
                torch.stack(feature_attentions, dim=0),
                torch.stack(time_attentions, dim=0),
            )

        predictions = [model(x, time_deltas) for model in self.models]
        return torch.stack(predictions, dim=0)

    def predict(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor,
        return_individual: bool = False,
        return_attention: bool = False,
    ) -> (
        np.ndarray
        | tuple[np.ndarray, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """Make predictions with the ensemble.

        Args:
            x (torch.Tensor):
                Input features.
            time_deltas (torch.Tensor):
                Time delta values.
            return_individual (bool):
                If True, return individual model predictions.
            return_attention (bool):
                If True and attention is enabled, return attention weights.

        Returns:
            Various tuple structures depending on arguments. For example:
            - mean_prediction, std if return_individual=False and return_attention=False
            - (predictions, feature_attentions, time_attentions, mean_prediction, std) if
              return_individual=True and return_attention=True, etc.
        """
        self.eval()
        with torch.no_grad():
            if return_attention and self.attention:
                predictions, feature_attentions, time_attentions = self(
                    x, time_deltas, return_attention=True
                )
                mean_prediction = torch.mean(predictions, dim=0).cpu().numpy()
                std_prediction = torch.std(predictions, dim=0).cpu().numpy()
                if return_individual:
                    return (
                        predictions.cpu().numpy(),
                        feature_attentions,
                        time_attentions,
                        mean_prediction,
                        std_prediction,
                    )
                return (
                    mean_prediction,
                    std_prediction,
                    torch.mean(feature_attentions, dim=0),
                    torch.mean(time_attentions, dim=0),
                )

            predictions = self(x, time_deltas).cpu()
            mean_prediction = torch.mean(predictions, dim=0).cpu().numpy()
            std_prediction = torch.std(predictions, dim=0).cpu().numpy()

            if return_individual:
                return predictions.cpu().numpy(), mean_prediction, std_prediction
            return mean_prediction, std_prediction

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        """Single training step for one model in the ensemble.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                (x, time_deltas, y).
            batch_idx (int):
                Batch index.
            optimizer_idx (int):
                Index specifying which model to update.

        Returns:
            torch.Tensor:
                Training loss for the chosen model.
        """
        assert isinstance(batch_idx, int)
        x_batch, time_batch, y_batch = batch
        model = self.models[optimizer_idx]
        y_hat = model(x_batch, time_batch)
        loss = nn.MSELoss()(y_hat, y_batch)
        self.log(f"train_loss_model_{optimizer_idx}", loss)

        # Log the average loss across all models for monitoring
        all_losses = [nn.MSELoss()(m(x_batch, time_batch), y_batch) for m in self.models]
        avg_loss = torch.mean(torch.stack(all_losses))
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validate all models in the ensemble on a batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                (x, time_deltas, y).
            batch_idx (int):
                Batch index.
        """
        assert isinstance(batch_idx, int)
        x_batch, time_batch, y_batch = batch
        ensemble_predictions = self(x_batch, time_batch)
        losses = [nn.MSELoss()(pred, y_batch) for pred in ensemble_predictions]
        avg_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", avg_loss)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create an optimiser for each model in the ensemble.

        Returns:
            list[torch.optim.Optimizer]:
                A list of PyTorch optimisers, one per model.
        """
        return [
            torch.optim.Adam(model.parameters(), lr=self.hparams.learning_rate)
            for model in self.models
        ]


def plot_feature_attention(
    feature_attn: torch.Tensor,
    feature_names: list[str] | None = None,
    id_val: int | None = None,
    save: str | None = None,
) -> None:
    """Plot feature attention weights as a bar chart.

    If 'id_val' is None, computes the average and std across the entire batch.
    Otherwise, plots the attention for the specified sample in the batch.

    Args:
        feature_attn (torch.Tensor):
            Attention weights of shape (batch_size, features, 1).
        feature_names (list[str] | None):
            Optional list of feature names for the x-axis.
        id_val (int | None):
            Index of a specific sample in the batch to visualise.
        save (str | None):
            If provided, saves the figure to this filepath before showing.
    """
    feature_attn_np = feature_attn.detach().cpu().numpy()

    if id_val is None:
        feature_attn_mean = feature_attn_np.mean(axis=0)  # (features, 1)
        feature_attn_std = feature_attn_np.std(axis=0)  # (features, 1)
    else:
        feature_attn_mean = feature_attn_np[id_val]
        feature_attn_std = np.zeros_like(feature_attn_mean)

    # Squeeze to 1D arrays
    feature_attn_mean = feature_attn_mean.squeeze()
    feature_attn_std = feature_attn_std.squeeze()

    plt.figure(figsize=(12, 8))
    plt.bar(
        x=range(len(feature_attn_mean)), height=feature_attn_mean, yerr=feature_attn_std, capsize=5
    )

    if feature_names is not None:
        plt.xticks(range(len(feature_names)), feature_names, rotation="vertical")
    else:
        plt.xticks(range(len(feature_attn_mean)), range(len(feature_attn_mean)))

    plt.xlabel("Features")
    plt.ylabel("Attention Weight")
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()


def plot_time_attention_line_graph(
    time_attn_weights: torch.Tensor, seq_idx: int | None = None, save: str | None = None
) -> None:
    """Plot a line graph of time-level attention weights with optional error bars.

    If 'seq_idx' is None, computes the mean and std across all sequences in the batch.
    Otherwise, plots only the specified sequence.

    Args:
        time_attn_weights (torch.Tensor):
            Attention weights of shape (batch_size, seq_len).
        seq_idx (int | None):
            Index of a specific sequence in the batch to visualise.
        save (str | None):
            If provided, saves the figure to this filepath before showing.
    """
    if seq_idx is None:
        attn_mean = time_attn_weights.mean(dim=0).detach().cpu().numpy()
        attn_std = time_attn_weights.std(dim=0).detach().cpu().numpy()
    else:
        attn_mean = time_attn_weights[seq_idx].detach().cpu().numpy()
        attn_std = np.zeros_like(attn_mean)

    plt.figure(figsize=(10, 4))
    plt.errorbar(
        range(attn_mean.shape[0]),
        attn_mean,
        yerr=attn_std,
        capsize=5,
        marker="o",
        linestyle="-",
        label="Attention Weights",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()


if __name__ == "__main__":
    x_data = torch.randn(32, 10, 16)  # (batch_size=32, seq_len=10, input_size=16)
    time_data = torch.randn(32, 10, 1)  # (batch_size=32, seq_len=10, 1 for time deltas)
    col_names = [f"name_{i}" for i in range(16)]

    ensemble_model = TimeAwareLSTMEnsemble(
        input_size=16,
        window_size=10,
        hidden_size=128,
        num_layers=2,
        output_size=1,
        learning_rate=0.001,
        dropout=0.5,
        attention=True,
    )

    # Extract attention from the ensemble (averaged across models)
    _feature_attn_weights, _time_attn_weights = ensemble_model.extract_attention(x_data, time_data)

    # Plot feature and time attention
    plot_feature_attention(_feature_attn_weights, feature_names=col_names)
    plot_time_attention_line_graph(_time_attn_weights)
