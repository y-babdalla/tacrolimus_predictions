"""Implementation of attention-based LSTM models using PyTorch Lightning."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as f

from src.models.time_aware_lstm import plot_feature_attention, plot_time_attention_line_graph


class AttentionLSTMModel(pl.LightningModule):
    """An LSTM-based model with optional feature-level and time-level attention.

    Args:
        input_size (int):
            Number of features in the input sequence.
        window_size (int):
            Number of time steps in the input sequence.
        hidden_size (int):
            Number of features in the hidden state of the LSTM.
        num_layers (int):
            Number of recurrent layers in the LSTM.
        output_size (int):
            Number of output features (e.g., 1 for regression).
        learning_rate (float):
            Learning rate for the optimiser.
        dropout (float):
            Dropout probability for the LSTM and linear layers.
        attention (bool):
            If True, uses feature-level and time-level attention.
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
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.attention = attention

        if self.attention:
            self.feature_attention = nn.Linear(window_size, 1)
            self.time_attention = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).
            return_attention (bool):
                If True, returns both feature and time attention weights.

        Returns:
            torch.Tensor or tuple of torch.Tensor:
                If `return_attention` is False, returns the model output of shape (batch_size,).
                If `return_attention` is True, returns (output, feature_attention_weights,
                time_attention_weights).
        """
        feature_attn_weights = time_attn_weights = None

        if self.attention:
            attn_scores = self.feature_attention(x.transpose(1, 2))
            feature_attn_weights = f.softmax(attn_scores, dim=1)
            x = x * feature_attn_weights.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        if self.attention:
            time_attn_scores = self.time_attention(lstm_out).squeeze(-1)
            time_attn_weights = f.softmax(time_attn_scores, dim=1)
            lstm_out = torch.sum(time_attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]

        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out).squeeze(-1)

        if self.attention and return_attention:
            return output, feature_attn_weights, time_attn_weights

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Switch the model to evaluation mode and make predictions.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor:
                Model predictions.
        """
        self.eval()
        with torch.no_grad():
            return self(x)

    def extract_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract both feature and time attention weights without returning predictions.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                (feature_attention_weights, time_attention_weights).
        """
        _, feature_attn, time_attn = self.forward(x, return_attention=True)
        return feature_attn, time_attn

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]):
                A tuple containing (x, y) for the batch.
            batch_idx (int):
                Batch index.

        Returns:
            torch.Tensor:
                The training loss.
        """
        assert isinstance(batch_idx, int)
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]):
                A tuple containing (x, y) for the batch.
            batch_idx (int):
                Batch index.
        """
        assert isinstance(batch_idx, int)
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure the optimiser.

        Returns:
            list[torch.optim.Optimizer]:
                A list of optimisers for PyTorch Lightning.
        """
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)]


class AttentionLSTMEnsemble(pl.LightningModule):
    """An ensemble of AttentionLSTMModel instances.

    Args:
        input_size (int):
            Number of features in the input sequence.
        window_size (int):
            Number of time steps in the input sequence.
        hidden_size (int):
            Number of features in the hidden state of each LSTM.
        num_layers (int):
            Number of recurrent layers in each LSTM.
        output_size (int):
            Number of output features (e.g., 1 for regression).
        learning_rate (float):
            Learning rate for the optimisers.
        dropout (float):
            Dropout probability for each model in the ensemble.
        attention (bool):
            If True, enables attention in each model.
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
                AttentionLSTMModel(
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
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through each model in the ensemble.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).
            return_attention (bool):
                If True and attention is enabled, return all models' attention weights.

        Returns:
            torch.Tensor or tuple of torch.Tensor:
                If `return_attention` is False, returns stacked predictions from each model
                of shape (num_models, batch_size).
                If `return_attention` is True and attention is enabled, returns a triple of
                stacked predictions, feature attentions, and time attentions:
                (predictions, feature_attentions, time_attentions).
        """
        if return_attention and self.attention:
            predictions = []
            feature_attentions = []
            time_attentions = []
            for model in self.models:
                preds, feature_attn, time_attn = model(x, return_attention=True)
                predictions.append(preds)
                feature_attentions.append(feature_attn)
                time_attentions.append(time_attn)
            return (
                torch.stack(predictions, dim=0),
                torch.stack(feature_attentions, dim=0),
                torch.stack(time_attentions, dim=0),
            )

        predictions = [model(x) for model in self.models]
        return torch.stack(predictions, dim=0)

    def predict(
        self, x: torch.Tensor, return_individual: bool = False, return_attention: bool = False
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Predict using the ensemble, optionally returning individual and/or attention weights.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, seq_len, input_size).
            return_individual (bool):
                If True, returns individual model predictions in addition to the mean and standard
                deviation.
            return_attention (bool):
                If True and attention is enabled, returns the feature and time attention weights.

        Returns:
            Depending on the arguments, returns either:
            - (mean_prediction, std) if only the ensemble prediction is needed.
            - Individual predictions plus the mean and std if return_individual=True.
            - With return_attention, returns either a tuple with stacked predictions/attentions
              or a combination of mean/std and mean attention weights.
        """
        self.eval()
        with torch.no_grad():
            if return_attention and self.attention:
                predictions, feature_attentions, time_attentions = self(x, return_attention=True)
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

            predictions = self(x).cpu()
            mean_prediction = torch.mean(predictions, dim=0).cpu().numpy()
            std_prediction = torch.std(predictions, dim=0).cpu().numpy()
            if return_individual:
                return predictions.cpu().numpy(), mean_prediction, std_prediction
            return mean_prediction, std_prediction

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on one of the ensemble models.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]):
                A tuple containing (x, y) for the batch.
            batch_idx (int):
                Batch index.
            optimizer_idx (int):
                Index of the model to train in this step.

        Returns:
            torch.Tensor:
                The training loss for the chosen model.
        """
        assert isinstance(batch_idx, int)
        x, y = batch
        model = self.models[optimizer_idx]
        y_hat = model(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log(f"train_loss_model_{optimizer_idx}", loss)

        # Compute average loss across all models to log
        all_losses = [nn.MSELoss()(m(x), y) for m in self.models]
        avg_loss = torch.mean(torch.stack(all_losses))
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a validation step by evaluating all models in the ensemble.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]):
                A tuple containing (x, y) for the batch.
            batch_idx (int):
                Batch index.
        """
        assert isinstance(batch_idx, int)
        x, y = batch
        ensemble_predictions = self(x)
        losses = [nn.MSELoss()(pred, y) for pred in ensemble_predictions]
        avg_loss = torch.mean(torch.stack(losses))
        self.log("val_loss", avg_loss)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create an optimiser for each model in the ensemble.

        Returns:
            list[torch.optim.Optimizer]:
                One optimiser per ensemble model.
        """
        return [
            torch.optim.Adam(model.parameters(), lr=self.hparams.learning_rate)
            for model in self.models
        ]


if __name__ == "__main__":
    # Random input data: batch_size=32, seq_len=10, input_size=15
    x_data = torch.randn(32, 10, 15)
    col_names = [f"name_{i}" for i in range(15)]

    # Instantiate a single AttentionLSTMModel
    single_model = AttentionLSTMModel(
        input_size=x_data.shape[2],
        hidden_size=128,
        window_size=x_data.shape[1],
        num_layers=2,
        output_size=1,
        learning_rate=0.001,
        dropout=0.5,
        attention=False,
    )
    single_model(x_data)

    # Extract attention weights from the model (if attention is True)
    feature_weights, time_weights = single_model.extract_attention(x_data)
    plot_feature_attention(feature_weights, feature_names=col_names)
    plot_time_attention_line_graph(time_weights)
