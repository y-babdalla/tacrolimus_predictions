"""Module for LSTM transfer learning with partial parameter freezing."""

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LSTMTransferLearning(pl.LightningModule):
    """Freeze LSTM layers in a model for transfer learning, leaving the final layer trainable.

    Args:
        model (nn.Module):
            A pre-trained PyTorch module (e.g., an LSTM-based model) whose LSTM layers are to
            be frozen.
        time (bool):
            If True, expects the forward method to handle an additional `time_delta` argument.
    """

    def __init__(self, model: nn.Module, time: bool = False) -> None:
        """Initialise the transfer learning module and freeze layers as necessary."""
        super().__init__()
        self.model = model
        self.use_time = time

        for param in model.lstm.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, time_delta: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass, optionally passing time delta to the underlying model.

        Args:
            x (torch.Tensor):
                Input features of shape (batch_size, seq_len, input_size).
            time_delta (torch.Tensor | None):
                Time delta values of shape (batch_size, seq_len, 1), if self.use_time is True.

        Returns:
            torch.Tensor:
                Model output of shape (batch_size,).
        """
        if self.use_time and time_delta is not None:
            return self.model(x, time_delta)
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Single training step.

        Args:
            batch (tuple[torch.Tensor, ...]):
                If self.use_time is True, the batch contains (x, time_deltas, y).
                Otherwise, the batch contains (x, y).
            batch_idx (int):
                Batch index (unused).

        Returns:
            torch.Tensor:
                The training loss.
        """
        assert isinstance(batch_idx, int)
        if self.use_time:
            x_batch, time_deltas, y_batch = batch
            y_hat = self(x_batch, time_deltas)
        else:
            x_batch, y_batch = batch
            y_hat = self(x_batch)

        loss = nn.MSELoss()(y_hat, y_batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Single validation step.

        Args:
            batch (tuple[torch.Tensor, ...]):
                If self.use_time is True, the batch contains (x, time_deltas, y).
                Otherwise, the batch contains (x, y).
            batch_idx (int):
                Batch index (unused).
        """
        assert isinstance(batch_idx, int)
        if self.use_time:
            x_batch, time_deltas, y_batch = batch
            y_hat = self(x_batch, time_deltas)
        else:
            x_batch, y_batch = batch
            y_hat = self(x_batch)

        loss = nn.MSELoss()(y_hat, y_batch)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimiser to update only trainable parameters.

        Returns:
            torch.optim.Optimizer:
                The Adam optimiser for the remaining trainable parameters.
        """
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4
        )
