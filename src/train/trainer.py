from typing import Dict, Any
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.dataset import TacrolimusDataset
from src.models.lstm_attention_model import AttentionLSTMModel, AttentionLSTMEnsemble
from src.models.lstm_model import LSTMModel, LSTMEnsemble
from src.models.time_aware_lstm import TimeAwareLSTM, TimeAwareLSTMEnsemble
from src.models.tl_lstm import LSTMTransferLearning


def train_model(
        config: Dict[str, Any],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        groups: np.ndarray = None,
        time_diff: Any = None,
        lstm_model: str = 'attention',
        validation: bool = False,
) -> pl.LightningModule:
    """Train the LSTM model using train and validation sets.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        x_train (torch.Tensor): Training input features.
        y_train (torch.Tensor): Training target variable.
        groups (np.ndarray): Group labels for GroupShuffleSplit.
        time_diff (Any, optional): Time difference information.
        lstm_model (str, optional): Type of LSTM model to use.
        ensemble (bool, optional): Whether to use ensemble model.

    Returns:
        pl.LightningModule: Trained LSTM model.
    """
    # Create full dataset
    full_dataset = TacrolimusDataset(x_train, y_train, time_diff=time_diff)

    if validation:
        # Use GroupShuffleSplit to create validation set (10% of training data)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(gss.split(x_train, y_train, groups=groups))

        # Create train and validation datasets
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        val_loader = DataLoader(
            val_dataset,
            batch_size=2048,
            shuffle=False,
            num_workers=20,
            pin_memory=True,
        )
    else:
        train_dataset = full_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=20,
        pin_memory=True,
    )

    # Model initialization (unchanged)
    if lstm_model == "attention":
        model = AttentionLSTMModel(
            input_size=x_train.shape[2],
            window_size=config["window_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=1,
            learning_rate=config["learning_rate"],
            dropout=config["dropout"],
            attention=True,
        )
    elif lstm_model == "time_aware":
        model = TimeAwareLSTM(
            input_size=x_train.shape[2],
            window_size=config["window_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=1,
            learning_rate=config["learning_rate"],
            dropout=config["dropout"],
            attention=True,
        )

    if validation:
        early_stopping = EarlyStopping(monitor="val_loss", patience=30, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        logger = TensorBoardLogger("tb_logs", name=lstm_model)

        trainer = pl.Trainer(
            max_epochs=300,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            accelerator='gpu',
            devices=[0],
        )

        trainer.fit(model, train_loader, val_loader)
    else:
        early_stopping = EarlyStopping(monitor="train_loss", patience=20, mode="min")
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="train_loss",
            mode="min",
        )

        logger = TensorBoardLogger("tb_logs", name=lstm_model)

        trainer = pl.Trainer(
            max_epochs=200,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            accelerator='gpu',
            devices=[0],
        )
        trainer.fit(model, train_loader)

    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    return best_model


def train_tl_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    tl_model: pl.LightningModule,
    time_diff: Any = None,
) -> pl.LightningModule:
    """Train the LSTM model using train and validation sets.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        x_train (torch.Tensor): Training input features.
        y_train (torch.Tensor): Training target variable.

    Returns:
        pl.LightningModule: Trained LSTM model.
    """
    train_dataset = TacrolimusDataset(x_train, y_train, time_diff=time_diff)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2048,
        shuffle=True,
        num_workers=40,
        pin_memory=True,
    )

    model = LSTMTransferLearning(tl_model, time=True if time_diff is not None else False)
    early_stopping = EarlyStopping(monitor="train_loss", patience=20, mode="min")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="train_loss",
        mode="min",
    )

    logger = TensorBoardLogger("tb_logs", name="lstm_model")

    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        accelerator='gpu',
        devices=[0],
    )

    trainer.fit(model, train_loader)

    best_model = model.load_from_checkpoint(checkpoint_callback.best_model_path, model=model.model)

    return best_model