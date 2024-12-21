import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMTransferLearning(pl.LightningModule):
    def __init__(self, model, time=False):
        super().__init__()
        self.model = model
        self.time = time
        # Freeze all LSTM layers, except for the final, fully connected layer
        for param in model.lstm.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    def forward(self, x, time_delta=None):
        if self.time:
            return self.model(x, time_delta)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.time:
            x, time_deltas, y = batch
            print(x.shape, time_deltas.shape, y.shape)
            y_hat = self(x, time_deltas)
        else:
            x, y = batch
            y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.time:
            x, time_deltas, y = batch
            y_hat = self(x, time_deltas)
        else:
            x, y = batch
            y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)
