import torch
from torch.utils.data import Dataset


class TacrolimusDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, time_diff: torch.Tensor = None):
        self.X = x.float()
        self.y = y.float()
        self.time_diff = time_diff.float() if time_diff is not None else None

        # Ensure X and y have the same first dimension
        assert len(self.X) == len(self.y), "X and y must have the same length"

        # If time_diff is provided, ensure it has the same length as X and y
        if self.time_diff is not None:
            assert len(self.X) == len(self.time_diff), "time_diff must have the same length as X and y"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.time_diff is not None:
            return self.X[idx], self.time_diff[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]