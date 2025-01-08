"""Script containing the pytorch dataset class for tacrolimus data."""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TacrolimusDataset(Dataset):
    """Dataset class for tacrolimus data that can handle time differences."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, time_diff: torch.Tensor = None) -> None:
        self.X = x.float()
        self.y = y.float()
        self.time_diff = time_diff.float() if time_diff is not None else None

        assert len(self.X) == len(self.y), "X and y must have the same length"

        if self.time_diff is not None:
            assert len(self.X) == len(
                self.time_diff
            ), "time_diff must have the same length as X and y"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]:
        if self.time_diff is not None:
            return self.X[idx], self.time_diff[idx], self.y[idx]
        return self.X[idx], self.y[idx]
