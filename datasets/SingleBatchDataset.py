from typing import TypeVar

from torch.utils.data import DataLoader, Dataset

T_co = TypeVar("T_co", covariant=True)


class SingleBatchDataset(Dataset[T_co]):
    def __init__(self, data_loader: DataLoader[T_co], length: int):
        self.sample = next(iter(data_loader))
        self.length = length

    def __getitem__(self, index) -> T_co:
        return self.sample

    def __len__(self) -> int:
        return self.length
