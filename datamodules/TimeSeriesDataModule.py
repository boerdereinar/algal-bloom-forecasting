from argparse import ArgumentParser
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples
from torchvision.transforms import Compose

from edegruyl.datasets import TimeSeriesDataset
from edegruyl.samplers import RandomBatchForecastingGeoSampler
from edegruyl.transforms import ConvertToLabel, KeepTensors
from edegruyl.transforms.ToDType import ToDType


class TimeSeriesDataModule(LightningDataModule):
    """Datamodule for classifying time series data."""
    input_dataset: TimeSeriesDataset
    label_dataset: TimeSeriesDataset
    dataset: GeoDataset
    train_sampler: RandomBatchForecastingGeoSampler

    def __init__(
            self,
            input_dir: str,
            label_dir: str,
            patch_size: int = 256,
            batch_size: int = 64,
            length: int = 1000,
            look_back: int = 5,
            look_ahead: int = 5,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.save_hyperparameters(ignore=["input_dir", "label_dir"] + list(kwargs))

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("-i", "--input-dir", type=str, help="The input directory.", required=True)
        parser.add_argument("-l", "--label-dir", type=str, help="The label directory.", required=True)
        parser.add_argument("--patch-size", type=int, help="The size of the sampled patches in pixels.", default=256)
        parser.add_argument("--batch-size", type=int, help="The size of the batches.", default=64)
        parser.add_argument("--length", type=int, help="The number of samples per epoch.", default=1000)
        parser.add_argument("--look-back", type=int, help="The number of days to look back.", default=5)
        parser.add_argument("--look-ahead", type=int, help="The number of days to look ahead.", default=5)
        return parent_parser

    def setup(self, stage: str) -> None:
        self.input_dataset = TimeSeriesDataset(self.input_dir,
                                               transforms=Compose([KeepTensors(), ToDType(torch.float32)]))
        self.label_dataset = TimeSeriesDataset(self.label_dir, self.input_dataset.crs, self.input_dataset.res,
                                               Compose([ConvertToLabel(), KeepTensors(), ToDType(torch.float32)]))
        self.dataset = self.input_dataset ^ self.label_dataset

        bounds = self.input_dataset.bounds & self.label_dataset.bounds
        self.train_sampler = RandomBatchForecastingGeoSampler(self.dataset, **self.hparams, roi=bounds)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            collate_fn=stack_samples
        )
