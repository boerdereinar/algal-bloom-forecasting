from argparse import ArgumentParser
from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, stack_samples

from edegruyl.datasets import TimeSeriesDataset
from edegruyl.samplers import RandomForecastingGeoSampler
from edegruyl.transforms import ConvertToLabel


class TimeSeriesDataModule(LightningDataModule):
    """Datamodule for classifying time series data."""
    input_dataset: TimeSeriesDataset
    label_dataset: TimeSeriesDataset
    dataset: GeoDataset
    train_sampler: RandomForecastingGeoSampler

    def __init__(
        self,
        input_dir: str,
        label_dir: str,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.label_dir = label_dir

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("-i", "--input-dir", type=str, help="The input directory.", required=True)
        parser.add_argument("-l", "--label-dir", type=str, help="The label directory.", required=True)
        return parent_parser

    def setup(self, stage: str) -> None:
        self.input_dataset = TimeSeriesDataset(self.input_dir)
        self.label_dataset = TimeSeriesDataset(self.label_dir, self.input_dataset.crs, self.input_dataset.res,
                                               ConvertToLabel())
        self.dataset = self.input_dataset ^ self.label_dataset

        self.train_sampler = RandomForecastingGeoSampler(self.dataset, 256, 1000, 5, 5)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=self.train_sampler,
            collate_fn=stack_samples
        )
