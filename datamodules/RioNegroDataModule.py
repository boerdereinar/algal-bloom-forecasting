from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import GeoSampler, BatchGeoSampler, GridGeoSampler, RandomBatchGeoSampler, Units

from edegruyl.datasets import RioNegroDataset


class RioNegroDataModule(LightningDataModule):
    """Data module for Rio Negro data."""
    dataset: RioNegroDataset
    train_sampler: BatchGeoSampler
    val_sampler: GeoSampler

    def __init__(
            self,
            root: str,
            reservoir: str,
            window_size: int,
            prediction_horizon: int,
            train_test_split: float = 0.8,
            size: int = 256,
            batch_size: int = 6,
            length: int = 1000,
            num_workers: int = 0,
            **kwargs: Any
    ) -> None:
        super().__init__()

        self.root = root
        self.reservoir = reservoir

        self.save_hyperparameters(ignore=["root", "reservoir", "kwargs"])

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--root", type=str, help="The root directory.", required=True)
        parser.add_argument("--reservoir", type=str, help="The reservoir to train for.", required=True)
        parser.add_argument("--window-size", type=int, help="The window size.", required=True)
        parser.add_argument("--prediction-horizon", type=int, help="The prediction horizon", required=True)
        parser.add_argument("--train-test-split", type=float, default=0.8,
                            help="The ratio between the number of training and test samples.")
        parser.add_argument("--size", type=int, help="The size of the sampled patches in pixels.", default=256)
        parser.add_argument("--batch-size", type=int, help="The size of the batches.", default=6)
        parser.add_argument("--length", type=int, help="The number of samples per epoch.", default=1000)
        parser.add_argument("--num-workers", type=int, help="The number of workers to load the date with", default=0)
        return parent_parser

    def setup(self, stage: str) -> None:
        self.dataset = RioNegroDataset(self.root, self.reservoir, **self.hparams)

        # Train-test split
        t = (self.dataset.roi.maxt - self.dataset.roi.mint) * self.hparams.train_test_split + self.dataset.roi.mint
        mint = datetime.fromtimestamp(t).replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = mint - timedelta(microseconds=1)

        # Training sampler
        train_roi = BoundingBox(*self.dataset.roi[:5], maxt.timestamp())
        self.train_sampler = RandomBatchGeoSampler(self.dataset.biological_unprocessed, **self.hparams, roi=train_roi)

        # Validation sampler
        val_roi = BoundingBox(*self.dataset.roi[:4], mint.timestamp(), self.dataset.roi.maxt)
        size = tuple(self.dataset.roi[3, 1] - self.dataset.roi[2, 0])
        self.val_sampler = GridGeoSampler(self.dataset.biological_unprocessed, size, 1, val_roi, Units.CRS)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_sampler=self.train_sampler, num_workers=self.hparams.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, sampler=self.val_sampler, num_workers=self.hparams.num_workers)
