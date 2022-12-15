from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox
from torchgeo.samplers import GeoSampler, BatchGeoSampler, GridGeoSampler, RandomBatchGeoSampler, Units

from edegruyl.datasets import RioNegroDataset


class RioNegroDataModule(LightningDataModule):
    """A PyTorch `DataModule` for loading and preparing data from the Rio Negro dataset.

    This class extends the `DataModule` class from the PyTorch Lightning framework
    and provides methods for loading and preparing data from the Rio Negro dataset
    for use in training and evaluating machine learning models. It also provides
    methods for specifying and parsing command-line arguments that are specific to
    this dataset.

    Attributes:
        dataset: An instance of the `RioNegroDataset` class, representing the
            dataset that this data module loads and prepares data from.
        train_sampler: An instance of the `RandomBatchGeoSampler` class,
            representing the sampler that is used to sample data for training.
        val_sampler: An instance of the `GridGeoSampler` class, representing
            the sampler that is used to sample data for validation.
    """
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
            batch_size: int = 1,
            length: int = 1000,
            num_workers: int = 0,
            **kwargs: Any
    ) -> None:
        """Constructs a new `RioNegroDataModule` instance.

        Args:
            root: The root directory where the data is stored.
            reservoir: The specific reservoir of data to load.
            window_size: The window size to use when sampling data.
            prediction_horizon: The prediction horizon to use when sampling data.
            train_test_split: The ratio between the number of training and test
                samples. The default value is 0.8, meaning that 80% of the data
                will be used for training and 20% for testing.
            size: The size of the patches of data that will be sampled. The
                default value is 256.
            batch_size: The size of the batches that will be sampled. The
                default value is 1.
            length: The number of samples that will be generated per epoch.
                The default value is 1000.
            num_workers: The number of workers to use for parallel data loading.
                The default value is 0.
            kwargs: Additional keyword arguments.
        """
        super().__init__()

        self.save_hyperparameters(ignore=list(kwargs))

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds command-line arguments specific to this data module to an
        `ArgumentParser` object.

        This method adds several command-line arguments that are specific to the
        Rio Negro dataset and the `RioNegroDataModule` class. These arguments
        can be used to specify the root directory, the reservoir, the window size,
        the prediction horizon, and other hyperparameters.

        Args:
            parent_parser: The `ArgumentParser` object to add the arguments to.

        Returns:
            The same `ArgumentParser` object that was passed in, with the
            additional arguments added to it.
        """
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--root", type=str, help="The root directory.", required=True)
        parser.add_argument("--reservoir", type=str, help="The reservoir to train for.", required=True)
        parser.add_argument("--window-size", type=int, help="The window size.", required=True)
        parser.add_argument("--prediction-horizon", type=int, help="The prediction horizon", required=True)
        parser.add_argument("--train-test-split", type=float, default=0.8,
                            help="The ratio between the number of training and test samples.")
        parser.add_argument("--size", type=int, help="The size of the sampled patches in pixels.", default=256)
        parser.add_argument("--batch-size", type=int, help="The size of the batches.", default=1)
        parser.add_argument("--length", type=int, help="The number of samples per epoch.", default=1000)
        parser.add_argument("--num-workers", type=int, help="The number of workers to load the date with", default=0)
        return parent_parser

    def setup(self, stage: str) -> None:
        """Sets up the data module, preparing it for use.

        This method is called by the PyTorch Lightning framework to set up the
        data module and prepare it for use. It creates an instance of the
        `RioNegroDataset` class, which represents the dataset that this data module
        will load and prepare data from. It also creates instances of the
        `RandomBatchGeoSampler` and `GridGeoSampler` classes, which are used to
        sample data for training and validation, respectively.

        Args:
            stage: The stage of the training process that this data module is being
                set up for. This can be one of "fit", "test", or "predict".
        """
        self.dataset = RioNegroDataset(**self.hparams)

        # Train-test split
        roi = self.dataset.roi
        t = (roi.maxt - roi.mint) * self.hparams.train_test_split + roi.mint  # type: ignore
        mint = datetime.fromtimestamp(t).replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = mint - timedelta(microseconds=1)

        # Training sampler
        train_roi = BoundingBox(roi.minx, roi.maxx, roi.miny, roi.maxy, roi.mint, maxt.timestamp())
        self.train_sampler = RandomBatchGeoSampler(
            self.dataset.biological_unprocessed,
            size=self.hparams.size,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            length=self.hparams.length,  # type: ignore
            roi=train_roi
        )

        # Validation sampler
        val_roi = BoundingBox(roi.minx, roi.maxx, roi.miny, roi.maxy, mint.timestamp(), roi.maxt)
        minx, maxx, miny, maxy = self.dataset.roi[:4]
        self.val_sampler = GridGeoSampler(
            self.dataset.biological_unprocessed,
            size=(maxy - miny, maxx - minx),
            stride=1,
            roi=val_roi,
            units=Units.CRS
        )

    def train_dataloader(self) -> DataLoader:
        """Returns a PyTorch `DataLoader` instance for the training set.

        Returns:
            A PyTorch `DataLoader` instance for the training set.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.hparams.num_workers  # type: ignore
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a PyTorch `DataLoader` instance for the validation set.

        Returns:
            A PyTorch `DataLoader` instance for the validation set.
        """
        return DataLoader(
            self.dataset,
            sampler=self.val_sampler,
            num_workers=self.hparams.num_workers  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:
        """Returns a PyTorch `DataLoader` instance for the test set.

        Returns:
            A PyTorch `DataLoader` instance for the test set.
        """
        return self.val_dataloader()
