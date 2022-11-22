from argparse import ArgumentParser
from typing import Any, Sequence, Dict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler
from torchgeo.datasets import BoundingBox, stack_samples, GeoDataset
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler

from edegruyl.datasets.ChlorophyllADataset import ChlorophyllADataset
from edegruyl.datasets.Sentinel2Dataset import Sentinel2Dataset


class ChlorophyllADataModule(LightningDataModule):
    sentinel2: GeoDataset
    chlorophyll: GeoDataset
    dataset: GeoDataset
    train_sampler: Sampler[Sequence]
    val_sampler: Sampler

    def __init__(
            self,
            data_dir: str,
            label_dir: str,
            batch_size: int = 64,
            num_workers: int = 0,
            patch_size: int = 256,
            length: int = 1000,
            stride: int = 128,
            **kwargs: Any) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.length = length
        self.stride = stride
        self.kwargs = kwargs

    @staticmethod
    def add_datamodule_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser

    @staticmethod
    def _train_test_split(bounds: BoundingBox) -> (BoundingBox, BoundingBox):
        return bounds, bounds

    @staticmethod
    def _image_to_label(x: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            label=x["image"],
            crs_label=x["crs"],
            bbox_label=x["bbox"])

    def setup(self, stage: str) -> None:
        self.sentinel2 = Sentinel2Dataset(self.data_dir)
        self.chlorophyll = ChlorophyllADataset(self.label_dir, self.sentinel2.crs, self.sentinel2.res,
                                               transforms=self._image_to_label)
        self.dataset = self.sentinel2 & self.chlorophyll

        train_roi, test_roi = self._train_test_split(self.dataset.bounds)
        self.train_sampler = RandomBatchGeoSampler(
            self.sentinel2, self.patch_size, self.batch_size, self.length, train_roi)
        self.val_sampler = GridGeoSampler(self.sentinel2, self.patch_size, self.stride, test_roi)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples)

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.dataset,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples)
