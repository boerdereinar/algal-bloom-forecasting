import glob
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, List, Optional, Tuple

import rasterio
from joblib import Parallel, delayed
from rasterio.profiles import DefaultGTiffProfile
from torchgeo.datasets import BoundingBox
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset, LandCoverDataset
from edegruyl.preprocessing import Preprocessor
from edegruyl.preprocessing.strategies import LinearStrategy, LookBackStrategy, Strategy
from edegruyl.transforms import ClassMask


class InterpolationStrategy(Enum):
    LookBack = "Use the last known samples to fill in missing samples."
    Linear = "Use linear interpolation to fill in missing samples."

    def __str__(self):
        return self.name


class File:
    def __init__(self, file: str, dataset: str, date: datetime):
        self.file = file
        self.dataset = dataset
        self.date = date


class InterpolatePreprocessor(Preprocessor):
    """Preprocessor for interpolating missing data."""

    # filename_regex = re.compile(r"^.*[\\/](?P<reservoir>.+)_(?P<date>\d{8})T\d{6}_(?P<dataset>.+)\.tif$")
    date_format = "%Y%m%d"

    def __init__(
            self,
            source_dir: str,
            target_dir: str,
            land_use: str,
            interpolation_strategy: InterpolationStrategy,
            num_workers: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(source_dir, target_dir)
        self.land_use = land_use
        self.num_workers = num_workers
        self.interpolation_strategy = interpolation_strategy
        self.kwargs = kwargs

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        super(InterpolatePreprocessor, InterpolatePreprocessor).add_preprocessor_specific_args(parent_parser)
        parent_parser.add_argument("--land-use", type=str, required=True, help="The path to the land use dataset.")
        parent_parser.add_argument("-st", "--interpolation-strategy",
                                   metavar="STRATEGY",
                                   type=lambda s: InterpolationStrategy[s],
                                   choices=list(InterpolationStrategy),
                                   help="The strategy used to interpolate the data.\nAvailable strategies:\n" +
                                        "\n".join(f"\t{v.name}: {v.value}" for v in InterpolationStrategy))
        parent_parser.add_argument("--num-workers", type=int, default=None,
                                   help="The number of workers to use to process the data with.")
        return parent_parser

    @property
    def strategy(self) -> Strategy:
        """Get a strategy"""
        match self.interpolation_strategy:
            case InterpolationStrategy.LookBack:
                return LookBackStrategy(**self.kwargs)
            case InterpolationStrategy.Linear:
                return LinearStrategy(**self.kwargs)

    def preprocess(self) -> None:
        """Preprocess the datasets."""
        reservoirs = [d for d in os.listdir(self.source_dir) if os.path.isdir(os.path.join(self.source_dir, d))]

        Parallel(self.num_workers)(
            delayed(self._preprocess_reservoir)(reservoir, i) for i, reservoir in enumerate(reservoirs)
        )

    def _preprocess_reservoir(self, reservoir: str, idx: int) -> None:
        """Preprocess a single reservoir."""
        biological = BiologicalDataset(os.path.join(self.source_dir, reservoir))
        land_use = LandCoverDataset(self.land_use, transforms=ClassMask(1))
        dataset = biological & land_use

        with rasterio.open(next(biological.index.intersection(tuple(dataset.bounds), "raw"))) as src:
            profile = src.profile

        bboxs = sorted(
            (BoundingBox(*item.bounds) for item in biological.index.intersection(tuple(dataset.bounds), True)),
            key=lambda bbox: bbox.mint
        )

        pbar = tqdm(
            total=(datetime.fromtimestamp(bboxs[-1].mint) - datetime.fromtimestamp(bboxs[0].mint)).days + 1,
            desc=f"{reservoir.capitalize():20}",
            unit="days",
            position=idx
        )

        # Create directory if it does not exist
        os.makedirs(os.path.join(self.target_dir, reservoir), exist_ok=True)

        strategy = self.strategy
        t_prev = None

        for bbox in bboxs:
            data = dataset[bbox]
            date = datetime.fromtimestamp(bbox.mint)

            image = data["image"]
            mask = data["mask"][0]

            target_file_template = os.path.join(self.target_dir, reservoir, f"{reservoir}_%s_biological.tif")

            if t_prev is None:
                pbar.set_postfix({"date": date.strftime(self.date_format)})
                # Handle first file in the reservoir
                image = strategy.first(image, date, mask)
                target_file = target_file_template % date.strftime(self.date_format)

                # Write the original data
                with rasterio.open(target_file, "w", **profile) as target:
                    target.write(image)

                # Update progress
                pbar.update(1)
            else:
                interpolated_data = strategy.interpolate(image, date, mask)

                for dt, x in enumerate(interpolated_data):
                    t = (t_prev + timedelta(dt + 1))
                    pbar.set_postfix({"date": t})
                    target_file = target_file_template % t.strftime(self.date_format)

                    # Write the interpolated data
                    with rasterio.open(target_file, "w", **profile) as target:
                        target.write(interpolated_data[dt])

                    pbar.update(1)

            t_prev = date
