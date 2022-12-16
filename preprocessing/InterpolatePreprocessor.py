import glob
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Tuple, List

import rasterio
from joblib import Parallel, delayed
from tqdm import tqdm

from edegruyl.preprocessing import Preprocessor
from edegruyl.preprocessing.strategies import Strategy, LookBackStrategy, LinearStrategy


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

    filename_regex = re.compile(r"^.*[\\/](?P<reservoir>.+)_(?P<date>\d{8})T\d{6}_(?P<dataset>.+)\.tif$")
    date_format = "%Y%m%d"

    def __init__(
            self,
            source_dir: str,
            target_dir: str,
            interpolation_strategy: InterpolationStrategy,
            num_workers: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(source_dir, target_dir)
        self.num_workers = num_workers
        self.interpolation_strategy = interpolation_strategy
        self.kwargs = kwargs

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        super(InterpolatePreprocessor, InterpolatePreprocessor).add_preprocessor_specific_args(parent_parser)
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
        reservoirs = defaultdict(list[File])
        for file in glob.iglob(self.source_dir, recursive=True):
            match = self.filename_regex.match(file)
            if match:
                reservoirs[match["reservoir"]].append(File(
                    file,
                    match["dataset"],
                    datetime.strptime(match["date"], self.date_format)
                ))

        Parallel(self.num_workers)(
            delayed(self._preprocess_reservoir)(item, i) for i, item in enumerate(reservoirs.items())
        )

    def _preprocess_reservoir(self, item: Tuple[str, List[File]], idx: int):
        """Preprocess a single reservoir."""
        reservoir, files = item

        pbar = tqdm(
            total=(files[-1].date - files[0].date).days + 1,
            desc=f"{reservoir.capitalize():20}",
            unit="days",
            position=idx
        )

        # Create directory if it does not exist
        os.makedirs(os.path.join(self.target_dir, reservoir), exist_ok=True)

        strategy = self.strategy
        t_prev = None
        data_prev = None

        for file in files:
            target_file_template = os.path.join(self.target_dir, reservoir, f"{reservoir}_%s_{file.dataset}.tif")

            with rasterio.open(file.file) as src:
                data = src.read()

                if data_prev is None or t_prev is None:
                    pbar.set_postfix({"date": file.date.strftime(self.date_format)})
                    # Handle first file in the reservoir
                    strategy.first(data)
                    target_file = target_file_template % file.date.strftime(self.date_format)

                    # Write the original data
                    with rasterio.open(target_file, "w", **src.profile) as target:
                        target.write(data)

                    # Update progress
                    pbar.update(1)
                else:
                    interpolated_data = strategy.interpolate(data_prev, t_prev, data, file.date)

                    for dt, x in enumerate(interpolated_data):
                        t = (t_prev + timedelta(dt + 1))
                        pbar.set_postfix({"date": t})
                        target_file = target_file_template % t.strftime(self.date_format)

                        # Write the interpolated data
                        with rasterio.open(target_file, "w", **src.profile) as target:
                            target.write(interpolated_data[dt])

                        pbar.update(1)

                t_prev = file.date
                data_prev = data
