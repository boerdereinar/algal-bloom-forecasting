import glob
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
from functools import reduce
from typing import Any, Optional, Tuple, List

import numpy as np
import rasterio
from joblib import Parallel, delayed
from tqdm import tqdm

from edegruyl.preprocessing import Preprocessor


class InterpolationStrategy(Enum):
    LookBack = "Use the last known sample to fill in missing samples."

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
    strategy: "Strategy"

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

        match interpolation_strategy:
            case InterpolationStrategy.LookBack:
                self.strategy = LookBackStrategy(**kwargs)

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

        for k, v in reservoirs.items():
            reservoirs[k] = sorted(v, key=lambda x: x.date)

        Parallel(self.num_workers)(
            delayed(self._preprocess_reservoir)(item, i) for i, item in enumerate(reservoirs.items())
        )

    def _preprocess_reservoir(self, item: Tuple[str, List[File]], idx: int):
        """Preprocess a single reservoir."""
        reservoir, files = item

        pbar = tqdm(
            total=(files[-1].date - files[0].date).days + 1,
            desc=reservoir.capitalize(),
            unit="day",
            position=idx
        )

        # Create directory if it does not exist
        os.makedirs(os.path.join(self.target_dir, reservoir), exist_ok=True)

        t_prev = None
        data_prev = None

        for file in files:
            target_file_template = os.path.join(self.target_dir, reservoir, f"{reservoir}_%s_{file.dataset}.tif")

            with rasterio.open(file.file) as src:
                data = src.read()

                if data_prev is None or t_prev is None:
                    pbar.set_postfix({"date": file.date.strftime(self.date_format)})
                    # Handle first file in the reservoir
                    self.strategy.first(data)
                    target_file = target_file_template % file.date.strftime(self.date_format)

                    # Write the original data
                    with rasterio.open(target_file, "w", **src.profile) as target:
                        target.write(data)

                    # Update progress
                    pbar.update(1)
                else:
                    for t in (t_prev + timedelta(days) for days in range(1, (file.date - t_prev).days + 1)):
                        pbar.set_postfix({"date": t.strftime(self.date_format)})
                        interpolated_data = self.strategy.interpolate(data_prev, t_prev, data, file.date, t)
                        target_file = target_file_template % t.strftime(self.date_format)

                        # Write the interpolated data
                        with rasterio.open(target_file, "w", **src.profile) as target:
                            target.write(interpolated_data)
                        pbar.update(1)

                t_prev = file.date
                data_prev = data


class Strategy(ABC):
    """Interpolation strategy."""
    @abstractmethod
    def first(self, data: np.ndarray) -> None:
        """Process the first sample."""
        ...

    @abstractmethod
    def interpolate(
            self,
            data_prev: np.ndarray,
            t_prev: datetime,
            data_next: np.ndarray,
            t_next: datetime,
            t: datetime
    ) -> np.ndarray:
        """
        Interpolate a sample in between two other samples.

        Args:
            data_prev: The previous sample
            t_prev: The previous timestamp
            data_next: The next sample
            t_next: The next timestamp
            t: The current timestamp

        Returns:
            The interpolated sample.
        """
        ...


class LookBackStrategy(Strategy):
    """Interpolation strategy that uses the last known sample to fill in missing samples."""
    data = None

    def __init__(self, buffer_size: int = 30, **kwargs: Any):
        """
        Initializes the look back strategy.

        Args:
            buffer_size: The number of days to look back for samples to fill in missing values.
        """
        self.buffer = deque(maxlen=buffer_size)

    def first(self, data: np.ndarray) -> None:
        self.buffer.clear()
        self.buffer.append(data)

    def interpolate(
            self,
            data_prev: np.ndarray,
            t_prev: datetime,
            data_next: np.ndarray,
            t_next: datetime,
            t: datetime
    ) -> np.ndarray:
        if self.data is None:
            self.data = reduce(lambda a, b: np.where(np.isnan(a), b, a), reversed(self.buffer))
            self.buffer.append(data_next)

        if t_next == t:
            last = np.where(np.isnan(data_next), self.data, data_next)
            self.data = None
            return last
        return self.data
