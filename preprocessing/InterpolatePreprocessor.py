import glob
import os
import queue
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from functools import reduce
from itertools import groupby
from typing import Any, Tuple, Dict, Iterator, Sequence

import numpy as np
import rasterio
from rasterio.io import DatasetReader, DatasetWriter
from tqdm import tqdm

from edegruyl.preprocessing import Preprocessor


class InterpolationStrategy(Enum):
    LookBack = "Use the last known sample to fill in missing samples."

    def __str__(self):
        return self.name


class InterpolatePreprocessor(Preprocessor):
    """Preprocessor for interpolating missing data."""

    filename_regex = re.compile(r"^.*[\\/](?P<reservoir>.+)[\\/]"
                                r"(?P<pre>.+_)(?P<date>\d{8})T\d{6}(?P<post>_.+\.tif)$")
    date_format = "%Y%m%d"
    strategy: "Strategy"

    def __init__(
            self,
            source_dir: str,
            target_dir: str,
            interpolation_strategy: InterpolationStrategy,
            max_workers: int,
            **kwargs
    ) -> None:
        super().__init__(source_dir, target_dir)
        self.max_workers = max_workers

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
        parent_parser.add_argument("--max-workers", type=int, default=100,
                                   help="The maximum number of worker threads used for writing the files.")
        return parent_parser

    def preprocess(self) -> None:
        """Preprocess the datasets."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Group files by reservoir
            reservoirs = {x: list(xs) for x, xs in groupby(
                ((f, self.filename_regex.match(f)) for f in glob.glob(self.source_dir, recursive=True)),
                lambda f: f[1]["reservoir"])}

            total = 0
            for x in reservoirs.values():
                t0 = datetime.strptime(x[0][1]["date"], self.date_format)
                t1 = datetime.strptime(x[-1][1]["date"], self.date_format)
                total += (t1 - t0).days + 1

            pbar = tqdm(total=total)

            # Loop through all reservoirs
            for reservoir, files in reservoirs.items():
                # Create directory
                os.makedirs(os.path.join(self.target_dir, reservoir), exist_ok=True)

                data_prev = None
                t_prev = None

                for match, data_cur, descriptions, profile in self._load_files(executor, files):
                    # Get the date
                    t_cur = datetime.strptime(match["date"], self.date_format)

                    if data_prev is None:
                        # Handle first file in the reservoir
                        self.strategy.first(data_cur)
                        # Target filename
                        target_file = os.path.join(
                            self.target_dir,
                            reservoir,
                            f"{match['pre']}{t_cur.strftime(self.date_format)}{match['post']}"
                        )
                        # Save file in a new worker thread
                        executor.submit(self._write_file, target_file, profile, data_cur)
                        pbar.update(1)
                    else:
                        # Process all missing samples including the last sample
                        for t_i in [t_prev + timedelta(days=dt) for dt in range(1, (t_cur - t_prev).days + 1)]:
                            pbar.set_description(f"{reservoir} {t_i:%Y-%m-%d}")
                            data = self.strategy.interpolate(data_prev, t_prev, data_cur, t_cur, t_i)
                            # Target filename
                            target_file = os.path.join(
                                self.target_dir,
                                reservoir,
                                f"{match['pre']}{t_i.strftime(self.date_format)}{match['post']}"
                            )
                            # Save file in a new worker thread
                            executor.submit(self._write_file, target_file, profile, data, descriptions)
                            pbar.update(1)

                    # Update previous sample
                    data_prev = data_cur
                    t_prev = t_cur

    def _load_files(self, executor: ThreadPoolExecutor, files) -> Iterator[Tuple[re.Match, np.ndarray, Dict[str, Any]]]:
        """Prefetch files."""
        prefetched = queue.SimpleQueue()

        files_iterator = iter(files)
        for _ in range(20):
            file = next(files_iterator, None)
            if not file:
                break
            prefetched.put(executor.submit(self._read_file, file[0], file[1]))

        while not prefetched.empty():
            yield prefetched.get(True).result()
            file = next(files_iterator, None)
            if file:
                prefetched.put(executor.submit(self._read_file, file[0], file[1]))

    @staticmethod
    def _read_file(file: str, match: re.Match) -> Tuple[re.Match, np.ndarray, Sequence[str], Dict[str, Any]]:
        with rasterio.open(file) as src:  # type: DatasetReader
            return match, src.read(), src.descriptions, src.profile

    @staticmethod
    def _write_file(file: str, profile: Dict[str, Any], data: np.ndarray, descriptions: Sequence[str]) -> None:
        with rasterio.open(file, "w", **profile) as target:  # type: DatasetWriter
            target.write(data)
            target.descriptions = descriptions


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
