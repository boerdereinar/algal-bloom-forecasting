import glob
import os
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import dateutil.parser
import rasterio
from rasterio import RasterioIOError
from rasterio.io import DatasetReader, DatasetWriter
from tqdm import tqdm


class Preprocessor(ABC):
    filename_glob = "*"
    date_format = "%Y%m%d"

    @property
    @abstractmethod
    def description_regex(self):
        """
        Regular expression used to extract date from description.
        The expression must contain the named group `date`.
        """
        ...

    def __init__(self, source_dir: str, target_dir: str):
        """Initialize the preprocessor.

        Args:
            source_dir: The source directory to look for files in.
            target_dir: The target directory to put the processed files in, preserving the directory structure.
        """
        self.source_dir = source_dir
        self.target_dir = target_dir

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument("-s", "--source-dir", type=str, required=True,
                                   help="The path to the source directory.")
        parent_parser.add_argument("-t", "--target-dir", type=str, required=True,
                                   help="The parth to the target directory.")
        return parent_parser

    def preprocess(self):
        """Runs the preprocessor."""
        pathname = os.path.join(self.source_dir, "**", self.filename_glob)
        description_regex = re.compile(self.description_regex, re.VERBOSE)

        progress = tqdm(list(glob.iglob(pathname, recursive=True)))
        for filepath in progress:
            try:
                source_dir, file = os.path.split(filepath)
                target_dir = os.path.join(self.target_dir, os.path.relpath(source_dir, self.source_dir))
                file, _ = os.path.splitext(file)
                progress.set_description(file)

                os.makedirs(target_dir, exist_ok=True)
                with rasterio.open(filepath) as src:  # type: DatasetReader
                    profile = src.profile
                    profile.update(dtype=src.dtypes[0], count=1)
                    for i in range(src.count):
                        # Parse timestamps
                        description_match = re.match(description_regex, src.descriptions[i])
                        date = dateutil.parser.parse(description_match.group("date"))

                        # Create target file
                        file = os.path.join(
                            target_dir,
                            f"{file}_{i}_{date.strftime(self.date_format)}.tif")
                        with rasterio.open(file, "w", **profile) as target:  # type: DatasetWriter
                            target.write(src.read(i + 1), 1)
            except RasterioIOError:
                continue
