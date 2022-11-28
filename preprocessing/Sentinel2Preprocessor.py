from argparse import ArgumentParser
from typing import Any

from edegruyl.preprocessing.Preprocessor import Preprocessor


class Sentinel2Preprocessor(Preprocessor):
    """Preprocessor for Sentinel2 datasets."""
    filename_glob = "*Sentinel2*.tif"
    description_regex = r"^.*_(?P<date>\d{8}T\d{6})_\d{8}T\d{6}.*$"
    date_format = "%Y%m%dT%H%M%S"

    def __init__(self, source_dir: str, target_dir: str, **kwargs: Any):
        """Initialize the preprocessor for Sentinel2 data.

        Args:
            source_dir: The source directory to look for files in.
            target_dir: The target directory to put the processed files in, preserving the directory structure.
        """
        super().__init__(source_dir, target_dir)

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return super(Sentinel2Preprocessor, Sentinel2Preprocessor).add_preprocessor_specific_args(parent_parser)
