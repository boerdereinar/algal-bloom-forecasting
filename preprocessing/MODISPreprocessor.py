from argparse import ArgumentParser
from typing import Any

from edegruyl.preprocessing.Preprocessor import Preprocessor


class MODISPreprocessor(Preprocessor):
    """Preprocessor for MODIS datasets."""
    filename_glob = "*.tif"
    description_regex = r"^\d{2}_\d{2}_(?P<date>\d{4}_\d{2}_\d{2})$"

    def __init__(self, source_dir: str, target_dir: str, **kwargs: Any):
        """Initialize the preprocessor for MODIS data.

        Args:
            source_dir: The source directory to look for files in.
            target_dir: The target directory to put the processed files in, preserving the directory structure.
        """
        super().__init__(source_dir, target_dir)

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return super(MODISPreprocessor, MODISPreprocessor).add_preprocessor_specific_args(parent_parser)
