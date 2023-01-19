from abc import ABC, abstractmethod
from argparse import ArgumentParser


class Preprocessor(ABC):
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
                                   help="The path to the target directory.")
        return parent_parser

    @abstractmethod
    def preprocess(self):
        """Runs the preprocessor."""
        ...
