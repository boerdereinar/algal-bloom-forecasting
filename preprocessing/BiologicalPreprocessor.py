import glob
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple

import rasterio
from joblib import Parallel, delayed
from rasterio.io import DatasetReader, DatasetWriter
from tqdm import tqdm


class BiologicalPreprocessor:
    """Preprocessor for the biological modalities."""

    path = "02_DATA/01_BIOPHYSICAL PARAMETERS/BIOLOGICAL PROPERTIES"
    filename_regex = r"^.*[\\/](?P<modality>.+)[\\/]{1,2}" \
                     r"(?P<reservoir>.+)[\\/]{1,2}" \
                     r".*_(?P<season>[A-Za-z]+_\d{4})\.tif$"
    description_regex = r"^.*_(?P<date>\d{8}T\d{6})_\d{8}T\d{6}.*$"

    def __init__(self, source_dir: str, target_dir: str, num_workers: int = None):
        """
        Initialize the biological processor.

        Args:
            source_dir: The path to the source directory.
            target_dir: The path to the target directory.
            num_workers: The number of workers to use to process the data with.
        """
        self.source_dir = os.path.join(source_dir, self.path, "**/*.tif")
        self.target_dir = target_dir
        self.num_workers = num_workers

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument("-s", "--source-dir", type=str, required=True,
                                   help="The path to the source directory.")
        parent_parser.add_argument("-t", "--target-dir", type=str, required=True,
                                   help="The parth to the target directory.")
        parent_parser.add_argument("--num-workers", type=int, default=None,
                                   help="The number of workers to use to process the data with.")
        return parent_parser

    def preprocess(self):
        """
        Preprocess the biological properties into individual tif files with each modality as a band in the file.
        """
        filename_regex = re.compile(self.filename_regex)

        files = defaultdict(list)

        # Group all the files by their reservoir and season.
        for file in glob.glob(self.source_dir, recursive=True):
            match = filename_regex.match(file)
            if match:
                files[tuple(match.groupdict().values())[1:]].append((match.group("modality"), file))

        # Process all the grouped files
        Parallel(self)(delayed(self._process_data)(reservoir, group) for (reservoir, _), group in tqdm(files.items()))

    def _process_data(self, reservoir: str, group: List[Tuple[str, str]]):
        """
        Processes a single group of data for a season for a reservoir.

        Args:
            reservoir: The reservoir.
            group: The grouped files as a list of tuples (modality, filename).
        """
        description_regex = re.compile(self.description_regex)

        # Open all files
        datasets: List[DatasetReader] = [rasterio.open(file) for _, file in group]
        profile = datasets[0].profile
        profile["count"] = len(datasets)

        # Create the target directory if necessary
        target_dir = os.path.join(self.target_dir, "biological", reservoir.lower())
        os.makedirs(target_dir, exist_ok=True)

        # Loop through all bands
        for i in range(datasets[0].count):
            # Get the date from the band description
            date = description_regex.match(datasets[0].descriptions[i]).group("date")

            # Start writing to the file
            target_file = os.path.join(target_dir, f"{reservoir.lower()}_{date}_biological.tif")
            with rasterio.open(target_file, "w", **profile) as writer:  # type: DatasetWriter
                for j, ds in enumerate(datasets):
                    # Read the data in the band
                    data = ds.read(i + 1)
                    # Write the data to the new file
                    writer.write(data, j + 1)
                    # Write the modality
                    writer.set_band_description(j + 1, group[j][0].lower())
