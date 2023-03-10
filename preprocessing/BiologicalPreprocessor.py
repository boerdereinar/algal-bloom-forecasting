import glob
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from joblib import Parallel, delayed
from rasterio.io import DatasetReader, DatasetWriter
from tqdm import tqdm

from edegruyl.preprocessing import Preprocessor


class BiologicalPreprocessor(Preprocessor):
    """Preprocessor for the biological modalities."""

    path = "02_DATA/01_BIOPHYSICAL PARAMETERS/BIOLOGICAL PROPERTIES"
    file_glob = "*.tif"
    filename_regex = re.compile(r"^.*(?:\\\\?|/)(?P<modality>.+)(?:\\\\?|/)"
                                r"(?P<reservoir>.+)(?:\\\\?|/)"
                                r".*_(?P<season>[A-Za-z]+_\d{4})\.tif$")
    description_regex = re.compile(r"^.*_(?P<date>\d{8}T\d{6})_\d{8}T\d{6}.*$")

    def __init__(self, source_dir: str, target_dir: str, num_workers: Optional[int] = None):
        """
        Initialize the biological processor.

        Args:
            source_dir: The path to the source directory.
            target_dir: The path to the target directory.
            num_workers: The number of workers to use to process the data with.
        """
        super().__init__(source_dir, target_dir)
        self.num_workers = num_workers

    @staticmethod
    def add_preprocessor_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        super(BiologicalPreprocessor, BiologicalPreprocessor).add_preprocessor_specific_args(parent_parser)
        parent_parser.add_argument("--num-workers", type=int, default=None,
                                   help="The number of workers to use to process the data with.")
        return parent_parser

    def preprocess(self):
        """
        Preprocess the biological properties into individual tif files with each modality as a band in the file.
        """
        files = defaultdict(list)

        # Group all the files by their reservoir and season.
        for file in glob.glob(os.path.join(self.source_dir, "**", self.file_glob), recursive=True):
            match = self.filename_regex.match(file)
            if match:
                files[tuple(match.groupdict().values())[1:]].append((match.group("modality"), file))

        # Process all the grouped files
        Parallel(self.num_workers)(
            delayed(self._process_data)(reservoir, group) for (reservoir, _), group in tqdm(files.items())
        )

    def _process_data(self, reservoir: str, group: List[Tuple[str, str]]):
        """
        Processes a single group of data for a season for a reservoir.

        Args:
            reservoir: The reservoir.
            group: The grouped files as a list of tuples (modality, filename).
        """
        # Open all files
        datasets: List[DatasetReader] = [rasterio.open(file) for _, file in group]
        profile = datasets[0].profile
        profile["count"] = len(datasets)
        profile["dtype"] = np.float32

        # Create the target directory if necessary
        target_dir = os.path.join(self.target_dir, "biological", reservoir.lower())
        os.makedirs(target_dir, exist_ok=True)

        # Loop through all bands
        for i in range(datasets[0].count):
            # Get the date from the band description
            match = self.description_regex.match(datasets[0].descriptions[i])
            if match is None:
                continue

            date = match.group("date")

            # Start writing to the file
            target_file = os.path.join(target_dir, f"{reservoir.lower()}_{date}_biological.tif")
            writer: DatasetWriter
            with rasterio.open(target_file, "w", **profile) as writer:
                for j, ds in enumerate(datasets):
                    # Read the data in the band
                    data = ds.read(i + 1).astype(np.float32)
                    # Write the data to the new file
                    writer.write(data, j + 1)
                    # Write the modality
                    writer.set_band_description(j + 1, group[j][0].lower())
