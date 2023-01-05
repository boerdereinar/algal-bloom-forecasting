import os.path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import torch
from joblib import Parallel, delayed
from torchgeo.datasets import BoundingBox, GeoDataset
from torchvision.transforms import Compose
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset, LandCoverDataset
from edegruyl.transforms import ClassMask, Clip, Normalize
from edegruyl.transforms.Bin import Bin
from edegruyl.utils.tqdmutils import tqdm_joblib


class RioNegroDataset(GeoDataset):
    biological_unprocessed: BiologicalDataset
    biological_processed: Optional[BiologicalDataset] = None
    water_use: LandCoverDataset

    CLIP = torch.tensor([20, 100, 80])
    MEAN = torch.tensor([2.68, 19.90, 27.89])
    STD = torch.tensor([1.37, 63.00, 12.72])
    BINS = torch.tensor([0, 20, 50, 80, 100])

    def __init__(
            self,
            root: str,
            reservoir: str,
            window_size: int,
            prediction_horizon: int,
            load_processed: bool = True,
            classify: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.load_processed = load_processed
        self.classify = classify

        # Transforms
        self.clip = Clip(self.CLIP)
        self.normalize = Normalize(self.MEAN, self.STD)
        self.bin = Bin(self.BINS)
        self.water_mask = ClassMask(1)

        # Load the datasets
        self.load_datasets(root, reservoir)
        self.dataset = self.biological_unprocessed & self.water_use

        # Update index
        self.res = self.dataset.res
        self._crs = self.dataset.crs
        self.index = self.dataset.index

        # Prevent out of range samples
        dt = timedelta(days=window_size + prediction_horizon).total_seconds()
        roi = self.dataset.bounds
        if self.load_processed:
            roi &= self.biological_processed.bounds
        self.roi = BoundingBox(roi.minx, roi.maxx, roi.miny, roi.maxy, roi.mint + dt, roi.maxt)

    def load_datasets(self, root: str, reservoir: str) -> None:
        """
        Load the datasets in parallel.

        Args:
            root: The root directory where the data is stored.
            reservoir: The specific reservoir of data to load.
        """
        with tqdm_joblib(tqdm(desc="Loading datasets", total=3)):
            jobs = [
                delayed(BiologicalDataset)(
                    os.path.join(root, "biological", reservoir),
                    transforms=self.clip
                ),
                delayed(LandCoverDataset)(
                    os.path.join(root, "land_use"),
                    transforms=self.water_mask
                )
            ]

            if self.load_processed:
                jobs.append(delayed(BiologicalDataset)(
                    os.path.join(root, "biological_processed", reservoir),
                    transforms=Compose([self.clip, self.normalize])
                ))

            datasets = Parallel(len(jobs), batch_size=1)(jobs)

            self.biological_unprocessed = datasets[0]
            self.water_use = datasets[1]

            if self.load_processed:
                self.biological_processed = datasets[2]

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        item = self.dataset[query]

        # Get the chlorophyll-a band
        ground_truth = item["image"][1].unsqueeze(0)
        if self.classify:
            ground_truth = self.bin(ground_truth)

        # Get the water mask
        water_mask = item["mask"].unsqueeze(0)

        # Early return if window size is zero
        if self.window_size == 0:
            return {"ground_truth": ground_truth, "water_mask": water_mask}

        mint = datetime.fromtimestamp(query.mint).replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = datetime.fromtimestamp(query.maxt).replace(hour=23, minute=59, second=59, microsecond=999999)

        samples = []
        for i in reversed(range(self.window_size)):
            dt = timedelta(days=self.prediction_horizon + i)
            t0 = (mint - dt).timestamp()
            t1 = (maxt - dt).timestamp()
            bbox = BoundingBox(query.minx, query.maxx, query.miny, query.maxy, t0, t1)
            if self.load_processed:
                # Get a preprocessed sample
                samples.append(self.biological_processed[bbox]["image"])
            elif any(True for _ in self.index.intersection(bbox)):
                # Get a real sample if it exists and normalize it
                samples.append(self.normalize(self.biological_unprocessed[bbox])["image"])
            else:
                # Generate a non-existent sample
                samples.append(torch.tensor([torch.nan]).expand_as(item["image"]))

        return {"images": torch.stack(samples), "ground_truth": ground_truth, "water_mask": water_mask}
