import os.path
from datetime import timedelta, datetime
from functools import partial
from typing import Dict, Any

import torch
from joblib import Parallel, delayed
from torchgeo.datasets import GeoDataset, BoundingBox
from torchvision.transforms import Compose
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset
from edegruyl.transforms import Normalize, Clip
from edegruyl.utils.tqdmutils import tqdm_joblib


class RioNegroDataset(GeoDataset):
    biological_unprocessed: BiologicalDataset
    biological_processed: BiologicalDataset

    def __init__(self, root: str, reservoir: str, window_size: int, prediction_horizon: int, **kwargs: Any):
        super().__init__()
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.clip = Clip(torch.tensor([8.74785, 240.943, 72.4036]))
        self.normalize = Normalize(torch.tensor([8.74785, 240.943, 72.4036]))

        # Load the datasets
        self.load_datasets(root, reservoir)

        # Update index
        self.index = self.biological_unprocessed.index

        # Description of bands
        self.all_bands = self.biological_unprocessed.all_bands

        # Prevent out of range samples
        dt = timedelta(days=window_size + prediction_horizon).total_seconds()
        roi = self.biological_unprocessed.bounds & self.biological_processed.bounds
        self.roi = BoundingBox(roi.minx, roi.maxx, roi.miny, roi.maxy, roi.mint + dt, roi.maxt)

    def load_datasets(self, root: str, reservoir: str):
        with tqdm_joblib(tqdm(desc="Loading datasets", total=2)):
            datasets = Parallel(4, batch_size=1)([
                delayed(BiologicalDataset)(
                    os.path.join(root, "biological", reservoir),
                    transforms=self.clip
                ),
                delayed(BiologicalDataset)(
                    os.path.join(root, "biological_processed", reservoir),
                    transforms=Compose([self.clip, self.normalize])
                )
            ])

            self.biological_processed = datasets[0]
            self.biological_unprocessed = datasets[1]

    def load_dataset(self, cls, name):
        return lambda *args, **kwargs: setattr(self, name, cls(*args, **kwargs))

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        ground_truth = self.biological_unprocessed[query]["image"][1].unsqueeze(0)

        mint = datetime.fromtimestamp(query.mint).replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = datetime.fromtimestamp(query.maxt).replace(hour=23, minute=59, second=59, microsecond=999999)

        samples = []
        for i in reversed(range(self.window_size)):
            dt = timedelta(days=self.prediction_horizon + i)
            t0 = (mint - dt).timestamp()
            t1 = (maxt - dt).timestamp()
            bbox = BoundingBox(query.minx, query.maxx, query.miny, query.maxy, t0, t1)
            samples.append(self.biological_processed[bbox]["image"])

        return {"images": torch.stack(samples), "ground_truth": ground_truth}
