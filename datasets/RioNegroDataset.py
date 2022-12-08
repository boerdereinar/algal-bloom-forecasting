import os.path
import threading
import time
from datetime import timedelta, datetime
from typing import Dict, Any

import torch
from torchgeo.datasets import GeoDataset, BoundingBox
from torchvision.transforms import Compose
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset
from edegruyl.transforms import Normalize


class RioNegroDataset(GeoDataset):
    biological_unprocessed: BiologicalDataset
    biological_processed: BiologicalDataset

    def __init__(self, root: str, reservoir: str, window_size: int, prediction_horizon: int, **kwargs: Any):
        super().__init__()
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
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
        self.roi = BoundingBox(*roi[:4], roi.mint + dt, roi.maxt)

    def load_datasets(self, root: str, reservoir: str):
        print("Loading datasets...")

        loaders = [
            self.load_biological_unprocessed,
            self.load_biological_processed
        ]

        # Load the datasets whilst updating the timer of the progress bar
        for loader in (pbar := tqdm(loaders)):
            x = threading.Thread(target=loader, args=[pbar, root, reservoir])
            x.start()
            while x.is_alive():
                time.sleep(0.1)
                pbar.refresh()
            x.join()

    def load_biological_unprocessed(self, pbar: tqdm, root: str, reservoir: str):
        pbar.set_description("Unprocessed biological dataset")
        self.biological_unprocessed = BiologicalDataset(
            os.path.join(root, "biological", reservoir),
            transforms=Compose([self.normalize])
        )

    def load_biological_processed(self, pbar: tqdm, root: str, reservoir: str):
        pbar.set_description("Processed biological dataset")
        self.biological_processed = BiologicalDataset(
            os.path.join(root, "biological_processed", reservoir),
            transforms=Compose([self.normalize])
        )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        ground_truth = self.biological_unprocessed[query]["image"][1]

        mint = datetime.fromtimestamp(query.mint).replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = datetime.fromtimestamp(query.maxt).replace(hour=23, minute=59, second=59, microsecond=999999)

        samples = []
        for i in reversed(range(self.window_size)):
            dt = timedelta(days=self.prediction_horizon + i)
            bbox = BoundingBox(*query[:4], (mint - dt).timestamp(), (maxt - dt).timestamp())
            samples.append(self.biological_processed[bbox]["image"])

        return {"images": torch.stack(samples), "ground_truth": ground_truth}
