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
from edegruyl.utils.tqdmutils import tqdm_joblib


class RioNegroDataset(GeoDataset):
    biological_unprocessed: BiologicalDataset
    biological_processed: Optional[BiologicalDataset] = None
    water_use: LandCoverDataset

    def __init__(
            self,
            root: str,
            reservoir: str,
            window_size: int,
            prediction_horizon: int,
            load_processed: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.load_processed = load_processed
        self.clip = Clip(torch.tensor([20, 150, 100]))
        self.normalize = Normalize(torch.tensor([20, 150, 100]))
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

    def load_datasets(self, root: str, reservoir: str):
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

    def load_dataset(self, cls, name):
        return lambda *args, **kwargs: setattr(self, name, cls(*args, **kwargs))

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        item = self.dataset[query]
        ground_truth = item["image"][1].unsqueeze(0)
        water_mask = item["mask"].unsqueeze(0)

        mint = datetime.fromtimestamp(query.mint).replace(hour=0, minute=0, second=0, microsecond=0)
        maxt = datetime.fromtimestamp(query.maxt).replace(hour=23, minute=59, second=59, microsecond=999999)

        samples = []
        for i in reversed(range(self.window_size)):
            dt = timedelta(days=self.prediction_horizon + i)
            t0 = (mint - dt).timestamp()
            t1 = (maxt - dt).timestamp()
            bbox = BoundingBox(query.minx, query.maxx, query.miny, query.maxy, t0, t1)
            if self.biological_processed:
                samples.append(self.biological_processed[bbox]["image"])
            elif any(True for _ in self.index.intersection(bbox)):
                samples.append(self.biological_unprocessed[bbox]["image"])
            else:
                samples.append(torch.full_like(item["image"], torch.nan))

        return {"images": torch.stack(samples), "ground_truth": ground_truth, "water_mask": water_mask}
