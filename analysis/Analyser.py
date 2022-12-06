import os.path
from argparse import ArgumentParser
from typing import Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from joblib import Parallel, delayed
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset


class Analyser:
    stats: Dict[str, np.ndarray]
    hists: Sequence[Tuple[np.ndarray, np.ndarray]]
    thresh: Sequence[float]
    thresh_stats: Dict[str, np.ndarray]
    thresh_hists: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(self, root: str, reservoir: str):
        path = os.path.join(root, "biological", reservoir)
        self.dataset = BiologicalDataset(path)

    @staticmethod
    def add_analyser_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser.add_argument("root", type=str, help="The root directory.")
        parent_parser.add_argument("reservoir", type=str, help="The reservoir to analyse.")
        return parent_parser

    def analyse(self):
        intersections = list(self.dataset.index.intersection(self.dataset.index.bounds, objects=True))
        data = Parallel(8)(delayed(self._load_file)(item.object) for item in tqdm(intersections))
        data = np.stack(data, axis=1)

        ax = (1, 2, 3)
        self.stats = {
            "Total": (total := np.repeat(data[0].size, len(data))),
            "Non-NaN Values": (non_nan := np.count_nonzero(~np.isnan(data), axis=ax)),
            "Non-NaN Ratio": non_nan / total,
            "Min": np.nanmin(data, axis=ax),
            "Max": np.nanmax(data, axis=ax),
            "Mean": np.nanmean(data, axis=ax),
            "Median": np.nanmedian(data, axis=ax),
            "Standard Deviation": np.nanstd(data, axis=ax)
        }

        self.hists = [np.histogram(x[~np.isnan(x)], bins=100) for x in data]

        # Threshold
        self.thresh = self.stats["Mean"] + 3.5 * self.stats["Standard Deviation"]
        below_thresh = data <= self.thresh[:, None, None, None]
        above_thresh = data > self.thresh[:, None, None, None]

        self.thresh_stats = {
            "Threshold": self.thresh,
            "Above Threshold": np.count_nonzero(above_thresh, axis=ax),
            "Min": np.min(data, axis=ax, initial=0, where=below_thresh),
            "Max": np.max(data, axis=ax, initial=0, where=below_thresh),
            "Mean": np.mean(data, axis=ax, where=below_thresh),
            "Standard Deviation": np.std(data, axis=ax, where=below_thresh)
        }

        self.thresh_hists = [np.histogram(x[below_thresh[i]], bins=100) for i, x in enumerate(data)]

        self.print_summary()
        self.plot_histograms()

    def print_summary(self):
        for i, band in enumerate(self.dataset.all_bands):
            print(f"╔═══════════════════════════════════════════╗")
            print(f"║{            band.capitalize()        :^43}║")
            print(f"╠═══════════════════════════════════════════╣")
            xs = (f"║ {key           :<20} {value[i]     :<20g} ║" for key, value in self.stats.items())
            print("\n".join(xs))
            print(f"╠═══════════════ Thresholded ═══════════════╣")
            ys = (f"║ {key           :<20} {value[i]     :<20g} ║" for key, value in self.thresh_stats.items())
            print("\n".join(ys))
            print(f"╚═══════════════════════════════════════════╝")

    def plot_histograms(self):
        for i, band in enumerate(self.dataset.all_bands):
            fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
            fig.suptitle(band.capitalize())
            ax1.stairs(*self.hists[i])

            ax2.set_title(fr"Threshold = {self.thresh[i]:.5f} (3.5$\sigma$)")
            ax2.stairs(*self.thresh_hists[i])
            plt.show()

    @staticmethod
    def _load_file(file: str):
        with rasterio.open(file) as src:
            return src.read()
