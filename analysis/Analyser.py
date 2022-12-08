import os.path
from argparse import ArgumentParser
from datetime import datetime
from typing import Sequence, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
from rasterio.windows import from_bounds
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset


class Analyser:
    """A class for analyzing data from a biological dataset.

    Attributes:
        total_samples: The total number of measurements.
        stats: A dictionary containing the calculated statistics. The keys are
            the names of the statistics and the values are NumPy arrays with
            the corresponding values.
        measurements: A NumPy array containing the total number of measurements
            at each pixel location.
        hists: A list of tuples, where each tuple contains the binned data and
            the bin edges for a histogram.
        thresh: A NumPy array containing the calculated threshold values.
        thresh_stats: A dictionary containing the statistics calculated using
            the threshold values. The keys and values have the same meaning
            as in the `stats` attribute.
        thresh_hists: A list of tuples containing the histogram data calculated
            using the threshold values. The tuples have the same format as in
            the `hists` attribute.
    """
    total_samples: int
    stats: Dict[str, np.ndarray]
    measurements: np.ndarray
    hists: Sequence[Tuple[np.ndarray, np.ndarray]]
    thresh: Sequence[float]
    thresh_stats: Dict[str, np.ndarray]
    thresh_hists: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(self, root: str, reservoir: str, land_cover: str):
        """Constructs a new `Analyser` object.

        Args:
            root: The root directory where the data is stored.
            reservoir: The specific reservoir of data to analyze.
            land_cover: The path to the land coverage tif file.
        """
        path = os.path.join(root, "biological", reservoir)
        self.dataset = BiologicalDataset(path)

        # Load the land coverage
        with rasterio.open(land_cover) as src:
            bounds = self.dataset.bounds
            out_width = round((bounds.maxx - bounds.minx) / self.dataset.res)
            out_height = round((bounds.maxy - bounds.miny) / self.dataset.res)
            out_shape = (1, out_height, out_width)
            window = from_bounds(bounds.minx, bounds.miny, bounds.maxx, bounds.maxy, src.transform)
            coverage = src.read(out_shape=out_shape, window=window)[0]
            self.water_coverage = coverage == 33

    @staticmethod
    def add_analyser_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments specific to the `Analyser` class to an existing `ArgumentParser` object.

        Args:
            parent_parser: The `ArgumentParser` object to which the new arguments should be added.

        Returns:
            The modified `ArgumentParser` object.
        """
        parent_parser.add_argument("root", type=str, help="The root directory.")
        parent_parser.add_argument("reservoir", type=str, help="The reservoir to analyse.")
        parent_parser.add_argument("land_cover", type=str, help="The path to the land coverage tif file.")
        return parent_parser

    def analyse(self):
        """Performs the data analysis.

        This method calculates various statistics and histograms from the data
        in the `dataset` attribute, and stores the results in the `measurements`, `stats`,
        `hists`, `thresh`, `thresh_stats`, and `thresh_hists` attributes.
        """
        intersections = list(self.dataset.index.intersection(self.dataset.index.bounds, objects=True))
        data = Parallel(8)(delayed(self._load_file)(item.object) for item in tqdm(intersections))
        data = np.stack(data, axis=1)
        is_not_nan = ~np.isnan(data)

        self.total_samples = data.shape[1]
        self.measurements = np.count_nonzero(is_not_nan, axis=1)

        ax = (1, 2, 3)
        self.stats = {
            "Total": (total := np.repeat(data[0].size, len(data))),
            "Non-NaN Values": (non_nan := np.count_nonzero(is_not_nan, axis=ax)),
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
        self.plot_sparsity()

    def print_summary(self):
        """Prints a summary of the analysis results to the console."""
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
        """Plots the histograms of the analysis results using Matplotlib."""
        for i, band in enumerate(self.dataset.all_bands):
            fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
            fig.suptitle(band.capitalize())
            ax1.stairs(*self.hists[i])

            ax2.set_title(fr"Threshold = {self.thresh[i]:.5f} (3.5$\sigma$)")
            ax2.stairs(*self.thresh_hists[i])
            plt.show()

    def plot_sparsity(self):
        """Plots the temporal and spatial sparsity of the dataset using Matplotlib."""
        dates = sorted(datetime.fromtimestamp(item.bounds[4]).replace(hour=0, minute=0, second=0, microsecond=0)
                       for item in self.dataset.index.intersection(self.dataset.index.bounds, True))

        # Plot temporal sparsity
        dt = [(d2 - d1).days for d1, d2 in zip(dates, dates[1:])]
        binned_dt = np.bincount(dt)
        plt.title("Days between samples")
        plt.xlabel("days")
        plt.ylabel("number of samples")
        plt.bar(range(len(binned_dt)), binned_dt)
        plt.show()

        # Plot spatial sparsity
        for i, band in enumerate(self.dataset.all_bands):
            masked = np.ma.masked_where(~self.water_coverage, self.total_samples - self.measurements[i])
            plt.title(f"{band.capitalize()} missing spatial samples")
            plt.imshow(masked, norm=LogNorm(), cmap="jet")
            plt.colorbar()
            plt.show()

    @staticmethod
    def _load_file(file: str):
        with rasterio.open(file) as src:
            return src.read()
