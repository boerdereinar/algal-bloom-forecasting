import os.path
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
from rasterio.windows import from_bounds
from tqdm import tqdm

from edegruyl.datasets import BiologicalDataset, RioNegroDataset


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
    thresh: np.ndarray
    thresh_stats: Dict[str, np.ndarray]
    thresh_hists: Sequence[Tuple[np.ndarray, np.ndarray]]

    def __init__(
            self,
            root: str,
            reservoir: str,
            land_cover: str,
            save_dir: Optional[str] = None,
            save_plots: bool = False,
            exclude_titles: bool = False,
            **kwargs: Any
    ):
        """Constructs a new `Analyser` object.

        Args:
            root: The root directory where the data is stored.
            reservoir: The specific reservoir of data to analyze.
            land_cover: The path to the land coverage tif file.
            save_dir: The save directory.
            save_plots: Whether to save the plots.
            exclude_titles: Whether to exclude the titles from the plots.
        """
        self.reservoir = reservoir
        self.save_dir = save_dir
        self.save_plots = save_plots
        self.exclude_titles = exclude_titles

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
        parent_parser.add_argument("--save-dir", type=str, help="The save directory for the plots.")
        parent_parser.add_argument("--save-plots", action="store_true", help="Whether to save the plots.")
        parent_parser.add_argument("--exclude-titles", action="store_true",
                                   help="Whether to exclude the titles from the plots.")
        return parent_parser

    def analyse(self):
        """Performs the data analysis.

        This method calculates various statistics and histograms from the data
        in the `dataset` attribute, and stores the results in the `measurements`, `stats`,
        `hists`, `thresh`, `thresh_stats`, and `thresh_hists` attributes.
        """
        intersections = list(self.dataset.index.intersection(self.dataset.index.bounds, objects=True))
        data = Parallel(8)(delayed(self._load_file)(item.object) for item in tqdm(intersections))
        data = np.stack(data, axis=1)  # type: ignore
        data = data[:, :, self.water_coverage]

        self.total_samples = data.shape[1]
        self.measurements = np.count_nonzero(~np.isnan(data), axis=1)

        ax = (1, 2)
        self.stats = {
            "Total": (total := np.repeat(data[0].size, len(data))),
            "Non-NaN Values": (non_nan := np.count_nonzero(self.measurements, axis=1)),
            "Non-NaN Ratio": non_nan / total,
            "Min": np.nanmin(data, axis=ax),
            "Max": np.nanmax(data, axis=ax),
            "Mean": np.nanmean(data, axis=ax),
            "Median": np.nanmedian(data, axis=ax),
            "Standard Deviation": np.nanstd(data, axis=ax)
        }

        self.hists = [
            np.histogram(x[~np.isnan(x)], bins=np.geomspace(1, self.stats["Max"][i], 50))
            for i, x in enumerate(data)
        ]

        # Threshold
        self.thresh = self.stats["Mean"] + 3.5 * self.stats["Standard Deviation"]
        below_thresh = data <= self.thresh[:, None, None]
        above_thresh = data > self.thresh[:, None, None]

        self.thresh_stats = {
            "Threshold": self.thresh,
            "Above Threshold": np.count_nonzero(above_thresh, axis=ax),
            "Min": np.min(data, axis=ax, initial=0, where=below_thresh),
            "Max": np.max(data, axis=ax, initial=0, where=below_thresh),
            "Mean": np.mean(data, axis=ax, where=below_thresh),
            "Standard Deviation": np.std(data, axis=ax, where=below_thresh)
        }

        self.thresh_hists = [np.histogram(x[below_thresh[i]], bins=100)
                             for i, x in enumerate(data)]

        # Styling
        # plt.rc("font", size=20)
        # plt.rc("axes", linewidth=1)
        # plt.rc("xtick.major", size=10)
        # plt.rc("xtick.minor", size=5)

        # self.print_summary()
        self.plot_histograms()
        self.plot_sparsity()

    def print_summary(self):
        """Prints a summary of the analysis results to the console."""
        for i, band in enumerate(self.dataset.all_bands):
            print(f"???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????")
            print(f"???{            band.capitalize()        :^43}???")
            print(f"???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????")
            xs = (f"??? {key           :<20} {value[i]     :<20g} ???" for key, value in self.stats.items())
            print("\n".join(xs))
            print(f"???????????????????????????????????????????????? Thresholded ????????????????????????????????????????????????")
            ys = (f"??? {key           :<20} {value[i]     :<20g} ???" for key, value in self.thresh_stats.items())
            print("\n".join(ys))
            print(f"???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????")

    def plot_histograms(self):
        """Plots the histograms of the analysis results using Matplotlib."""
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=200)
        plt.ylabel("density")
        ax.set_xscale("log")

        for i, band in enumerate(self.dataset.all_bands):
            h, bins = self.hists[i]
            h = h / h.max(initial=1)
            bins = (bins[1:] + bins[:-1]) / 2

            p = plt.plot(bins, h, label=band)
            plt.axvline(RioNegroDataset.CLIP[i], color=p[0].get_color(), linestyle="dashed")

        plt.legend()
        fig.tight_layout()

        if self.save_plots and self.save_dir:
            plt.savefig(os.path.join(self.save_dir, f"{self.reservoir}_hist.png"))
        plt.show()

    def plot_sparsity(self):
        """Plots the temporal and spatial sparsity of the dataset using Matplotlib."""
        dates = sorted(
            datetime.fromtimestamp(item.bounds[4]).replace(hour=0, minute=0, second=0, microsecond=0)  # type: ignore
            for item in self.dataset.index.intersection(self.dataset.index.bounds, True)
        )

        # Temporal sparsity
        dt = [(d2 - d1).days for d1, d2 in zip(dates, dates[1:])]
        binned_dt = np.bincount(dt)

        # Plot temporal/spatial sparsity
        for i, band in enumerate(self.dataset.all_bands):
            # Spatial sparsity
            missing_samples = np.ma.masked_where(~self.water_coverage, np.zeros(self.water_coverage.shape))
            missing_samples[np.where(self.water_coverage)] = self.total_samples - self.measurements[i]

            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[2, 3], figsize=(8, 3), dpi=200)

            if not self.exclude_titles:
                ax1.set_title("Temporal sparsity")
                ax2.set_title(f"Spatial sparsity ({band.capitalize()})")

            ax1.set_xlabel("days")
            ax1.set_ylabel("samples")
            ax1.bar(range(len(binned_dt)), binned_dt)

            ax2.set_xticks([])
            ax2.set_yticks([])
            im = ax2.imshow(missing_samples, norm=LogNorm(10))

            fig.tight_layout()
            fig.subplots_adjust(right=0.92)
            cax = fig.add_axes([0.92, 0.195, 0.02, 0.685])
            fig.colorbar(im, cax=cax)

            if self.save_plots and self.save_dir:
                path = os.path.join(self.save_dir, f"{self.reservoir}_{band}_spatial_temporal_sparsity.png")
                plt.savefig(path, transparent=True)
            fig.show()

    @staticmethod
    def _load_file(file: str):
        with rasterio.open(file) as src:
            return src.read()
