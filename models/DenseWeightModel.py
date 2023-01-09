from argparse import ArgumentParser
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from KDEpy import FFTKDE
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from edegruyl.datasets import RioNegroDataset


class DenseWeightModel(LightningModule):
    """
    Model for training the DenseWeight loss function.

    Based on the implementation at https://github.com/SteiMi/denseweight.
    """
    def __init__(self, alpha: float = 1.0, res: int = 1024, eps: float = 1e-6, **kwargs: Any) -> None:
        """
        Args:
            alpha: Alpha value for DenseWeight. Adjusts the intensity of density-based weighting.
            res: The resolution of the grid returned by the KDE function.
            eps: Epsilon for DenseWeight. Sets the minimum weight a data point can receive.
        """
        super().__init__()

        self.save_hyperparameters(ignore=list(kwargs))

        # Turn off automatic optimization
        self.automatic_optimization = False

        self.bins = nn.Parameter(torch.empty((res,)), False)
        self.weights = nn.Parameter(torch.empty((res,)), False)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--alpha", type=float, default=1., help="Alpha value for DenseWeight. Adjusts the "
                                                                    "intensity of density-based weighting.")
        parser.add_argument("--res", type=int, default=1024, help="The resolution of the grid returned by the KDE "
                                                                  "function.")
        parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for DenseWeight. Sets the minimum weight "
                                                                    "a data point can receive.")
        return parent_parser

    def forward(self, predicted: Tensor, expected: Tensor) -> Tensor:
        indexes = torch.clamp(torch.bucketize(expected, self.bins), 0, self.hparams.res - 1)
        weights = self.weights[indexes]

        weighted_mse = ((expected - predicted) ** 2 * weights).mean()
        return weighted_mse.nan_to_num()

    def configure_optimizers(self) -> Any:
        return None

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        self.optimizers().step()

        ground_truth = train_batch["ground_truth"]
        observed = ~torch.isnan(ground_truth)
        return {"observed_y": ground_truth[observed]}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        data = torch.cat([x["observed_y"] for x in outputs]).numpy()

        kernel = FFTKDE(bw="silverman")
        kernel.fit(data)

        bins, p = kernel.evaluate(self.hparams.res)

        # Normalize between 0 and 1
        min_p = p.min()
        max_p = p.max()
        p = (p - min_p) / (max_p - min_p)

        w = np.maximum(1 - self.hparams.alpha * p, self.hparams.eps)
        mean_w = w.mean()
        w /= mean_w

        self.bins = nn.Parameter(torch.tensor(bins), False)
        self.weights = nn.Parameter(torch.tensor(w), False)

        # Plot the relevance and chlorophyll-a density
        h, _ = np.histogram(data, bins, density=True)
        x1 = bins * RioNegroDataset.CLIP[1].item()
        x2 = (bins[1:] + bins[:-1]) / 2 * RioNegroDataset.CLIP[1].item()

        fig, ax1 = plt.subplots(dpi=200, figsize=(6, 3))
        ax2 = ax1.twinx()

        # Axis configuration
        ax1.set_xlim([1, RioNegroDataset.CLIP[1].item()])
        ax1.set_xscale("log")
        ax1.set_xlabel("concentration (μg/L)")

        # Plot relevance
        ax1.set_ylabel("relevance", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        p1, = ax1.plot(x1, w, color="tab:red", label="DenseWeight")

        # Plot chlorophyll-a density
        ax2.set_ylabel("density", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        p2, = ax2.plot(x2, h, color="tab:blue", label="Chlorophyll-A")

        ax1.legend(handles=[p1, p2], loc="center right")
        fig.tight_layout()
        plt.show()
