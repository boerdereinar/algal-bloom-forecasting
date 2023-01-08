from argparse import ArgumentParser
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from KDEpy import FFTKDE
from pytorch_lightning import LightningModule
from torch import Tensor, nn


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

        plt.plot(bins, w)
        plt.show()
