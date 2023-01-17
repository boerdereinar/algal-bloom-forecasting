import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from edegruyl.datasets import RioNegroDataset
from edegruyl.models import DenseWeightModel, UNet
from edegruyl.utils.modelutils import extract_batch, mask_output
from edegruyl.utils.plotutils import figure_to_image, log_figure, plot_predicted, plot_rmse


class UNetModel(LightningModule):
    """A UNet-based classifier for image segmentation."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            patience: int = 3,
            dense_weight: Optional[str] = None,
            seed: int = 42,
            save_dir: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the UNet-model.

        Args:
            window_size: The size of the window that the classifier will use to look at the input data.
            num_bands: The number of bands (i.e. channels) in the input data.
            size: The size of the image that the classifier will be trained on.
            learning_rate: The learning rate of the optimizer.
            momentum: The momentum of the optimizer.
            patience: The number of epochs with no improvement after which learning rate will be reduced.
            dense_weight: The path to the checkpoint of the DenseWeightModel.
            seed: The seed for the global random state.
            save_dir: The save directory for the output of the test run.
            **kwargs:
        """
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        pl.seed_everything(seed)

        num_classes = len(RioNegroDataset.BINS) + 1

        in_channels = window_size * num_bands
        out_channels = 1
        self.model = UNet(in_channels, out_channels)

        self.dense_weight = dense_weight and DenseWeightModel.load_from_checkpoint(dense_weight)
        self.accuracy = Accuracy(num_classes=num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning-rate", type=float, help="The learning rate of the optimizer.", default=1e-3)
        parser.add_argument("--momentum", type=float, help="The momentum of the optimizer.", default=0.9)
        parser.add_argument("--patience", type=int, default=3, help="The number of epochs with no improvement after "
                                                                    "which learning rate will be reduced.")
        parser.add_argument("--dense-weight", type=str, help="The path to the checkpoint of the DenseWeightModel.")
        parser.add_argument("--seed", type=int, default=42, help="The seed for the global random state.")
        parser.add_argument("--save-dir", type=str, help="The save directory for the plots.")
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the UNet model.

        Args:
            x: The input tensor to the model, of shape (batch_size, window_size, num_bands, size, size).

        Returns:
            Tensor: The output of the model, of shape (batch_size, 1, size, size).
        """
        x = torch.flatten(x, 1, 2)
        x = self.model(x)

        return x

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            patience=self.hparams.patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def loss(self, predicted: Tensor, expected: Tensor) -> Tuple[Tensor, Tensor]:
        mse = mse_loss(predicted, expected).nan_to_num()

        if self.dense_weight is not None:
            return self.dense_weight.forward(predicted, expected), mse

        return mse, mse

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(train_batch)
        y_hat = self.forward(x)

        y_hat, y = mask_output(y_hat, y, observed)
        loss, loss_to_log = self.loss(y_hat, y)

        self.log("train_loss", loss_to_log)

        return loss

    def validation_step(self, val_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(val_batch)
        y_hat = self.forward(x)

        y_hat, y = mask_output(y_hat, y, observed)
        loss, loss_to_log = self.loss(y_hat, y)

        self.log("val_loss", loss_to_log)

        return loss

    def test_step(self, test_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, water_mask, observed_y = extract_batch(test_batch)
        y_hat = self.forward(x)
        y_hat[~water_mask] = 0

        # Log images
        for i in range(len(y)):
            fig = plot_predicted(y_hat[i, 0], y[i, 0])
            path = self.hparams.save_dir and os.path.join(self.hparams.save_dir, f"test_{batch_idx}_{i}.png")
            log_figure(fig, self.logger, "test_predicted", path)

        # Compute per-element squared error
        squared_error = torch.full_like(y, torch.nan)
        squared_error[observed_y] = (y[observed_y] - y_hat[observed_y]) ** 2

        return squared_error

    def test_epoch_end(self, outputs: List[Tensor]) -> None:
        outputs = torch.cat(outputs)[:, 0]
        rmse = torch.nanmean(outputs, 0).nan_to_num().sqrt()
        global_rmse = torch.nanmean(outputs).nan_to_num().sqrt()

        fig = plot_rmse(rmse, global_rmse)
        _ = figure_to_image(fig)
        path = os.path.join(self.hparams.save_dir, "rmse_loss_validation.png")  # type: ignore
        log_figure(fig, self.logger, "test_rmse", path)
