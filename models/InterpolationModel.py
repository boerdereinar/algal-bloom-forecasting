import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import SGD

from edegruyl.models import DenseWeightModel, UNet
from edegruyl.models.InterpolationLayer import InterpolationNetwork, reshape_input, reshape_output
from edegruyl.utils.modelutils import extract_batch, mask_output


class InterpolationModel(LightningModule):
    """Interpolation model."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            masked: bool = False,
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            patience: int = 3,
            dense_weight: Optional[str] = None,
            seed: int = 42,
            save_dir: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the Interpolation-model.

        Args:
            window_size: The size of the window that the classifier will use to look at the input data.
            num_bands: The number of bands (i.e. channels) in the input data.
            size: The size of the image that the classifier will be trained on.
            masked: Whether the interpolation is only applied to the area masked by the water mask.
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

        self.interp_net = InterpolationNetwork(num_bands, window_size, window_size)

        in_channels = 3 * window_size * num_bands
        self.model = UNet(in_channels, 1)

        self.dense_weight = dense_weight and DenseWeightModel.load_from_checkpoint(dense_weight)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--masked", action="store_true", help="Whether the interpolation is only applied to the "
                                                                  "area masked by the water mask.")
        parser.add_argument("--learning-rate", type=float, help="The learning rate of the optimizer.", default=1e-3)
        parser.add_argument("--momentum", type=float, help="The momentum of the optimizer.", default=0.9)
        parser.add_argument("--patience", type=int, default=3, help="The number of epochs with no improvement after "
                                                                    "which learning rate will be reduced.")
        parser.add_argument("--dense-weight", type=str, help="The path to the checkpoint of the DenseWeightModel.")
        parser.add_argument("--seed", type=int, default=42, help="The seed for the global random state.")
        parser.add_argument("--save-dir", type=str, help="The save directory for the plots.", default=None)
        return parent_parser

    def forward(self, x: Tensor, water_mask: Tensor, observed_x: Tensor) -> Tensor:
        """
        Args:
            x: The images tensor of shape (batch, window_size, num_bands, height, width)
            water_mask: The water mask tensor of shape (batch, height, width)
            observed_x: A tensor of shape (batch_size, window_size, bands, height, width) indicating which values in
                the images tensor are observed (not NaN).

        Returns:
            The predicted tensor of shape (batch, 1, height, width).
        """
        batch_size, observed_points, num_features, height, width = x.shape
        water_mask_indices = torch.where(water_mask)

        # Reshape input
        x_t, d, m = reshape_input(x, water_mask_indices, observed_x, self.hparams.masked)

        # Feed into interpolation network
        y_interp = self.interp_net(x_t, d, m)

        # Reshape interpolated output
        y_interp = reshape_output(y_interp, water_mask_indices, batch_size, height, width, self.hparams.masked)
        y_interp = y_interp.reshape(batch_size, -1, height, width)

        # Feed into UNet
        y_hat = self.model(y_interp)
        y_hat = torch.sigmoid(y_hat)

        return y_hat

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

    def loss(self, predicted: Tensor, expected: Tensor) -> Tensor:
        if self.dense_weight is not None:
            return self.dense_weight.forward(predicted, expected)

        return mse_loss(predicted, expected).nan_to_num()

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y, water_mask, observed_x, observed_y = extract_batch(batch, True)
        batch_size, observed_points, num_features, height, width = x.shape
        water_mask_indices = torch.where(water_mask)

        # Reshape input
        x_t, d, m = reshape_input(x, water_mask_indices, observed_x, self.hparams.masked)

        # Hold out 20% of the input data for reconstruction
        m_holdout = (torch.rand(m.shape).to(x) > 0.2) & m
        held_out = m_holdout ^ m

        # Feed into interpolation and reconstruction network
        y_interp = self.interp_net(x_t, d, m)
        y_reconst = self.interp_net(x_t, d, m_holdout, True)

        # Compute reconstruction loss
        reconst_loss = mse_loss(y_reconst[held_out], x_t[held_out])

        # Reshape interpolated output
        y_interp = reshape_output(y_interp, water_mask_indices, batch_size, height, width, self.hparams.masked)
        y_interp = y_interp.reshape(batch_size, -1, height, width)

        # Feed into UNet
        y_hat = self.model(y_interp)
        y_hat = torch.sigmoid(y_hat)

        # Compute prediction loss
        y_hat, y = mask_output(y_hat, y, observed_y)
        loss = self.loss(y_hat, y)

        return reconst_loss, loss

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        reconst_loss, loss = self.compute_loss(train_batch)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_reconst_loss", reconst_loss, prog_bar=True)

        return reconst_loss + loss

    def validation_step(self, val_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        reconst_loss, loss = self.compute_loss(val_batch)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_reconst_loss", reconst_loss, prog_bar=True)

        return reconst_loss + loss

    def test_step(self, test_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, water_mask, observed_x, observed_y = extract_batch(test_batch, True)
        y_hat = self.forward(x, water_mask, observed_x)

        # Log images
        logger = self.logger
        if isinstance(logger, WandbLogger):
            cm = plt.get_cmap("viridis")
            for i in range(len(y)):
                predicted = np.pad(cm(y_hat[i, 0].cpu(), bytes=True), ((2,), (2,), (0,)), constant_values=255)
                expected = np.pad(cm(y[i, 0].cpu(), bytes=True), ((2,), (2,), (0,)), constant_values=255)
                img = np.hstack((predicted, expected))
                logger.log_image("test_predicted", [wandb.Image(img)])
        elif self.hparams.save_dir:
            for i in range(len(y)):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
                ax1.set_title("predicted")
                im = ax1.imshow(y_hat[i, 0].cpu(), vmin=0, vmax=1, interpolation=None)
                ax2.set_title("expected")
                ax2.imshow(y[i, 0].cpu(), vmin=0, vmax=1, interpolation=None)

                # Add colorbar
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                # Save figure
                fig.savefig(os.path.join(self.hparams.save_dir, f"test_{batch_idx}_{i}.png"))
                plt.close()

        # Compute per-element squared error
        squared_error = torch.empty_like(y)
        squared_error[:] = torch.nan
        squared_error[torch.where(observed_y)] = (y[observed_y] - y_hat[observed_y]) ** 2

        return squared_error

    def test_epoch_end(self, outputs: List[Tensor]) -> None:
        outputs = torch.cat(outputs)[:, 0]
        rmse = torch.nanmean(outputs, 0).sqrt()

        logger = self.logger
        if isinstance(logger, WandbLogger):
            cm = plt.get_cmap("viridis")
            norm = rmse / rmse.max()
            logger.log_image("test_rmse", [wandb.Image(cm(norm.cpu(), bytes=True), caption="RMSE loss")])
        elif self.hparams.save_dir:  # type: ignore
            plt.figure(figsize=(8, 4))
            plt.title("RMSE loss")
            plt.imshow(rmse.cpu(), norm=LogNorm())
            plt.colorbar()
            path = os.path.join(self.hparams.save_dir, "rmse_loss_validation.png")  # type: ignore
            plt.savefig(path, transparent=True)
