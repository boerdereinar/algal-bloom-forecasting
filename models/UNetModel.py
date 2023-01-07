import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb
from matplotlib.colors import LogNorm
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

from edegruyl.datasets import RioNegroDataset
from edegruyl.models import UNet
from edegruyl.utils.modelutils import extract_batch, mask_output


class UNetModel(LightningModule):
    """A UNet-based classifier for image segmentation."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            learning_rate: float = 1e-4,
            momentum: float = 0.9,
            patience: int = 3,
            classify: bool = False,
            seed: int = 42,
            save_dir: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the UNet-classifier.

        Args:
            window_size: The size of the window that the classifier will use to look at the input data.
            num_bands: The number of bands (i.e. channels) in the input data.
            size: The size of the image that the classifier will be trained on.
            learning_rate: The learning rate of the optimizer.
            momentum: The momentum of the optimizer.
            patience: The number of epochs with no improvement after which learning rate will be reduced.
            classify: Whether to use classification instead of regression.
            seed: The seed for the global random state.
            save_dir: The save directory for the output of the test run.
            **kwargs:
        """
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        pl.seed_everything(seed)

        num_classes = len(RioNegroDataset.BINS) + 1

        in_channels = window_size * num_bands
        out_channels = num_classes if classify else 1
        self.model = UNet(in_channels, out_channels)

        self.accuracy = Accuracy(num_classes=num_classes)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--learning-rate", type=float, help="The learning rate of the optimizer.", default=1e-4)
        parser.add_argument("--momentum", type=float, help="The momentum of the optimizer.", default=0.9)
        parser.add_argument("--patience", type=int, default=3, help="The number of epochs with no improvement after "
                                                                    "which learning rate will be reduced.")
        parser.add_argument("--seed", type=int, default=42, help="The seed for the global random state.")
        parser.add_argument("--save-dir", type=str, help="The save directory for the plots.", default=None)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the UNet model.

        Args:
            x: The input tensor to the model, of shape (batch_size, window_size, num_bands, size, size).

        Returns:
            Tensor: The output of the model, of shape (batch_size, 1, size, size).
        """
        x = torch.flatten(x, 1, 2)
        return self.model(x)

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

    def loss(self, predictions: Tensor, expected: Tensor) -> Tensor:
        if self.hparams.classify:
            return cross_entropy(predictions, expected).nan_to_num()

        return mse_loss(predictions, expected).nan_to_num()

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(train_batch, classify=self.hparams.classify)
        y_hat = self(x)

        y_hat, y = mask_output(y_hat, y, observed, self.hparams.classify)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss)
        if self.hparams.classify:
            accuracy = self.accuracy(y_hat, y) if y_hat.numel() > 0 else torch.tensor(0.).to(x)
            self.log("train_accuracy", accuracy, prog_bar=True)

        return loss

    def validation_step(self, val_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(val_batch, classify=self.hparams.classify)
        y_hat = self(x)

        y_hat, y = mask_output(y_hat, y, observed, self.hparams.classify)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss)
        if self.hparams.classify:
            accuracy = self.accuracy(y_hat, y) if y_hat.numel() > 0 else torch.tensor(0.).to(x)
            self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def test_step(self, test_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(test_batch, classify=self.hparams.classify)
        y_hat = self(x)

        if self.hparams.classify:
            raise NotImplementedError()

        # Log images
        logger = self.logger
        if isinstance(logger, WandbLogger):
            for i in range(len(y)):
                logger.log_table(
                    "test_predicted",
                    ["Predicted", "Expected"],
                    [[wandb.Image(y_hat[i], "F"), wandb.Image(y[i], "F")]]
                )
        elif self.hparams.save_dir:
            for i in range(len(y)):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
                ax1.set_title("predicted")
                im = ax1.imshow(y_hat[i, 0].cpu(), vmin=0, vmax=RioNegroDataset.CLIP[1], interpolation=None)
                ax2.set_title("expected")
                ax2.imshow(y[i, 0].cpu(), vmin=0, vmax=RioNegroDataset.CLIP[1], interpolation=None)

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
        squared_error[torch.where(observed)] = (y[observed] - y_hat[observed]) ** 2

        return squared_error

    def test_epoch_end(self, outputs: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(outputs, List):
            outputs = torch.cat(outputs)[:, 0]
        rmse = torch.nanmean(outputs, 0).sqrt()

        logger = self.logger
        if isinstance(logger, WandbLogger):
            logger.log_image("test_rmse", [wandb.Image(rmse, "F", "RMSE loss")])
        elif self.hparams.save_dir:  # type: ignore
            plt.figure(figsize=(8, 4))
            plt.title("RMSE loss")
            plt.imshow(rmse.cpu(), norm=LogNorm())
            plt.colorbar()
            path = os.path.join(self.hparams.save_dir, "rmse_loss_validation.png")  # type: ignore
            plt.savefig(path, transparent=True)
