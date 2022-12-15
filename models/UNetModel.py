from argparse import ArgumentParser
from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim import Adam, Optimizer

from edegruyl.models import UNet


class UNetModel(LightningModule):
    """A UNet-based classifier for image segmentation."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            learning_rate: float = 1e-4,
            **kwargs: Any
    ) -> None:
        """
        Initialize the UNet-classifier.

        Args:
            window_size (int): The size of the window that the classifier will use to look at the input data.
            num_bands (int): The number of bands (i.e. channels) in the input data.
            size (int): The size of the image that the classifier will be trained on.
            learning_rate (float): The learning rate to use for training.
            **kwargs:
        """
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        in_channels = window_size * num_bands
        self.model = UNet(in_channels, 1)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("-lr", "--learning-rate", type=float, help="The learning rate of the model.", default=1e-4)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the UNet model.

        Args:
            x (Tensor): The input tensor to the model, of shape (batch_size, window_size, num_bands, size, size).

        Returns:
            Tensor: The output of the model, of shape (batch_size, 1, size, size).
        """
        x = torch.flatten(x, 1, 2)
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _preprocess_batch(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Preprocess the given batch of data.

        Args:
            batch: A dictionary of tensors containing the images and ground truth labels.

        Returns:
            A tuple of the input tensor, the predicted tensor and the mask for all valid labels.
        """
        x = batch["images"]
        y = batch["ground_truth"]

        # Mask for output NaN's
        mask = ~y.isnan()

        # Remove NaN's from the input and output tensors
        x = x.nan_to_num()
        y = y.nan_to_num()

        return x, y, mask

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Computes the loss for a training batch.

        This method is called on each training batch during the training process. It computes the loss
        for the batch using the `_compute_loss` method and then logs the loss.

        Args:
            train_batch (Dict[str, Tensor]): A dictionary of tensors containing the training batch data.
            batch_idx (int): The index of the training batch.

        Returns:
            Tensor: The loss for the training batch.
        """
        x, y, mask = self._preprocess_batch(train_batch)
        y_hat = self(x)
        loss = mse_loss(y_hat[mask], y[mask]).nan_to_num()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Computes the loss for a validation batch.

        This method is called on each validation batch during the training process. It computes the loss
        for the batch using the `_compute_loss` method and then logs the loss.

        Args:
            val_batch (Dict[str, Tensor]): A dictionary of tensors containing the validation batch data.
            batch_idx (int): The index of the validation batch.

        Returns:
            Tensor: The loss for the validation batch.
        """
        x, y, mask = self._preprocess_batch(val_batch)
        y_hat = self(x)
        loss = mse_loss(y_hat[mask], y[mask]).nan_to_num()

        self.log("val_loss", loss)
        return loss
