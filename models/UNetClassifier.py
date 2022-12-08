from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from edegruyl.models import UNet
from edegruyl.models.functional import mse_loss


class UNetClassifier(LightningModule):
    """A UNet-based classifier for image segmentation.

    This class extends the PyTorch Lightning `LightningModule` class to provide useful abstractions
    for training models. It uses a UNet model to perform image segmentation on input images.

    Args:
        window_size (int): The size of the window that the classifier will use to look at the input data.
        num_bands (int): The number of bands (i.e. channels) in the input data.
        size (int): The size of the image that the classifier will be trained on.
        learning_rate (float): The learning rate to use for training.
    """
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            learning_rate: float = 0.01,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        in_channels = window_size * num_bands
        self.model = UNet(in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the UNet model.

        Args:
            x (Tensor): The input tensor to the model, of shape (batch_size, window_size * num_bands, size, size).

        Returns:
            Tensor: The output of the model, of shape (batch_size, 1, size, size).
        """
        x = torch.flatten(x, 1, 2)
        return self.model(x)

    def _compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Computes the loss for a given batch of data.

        Args:
            batch (Dict[str, Tensor]): A dictionary of tensors containing the images and ground truth labels.

        Returns:
            Tensor: The mean squared error between the predicted and ground truth labels.
        """
        x = torch.nan_to_num(batch["images"])
        y = batch["ground_truth"]
        y_hat = self.forward(x)
        return mse_loss(y_hat, y)

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
        loss = self._compute_loss(train_batch)

        self.log("train_loss", loss)
        return loss

    def test_step(self, test_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Computes the loss for a test batch.

        This method is called on each test batch during the training process. It computes the loss
        for the batch using the `_compute_loss` method and then logs the loss.

        Args:
            test_batch (Dict[str, Tensor]): A dictionary of tensors containing the test batch data.
            batch_idx (int): The index of the test batch.

        Returns:
            Tensor: The loss for the test batch.
        """
        loss = self._compute_loss(test_batch)

        self.log("test_loss", loss)
        return loss
