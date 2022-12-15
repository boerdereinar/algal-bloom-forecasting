from argparse import ArgumentParser
from typing import Any, Dict

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer


class LinearModel(LightningModule):
    """A linear model."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            learning_rate: float = 0.01,
            **kwargs: Any
    ) -> None:
        super(LinearModel, self).__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        n_in = window_size * num_bands * size * size
        n_out = size * size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, n_out),
            nn.Unflatten(1, (size, size))
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("-lr", "--learning-rate", type=float, help="The learning rate of the model.", default=0.01)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def _compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """Computes the loss for a given batch of data.

        Args:
            batch (Dict[str, Tensor]): A dictionary of tensors containing the images and ground truth labels.

        Returns:
            Tensor: The mean squared error between the predicted and ground truth labels.
        """
        x = batch["images"]
        y = batch["ground_truth"]

        # Mask for output NaN's
        mask = ~y.isnan()

        # Remove NaN's from the input and output tensors
        x = x.nan_to_num()
        y = y.nan_to_num()

        y_hat = self.forward(x)
        return mse_loss(y_hat[mask], y[mask])

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._compute_loss(train_batch)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._compute_loss(val_batch)

        self.log("val_loss", loss)
        return loss
