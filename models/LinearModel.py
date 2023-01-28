from argparse import ArgumentParser
from typing import Any, Dict

from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn.functional import mse_loss
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from edegruyl.utils.modelutils import extract_batch


class LinearModel(LightningModule):
    """A linear model."""
    def __init__(
            self,
            window_size: int,
            learning_rate: float = 0.01,
            **kwargs: Any
    ) -> None:
        super(LinearModel, self).__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        n_in = window_size * 3
        self.fc = nn.Linear(n_in, 1)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("-lr", "--learning-rate", type=float, help="The learning rate of the model.", default=0.01)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        batch_size, window_size, num_bands, height, width = x.shape

        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(batch_size * height * width, window_size * num_bands)
        x = self.fc(x)
        x = x.reshape(batch_size, window_size, height, width)

        return x

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(train_batch)
        y_hat = self(x)
        loss = mse_loss(y_hat[observed], y[observed]).nan_to_num()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, _, observed = extract_batch(val_batch)
        y_hat = self(x)
        loss = mse_loss(y_hat[observed], y[observed]).nan_to_num()

        self.log("val_loss", loss)
        return loss
