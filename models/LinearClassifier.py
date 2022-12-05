from argparse import ArgumentParser
from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from edegruyl.models.functional import mse_loss


class LinearClassifier(LightningModule):
    """A linear model."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            learning_rate: float = 0.01,
            **kwargs: Any
    ) -> None:
        super(LinearClassifier, self).__init__()
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
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x = torch.nan_to_num(train_batch["images"])
        y = train_batch["ground_truth"]
        res = self.forward(x)

        loss = mse_loss(res, y)

        self.log("train_loss", loss)
        return loss
