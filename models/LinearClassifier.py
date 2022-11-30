from argparse import ArgumentParser
from typing import Any, Dict

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer


class LinearClassifier(LightningModule):
    """A linear model."""
    def __init__(
            self,
            input_channels: int,
            input_size: int,
            learning_rate: float = 0.01,
            **kwargs: Any
    ) -> None:
        super(LinearClassifier, self).__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        n = input_channels * input_size * input_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n, n),
            nn.Unflatten(1, (input_channels, input_size, input_size))
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--input-channels", type=int, required=True,
                            help="The number of input channels of the model.")
        parser.add_argument("--input-size", type=int, help="The size of the input of the model.", required=True)
        parser.add_argument("-lr", "--learning-rate", type=float, help="The learning rate of the model.", default=0.01)
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Tensor:
        x = torch.nan_to_num(train_batch["image"])
        y = train_batch["label"]
        res = self.forward(x)

        mask = ~torch.isnan(y)
        out = (res[mask] - y[mask]) ** 2
        loss = out.mean()

        self.log("train_loss", loss)
        return loss
