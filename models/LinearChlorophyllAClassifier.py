from argparse import ArgumentParser
from typing import Tuple, Any

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer


class LinearChlorophyllAClassifier(LightningModule):
    def __init__(self, learning_rate: float = 0.01, **kwargs: Any):
        super(LinearChlorophyllAClassifier, self).__init__()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        return x

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = train_batch
        res = self.forward(x)
        loss = mse_loss(res, y)
        self.log("train_loss", loss)
        return loss
