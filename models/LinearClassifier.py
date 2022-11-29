from argparse import ArgumentParser
from typing import Tuple, Any, Dict

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer


class LinearClassifier(LightningModule):
    """A linear model."""
    def __init__(self, learning_rate: float = 0.01, **kwargs: Any):
        super(LinearClassifier, self).__init__()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("-lr", "--learning-rate", type=float, default=0.01, help="The learning rate of the model.")
        return parent_parser

    def forward(self, x: Tensor) -> Tensor:
        return x

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Tensor:
        x = train_batch["image"]
        y = train_batch["label"]
        res = self.forward(x)
        loss = mse_loss(res, y)
        self.log("train_loss", loss)
        return loss
