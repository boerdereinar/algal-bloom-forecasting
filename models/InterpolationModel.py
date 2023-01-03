from argparse import ArgumentParser
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss

from edegruyl.models import UNet
from edegruyl.models.InterpolationLayer import InterpolationNetwork, reshape_input, reshape_output
from edegruyl.utils.modelutils import extract_batch


class InterpolationModel(LightningModule):
    """Interpolation model."""
    def __init__(
            self,
            window_size: int,
            num_bands: int,
            size: int,
            masked: bool = False,
            learning_rate: float = 1e-4,
            momentum: float = 0.9,
            patience: int = 3,
            save_dir: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=list(kwargs))

        self.interp_net = InterpolationNetwork(num_bands, window_size, window_size)

        in_channels = 3 * window_size * num_bands
        self.unet = UNet(in_channels, 1)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--masked", action="store_true", help="Whether the interpolation is only applied to the "
                                                                  "area masked by the water mask.")
        parser.add_argument("--learning-rate", type=float, help="The learning rate of the optimizer.", default=1e-4)
        parser.add_argument("--momentum", type=float, help="The momentum of the optimizer.", default=0.9)
        parser.add_argument("--patience", type=int, default=3, help="The number of epochs with no improvement after "
                                                                    "which learning rate will be reduced.")
        parser.add_argument("--save-dir", type=str, help="The save directory for the plots.", default=None)
        return parent_parser

    def forward(self, x: Tensor, water_mask: Tensor, observed_x: Tensor, masked: bool = False) -> Tensor:
        """

        Args:
            x: The images tensor of shape (batch, window_size, num_bands, height, width)
            water_mask: The water mask tensor of shape (batch, height, width)
            observed_x: A tensor of shape (batch_size, window_size, bands, height, width) indicating which values in
                the images tensor are observed (not NaN).
            masked: Whether the interpolation is only applied to the area masked by the water_mask.

        Returns:
            The predicted tensor of shape (batch, 1, height, width).
        """
        batch_size, observed_points, num_features, height, width = x.shape
        water_mask_indices = torch.where(water_mask)

        # Reshape input
        x_t, d, m = reshape_input(x, water_mask_indices, observed_x, masked)

        # Feed into interpolation network
        y_interp = self.interp_net(x_t, d, m)

        # Reshape interpolated output
        y_interp = reshape_output(y_interp, water_mask_indices, batch_size, height, width, masked)
        y_interp = y_interp.reshape(batch_size, -1, height, width)

        # TODO: Feed into U-Net

    def training_step(self, train_batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        x, y, water_mask, observed_x, observed_y = extract_batch(train_batch, True)
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
        mse_reconst = mse_loss(y_reconst[held_out], x_t[held_out])

        # Reshape interpolated output
        y_interp = reshape_output(y_interp, water_mask_indices, batch_size, height, width, self.hparams.masked)
        y_interp = y_interp.reshape(batch_size, -1, height, width)

        # TODO: Feed into U-Net
