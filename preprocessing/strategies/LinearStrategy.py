from datetime import datetime
from typing import Any

import torch
from torch import Tensor
from torch.nn.functional import interpolate

from edegruyl.preprocessing.strategies import Strategy


class LinearStrategy(Strategy):
    """Interpolation strategy that uses linear interpolation to fill in missing samples."""

    def __init__(self, **kwargs: Any):
        """
        Initializes the linear strategy.
        """
        ...

    def first(self, data: torch.Tensor) -> None:
        ...

    def interpolate(
            self,
            data_prev: torch.Tensor,
            t_prev: datetime,
            data_next: torch.Tensor,
            t_next: datetime
    ) -> Tensor:
        days = (t_next - t_prev).days

        data = torch.stack((data_prev, data_next), 1)[None, :]
        interpolated = interpolate(data, (days + 1, *data.shape[-2:]), mode="trilinear")
        return interpolated[0, :, 1:].permute((1, 0, 2, 3))
