from datetime import datetime
from typing import Any

import numpy as np
import torch
from scipy.interpolate import griddata
from torch import Tensor
from torch.nn.functional import interpolate

from edegruyl.preprocessing.strategies import Strategy


class LinearStrategy(Strategy):
    """Interpolation strategy that uses linear interpolation to fill in missing samples."""

    data_prev: Tensor

    def __init__(self, **kwargs: Any):
        """
        Initializes the linear strategy.
        """
        ...

    def first(self, data: Tensor, t: datetime, mask: Tensor) -> Tensor:
        data = self._interpolate_single(data, mask)

        self.data_prev = data
        self.t_prev = t

        return data

    def interpolate(self, data: Tensor, t: datetime, mask: Tensor) -> Tensor:
        days = (t - self.t_prev).days

        data = self._interpolate_single(data, mask)

        stacked = torch.stack((self.data_prev, data), 1)[None, :]
        interpolated = interpolate(stacked, (days + 1, *data.shape[-2:]), mode="trilinear")

        self.data_prev = data
        self.t_prev = t
        return interpolated[0, :, 1:].permute((1, 0, 2, 3))

    @staticmethod
    def _interpolate_single(data: Tensor, mask: Tensor) -> Tensor:
        data = data.numpy()

        for i in range(len(data)):
            non_nan = np.where(~np.isnan(data[i]))
            if len(non_nan[0]) >= 4:
                data[i] = griddata(np.transpose(non_nan), data[i][non_nan], tuple(np.indices(data[i].shape)), "linear")
            else:
                data[i] = 0

        data = torch.tensor(data)
        data[:, ~mask] = torch.nan

        return data
