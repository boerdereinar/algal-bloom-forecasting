import numpy as np
import torch
from torch import Tensor


def mse_loss(preds: Tensor, target: Tensor, squared: bool = True) -> torch.Tensor:
    """
    Computes the mean squared error.

    Args:
        preds: Predicted tensor.
        target: Ground truth tensor.
        squared: Returns RMSE value if set to false.

    Returns:
        The mean squared error.

    References:
        https://github.com/Lightning-AI/metrics/blob/24ea1e9cc4604de1c84bd452d6d1f59da502bda0/src/torchmetrics/functional/regression/mse.py
    """
    diff = preds - target
    mean_squared_error = diff.square().nanmean()
    res = mean_squared_error if squared else mean_squared_error.sqrt()
    return res if not np.isnan(res) else 0

