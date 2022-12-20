from typing import Any, Dict

import torch
from torch import Tensor


class Clip:
    """
    A class that clips tensor values in a dictionary.

    Attributes:
        max: A tensor containing the maximum value that a tensor can have.
    """
    def __init__(self, max: Tensor):
        """
        Initializes the Clip class.

        Args:
            max: A tensor containing the maximum value that a tensor can have.
        """
        self.max = max

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clips the tensor values in a dictionary.

        Args:
            x: A dictionary containing keys with tensor values.

        Returns:
            A dictionary with the same keys as the input dictionary, but with the tensor values clipped.
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                max_reshaped = self.max[:, None, None]
                data[key] = torch.minimum(value, max_reshaped)
            else:
                data[key] = value

        return data
