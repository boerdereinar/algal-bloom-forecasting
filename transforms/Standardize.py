from typing import Any, Dict

from torch import Tensor


class Standardize:
    """
    A class that standardizes tensor values in a dictionary.
    """
    def __init__(self, mean: Tensor, std: Tensor):
        """
        Initializes the Standardize class.

        Args:
            mean: A tensor containing the means of each channel.
            std: A tensor containing the standard deviations of each channel.
        """
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardizes the tensor values in a dictionary.

        Args:
            x: A dictionary containing keys with tensor values.

        Returns:
            A dictionary with the same keys as the input dictionary, but with the tensor values standardized.
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                data[key] = (value - self.mean) / self.std
            else:
                data[key] = value

        return data
