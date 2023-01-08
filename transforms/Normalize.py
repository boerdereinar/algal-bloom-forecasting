from typing import Any, Dict

from torch import Tensor


class Normalize:
    """
    A class that normalizes tensor values in a dictionary.
    """
    def __init__(self, max: Tensor):
        """
        Initializes the Normalize class.

        Args:
            max: A tensor containing the maximum values of each channel.
        """
        self.max = max[:, None, None]

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the tensor values in a dictionary.

        Args:
            x: A dictionary containing keys with tensor values.

        Returns:
            A dictionary with the same keys as the input dictionary, but with the tensor values normalized.
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                data[key] = value / self.max
            else:
                data[key] = value

        return data
