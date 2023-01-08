from typing import Any, Dict, overload

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

    @overload
    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes the tensor values in a dictionary.

        Args:
            x: A dictionary containing keys with tensor values.

        Returns:
            A dictionary with the same keys as the input dictionary, but with the tensor values normalized.
        """
        ...

    @overload
    def __call__(self, x: Tensor) -> Tensor:
        """
        Normalizes the tensor.

        Args:
            x: A tensor.

        Returns:
            The normalized tensor.
        """
        ...

    def __call__(self, x: Dict[str, Any] | Tensor) -> Dict[str, Any] | Tensor:
        if isinstance(x, Tensor):
            return x / self.max

        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                data[key] = value / self.max
            else:
                data[key] = value

        return data
