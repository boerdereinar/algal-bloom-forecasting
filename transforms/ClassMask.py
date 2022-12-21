from typing import Any, Dict

from torch import Tensor


class ClassMask:
    """
    A class that masks tensor values in a dictionary.

    Attributes:
        cls: The integer value to mask in the tensors.
    """
    def __init__(self, cls: int):
        """
        Initializes the ClassMask class.

        Args:
            cls: The integer value to mask in the tensors.
        """
        self.cls = cls

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask the tensors in the input dictionary.

        Args:
            x: A dictionary containing keys with tensor values.

        Returns:
            A dictionary with the same keys as the input dictionary, but with the tensor values masked.
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                data[key] = value == self.cls
            else:
                data[key] = value

        return data
