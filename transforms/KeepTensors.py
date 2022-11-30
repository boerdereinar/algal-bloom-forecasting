from typing import Dict, Any

from torch import Tensor


class KeepTensors:
    """
    Keep only the tensors in the result of the datasets.
    This transform is intended to remove frozen dataclasses from the results which cannot be moved to a device.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param x: Input data
        :return: The input data with all non-tensor items removed.
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                data[key] = value

        return data
