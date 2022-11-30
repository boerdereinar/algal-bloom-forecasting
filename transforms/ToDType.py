from typing import Dict, Any

from torch import Tensor


class ToDType:
    """
    Convert the tensors to the given d-type.
    """
    def __init__(self, dtype: Any):
        super().__init__()
        self.dtype = dtype

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param x: Input data
        :return: The input data with all tensors converted to float32.
        """
        data: Dict[str, Any] = dict()
        for key, value in x.items():
            if isinstance(value, Tensor):
                data[key] = value.type(self.dtype)
            else:
                data[key] = value

        return data
