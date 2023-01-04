import torch
from torch import Tensor


class Bin:
    """
    A class that bins a tensor.
    """
    def __init__(self, bins: Tensor):
        """
        Initializes the Bin class.

        Args:
            bins: A tensor containing the bins.
        """
        self.bins = bins

    def __call__(self, x: Tensor) -> Tensor:
        """
        Bins the tensor.

        Args:
            x: A tensor.

        Returns:
            A binned tensor.
        """
        binned = torch.bucketize(x, self.bins)
        binned[torch.isnan(x)] = -1
        return binned
