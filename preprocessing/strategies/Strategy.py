from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from torch import Tensor


class Strategy(ABC):
    t_prev: datetime

    """Interpolation strategy."""
    @abstractmethod
    def first(self, data: Tensor, t: datetime, mask: Tensor) -> Tensor:
        """Process the first sample."""
        ...

    @abstractmethod
    def interpolate(self, data: Tensor, t: datetime, mask: Tensor) -> List[Tensor] | Tensor:
        """
        Interpolate samples in between two other samples.

        Args:
            data: The next sample
            t: The next timestamp
            mask: The mask for the valid values

        Returns:
            The interpolated samples.
        """
        ...
