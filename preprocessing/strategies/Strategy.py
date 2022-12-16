from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import numpy as np


class Strategy(ABC):
    """Interpolation strategy."""
    @abstractmethod
    def first(self, data: np.ndarray) -> None:
        """Process the first sample."""
        ...

    @abstractmethod
    def interpolate(
            self,
            data_prev: np.ndarray,
            t_prev: datetime,
            data_next: np.ndarray,
            t_next: datetime
    ) -> List[np.ndarray]:
        """
        Interpolate samples in between two other samples.

        Args:
            data_prev: The previous sample
            t_prev: The previous timestamp
            data_next: The next sample
            t_next: The next timestamp

        Returns:
            The interpolated samples.
        """
        ...
