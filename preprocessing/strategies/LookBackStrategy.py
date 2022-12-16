from collections import deque
from datetime import datetime
from functools import reduce
from typing import Any, Optional, List

import numpy as np

from edegruyl.preprocessing.strategies import Strategy


class LookBackStrategy(Strategy):
    """Interpolation strategy that uses the last known sample to fill in missing samples."""
    data: Optional[np.ndarray]

    def __init__(self, buffer_size: int = 30, **kwargs: Any):
        """
        Initializes the look back strategy.

        Args:
            buffer_size: The number of days to look back for samples to fill in missing values.
        """
        self.buffer = deque(maxlen=buffer_size)

    def first(self, data: np.ndarray) -> None:
        self.buffer.append(data)

    def interpolate(
            self,
            data_prev: np.ndarray,
            t_prev: datetime,
            data_next: np.ndarray,
            t_next: datetime
    ) -> List[np.ndarray]:
        days = (t_next - t_prev).days
        interpolated = []

        for i in range(days):
            if len(self.buffer) == self.buffer.maxlen and self.buffer[0] is not None:
                self.data = None

            self.buffer.append(data_next if i == days - 1 else None)

            if self.data is None:
                self.data = reduce(lambda a, b: np.where(np.isnan(a), b, a), filter(id, reversed(self.buffer)))

            interpolated.append(self.data)

        return interpolated
