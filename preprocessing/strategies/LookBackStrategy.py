from collections import deque
from datetime import datetime
from functools import reduce
from itertools import islice
from typing import Any, List, Optional

import numpy as np

from edegruyl.preprocessing.strategies import Strategy


class LookBackStrategy(Strategy):
    """Interpolation strategy that uses the last known sample to fill in missing samples."""
    data: Optional[np.ndarray] = None

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
            # If the buffer is full
            # And an existing data point is going to be removed
            # And the buffer still contains data points
            # Then reset the combined data
            if len(self.buffer) == self.buffer.maxlen and self.buffer[0] is not None and \
               any(x is not None for x in islice(self.buffer, 1, None)):
                self.data = None
            # If a data point is going to be inserted, reset the combined data
            if i == days - 1:
                self.data = None

            self.buffer.append(data_next if i == days - 1 else None)

            if self.data is None:
                self.data = reduce(
                    lambda a, b: np.where(np.isnan(a), b, a),
                    filter(lambda a: a is not None, reversed(self.buffer))
                )

            interpolated.append(self.data)

        return interpolated
