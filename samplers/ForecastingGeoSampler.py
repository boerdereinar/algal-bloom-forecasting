from datetime import datetime
from random import random
from typing import Iterator, Optional, Union, Tuple, Sequence

from torchgeo.datasets import GeoDataset, BoundingBox
from torchgeo.samplers import RandomGeoSampler, Units

from edegruyl.samplers.constants import Resolution
from edegruyl.utils.datetimeutils import adjust_resolution, timedelta_from_resolution, disambiguate_datetime


class ForecastingGeoSampler(RandomGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: int,
        look_back: int,
        look_ahead: int,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        res: Resolution = Resolution.Days
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
            look_back: the number of resolution units to look backwards
            look_ahead: the number of resolution units to look forwards
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            res: defines the resolution of the dates
        """
        super().__init__(dataset, size, length, roi, units)
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.res = res

    def __iter__(self) -> Iterator[Sequence[Sequence[BoundingBox]]]:
        for item in super().__iter__():
            minx, maxx, miny, maxy, _, _ = item
            tn = [BoundingBox(minx, maxx, miny, maxy, t1.timestamp(), t2.timestamp())
                  for t1, t2 in self._get_random_timestamps()]
            yield [tn[:self.look_back], tn[self.look_back:]]

    def _get_random_timestamps(self) -> Sequence[Tuple[datetime, datetime]]:
        mint, maxt = self.index.bounds[-2:]
        mint = adjust_resolution(datetime.fromtimestamp(mint), self.res)
        maxt = adjust_resolution(datetime.fromtimestamp(maxt), self.res)

        dt = timedelta_from_resolution(self.res)
        tdt = dt * (self.look_back + self.look_ahead)

        if maxt - mint < tdt:
            raise IndexError("")

        t0 = adjust_resolution(mint + (maxt - mint - tdt) * random(), self.res)
        tn = [disambiguate_datetime(t0 + dt * i, self.res) for i in range(self.look_back + self.look_ahead)]
        return tn
