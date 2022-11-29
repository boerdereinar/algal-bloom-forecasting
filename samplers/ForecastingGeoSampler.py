from datetime import datetime
from random import random
from typing import Iterator, Optional, Union, Tuple, Sequence

from torchgeo.datasets import GeoDataset, BoundingBox
from torchgeo.samplers import RandomGeoSampler, Units

from edegruyl.samplers.constants import Resolution
from edegruyl.utils.datetimeutils import adjust_resolution, timedelta_from_resolution, disambiguate_datetime
from edegruyl.utils.samplerutils import get_random_bounding_boxes


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
            bboxs = get_random_bounding_boxes(item, self.look_back, self.look_ahead, self.res)
            yield [bboxs[:self.look_back], bboxs[self.look_back:]]
