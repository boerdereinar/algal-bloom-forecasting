from typing import Union, Tuple, Optional, Iterator, Sequence

from torchgeo.datasets import GeoDataset, BoundingBox
from torchgeo.samplers import RandomBatchGeoSampler, Units

from edegruyl.samplers.constants import Resolution
from edegruyl.utils.samplerutils import get_random_bounding_boxes


class RandomBatchForecastingGeoSampler(RandomBatchGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        batch_size: int,
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
            batch_size: number of samples per batch
            length: number of random samples to draw per epoch
            look_back: the number of resolution units to look backwards
            look_ahead: the number of resolution units to look forwards
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            res: defines the resolution of the dates
        """
        super().__init__(dataset, size, length, batch_size, roi, units)
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.res = res

    def __iter__(self) -> Iterator[Sequence[Sequence[Sequence[BoundingBox]]]]:
        for item in super().__iter__():
            batch = []
            for bbox in item:
                bounds = BoundingBox(*bbox[:4], *self.roi[-2:])
                bboxs = get_random_bounding_boxes(bounds, self.look_back, self.look_ahead, self.res)
                batch.append([bboxs[:self.look_back], bboxs[self.look_back:]])
            yield batch
