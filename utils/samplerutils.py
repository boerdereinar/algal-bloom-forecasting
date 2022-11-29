from datetime import datetime
from random import random
from typing import Sequence

from torchgeo.datasets import BoundingBox

from edegruyl.samplers.constants import Resolution
from edegruyl.utils.datetimeutils import adjust_resolution, timedelta_from_resolution, disambiguate_datetime


def get_random_bounding_boxes(
        bounds: BoundingBox,
        look_back: int,
        look_ahead: int,
        res: Resolution
) -> Sequence[BoundingBox]:
    mint = adjust_resolution(datetime.fromtimestamp(bounds.mint), res)
    maxt = adjust_resolution(datetime.fromtimestamp(bounds.maxt), res)

    dt = timedelta_from_resolution(res)
    tdt = dt * (look_back + look_ahead)

    if maxt - mint < tdt:
        raise IndexError("Time range does not fit within the bounds.")

    t0 = adjust_resolution(mint + (maxt - mint - tdt) * random(), res)
    return [BoundingBox(*bounds[:4], t0.timestamp(), t1.timestamp()) for t0, t1 in
            (disambiguate_datetime(t0 + dt * i, res) for i in range(look_back + look_ahead))]
