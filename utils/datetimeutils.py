from datetime import datetime, timedelta
from typing import Tuple

from edegruyl.samplers.constants import Resolution


def adjust_resolution(t: datetime, res: Resolution) -> datetime:
    """Adjusts the resolution of a datetime.

    Args:
        t: The datetime to adjust.
        res: The resolution.

    Returns:
        The adjusted datetime.
    """
    if res >= Resolution.Days:
        t = t.replace(hour=0)
    if res >= Resolution.Hours:
        t = t.replace(minute=0, second=0, microsecond=0)

    return t


def timedelta_from_resolution(res: Resolution) -> timedelta:
    """Gets the timedelta relating to the resolution.

    Args:
        res: The resolution.

    Returns:
        The timedelta of one unit of the resolution.

    Raises:
        ValueError: if the given resolution is invalid.
    """
    if res == Resolution.Days:
        return timedelta(days=1)
    if res == Resolution.Hours:
        return timedelta(hours=1)

    raise ValueError("The given resolution is invalid.")


def disambiguate_datetime(t: datetime, res: Resolution) -> Tuple[datetime, datetime]:
    """Gets the minimum and maximum date given the resolution.

    Args:
        t: The time to disambiguate.
        res: The resolution.

    Returns:
        The minimum and maximum date given the date.

    Raises:
        ValueError: if the given resolution is invalid.
    """
    if res == Resolution.Days:
        return adjust_resolution(t, res), t.replace(hour=23, minute=59, second=59, microsecond=999999)
    if res == Resolution.Hours:
        return adjust_resolution(t, res), t.replace(minute=59, second=59, microsecond=999999)

    raise ValueError("The given resolution is invalid.")