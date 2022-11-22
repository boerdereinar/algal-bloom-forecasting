from typing import Optional, Callable, Dict, Any

from rasterio import CRS
from torchgeo.datasets import RasterDataset


class ChlorophyllADataset(RasterDataset):
    filename_regex = r"^[a-zA-Z]+_(?P<date>\d{8}T\d{6})_[a-zA-Z]+_(?P<resolution>\d+)m\..+$"
    date_format = "%Y%m%dT%H%M%S"

    def __init__(
        self,
        root: str,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(root, crs, res, transforms, cache)
        self.kwargs = kwargs
