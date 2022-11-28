from typing import Dict, Any, Optional, Callable, Sequence, Union

from rasterio import CRS
from rtree import Index
from torchgeo.datasets import RasterDataset, BoundingBox, GeoDataset

from edegruyl.datasets import CombinedDataset


class TimeSeriesDataset(RasterDataset):
    """Abstract base class for :class:`RasterDataset` stored as raster files."""
    def __init__(
            self,
            root: str,
            crs: Optional[CRS] = None,
            res: Optional[float] = None,
            transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
            cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(root, crs, res, transforms, cache)

        # Initialize the time index
        self.time_index = Index(interleaved=False)
        self.time_index.insert(0, [*self.index.bounds[-2:], 0, 0], (None, None))

        # Populate the time index
        i = 0
        for file in self.index.intersection(self.index.bounds, True):
            mint, maxt = file.bounds[-2:]
            filepath = file.object

            bbox = [mint, maxt, 0, 0]
            for item in self.time_index.intersection(bbox, True):
                self.time_index.delete(item.id, item.bounds)
                v0, v1 = item.object
                t0, t1 = item.bounds[:2]
                if t0 < mint < t1:
                    self.time_index.insert(0, [t0, mint, 0, 0], (v0, filepath))
                if t0 < maxt < t1:
                    self.time_index.insert(0, [maxt, t1, 0, 0], (v1, filepath))
            i += 1

    def _getitem(self, query: BoundingBox) -> Dict[str, Any]:
        try:
            return super().__getitem__(query)
        except IndexError:
            bbox = (query.mint, query.maxt, 0, 0)
            time_range = next(self.time_index.intersection(bbox, True), None)
            if not time_range:
                raise IndexError(f"query: {query} not found in index with bounds: {self.bounds}")

            left, right = time_range.object
            data = super()._merge_files([left], query)

            key = "image" if self.is_image else "mask"
            sample = {key: data, "crs": self.crs, "bbox": query}

            if self.transforms is not None:
                sample = self.transforms(sample)

            return sample

    def __xor__(self, other: GeoDataset) -> CombinedDataset:
        return CombinedDataset(self, other)

    def __getitem__(self, query: Union[BoundingBox, Sequence[BoundingBox]]) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: A single tuple or a list of tuples with (minx, maxx, miny, maxy, mint, maxt) coordinates to index.

        Returns:
            Sample of image/mask and metadata at that index.

        Raises:
            IndexError: if query is not found in the index
        """
        if isinstance(query, BoundingBox):
            return self._getitem(query)

        samples = [self._getitem(q) for q in query]

        key = "image" if self.is_image else "mask"
        return {key: samples[0], "crs": self.crs, "bbox": query}
