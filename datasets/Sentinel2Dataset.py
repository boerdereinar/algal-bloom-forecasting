import glob
import os
import re
import sys
from functools import lru_cache, reduce
from itertools import groupby
from typing import Any, Dict, Optional, Callable, cast, Sequence

import numpy as np
import rasterio
from rasterio import CRS, DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from torch import Tensor
from torchgeo.datasets import GeoDataset, BoundingBox
from torchgeo.datasets.utils import disambiguate_timestamp


class Sentinel2Dataset(GeoDataset):
    filename_glob = "*"
    filename_regex = r"^.*\.tif$"
    description_regex = r"^.*_(?P<date_from>\d{8}T\d{6})_(?P<date_to>\d{8}T\d{6})_T21HWD$"
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
        """
        Initialize a new Dataset instance. This class is a modification of `torchgeo.datasets.RasterDataset` that
        supports multiple samples per file.

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
        super().__init__(transforms)

        self.root = root
        self.cache = cache
        self.kwargs = kwargs

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        description_regex = re.compile(self.description_regex, re.VERBOSE)

        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:  # type: DatasetReader
                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        descriptions = src.descriptions

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    for j, description in enumerate(descriptions):
                        description_match = re.match(description_regex, description)
                        mint: float = 0
                        maxt: float = sys.maxsize
                        if description_match is not None:
                            date_from = description_match.group("date_from")
                            date_to = description_match.group("date_to")
                            mint, _ = disambiguate_timestamp(date_from, self.date_format)
                            _, maxt = disambiguate_timestamp(date_to, self.date_format)
                            if mint > maxt:
                                mint, maxt = maxt, mint

                        coords = (minx, maxx, miny, maxy, mint, maxt)
                        self.index.insert(i, coords, dict(filepath=filepath, index=j+1))
                        i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        hits = self.index.intersection(tuple(query), objects=True)
        files: Sequence[Dict[str, Any]] = [hit.object for hit in hits]

        if not files:
            raise IndexError(f"query: {query} not found in index with bounds: {self.bounds}")

        data = self._merge_files(files, query)
        sample = {"image": data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(self, files: Sequence[Dict[str, Any]], query: BoundingBox) -> Tensor:
        """Load and merge one or more files.

        Args:
            files: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        # Group file by filepath
        grouped_files = dict((k, [v["index"] for v in vs]) for k, vs in groupby(files, key=lambda x: x["filepath"]))

        if self.cache:
            vrt_fhs = [(self._cached_load_warp_file(fp), vs) for fp, vs in grouped_files.items()]
        else:
            vrt_fhs = [(self._load_warp_file(fp), vs) for fp, vs in grouped_files.items()]

        def copy(x, y):
            return np.where(np.isnan(x), y, x)

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        out_width = round((query.maxx - query.minx) / self.res)
        out_height = round((query.maxy - query.miny) / self.res)
        out_shape = (out_height, out_width)

        tensors = [f.read(v, out_shape=out_shape, window=from_bounds(*bounds, f.transform))[None, :]
                   for f, vs in vrt_fhs for v in vs]

        if len(tensors) == 1:
            return Tensor(tensors[0])
        return Tensor(reduce(copy, tensors)[None, :])

    @lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src
