from typing import Callable, Any, Dict, Sequence, Union

from torchgeo.datasets import UnionDataset, GeoDataset, BoundingBox, merge_samples


class CombinedDataset(UnionDataset):
    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[[Sequence[Dict[str, Any]]], Dict[str, Any]] = merge_samples
    ) -> None:
        super().__init__(dataset1, dataset2, collate_fn)

    def __getitem__(self, queries: Sequence[Union[BoundingBox, Sequence[BoundingBox]]]) -> Dict[str, Any]:
        # All datasets are guaranteed to have a valid query
        samples = [ds[queries[i]] for i, ds in enumerate(self.datasets)]

        return self.collate_fn(samples)
