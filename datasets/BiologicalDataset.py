from torchgeo.datasets import RasterDataset


class BiologicalDataset(RasterDataset):
    """Dataset for biological data."""

    filename_glob = "*_biological.tif"
    filename_regex = r".*_(?P<date>\d{8})"
    all_bands = ["cdom", "chlorophyll-a", "turbidity"]
