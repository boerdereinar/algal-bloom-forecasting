from torchgeo.datasets import RasterDataset


class LandCoverDataset(RasterDataset):
    """Dataset for land coverage data"""

    date_format = "%Y"
    filename_glob = "*.tif"
    filename_regex = r".*(?P<date>\d{4})_\d{4}CRN\.tif$"
    is_image = False
    classes = [
        "Water",
        "Implanted Forests",
        "Native Forests",
        "Native Grasslands/Rangelands",
        "Agriculture crops",
        "Bare soil/Built-up",
        "Wetlands"
    ]
