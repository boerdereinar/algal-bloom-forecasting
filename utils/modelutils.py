from typing import Dict, Tuple

from torch import Tensor


def extract_batch(batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extracts and processes tensors from a batch of data.

    Args:
        batch: A dictionary containing keys "images", "ground_truth", and "mask".
            "images": A tensor of shape (batch_size, channels, height, width) representing the images in the batch.
            "ground_truth": A tensor of shape (batch_size, channels, height, width) representing the ground truth
                labels for the images.
            "mask": A tensor of shape (batch_size, height, width) representing the mask applied to the images.

    Returns:
        A tuple containing the processed tensors (x, y, m, observed).
        "x": The images tensor with any NaN values replaced with zeros.
        "y": The ground truth labels tensor with any NaN values replaced with zeros.
        "m": The mask tensor.
        "observed": A tensor of shape (batch_size, channels, height, width) indicating which values in the ground
            truth labels tensor are observed (not NaN).
    """
    x = batch["images"]
    y = batch["ground_truth"]
    m = batch["mask"]

    # Observed values
    observed = ~y.isnan()

    # Remove NaNs
    x = x.nan_to_num()
    y = y.nan_to_num()

    return x, y, m, observed
