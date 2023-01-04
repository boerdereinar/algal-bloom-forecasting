from typing import Dict, Literal, Tuple, overload

import torch
from torch import Tensor


@overload
def extract_batch(
        batch: Dict[str, Tensor],
        mask_input: Literal[False] = False,
        classify: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Extracts and processes tensors from a batch of data.

    Args:
        batch: A dictionary containing keys "images", "ground_truth", and "water_mask".
            "images": A tensor of shape (batch_size, window_size, bands, height, width) representing the images in
                the batch.
            "ground_truth": A tensor of shape (batch_size, height, width) representing the ground truth labels for
                the images.
            "water_mask": A tensor of shape (batch_size, height, width) representing the mask applied to the images.
        mask_input: Whether to return a tensor indicating which values in the images tensor are observed (not NaN).
        classify: Whether the input contains class labels.

    Returns:
        A tuple containing the processed tensors (x, y, m, observed_y).
        "x": The images tensor with any NaN values replaced with zeros.
        "y": The ground truth labels tensor with any NaN values replaced with zeros.
        "m": The water mask tensor.
        "observed_y": A tensor of shape (batch_size, height, width) indicating which values in the ground
            truth labels tensor are observed (not NaN).
    """
    ...


@overload
def extract_batch(
        batch: Dict[str, Tensor],
        mask_input: Literal[True],
        classify: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Extracts and processes tensors from a batch of data.

    Args:
        batch: A dictionary containing keys "images", "ground_truth", and "water_mask".
            "images": A tensor of shape (batch_size, window_size, bands, height, width) representing the images in
                the batch.
            "ground_truth": A tensor of shape (batch_size, height, width) representing the ground truth labels for
                the images.
            "water_mask": A tensor of shape (batch_size, height, width) representing the mask applied to the images.
        mask_input: Whether to return a tensor indicating which values in the images tensor are observed (not NaN).
        classify: Whether the input contains class labels.

    Returns:
        A tuple containing the processed tensors (x, y, m, observed_x, observed_y).
        "x": The images tensor with any NaN values replaced with zeros.
        "y": The ground truth labels tensor with any NaN values replaced with zeros.
        "m": The water mask tensor.
        "observed_x": A tensor of shape (batch_size, window_size, bands, height, width) indicating which values in
            the images tensor are observed (not NaN).
        "observed_y": A tensor of shape (batch_size, height, width) indicating which values in the ground
            truth labels tensor are observed (not NaN).
    """
    ...


def extract_batch(
        batch: Dict[str, Tensor],
        mask_input: bool = False,
        classify: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]:
    x = batch["images"]
    y = batch["ground_truth"]
    m = batch["water_mask"]

    # Observed values
    observed_y = y >= 0 if classify else ~y.isnan()
    observed_x = ~x.isnan() if mask_input else None

    # Remove NaNs
    x = x.nan_to_num()
    y = y.nan_to_num()

    if mask_input:
        return x, y, m, observed_x, observed_y  # type: ignore

    return x, y, m, observed_y


def mask_output(
        predictions: Tensor,
        expected: Tensor,
        observed: Tensor,
        classify: bool = False
) -> Tuple[Tensor, Tensor]:
    """
    Mask the output of a model's predictions and expected output based on the values in the "observed" tensor.

    Args:
        predictions: Tensor of model predictions with shape (batch_size, num_classes, height, width)
        expected: Tensor of expected output with shape (batch_size, height, width)
        observed: Tensor of observed output with shape (batch_size, height, width)
        classify: bool indicating whether to mask the output for classification.

    Returns:
        Tuple[Tensor, Tensor]: Masked version of the "predictions" and "expected" tensors.
    """
    if classify:
        m = torch.where(observed)
        m = (m[0], slice(None), m[1], m[2])
        return predictions[m], expected[observed]

    return predictions[observed], expected[observed]
