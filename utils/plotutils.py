import io
from typing import Optional

import wandb
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pytorch_lightning.loggers import Logger, WandbLogger
from torch import Tensor

from edegruyl.datasets import RioNegroDataset


def plot_predicted(predicted: Tensor, expected: Tensor) -> Figure:
    """
    Plots the predicted and expected values.

    Args:
        predicted: The predicted values.
        expected: The expected values.

    Returns:
        The figure with the plotted values.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200, figsize=(15, 4.8))

    max_chlorophyll = RioNegroDataset.CLIP[1]
    predicted *= max_chlorophyll
    expected *= max_chlorophyll

    ax1.set_title("predicted")
    ax1.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    im = ax1.imshow(predicted.cpu(), vmin=0, vmax=max_chlorophyll, interpolation=None)

    ax2.set_title("expected")
    ax2.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax2.imshow(expected.cpu(), vmin=0, vmax=max_chlorophyll, interpolation=None)

    # Add colorbar
    fig.tight_layout()
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.93, 0.11, 0.02, 0.78])
    fig.colorbar(im, cax=cbar_ax, label="μgL⁻¹")

    return fig


def plot_rmse(rmse: Tensor, global_rmse: Tensor) -> Figure:
    """
    Plots the RMSE loss.

    Args:
        rmse: The RMSE loss.
        global_rmse: The RMSE loss for the entire test run.

    Returns:
        The figure with the plotted loss.
    """
    max_chlorophyll = RioNegroDataset.CLIP[1]
    rmse *= max_chlorophyll
    global_rmse *= max_chlorophyll

    fig = plt.figure(dpi=200, figsize=(8, 4))
    plt.title(f"RMSE loss ({global_rmse} μgL⁻¹)")
    plt.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.imshow(rmse.cpu(), vmin=0, vmax=max_chlorophyll, interpolation=None)
    plt.colorbar(label="μgL⁻¹")

    return fig


def figure_to_image(fig: Figure) -> Image:
    """
    Converts a matplotlib Figure to a PIL Image.

    Args:
        fig: The figure to convert.

    Returns:
        The converted PIL Image.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    return Image.open(buffer)


def log_figure(fig: Figure, logger: Logger, key: str, path: Optional[str]) -> None:
    """
    Logs a figure either to a WandbLogger with key or writes the image to a path.

    Args:
        fig: The figure.
        logger: The logger used for the training.
        key: The key to use with the logger.
        path: The path to write the image to.
    """
    if isinstance(logger, WandbLogger):
        img = figure_to_image(fig)
        logger.log_image(key, [wandb.Image(img)])
    elif path:
        fig.savefig(path)

    plt.close(fig)
