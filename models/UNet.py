import torch
from torch import Tensor, nn


class UNet(nn.Module):
    """A UNet-based neural network for image segmentation.

    This class extends the PyTorch `nn.Module` class to create a UNet model for image segmentation.
    The UNet architecture is based on the implementation at https://github.com/milesial/Pytorch-UNet.

    Args:
        in_channels (int): The number of input channels to the model.
        out_channels (int): The number of output channels from the model.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """Performs the forward pass of the UNet model.

        Args:
            x (Tensor): The input tensor to the model, of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output of the model, of shape (batch_size, out_channels, height, width).
        """
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decode
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x


class DoubleConv(nn.Module):
    """A double convolutional layer with batch normalization and ReLU activation.

    This class extends the PyTorch `nn.Module` class to create a double convolutional layer with
    batch normalization and ReLU activation. It is used as a building block in the UNet model.

    Args:
        in_channels (int): The number of input channels to the layer.
        out_channels (int): The number of output channels from the layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        """Performs the forward pass of the double convolutional layer.

        Args:
            x (Tensor): The input tensor to the layer, of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output of the layer, of shape (batch_size, out_channels, height, width).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """A downsampling layer for the UNet model.

    This class extends the PyTorch `nn.Module` class to create a downsampling layer for the UNet model.
    It consists of a max pooling layer followed by a `DoubleConv` layer.

    Args:
        in_channels (int): The number of input channels to the layer.
        out_channels (int): The number of output channels from the layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Performs the forward pass of the downsampling layer.

        Args:
            x (Tensor): The input tensor to the layer, of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output of the layer, of shape (batch_size, out_channels, height / 2, width / 2).
        """
        return self.down(x)


class Up(nn.Module):
    """An upsampling layer for the UNet model.

    This class extends the PyTorch `nn.Module` class to create an upsampling layer for the UNet model.
    It consists of a convolutional transpose layer followed by a `DoubleConv` layer.

    Args:
        in_channels (int): The number of input channels to the layer.
        out_channels (int): The number of output channels from the layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Performs the forward pass of the upsampling layer.

        Args:
            x1 (Tensor): The first input tensor to the layer, of shape (batch_size, in_channels, height, width).
            x2 (Tensor): The second input tensor to the layer, of shape (batch_size, in_channels / 2, height, width).

        Returns:
            Tensor: The output of the layer, of shape (batch_size, out_channels, height * 2, width * 2).
        """
        x1 = self.up(x1)

        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
