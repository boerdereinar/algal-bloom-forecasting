from typing import Literal, Tuple, overload

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss


class InterpolationNetwork(nn.Module):
    """
    A neural network that interpolates between observed and reference points.

    Parameters:
        num_features (int): The number of features in the input data.
        observed_points (int): The number of observed points.
        reference_points (int): The number of reference points.
    """
    def __init__(
            self,
            num_features,
            observed_points,
            reference_points
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.observed_points = observed_points
        self.reference_points = reference_points

        self.interpolate = nn.Sequential(
            SingleChannelInterpolation(observed_points, reference_points),
            nn.Sigmoid(),
            CrossChannelInterpolation(num_features, observed_points, reference_points),
            nn.Sigmoid()
        )

        self.reconstruct = nn.Sequential(
            SingleChannelInterpolation(observed_points, reference_points, True),
            nn.Sigmoid(),
            CrossChannelInterpolation(num_features, observed_points, reference_points, True),
            nn.Sigmoid()
        )

    @overload
    def forward(
            self,
            x: Tensor,
            water_mask: Tensor,
            observed_x: Tensor,
            masked: bool = False,
            return_reconstructed: Literal[False] = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Interpolate and reconstruct the input data.

        Args:
            x: The images tensor of shape (batch, observed_points, num_features, height, width)
            water_mask: The water mask tensor of shape (batch, height, width)
            observed_x: A tensor of shape (batch_size, observed_points, num_features, height, width) indicating which
                values in the images tensor are observed (not NaN).
            masked: Whether the interpolation is only applied to the area masked by the water_mask.
            return_reconstructed: Whether to return the reconstructed Tensor.

        Returns:
            A tuple containing:
            The interpolated tensor features of shape (batch_size, 3, num_features, reference_points, height, width).
            The mse loss of the reconstructed tensor.
        """
        ...

    @overload
    def forward(
            self,
            x: Tensor,
            water_mask: Tensor,
            observed_x: Tensor,
            masked: bool = False,
            return_reconstructed: Literal[True] = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Interpolate and reconstruct the input data.

        Args:
            x: The images tensor of shape (batch_size, observed_points, num_features, height, width)
            water_mask: The water mask tensor of shape (batch_size, height, width)
            observed_x: A tensor of shape (batch_size, observed_points, num_features, height, width) indicating which
                values in the images tensor are observed (not NaN).
            masked: Whether the interpolation is only applied to the area masked by the water_mask.
            return_reconstructed: Whether to return the reconstructed Tensor.

        Returns:
            A tuple containing:
            The interpolated tensor features of shape (batch_size, 3, num_features, reference_points, height, width).
            The mse loss of the reconstructed tensor.
            The reconstructed images tensor with the same shape as the images tensor.
        """
        ...

    def forward(
            self,
            x: Tensor,
            water_mask: Tensor,
            observed_x: Tensor,
            masked: bool = False,
            return_reconstructed: bool = False
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        batch_size, observed_points, num_features, height, width = x.shape

        assert observed_points == self.observed_points
        assert num_features == self.num_features

        masked_indices = None

        # Reshapes the input tensors
        # x_t: (n, num_features, observed_points)
        # d:   (n, num_features, observed_points)
        # m:   (n, num_features, observed_points)
        # n:   (batch_size * height * width) if masked is False, (batch_size * masked area) otherwise.
        if masked:
            masked_indices = torch.where(water_mask)
            x_t = x.permute(0, 3, 4, 2, 1)[masked_indices]
            d = torch.arange(observed_points).to(x)
            d = d[None, None, :].expand(*x_t.shape)
            m = observed_x.permute(0, 3, 4, 2, 1)[masked_indices]
        else:
            x_t = x.permute(0, 3, 4, 2, 1).reshape(-1, num_features, observed_points)
            d = torch.arange(observed_points).to(x)
            d = d[None, None, :].expand(*x_t.shape)
            m = observed_x.permute(0, 3, 4, 2, 1)
            m = m.reshape(-1, num_features, observed_points)

        # Feed the result into the interpolation network
        # This returns a tensor of shape (3, n, num_features, reference_points)
        x_in = torch.stack((x_t, d, m))
        interp_out = self.interpolate(x_in)

        # Reshapes the interpolated tensor to (batch_size, reference_points, 3, num_features, height, width)
        if masked:
            y = torch.zeros((batch_size, height, width, 3, num_features, self.reference_points))
            y[masked_indices] = interp_out.permute(1, 0, 2, 3)
            y = y.permute(0, 5, 3, 4, 1, 2)
        else:
            y = interp_out.view(3, batch_size, height, width, num_features, self.reference_points)
            y = y.permute(1, 5, 0, 4, 2, 3)

        # Hold out 20% of the input data for reconstruction
        m_holdout = (torch.rand_like(m) > 0.2) & m

        # Feed the result into the reconstruction network
        # This returns a tensor of shape (n, num_features, observed_points)
        x_in = torch.stack((x_t, d, m_holdout))
        reconst_out = self.reconstruct(x_in)

        # Get which values were held out and compute the loss
        held_out = ~m_holdout & m
        mse = mse_loss(reconst_out[held_out], x_t[held_out])

        # Return the reconstructed images tensor.
        if return_reconstructed:
            # Reshapes the reconstructed tensor to (batch_size, observed_points, num_features, height, width)
            if masked:
                y_reconst = torch.zeros((batch_size, height, width, num_features, observed_points))
                y_reconst[masked_indices] = reconst_out
                y_reconst = y_reconst.permute(0, 4, 3, 1, 2)
            else:
                y_reconst = reconst_out.view(batch_size, height, width, num_features, observed_points)
                y_reconst = y_reconst.permute(0, 4, 3, 1, 2)

            return y, mse, y_reconst

        return y, mse


class SingleChannelInterpolation(nn.Module):
    """
    A neural network layer that interpolates between observed and reference points for a single channel.

    Parameters:
        observed_points (int): The number of observed points.
        reference_points (int): The number of reference points.
        reconstruction (bool, optional): Flag indicating whether the network is used for reconstruction.
            If set to True, the output will only be the interpolated points, whereas if set to False,
            the output will also include the transformed points. Default: False.
        kappa (float, optional): A hyperparameter that controls the strength of the transformation. Default: 10.
    """
    def __init__(
            self,
            observed_points: int,
            reference_points: int,
            reconstruction: bool = False,
            kappa: float = 10
    ) -> None:
        super().__init__()

        self.observed_points = observed_points
        self.reference_points = reference_points
        self.reconstruction = reconstruction
        self.kappa = kappa
        self.kernel = nn.Parameter(torch.zeros((self.observed_points,)))

    def forward(self, x: Tensor) -> Tensor:
        x_t = x[0]
        d = x[1]
        m = x[2]

        if self.reconstruction:
            output_dim = self.observed_points
            ref_t = d.unsqueeze(2).expand(-1, -1, output_dim, -1)
        else:
            output_dim = self.reference_points
            ref_t = torch.linspace(0, output_dim, output_dim)[None, :].to(x_t)

        x_t = x_t.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        d = d.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        m = m.unsqueeze(-1).expand(-1, -1, -1, output_dim)

        norm = (d - ref_t) * (d - ref_t)
        alpha = torch.log(1 + torch.exp(self.kernel))[None, None, :, None]
        w = torch.logsumexp(-alpha * norm + torch.log(m), dim=2)
        w1 = w.unsqueeze(2).expand(-1, -1, self.observed_points, -1)
        w1 = torch.exp(-alpha * norm + torch.log(m) - w1)
        y = torch.einsum("ijkl,ijkl->ijl", w1, x_t)

        if self.reconstruction:
            return torch.stack((y, w))

        w_t = torch.logsumexp(-self.kappa * alpha * norm + torch.log(m), dim=2)
        w_t = w_t.unsqueeze(2).expand(-1, -1, self.observed_points, -1)
        w_t = torch.exp(-self.kappa * alpha * norm + torch.log(m) - w_t)
        y_trans = torch.einsum("ijkl,ijkl->ijl", w_t, x_t)
        return torch.stack((y, w, y_trans))


class CrossChannelInterpolation(nn.Module):
    """
    A neural network layer that interpolates between observed and reference points for multiple channels.

    Parameters:
        num_features (int): The number of features in the input data.
        observed_points (int): The number of observed points.
        reference_points (int): The number of reference points.
        reconstruction (bool, optional): Flag indicating whether the network is used for reconstruction.
            If set to True, the output will only be the interpolated points, whereas if set to False,
            the output will also include the transformed points. Default: False.
    """
    def __init__(
            self,
            num_features: int,
            observed_points: int,
            reference_points: int,
            reconstruction: bool = False
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.output_dim = observed_points if reconstruction else reference_points
        self.reconstruction = reconstruction
        self.cross_channel_interpolation = nn.Parameter(torch.eye(self.num_features))

    def forward(self, x: Tensor) -> Tensor:
        y = x[0]
        w = x[1]

        intensity = torch.exp(w)

        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2 = w
        w = w.unsqueeze(-1).expand(-1, -1, -1, self.num_features)
        den = w.logsumexp(dim=2)
        w = torch.exp(w2 - den)
        mean = y.mean(dim=1)
        mean = mean[:, None, :].expand(-1, self.output_dim, -1)
        w2 = (w * (y - mean)) @ self.cross_channel_interpolation + mean
        rep1 = w2.permute(0, 2, 1)

        if self.reconstruction:
            return rep1

        y_trans = x[2]
        y_trans = y_trans - rep1
        return torch.stack((rep1, intensity, y_trans))
