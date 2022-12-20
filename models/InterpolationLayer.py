import torch
from torch import Tensor, nn


class InterpolationNetwork(nn.Module):
    def __init__(
            self,
            num_features,
            observed_points,
            reference_points,
            reconstruction=False
    ) -> None:
        super().__init__()

        self.interpolation = nn.Sequential(
            SingleChannelInterpolation(observed_points, reference_points, reconstruction),
            nn.Sigmoid(),
            CrossChannelInterpolation(num_features, reference_points, reconstruction),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.interpolation(x)


class SingleChannelInterpolation(nn.Module):
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
        x_t = x[:, 0]
        d = x[:, 1]
        m = x[:, 2]

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
            return torch.stack((y, w), dim=1)

        w_t = torch.logsumexp(-self.kappa * alpha * norm + torch.log(m), dim=2)
        w_t = w_t.unsqueeze(2).expand(-1, -1, self.observed_points, -1)
        w_t = torch.exp(-self.kappa * alpha * norm + torch.log(m) - w_t)
        y_trans = torch.einsum("ijkl,ijkl->ijl", w_t, x_t)
        return torch.stack((y, w, y_trans), dim=1)


class CrossChannelInterpolation(nn.Module):
    def __init__(
            self,
            num_features: int,
            reference_points: int,
            reconstruction: bool = False
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.reference_points = reference_points
        self.reconstruction = reconstruction
        self.cross_channel_interpolation = nn.Parameter(torch.eye(self.num_features))

    def forward(self, x: Tensor) -> Tensor:
        y = x[:, 0]
        w = x[:, 1]

        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2 = w
        w = w.unsqueeze(-1).expand(-1, -1, -1, self.num_features)
        den = w.logsumexp(dim=2)
        w = torch.exp(w2 - den)
        mean = y.mean(dim=1)
        mean = mean[:, None, :].expand(-1, self.reference_points, -1)
        w2 = (w * (y - mean)) @ self.cross_channel_interpolation + mean
        rep1 = w2.permute(0, 2, 1)

        if self.reconstruction:
            return rep1

        y_trans = x[:, 2]
        y_trans = y_trans - rep1
        return torch.stack((rep1, intensity, y_trans), dim=1)
