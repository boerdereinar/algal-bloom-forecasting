from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class SingleChannelInterpolation(Module):
    def __init__(
            self,
            window_size: int,
            prediction_horizon: int,
            reconstruction: bool = False,
            kappa: float = 10
    ) -> None:
        super().__init__()

        self.kappa = kappa
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.reconstruction = reconstruction
        self.kernel = Parameter(torch.zeros((self.window_size,)))

    def forward(self, x_t: Tensor, d: Tensor, m: Tensor) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        if self.reconstruction:
            output_dim = self.window_size
            ref_t = d.unsqueeze(2).expand(-1, -1, output_dim, -1)
        else:
            output_dim = self.prediction_horizon
            ref_t = torch.linspace(0, output_dim, output_dim)[None, :].to(x_t)

        x_t = x_t.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        d = d.unsqueeze(-1).expand(-1, -1, -1, output_dim)
        m = m.unsqueeze(-1).expand(-1, -1, -1, output_dim)

        norm = (d - ref_t) * (d - ref_t)
        alpha = torch.log(1 + torch.exp(self.kernel))[None, None, :, None]
        w = torch.logsumexp(-alpha * norm + torch.log(m), dim=2)
        w1 = w.unsqueeze(2).expand(-1, -1, self.window_size, -1)
        w1 = torch.exp(-alpha * norm + torch.log(m) - w1)
        y = torch.einsum("ijkl,ijkl->ijl", w1, x_t)

        if self.reconstruction:
            return y, w

        w_t = torch.logsumexp(-self.kappa * alpha * norm + torch.log(m), dim=2)
        w_t = w_t.unsqueeze(2).expand(-1, -1, self.window_size, -1)
        w_t = torch.exp(-self.kappa * alpha * norm + torch.log(m) - w_t)
        y_trans = torch.einsum("ijkl,ijkl->ijl", w_t, x_t)
        return y, w, y_trans


class CrossChannelInterpolation(Module):
    def __init__(
            self,
            n_features: int,
            window_size: int,
            prediction_horizon: int,
            reconstruction: bool = False
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.reconstruction = reconstruction
        self.cross_channel_interpolation = Parameter(torch.eye(self.n_features))

    def forward(self, y: Tensor, w: Tensor, y_trans: Optional[Tensor] = None) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2 = w
        w = w.unsqueeze(-1).expand(-1, -1, -1, self.n_features)
        den = w.logsumexp(dim=2)
        w = torch.exp(w2 - den)
        mean = y.mean(dim=1)
        mean = mean[:, None, :].expand(-1, self.prediction_horizon, -1)
        w2 = (w * (y - mean)) @ self.cross_channel_interpolation + mean
        rep1 = w2.permute(0, 2, 1)

        if self.reconstruction:
            return rep1

        y_trans = y_trans - rep1
        return rep1, intensity, y_trans
