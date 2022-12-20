import torch
from torch.nn import Module, Parameter


class SingleChannelInterpolation(Module):
    def __init__(self, window_size: int, prediction_horizon: int, reconstruction: bool = False):
        super().__init__()

        self.kappa = 10.0
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.reconstruction = reconstruction
        self.kernel = Parameter(torch.zeros((self.window_size,)))

    def forward(self, x_t, d, m):
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
            rep1 = torch.cat([y, w], 1)
        else:
            w_t = torch.logsumexp(-self.kappa * alpha * norm + torch.log(m), dim=2)
            w_t = w_t.unsqueeze(2).expand(-1, -1, self.window_size, -1)
            w_t = torch.exp(-self.kappa * alpha * norm + torch.log(m) - w_t)
            y_trans = torch.einsum("ijkl,ijkl->ijl", w_t, x_t)
            rep1 = torch.cat([y, w, y_trans], 1)

        return rep1
