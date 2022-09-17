from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class SequentialCVAE(nn.Module):
    def __init__(
        self,
        encoder_layers: List[nn.Module],
        decoder_layers: List[nn.Module],
        z_dim: int,
        free_nats: float = 0.0,
        z_std_bias_adjust: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.z_dim = z_dim
        self.free_nats = free_nats

        with torch.no_grad():
            # Initialize with low variance
            self.encoder[-1].bias[z_dim:] += z_std_bias_adjust

    def forward(
        self, cond: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, pred_mode=False
    ):
        kl_loss = None
        prior = self.get_prior(cond.shape[:-1], cond.device)

        # CVAE encoder: input target + conditioning
        if not pred_mode:
            posterior = self.encode(cond, x)
            kl_loss = self.kl_loss(posterior, prior, mask)

        # CVAE decoder: input conditioning, reconstruct target
        if pred_mode:
            z = prior.sample()
        else:
            z = posterior.rsample()
        pred = self.decode(cond, z)

        return pred, z, kl_loss

    def encode(self, cond: torch.Tensor, x: torch.Tensor):
        y = torch.cat([cond, x], dim=-1)
        y = self.encoder(y)
        mean, log_std = torch.split(y, y.shape[-1] // 2, dim=-1)
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def decode(self, cond: torch.Tensor, z: Optional[torch.Tensor] = None):
        if z is None:
            z = self.get_prior(cond.shape[:-1], cond.device).rsample()
        y = torch.cat([cond, z], dim=-1)
        y = self.decoder(y)
        return y

    def get_prior(self, shape=None, device=None):
        shape = (self.z_dim,) if shape is None else (*shape, self.z_dim)
        return torch.distributions.Normal(
            torch.zeros(shape, device=device), torch.ones(shape, device=device)
        )

    def kl_loss(
        self,
        posterior: torch.distributions.Distribution,
        prior: torch.distributions.Distribution,
        mask: torch.Tensor,
    ):
        kl = torch.distributions.kl_divergence(posterior, prior)
        kl = torch.maximum(kl, torch.tensor(self.free_nats, device=kl.device))
        kl = kl.mean(dim=-1)  # average over latent dimensions
        kl = kl * mask  # mask by position
        kl = kl.sum() / mask.sum()  # average over positions and batch
        return kl


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = F.mse_loss(input, target, reduction="none")
    mask = mask[(...,) + (None,) * (loss.ndim - mask.ndim)]
    loss = loss * mask
    loss = loss.view(loss.shape[0], -1).sum(-1) / mask.view(mask.shape[0], -1).sum(-1)
    return loss.mean()


def argmax_gather(scores, values):
    scores, values = torch.as_tensor(scores), torch.as_tensor(values)
    indices = scores.argmax(-1, keepdim=True)
    indices = indices[(..., *((None,) * (values.ndim - indices.ndim)))]
    indices = indices.expand(*scores.shape[:-1], 1, *values.shape[scores.ndim :])
    values = values.expand(*scores.shape, *values.shape[scores.ndim :])
    return values.gather(scores.ndim - 1, indices)
