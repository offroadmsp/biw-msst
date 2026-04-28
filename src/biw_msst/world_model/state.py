from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultiScaleState:
    """Structured latent belief state used by BIW-MSST."""

    h_fast: torch.Tensor
    h_med: torch.Tensor
    h_slow: torch.Tensor
    z_fast: torch.Tensor
    z_med: torch.Tensor
    z_slow: torch.Tensor
    uncertainty: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "h_fast": self.h_fast,
            "h_med": self.h_med,
            "h_slow": self.h_slow,
            "z_fast": self.z_fast,
            "z_med": self.z_med,
            "z_slow": self.z_slow,
            "uncertainty": self.uncertainty,
        }

    @property
    def batch_size(self) -> int:
        return self.h_fast.shape[0]


def init_state(batch_size: int, hidden_dim: int, latent_dim: int, device: torch.device) -> MultiScaleState:
    zeros_h = torch.zeros(batch_size, hidden_dim, device=device)
    zeros_z = torch.zeros(batch_size, latent_dim, device=device)
    uncertainty = torch.ones(batch_size, 1, device=device)
    return MultiScaleState(
        h_fast=zeros_h.clone(),
        h_med=zeros_h.clone(),
        h_slow=zeros_h.clone(),
        z_fast=zeros_z.clone(),
        z_med=zeros_z.clone(),
        z_slow=zeros_z.clone(),
        uncertainty=uncertainty,
    )
