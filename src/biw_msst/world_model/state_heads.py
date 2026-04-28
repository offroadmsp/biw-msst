from __future__ import annotations

import torch
from torch import nn

from .state import MultiScaleState


def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3-style symlog transform for stable regression."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


class StructuredStateHeads(nn.Module):
    """Geometry/place/relation/prediction/belief + goal/time structured heads."""

    def __init__(self, hidden_dim: int, latent_dim: int, place_dim: int = 64):
        super().__init__()
        fused_dim = hidden_dim * 3 + latent_dim * 3 + 1
        self.backbone = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
        )
        trunk = hidden_dim * 2

        self.geo_head = nn.Linear(trunk, 3)
        self.place_head = nn.Linear(trunk, place_dim)
        self.rel_head = nn.Linear(trunk, 4)
        self.pred_head = nn.Linear(trunk, latent_dim * 3)
        self.belief_head = nn.Linear(trunk, 4)

        self.goal_mask_head = nn.Sequential(
            nn.Linear(trunk, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.time_cell_head = nn.Linear(trunk, 16)

    def forward(self, state: MultiScaleState) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                state.h_fast,
                state.h_med,
                state.h_slow,
                state.z_fast,
                state.z_med,
                state.z_slow,
                state.uncertainty,
            ],
            dim=-1,
        )
        h = self.backbone(x)
        return {
            "s_geo": self.geo_head(h),
            "s_place": self.place_head(h),
            "s_rel": self.rel_head(h),
            "s_pred": self.pred_head(h),
            "s_belief": self.belief_head(h),
            "s_goal": self.goal_mask_head(h),
            "s_time": self.time_cell_head(h),
        }


def compute_structured_losses(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    l_rec = mse(symlog(preds["s_pred"]), symlog(targets["s_pred"]))
    l_geo = mse(symlog(preds["s_geo"]), symlog(targets["s_geo"]))
    l_goal_mask = bce(preds["s_goal"], targets["s_goal_mask"])

    losses = {
        "L_rec": l_rec,
        "L_place": mse(preds["s_place"], targets["s_place"]),
        "L_topo": mse(preds["s_rel"], targets["s_rel"]),
        "L_unc": mse(preds["s_belief"], targets["s_belief"]),
        "L_geo": l_geo,
        "L_goal_mask": l_goal_mask,
    }

    if "s_time" in targets:
        losses["L_time"] = mse(preds["s_time"], targets["s_time"])

    total = 0.0
    for key, val in losses.items():
        total = total + weights.get(key, 1.0) * val
    losses["L_total"] = total
    return losses
