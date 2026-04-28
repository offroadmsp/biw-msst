from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .world_model.model import WorldModel
from .world_model.rssm import balanced_kl
from .world_model.state_heads import compute_structured_losses


def compute_kl_loss(stats) -> torch.Tensor:
    """Aggregate balanced KL terms across fast/med/slow RSSM branches."""
    return (
        balanced_kl(stats.prior_fast, stats.post_fast)
        + balanced_kl(stats.prior_med, stats.post_med)
        + balanced_kl(stats.prior_slow, stats.post_slow)
    )


def train_step(
    model: WorldModel,
    batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_weights: dict[str, float] | None = None,
    kl_weight: float = 1.0,
) -> dict[str, float]:
    """Single optimization step using L_total + L_kl.

    Required batch keys:
    - obs: [B, T, C, H, W]
    - action: [B, A]
    - targets: dict for structured heads (must include s_goal_mask)
    """

    loss_weights = loss_weights or {}
    obs = batch["obs"]
    action = batch["action"]
    targets = batch["targets"]

    if not isinstance(obs, torch.Tensor) or not isinstance(action, torch.Tensor) or not isinstance(targets, dict):
        raise TypeError("batch must contain tensors for obs/action and dict for targets")

    model.train()
    prev_state = model.init_state(obs.shape[0], obs.device)

    optimizer.zero_grad(set_to_none=True)
    _, stats, preds = model.observe(obs, action, prev_state)

    structured_losses = compute_structured_losses(preds, targets, loss_weights)
    l_kl = compute_kl_loss(stats)
    loss = structured_losses["L_total"] + kl_weight * l_kl

    loss.backward()
    optimizer.step()

    metrics = {
        "L_total": float(structured_losses["L_total"].detach().item()),
        "L_kl": float(l_kl.detach().item()),
        "L_joint": float(loss.detach().item()),
    }
    for key, val in structured_losses.items():
        if key != "L_total":
            metrics[key] = float(val.detach().item())
    return metrics


def train_loop(
    model: WorldModel,
    dataloader: Iterable[dict[str, torch.Tensor | dict[str, torch.Tensor]]],
    optimizer: torch.optim.Optimizer,
    loss_weights: dict[str, float] | None = None,
    kl_weight: float = 1.0,
    max_steps: int | None = None,
) -> list[dict[str, float]]:
    """Minimal training loop for quick experiments."""
    history: list[dict[str, float]] = []
    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        metrics = train_step(model, batch, optimizer, loss_weights=loss_weights, kl_weight=kl_weight)
        history.append(metrics)
    return history


def build_adam(model: nn.Module, lr: float = 3e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)
