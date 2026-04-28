from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    """Straight-through surrogate for spikes."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = grad_output / (1 + alpha * x.abs()).pow(2)
        return grad, None, None


def spike_fn(x: torch.Tensor, threshold: float = 0.0, alpha: float = 5.0) -> torch.Tensor:
    return SurrogateSpike.apply(x, threshold, alpha)


@dataclass
class MCNState:
    v_basal: torch.Tensor
    v_apical: torch.Tensor
    v_soma: torch.Tensor
    spikes: torch.Tensor


class MultiCompartmentCell(nn.Module):
    """Basal + apical + soma MCN cell for hybrid spiking RSSM."""

    def __init__(self, input_dim: int, context_dim: int, hidden_dim: int, tau: float = 0.9):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.recurrent = nn.Linear(hidden_dim, hidden_dim)
        self.tau = tau

    def forward(self, x: torch.Tensor, context: torch.Tensor, prev: MCNState) -> MCNState:
        basal_cur = self.input_proj(x)
        apical_cur = self.context_proj(context)
        rec_cur = self.recurrent(prev.spikes)

        v_basal = self.tau * prev.v_basal + (1 - self.tau) * basal_cur
        v_apical = self.tau * prev.v_apical + (1 - self.tau) * apical_cur
        v_soma = self.tau * prev.v_soma + (1 - self.tau) * (v_basal + v_apical + rec_cur)
        spikes = spike_fn(v_soma)
        v_soma = v_soma * (1.0 - spikes)
        return MCNState(v_basal=v_basal, v_apical=v_apical, v_soma=v_soma, spikes=spikes)

    def init_state(self, batch_size: int, hidden_dim: int, device: torch.device) -> MCNState:
        zeros = torch.zeros(batch_size, hidden_dim, device=device)
        return MCNState(v_basal=zeros.clone(), v_apical=zeros.clone(), v_soma=zeros.clone(), spikes=zeros.clone())
