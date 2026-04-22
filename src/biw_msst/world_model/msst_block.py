from __future__ import annotations

import torch
from torch import nn


class ScaleRouter(nn.Module):
    """Learns weights over fast/med/slow streams."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.LayerNorm(token_dim * 3),
            nn.Linear(token_dim * 3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, 3),
        )

    def forward(self, fast: torch.Tensor, med: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fast, med, slow], dim=-1)
        return torch.softmax(self.router(x), dim=-1)


class EventGate(nn.Module):
    """Event-driven update gate for memory writes/fusion."""

    def __init__(self, token_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.gate(token)


class MultiScaleSTBlock(nn.Module):
    """Minimal multi-scale ST transformer with graph cross-attention."""

    def __init__(self, token_dim: int, num_heads: int = 4):
        super().__init__()
        self.self_fast = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.self_med = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.self_slow = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        self.cross_fast_med = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.cross_med_slow = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.graph_cross = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)

        self.router = ScaleRouter(token_dim)
        self.event_gate = EventGate(token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def _self_update(self, attn: nn.MultiheadAttention, tokens: torch.Tensor) -> torch.Tensor:
        out, _ = attn(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + out)

    def forward(
        self,
        fast_tokens: torch.Tensor,
        med_tokens: torch.Tensor,
        slow_tokens: torch.Tensor,
        graph_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = self._self_update(self.self_fast, fast_tokens)
        med = self._self_update(self.self_med, med_tokens)
        slow = self._self_update(self.self_slow, slow_tokens)

        fast2, _ = self.cross_fast_med(fast, med, med, need_weights=False)
        med2, _ = self.cross_med_slow(med, slow, slow, need_weights=False)
        fast = self.norm(fast + fast2)
        med = self.norm(med + med2)

        combined = torch.cat(
            [fast.mean(dim=1), med.mean(dim=1), slow.mean(dim=1)],
            dim=-1,
        )
        weights = self.router(fast.mean(dim=1), med.mean(dim=1), slow.mean(dim=1))
        fused = (
            weights[:, 0:1] * fast.mean(dim=1)
            + weights[:, 1:2] * med.mean(dim=1)
            + weights[:, 2:3] * slow.mean(dim=1)
        )

        gate = self.event_gate(fused)
        gated_query = (gate * fused).unsqueeze(1)
        graph_out, _ = self.graph_cross(gated_query, graph_tokens, graph_tokens, need_weights=False)
        return fast, med, slow, graph_out.squeeze(1)
