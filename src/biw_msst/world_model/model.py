from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .msst_block import MultiScaleSTBlock
from .rssm import HierarchicalRSSM, RSSMStats
from .state import MultiScaleState, init_state
from .state_heads import StructuredStateHeads


@dataclass
class WorldModelConfig:
    obs_dim: int = 256
    action_dim: int = 8
    hidden_dim: int = 256
    latent_dim: int = 64
    token_dim: int = 128
    graph_tokens: int = 16
    use_spiking_core: bool = False


class WorldModel(nn.Module):
    """MSST-BIW world model skeleton inspired by Dreamer-like training flow."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.LayerNorm(cfg.obs_dim),
            nn.Linear(cfg.obs_dim, cfg.token_dim),
            nn.GELU(),
            nn.Linear(cfg.token_dim, cfg.token_dim),
        )
        self.fast_pool = nn.Identity()
        self.med_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.slow_pool = nn.AvgPool1d(kernel_size=4, stride=4)

        self.graph_memory = nn.Parameter(torch.randn(1, cfg.graph_tokens, cfg.token_dim) * 0.02)
        self.st_block = MultiScaleSTBlock(cfg.token_dim)

        self.enc_to_rssm = nn.Linear(cfg.token_dim * 4, cfg.hidden_dim)
        self.rssm = HierarchicalRSSM(
            enc_dim=cfg.hidden_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            use_spiking_core=cfg.use_spiking_core,
        )
        self.decoder = nn.Linear(cfg.hidden_dim * 3 + cfg.latent_dim * 3, cfg.obs_dim)
        self.heads = StructuredStateHeads(cfg.hidden_dim, cfg.latent_dim)

    def _make_scales(self, enc_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # enc_seq: [B, T, D]
        fast = self.fast_pool(enc_seq)
        med = self.med_pool(enc_seq.transpose(1, 2)).transpose(1, 2)
        slow = self.slow_pool(enc_seq.transpose(1, 2)).transpose(1, 2)
        return fast, med, slow

    def _fuse_tokens(self, fast: torch.Tensor, med: torch.Tensor, slow: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        return torch.cat([fast.mean(1), med.mean(1), slow.mean(1), graph], dim=-1)

    def init_state(self, batch_size: int, device: torch.device | None = None) -> MultiScaleState:
        device = device if device is not None else next(self.parameters()).device
        return init_state(batch_size, self.cfg.hidden_dim, self.cfg.latent_dim, device)

    def observe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        prev_state: MultiScaleState,
    ) -> tuple[MultiScaleState, RSSMStats, dict[str, torch.Tensor]]:
        enc_seq = self.encoder(obs)
        fast, med, slow = self._make_scales(enc_seq)
        graph_tokens = self.graph_memory.expand(obs.shape[0], -1, -1)
        fast, med, slow, graph = self.st_block(fast, med, slow, graph_tokens)
        fused = self._fuse_tokens(fast, med, slow, graph)
        rssm_in = self.enc_to_rssm(fused)
        state, stats = self.rssm.observe(rssm_in, action, prev_state)
        preds = self.predict_heads(state)
        return state, stats, preds

    def imagine(self, action: torch.Tensor, prev_state: MultiScaleState) -> MultiScaleState:
        return self.rssm.imagine(action, prev_state)

    def decode(self, state: MultiScaleState) -> torch.Tensor:
        z = torch.cat([state.h_fast, state.h_med, state.h_slow, state.z_fast, state.z_med, state.z_slow], dim=-1)
        return self.decoder(z)

    def predict_heads(self, state: MultiScaleState) -> dict[str, torch.Tensor]:
        return self.heads(state)
