from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .spiking_core import MCNState, MultiCompartmentCell
from .state import MultiScaleState


@dataclass
class RSSMStats:
    prior_fast: torch.Tensor
    post_fast: torch.Tensor
    prior_med: torch.Tensor
    post_med: torch.Tensor
    prior_slow: torch.Tensor
    post_slow: torch.Tensor


class ScaleTransition(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim + action_dim, hidden_dim)
        self.prior = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior = nn.Linear(hidden_dim + input_dim, latent_dim * 2)

    def _sample(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = stats.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z, mean, logvar

    def forward_prior(self, x: torch.Tensor, action: torch.Tensor, h_prev: torch.Tensor):
        h = self.gru(torch.cat([x, action], dim=-1), h_prev)
        z, mean, logvar = self._sample(self.prior(h))
        return h, z, mean, logvar

    def forward_post(self, enc: torch.Tensor, h: torch.Tensor):
        z, mean, logvar = self._sample(self.posterior(torch.cat([h, enc], dim=-1)))
        return z, mean, logvar


class HierarchicalRSSM(nn.Module):
    """ANN first + optional spiking transition core for fast stream."""

    def __init__(self, enc_dim: int, action_dim: int, hidden_dim: int, latent_dim: int, use_spiking_core: bool = False):
        super().__init__()
        self.use_spiking_core = use_spiking_core
        self.fast = ScaleTransition(enc_dim, action_dim, hidden_dim, latent_dim)
        self.med = ScaleTransition(enc_dim + latent_dim, action_dim, hidden_dim, latent_dim)
        self.slow = ScaleTransition(enc_dim + latent_dim, action_dim, hidden_dim, latent_dim)
        self.fast_imagine_in = nn.Linear(latent_dim, enc_dim)
        self.med_imagine_in = nn.Linear(latent_dim * 2, enc_dim + latent_dim)
        self.slow_imagine_in = nn.Linear(latent_dim * 2, enc_dim + latent_dim)
        if use_spiking_core:
            self.spiking = MultiCompartmentCell(enc_dim + action_dim, latent_dim, hidden_dim)
            self.fast_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.spiking = None
            self.fast_proj = None

    @staticmethod
    def _uncertainty(*logvars: torch.Tensor) -> torch.Tensor:
        unc = sum(lv.exp().mean(dim=-1, keepdim=True) for lv in logvars) / len(logvars)
        return unc

    def observe(self, enc: torch.Tensor, action: torch.Tensor, prev: MultiScaleState) -> tuple[MultiScaleState, RSSMStats]:
        if self.use_spiking_core and self.spiking is not None and self.fast_proj is not None:
            spk_prev = MCNState(prev.h_fast, prev.h_fast, prev.h_fast, prev.h_fast)
            spk = self.spiking(torch.cat([enc, action], dim=-1), prev.z_slow, spk_prev)
            h_fast_seed = self.fast_proj(spk.v_soma)
            event_trigger = (spk.spikes.mean(dim=-1, keepdim=True) > 0).to(enc.dtype)
        else:
            h_fast_seed = prev.h_fast
            event_trigger = torch.ones(enc.shape[0], 1, device=enc.device, dtype=enc.dtype)

        h_fast, zf_prior, zf_mu_p, zf_lv_p = self.fast.forward_prior(enc, action, h_fast_seed)
        zf_post, zf_mu_q, zf_lv_q = self.fast.forward_post(enc, h_fast)

        med_in = torch.cat([enc, zf_post], dim=-1)
        h_med_new, zm_prior, zm_mu_p, zm_lv_p = self.med.forward_prior(med_in, action, prev.h_med)
        zm_post_new, zm_mu_q, zm_lv_q = self.med.forward_post(med_in, h_med_new)
        h_med = h_med_new * event_trigger + prev.h_med * (1 - event_trigger)
        zm_post = zm_post_new * event_trigger + prev.z_med * (1 - event_trigger)

        slow_in = torch.cat([enc, zm_post], dim=-1)
        h_slow_new, zs_prior, zs_mu_p, zs_lv_p = self.slow.forward_prior(slow_in, action, prev.h_slow)
        zs_post_new, zs_mu_q, zs_lv_q = self.slow.forward_post(slow_in, h_slow_new)
        h_slow = h_slow_new * event_trigger + prev.h_slow * (1 - event_trigger)
        zs_post = zs_post_new * event_trigger + prev.z_slow * (1 - event_trigger)

        uncertainty = self._uncertainty(zf_lv_q, zm_lv_q, zs_lv_q)
        state = MultiScaleState(h_fast, h_med, h_slow, zf_post, zm_post, zs_post, uncertainty)
        stats = RSSMStats(
            prior_fast=torch.cat([zf_mu_p, zf_lv_p], dim=-1),
            post_fast=torch.cat([zf_mu_q, zf_lv_q], dim=-1),
            prior_med=torch.cat([zm_mu_p, zm_lv_p], dim=-1),
            post_med=torch.cat([zm_mu_q, zm_lv_q], dim=-1),
            prior_slow=torch.cat([zs_mu_p, zs_lv_p], dim=-1),
            post_slow=torch.cat([zs_mu_q, zs_lv_q], dim=-1),
        )
        return state, stats

    def imagine(self, action: torch.Tensor, prev: MultiScaleState) -> MultiScaleState:
        fast_in = self.fast_imagine_in(prev.z_fast)
        med_in = self.med_imagine_in(torch.cat([prev.z_med, prev.z_fast], dim=-1))
        slow_in = self.slow_imagine_in(torch.cat([prev.z_slow, prev.z_med], dim=-1))
        h_fast, zf, _, zf_lv = self.fast.forward_prior(fast_in, action, prev.h_fast)
        h_med, zm, _, zm_lv = self.med.forward_prior(med_in, action, prev.h_med)
        h_slow, zs, _, zs_lv = self.slow.forward_prior(slow_in, action, prev.h_slow)
        uncertainty = self._uncertainty(zf_lv, zm_lv, zs_lv)
        return MultiScaleState(h_fast, h_med, h_slow, zf, zm, zs, uncertainty)


def balanced_kl(prior_stats: torch.Tensor, post_stats: torch.Tensor, dyn_scale: float = 0.5, rep_scale: float = 0.1) -> torch.Tensor:
    p_mu, p_lv = prior_stats.chunk(2, dim=-1)
    q_mu, q_lv = post_stats.chunk(2, dim=-1)

    def kl(mu1: torch.Tensor, lv1: torch.Tensor, mu2: torch.Tensor, lv2: torch.Tensor) -> torch.Tensor:
        v1, v2 = lv1.exp(), lv2.exp()
        return 0.5 * (lv2 - lv1 + (v1 + (mu1 - mu2).pow(2)) / (v2 + 1e-6) - 1.0).sum(dim=-1).mean()

    dyn = kl(q_mu.detach(), q_lv.detach(), p_mu, p_lv)
    rep = kl(q_mu, q_lv, p_mu.detach(), p_lv.detach())
    return dyn_scale * dyn + rep_scale * rep
