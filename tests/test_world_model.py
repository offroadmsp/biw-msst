import torch

from biw_msst.world_model.model import WorldModel, WorldModelConfig
from biw_msst.world_model.rssm import balanced_kl


def test_world_model_observe_and_decode_smoke():
    cfg = WorldModelConfig(obs_dim=48, action_dim=6, hidden_dim=64, latent_dim=16, token_dim=32)
    model = WorldModel(cfg)

    batch, time = 4, 8
    obs = torch.randn(batch, time, cfg.obs_dim)
    action = torch.randn(batch, cfg.action_dim)

    state0 = model.init_state(batch, obs.device)
    state1, stats, preds = model.observe(obs, action, state0)

    assert state1.h_fast.shape == (batch, cfg.hidden_dim)
    assert preds["s_place"].shape[0] == batch
    recon = model.decode(state1)
    assert recon.shape == (batch, cfg.obs_dim)

    kl = balanced_kl(stats.prior_fast, stats.post_fast)
    assert torch.isfinite(kl)


def test_world_model_spiking_mode_smoke():
    cfg = WorldModelConfig(
        obs_dim=48,
        action_dim=6,
        hidden_dim=64,
        latent_dim=16,
        token_dim=32,
        use_spiking_core=True,
    )
    model = WorldModel(cfg)
    batch, time = 2, 8
    obs = torch.randn(batch, time, cfg.obs_dim)
    action = torch.randn(batch, cfg.action_dim)
    state = model.init_state(batch, obs.device)
    next_state, _, _ = model.observe(obs, action, state)
    assert next_state.z_fast.shape == (batch, cfg.latent_dim)
