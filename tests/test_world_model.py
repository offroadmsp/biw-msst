import torch
from torch import nn

from biw_msst.world_model.model import WorldModel, WorldModelConfig
from biw_msst.world_model.rssm import balanced_kl
from biw_msst.world_model.spiking_core import MCNState
from biw_msst.world_model.state_heads import compute_structured_losses, symlog
from biw_msst.train import build_adam, train_step


def test_world_model_observe_and_decode_smoke():
    cfg = WorldModelConfig(
        obs_shape=(3, 64, 64),
        obs_dim=3 * 64 * 64,
        action_dim=6,
        hidden_dim=64,
        latent_dim=16,
        token_dim=32,
    )
    model = WorldModel(cfg)

    batch, time = 4, 8
    obs = torch.randn(batch, time, *cfg.obs_shape)
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
        obs_shape=(3, 64, 64),
        obs_dim=3 * 64 * 64,
        action_dim=6,
        hidden_dim=64,
        latent_dim=16,
        token_dim=32,
        use_spiking_core=True,
    )
    model = WorldModel(cfg)
    batch, time = 2, 8
    obs = torch.randn(batch, time, *cfg.obs_shape)
    action = torch.randn(batch, cfg.action_dim)
    state = model.init_state(batch, obs.device)
    next_state, _, _ = model.observe(obs, action, state)
    assert next_state.z_fast.shape == (batch, cfg.latent_dim)


class _ZeroSpikeCore(nn.Module):
    def forward(self, x: torch.Tensor, context: torch.Tensor, prev: MCNState) -> MCNState:
        zeros = torch.zeros_like(prev.v_soma)
        return MCNState(v_basal=zeros, v_apical=zeros, v_soma=zeros, spikes=zeros)


def test_event_driven_mask_keeps_med_slow_without_spikes():
    cfg = WorldModelConfig(
        obs_shape=(3, 64, 64),
        obs_dim=3 * 64 * 64,
        action_dim=6,
        hidden_dim=64,
        latent_dim=16,
        token_dim=32,
        use_spiking_core=True,
    )
    model = WorldModel(cfg)
    model.rssm.spiking = _ZeroSpikeCore()

    batch, time = 3, 8
    obs = torch.randn(batch, time, *cfg.obs_shape)
    action = torch.randn(batch, cfg.action_dim)
    prev = model.init_state(batch, obs.device)
    prev.h_med = torch.randn_like(prev.h_med)
    prev.h_slow = torch.randn_like(prev.h_slow)
    prev.z_med = torch.randn_like(prev.z_med)
    prev.z_slow = torch.randn_like(prev.z_slow)

    next_state, _, _ = model.observe(obs, action, prev)

    assert torch.allclose(next_state.h_med, prev.h_med)
    assert torch.allclose(next_state.h_slow, prev.h_slow)
    assert torch.allclose(next_state.z_med, prev.z_med)
    assert torch.allclose(next_state.z_slow, prev.z_slow)


def test_structured_heads_goal_time_and_symlog_losses():
    cfg = WorldModelConfig(
        obs_shape=(3, 64, 64),
        obs_dim=3 * 64 * 64,
        action_dim=6,
        hidden_dim=64,
        latent_dim=16,
        token_dim=32,
    )
    model = WorldModel(cfg)
    batch, time = 2, 8
    obs = torch.randn(batch, time, *cfg.obs_shape)
    action = torch.randn(batch, cfg.action_dim)
    state = model.init_state(batch, obs.device)
    next_state, _, preds = model.observe(obs, action, state)

    assert preds["s_goal"].shape == (batch, 1)
    assert preds["s_time"].shape == (batch, 16)

    targets = {
        "s_pred": torch.randn_like(preds["s_pred"]),
        "s_geo": torch.randn_like(preds["s_geo"]),
        "s_place": torch.randn_like(preds["s_place"]),
        "s_rel": torch.randn_like(preds["s_rel"]),
        "s_belief": torch.randn_like(preds["s_belief"]),
        "s_goal_mask": torch.rand_like(preds["s_goal"]),
        "s_time": torch.randn_like(preds["s_time"]),
    }
    losses = compute_structured_losses(preds, targets, weights={})

    assert "L_goal_mask" in losses
    assert "L_time" in losses
    assert torch.isfinite(losses["L_total"])

    transformed = symlog(torch.tensor([-10.0, 0.0, 10.0]))
    assert transformed[0] < 0 and transformed[2] > 0


def test_train_step_uses_balanced_kl_and_total_loss():
    cfg = WorldModelConfig(
        obs_shape=(3, 64, 64),
        obs_dim=3 * 64 * 64,
        action_dim=6,
        hidden_dim=64,
        latent_dim=16,
        token_dim=32,
        use_spiking_core=True,
    )
    model = WorldModel(cfg)
    optimizer = build_adam(model, lr=1e-3)

    batch_size, time = 2, 8
    obs = torch.randn(batch_size, time, *cfg.obs_shape)
    action = torch.randn(batch_size, cfg.action_dim)

    with torch.no_grad():
        init_state = model.init_state(batch_size, obs.device)
        _, _, preds = model.observe(obs, action, init_state)

    batch = {
        "obs": obs,
        "action": action,
        "targets": {
            "s_pred": torch.randn_like(preds["s_pred"]),
            "s_geo": torch.randn_like(preds["s_geo"]),
            "s_place": torch.randn_like(preds["s_place"]),
            "s_rel": torch.randn_like(preds["s_rel"]),
            "s_belief": torch.randn_like(preds["s_belief"]),
            "s_goal_mask": torch.rand_like(preds["s_goal"]),
            "s_time": torch.randn_like(preds["s_time"]),
        },
    }

    metrics = train_step(model, batch, optimizer, loss_weights={"L_goal_mask": 1.0}, kl_weight=0.5)

    assert "L_total" in metrics
    assert "L_kl" in metrics
    assert "L_joint" in metrics
    assert metrics["L_kl"] > 0
    assert metrics["L_joint"] >= metrics["L_total"]
