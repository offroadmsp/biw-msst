# BIW-MSST Experimental Scaffold

本仓库提供一个可扩展的 BIW-MSST 世界模型实验骨架，当前包含：

- **CNN 视觉编码器**（Go-Stanford 第一人称图像输入）
- **Multi-scale ST Transformer**（多尺度时空融合 + graph memory cross-attn）
- **Hierarchical RSSM**（fast / med / slow 三尺度动力学）
- **Hybrid Spiking Core（可选）**（事件驱动更新）
- **Structured State Heads**（`s_geo/s_place/s_rel/s_pred/s_belief/s_goal/s_time`）
- **训练循环工具**（`train_step/train_loop/compute_kl_loss`）

---

## 1. 环境准备

> 推荐 Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

如果你的环境没有自动安装 `pytest`，再执行：

```bash
pip install pytest
```

---

## 2. 代码结构（重点文件）

- `src/biw_msst/world_model/model.py`  
  主模型 `WorldModel`，包含 `observe/imagine/decode/predict_heads`。
- `src/biw_msst/world_model/rssm.py`  
  分层 RSSM + `balanced_kl` + 事件驱动 masking。
- `src/biw_msst/world_model/state_heads.py`  
  结构化输出头与损失（含 `symlog`）。
- `src/biw_msst/train.py`  
  训练 API：
  - `compute_kl_loss(stats)`
  - `train_step(...)`
  - `train_loop(...)`
  - `build_adam(...)`

---

## 3. 运行测试（快速确认可用）

```bash
pytest -q
```

如果你只想跑单个训练相关测试：

```bash
pytest -q tests/test_world_model.py::test_train_step_uses_balanced_kl_and_total_loss
```

---

## 4. 如何运行 `train.py`（最小可复现实例）

当前 `src/biw_msst/train.py` 是函数式训练入口，不是 CLI 脚本。推荐用下面方式快速跑通：

```bash
python - <<'PY'
import torch
from biw_msst.world_model.model import WorldModel, WorldModelConfig
from biw_msst.train import build_adam, train_step

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
optimizer = build_adam(model, lr=3e-4)

B, T = 4, 8
obs = torch.randn(B, T, *cfg.obs_shape)
action = torch.randn(B, cfg.action_dim)

# 先走一次 forward，拿到 heads 的 shape 生成 targets
with torch.no_grad():
    init_state = model.init_state(B, obs.device)
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

metrics = train_step(
    model,
    batch,
    optimizer,
    loss_weights={"L_goal_mask": 1.0, "L_geo": 1.0},
    kl_weight=0.5,
)

print(metrics)
PY
```

你会看到类似输出：

```text
{'L_total': ..., 'L_kl': ..., 'L_joint': ..., 'L_rec': ...}
```

其中：
- `L_total`：结构化 heads 的加权损失
- `L_kl`：三尺度 `balanced_kl` 的总和
- `L_joint`：实际反向传播损失 `L_total + kl_weight * L_kl`

---

## 5. 如何调试 `train.py`（重点）

### 5.1 用 `pdb` 单步调试

在你自己的脚本里插入：

```python
import pdb; pdb.set_trace()
metrics = train_step(model, batch, optimizer, kl_weight=0.5)
```

常用命令：
- `n`：下一行
- `s`：进入函数
- `p metrics`：打印变量
- `p batch["targets"].keys()`：检查 targets 键是否齐全

### 5.2 重点观察项

1. **输入维度**
   - `obs`: `[B, T, C, H, W]`
   - `action`: `[B, A]`
2. **targets 必填项**
   - `s_pred/s_geo/s_place/s_rel/s_belief/s_goal_mask`
   - 可选：`s_time`
3. **损失是否为有限值**
   - `metrics["L_joint"]` 不能是 `nan/inf`
4. **KL 是否生效**
   - `metrics["L_kl"]` 应大于 0（通常如此）

### 5.3 快速做梯度检查

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm().item())
        break
```

如果全部 `grad is None`，通常是：
- 没有执行 `loss.backward()`
- loss 张量被错误 `detach`
- batch targets 键缺失导致提前报错

### 5.4 打开异常追踪（定位 NaN）

```python
torch.autograd.set_detect_anomaly(True)
```

---

## 6. 用 `train_loop` 跑多步

```python
from biw_msst.train import train_loop

# 假设 dataloader 每次返回与 train_step 同格式的 batch dict
history = train_loop(
    model,
    dataloader,
    optimizer,
    loss_weights={"L_goal_mask": 1.0},
    kl_weight=0.5,
    max_steps=100,
)
print(history[-1])
```

---

## 7. 常见报错与解决

1. `KeyError: 's_goal_mask'`  
   你的 `targets` 缺少 `s_goal_mask`，这是 BCE loss 必需项。

2. `RuntimeError`（shape mismatch）  
   优先检查 `obs` 是否是 `[B, T, C, H, W]`，以及 `action` batch 维是否与 `obs` 一致。

3. loss 震荡很大  
   可先降低 `kl_weight`（例如从 `1.0` 到 `0.1`），并检查目标值范围是否合理。

---

## 8. 建议的实验起点

- **先关 spiking**：`use_spiking_core=False` 跑通训练
- **再开 spiking**：观察 `L_kl` 与 `L_joint` 变化
- **最后调权重**：逐步增加 `L_goal_mask`、`L_geo`、`L_time` 的权重

这样能更稳地定位问题来源（编码器 / RSSM / heads / loss）。
