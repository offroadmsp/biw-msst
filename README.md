# BIW-MSST Experimental Scaffold

This repository provides a compact implementation scaffold for the roadmap you described:

- **CNN vision encoder** for Go-Stanford egocentric images plus **Multi-scale ST Transformer** token fusion with graph-memory cross-attention.
- **Structured state heads** for geometry/place/relation/prediction/belief outputs.
- **Hierarchical RSSM** with an optional **hybrid spiking transition core** (multi-compartment cell + surrogate gradient).

## Layout

- `src/biw_msst/world_model/model.py`: main `WorldModel` with `observe`, `imagine`, `decode`, `predict_heads` interfaces.
- `src/biw_msst/world_model/msst_block.py`: multi-scale temporal/spatial fusion block.
- `src/biw_msst/world_model/rssm.py`: hierarchical prior/posterior latent dynamics + balanced KL.
- `src/biw_msst/world_model/spiking_core.py`: MCN-inspired spiking core.
- `src/biw_msst/world_model/state_heads.py`: structured output heads and combined losses.
- `src/biw_msst/configs/*.yaml`: baseline and ablation configs.

## Quick smoke test

```bash
python -m pytest -q
```
