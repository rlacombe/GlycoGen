# GlycoGen-MDLM (50M)

GlycoGen trained with the Masked Diffusion Language Model (MDLM) objective ([Sahoo et al., 2024](https://arxiv.org/abs/2406.07524)) on the leak-free CandyCrunch v2 benchmark (`378,880` train / `42,097` val / `76,774` test).

GlycoGen-MDLM shares the exact spectrum patch encoder, 8-layer bidirectional Transformer decoder, 74-token atomic IUPAC vocabulary, and AdamW optimization schedule of [`glycogen/dfm/`](../dfm/README.md), changing only the training objective:

1. **Discrete time grid:** `t` is sampled uniformly from `{1/T, 2/T, ..., T/T}` with `T = 128` rather than continuous `Uniform(0, 1)`.
2. **NELBO loss weighting:** masked-token cross-entropy is weighted by `1 / max(1 - t, 1/T)` per [Sahoo et al. (2024)](https://arxiv.org/abs/2406.07524) under the linear schedule `alpha_t = t`.

Applied at inference without retraining, Crystallizing Flows act as a drop-in sampler for the 128-step MDLM denoiser, reaching `89.73% ± 0.17%` at `K = 1`, `92.30% ± 0.12%` at `K = 8` (`5.0` effective passes, `0.199 ms/spec`), and `92.47% ± 0.11%` at `K = 32` (`17.4` effective passes) across three seeds.

## Training and evaluation

```bash
pip install -r ../../requirements.txt

# Train from scratch (seeds 0, 1, 2)
INIT_SEED=0 python train.py
INIT_SEED=1 python train.py
INIT_SEED=2 python train.py
```

To evaluate an existing checkpoint on all three sampler configurations without retraining:

```bash
# 1) Crystallizing Flows with t snapped to the 128-step training grid
PARETO_ONLY=1 SAMPLER=crystallizing_flows MDLM_GRID_SNAP=1 python train.py

# 2) Crystallizing Flows with continuous off-grid Euler steps
PARETO_ONLY=1 SAMPLER=crystallizing_flows MDLM_GRID_SNAP=0 python train.py

# 3) Standard CTMC Euler unmasking baseline
PARETO_ONLY=1 SAMPLER=ctmc_euler python train.py
```

Legacy sampler aliases (`soft_auto_commit_ee` for `crystallizing_flows` and `algorithm1` for `ctmc_euler`) remain supported for backward compatibility.

## Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `INIT_SEED` | `0` | Initialization and shuffle seed (`0`, `1`, `2`) |
| `EPOCHS` | `30` | Training epochs |
| `BATCH` | `1024` | Training batch size |
| `BASE_LR` | `3e-4` | Base learning rate (`LR_MAX = BASE_LR * sqrt(BATCH / 128) ≈ 8.49e-4`) |
| `T_GRID` | `128` | Discrete time-grid resolution |
| `MDLM_LOSS` | `weighted` | `weighted` (NELBO `1/(1-t)`) or `unweighted` |
| `MDLM_GRID_SNAP` | `0` | `1` snaps inference `t` to `{1/T, ..., 1}`; `0` passes continuous `t` directly |
| `SAMPLER` | `crystallizing_flows` | `crystallizing_flows`, `ctmc_euler`, or ablation samplers in `train.py` |
| `HF_DATASET_REPO` | `rlacombe/candycrunch-v2-benchmark` | Hugging Face dataset repository (`HF_REPO` also accepted) |
| `HF_OUTPUT_REPO` | `rlacombe/glycogen-mdlm-50m` | Hugging Face model repository for checkpoint push/pull |
| `PARETO_ONLY` | unset | Set `1` to skip training and run test evaluation from `HF_OUTPUT_REPO` |
| `EVAL_SUBSET` | `all` | `all`, `seen`, or `unseen` test-set filter |
| `EVAL_TRAP` | unset | Mass-analyzer filter (`0`=linear ion trap, `1`=Orbitrap, `2`=Bruker amaZon, `3`=other) |

## Outputs

Written to `/output/` on Hugging Face Jobs or `./output/` locally:

- `best_model.pt` — checkpoint with highest validation `K = 1` structural accuracy.
- `pareto_crystallizing_flows_grid_snap.json` — Crystallizing Flows with `MDLM_GRID_SNAP=1`.
- `pareto_crystallizing_flows_ood_t.json` — Crystallizing Flows with `MDLM_GRID_SNAP=0`.
- `pareto.json` — CTMC Euler sweep (`SAMPLER=ctmc_euler`).
- `training_log.jsonl` — per-eval-step validation log.
