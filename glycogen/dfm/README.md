# GlycoGen-DFM (50M)

GlycoGen trained with the Discrete Flow Matching (DFM) objective ([Gat et al., 2024](https://arxiv.org/abs/2407.15595)) on the leak-free CandyCrunch v2 benchmark (`378,880` train / `42,097` val / `76,774` test). After convergence, `train.py` runs a Pareto sweep across step budgets `K ∈ {1, 2, 4, 8, 16, 32, 64}` on the held-out test split.

## Headline test performance (3 seeds on 1× H200 BF16)

| Sampler | `K` | Mean effective NFE | Top-1 structural accuracy | Batched latency |
|---|---:|---:|---:|---:|
| **Crystallizing Flows** (`crystallizing_flows`) | `1` | `1.0` | `90.19% ± 0.04%` | `0.046 ms/spec` |
| **Crystallizing Flows** (`crystallizing_flows`) | `8` | `5.05` | **`92.51% ± 0.06%`** | `0.201 ms/spec` |
| **Crystallizing Flows** (`crystallizing_flows`) | `32` | `17.40` | **`92.67% ± 0.05%`** | `0.675 ms/spec` |
| **Crystallizing Flows** (`crystallizing_flows`) | `64` | `33.89` | **`92.69% ± 0.06%`** | `1.306 ms/spec` |

## Training from scratch

```bash
pip install -r ../../requirements.txt

# Primary run (seed 0) and multi-seed variance runs (seeds 1, 2)
INIT_SEED=0 python train.py
INIT_SEED=1 python train.py
INIT_SEED=2 python train.py
```

### Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `INIT_SEED` | `0` | Random initialization and data-order seed (`0`, `1`, `2`) |
| `EPOCHS` | `30` | Training epochs |
| `BATCH` | `1024` | Training batch size |
| `BASE_LR` | `3e-4` | Base learning rate (`LR_MAX = BASE_LR * sqrt(BATCH / 128) ≈ 8.49e-4`) |
| `EVAL_BATCH` | `256` | Validation batch size |
| `SAMPLER` | `crystallizing_flows` | Test-sweep sampler (`crystallizing_flows`, `ctmc_euler`, or ablation variants; legacy aliases `soft_auto_commit_ee` and `algorithm1` remain supported) |
| `HF_DATASET_REPO` | `rlacombe/candycrunch-v2-benchmark` | Hugging Face dataset repository (`HF_REPO` also accepted) |
| `HF_OUTPUT_REPO` | `rlacombe/glycogen-dfm-50m` | Hugging Face model repository for checkpoint push/pull (set empty to disable push) |
| `PARETO_ONLY` | unset | Set `1` to skip training, pull `best_model.pt` from `HF_OUTPUT_REPO`, and run test evaluation only |
| `EVAL_SUBSET` | `all` | Test subset filter: `all`, `seen`, or `unseen` |
| `EVAL_TRAP` | unset | Mass-analyzer filter (`0`=linear ion trap, `1`=Orbitrap, `2`=Bruker amaZon, `3`=other) |
| `TRAIN_TRAP_EXCLUDE` | unset | Exclude an instrument family during training for zero-shot analyzer transfer experiments |
| `BENCH_E2E` | `0` | Set `1` to benchmark raw-spectrum preprocessing and `batch = 1` serial streaming latency |
| `DUMP_PREDS` | `0` | Set `1` to write per-spectrum JSONL predictions |
| `LOG_COMMIT_CONF` | `0` | Set `1` to record commit-step confidence calibration logs |

## Outputs

Written to `/output/` when running on Hugging Face Jobs (`COMPUTE_BACKEND=hfjobs` or `HF_JOB=1`), or `./output/` locally:

- `best_model.pt` — checkpoint with highest validation `K = 1` top-1 structural accuracy (encoder/decoder weights, 74-token vocabulary, normalization statistics, and training config).
- `pareto_crystallizing_flows.json` (or `pareto.json` when `SAMPLER=ctmc_euler`) — per-`K` accuracy, effective NFE, batched latency, and eval batch size on `X_test`.
- `training_log.jsonl` — validation metrics logged every 500 steps.

## Architecture and training recipe

| Component / Hyperparameter | Setting |
|---|---|
| Spectrum input | 2,048-bin dual-channel m/z grid (`39.714–3000.0` Da: log1p-normalized intensity + sub-bin m/z remainder) grouped into `64` patches of `32` bins, plus `1` z-score standardized precursor m/z token (`65` encoder tokens total) |
| Encoder | 4-layer pre-LN GELU bidirectional Transformer (`d_model = 512`, `8` heads, `ffn = 2048`) |
| Decoder | 8-layer pre-LN GELU bidirectional Transformer (`d_model = 512`, `8` heads, `ffn = 2048`, `MAX_LEN = 96`) |
| Time conditioning | Single linear projection `nn.Linear(1, 512)` broadcast additively across sequence positions |
| Vocabulary | `74` tokens (`70` atomic IUPAC monosaccharide, linkage, and bracket tokens + `<pad>`, `<mask>`, `<bos>`, `<eos>`) |
| Total parameters | `~50M` |
| Objective | Mask-source DFM (`t = 0` fully masked, `t = 1` clean target, `kappa_t = t`; per-example `t ~ Uniform(0, 1)`, unweighted cross-entropy on masked non-special tokens) |
| Optimizer | AdamW (`lr_max = 8.49e-4`, `lr_floor = 2.55e-5`, 200-step linear warmup + cosine decay, `weight_decay = 0.01`, `grad_clip = 1.0`, BF16) |

## Evaluation CLI (`eval.py`)

`eval.py` wraps `train.py`'s `PARETO_ONLY=1` path with command-line flags:

```bash
# Full test set (76,774 spectra) with Crystallizing Flows
python eval.py --subset all --k pareto --sampler crystallizing_flows --output cf_pareto.json

# Standard CTMC Euler baseline
python eval.py --subset all --k pareto --sampler ctmc_euler --output euler_pareto.json

# Unseen-string subset (~571 spectra; add --use-graph-equality to separate novel topologies from unseen branch spellings)
python eval.py --subset unseen --k 32 --sampler crystallizing_flows --output unseen_k32.json

# Dry-run config check
python eval.py --subset unseen --k 1,8,32 --dry-run
```

Run the unit tests:

```bash
python -m pytest test_eval.py -v
```
