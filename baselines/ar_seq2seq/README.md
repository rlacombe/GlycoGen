# GlycoGen-AR Baseline (46.4M)

Architecture-matched causal autoregressive seq2seq baseline for GlycoGen, trained from scratch on the leak-free CandyCrunch v2 benchmark (`378,880` train / `42,097` val / `76,774` test).

GlycoGen-AR shares the exact `SpecEncoder` (2,048-bin dual-channel m/z grid grouped into `64` patches + precursor m/z token, 4-layer Transformer encoder), 8-layer Transformer decoder dimensions (`d_model = 512`, `8` heads, `ffn = 2048`), 74-token atomic IUPAC vocabulary, and AdamW optimization schedule of [`glycogen/dfm/`](../../glycogen/dfm/README.md), changing only the decoding paradigm:

1. **Causal decoder self-attention** (upper-triangular mask).
2. **No time-conditioning projection** (`46.4M` total parameters vs. `~50M` for DFM/MDLM).
3. **Teacher-forced next-token cross-entropy loss** (`ignore_index = PAD_ID`).

## Headline test performance (1× H200 BF16)

| Beam width | Top-1 structural accuracy | Mean NFE | Batched latency |
|---:|---:|---:|---:|
| `beam = 1` (greedy, 3 seeds) | `90.38% ± 0.05%` | `33.0` | `2.21 ms/spec` |
| `beam = 32` (seed 0) | `90.39%` | `1,045` | `72.66 ms/spec` |

Widening beam search from `1` to `32` leaves top-1 structural accuracy flat within `0.01` percentage points despite a `33×` latency increase.

## Training and evaluation

```bash
pip install -r ../../requirements.txt

# Train from scratch (seeds 0, 1, 2) + automatic beam sweep
INIT_SEED=0 python train.py
INIT_SEED=1 python train.py
INIT_SEED=2 python train.py

# Eval-only beam sweep from HF checkpoint (skip training)
PARETO_ONLY=1 python train.py
```

## Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `INIT_SEED` | `0` | Initialization and shuffle seed (`0`, `1`, `2`) |
| `EPOCHS` | `30` | Training epochs |
| `BATCH` | `1024` | Training batch size |
| `BASE_LR` | `3e-4` | Base learning rate (`LR_MAX = BASE_LR * sqrt(BATCH / 128) ≈ 8.49e-4`) |
| `VAL_EVAL_BATCH` | `128` | Validation beam-search batch size |
| `BEAM_LIST` | `1,2,4,8,16,32` | Beam widths evaluated on `X_test` |
| `PARETO_ONLY` | unset | Set `1` to skip training and pull `best_model.pt` from `HF_OUTPUT_REPO` |
| `HF_DATASET_REPO` | `rlacombe/candycrunch-v2-benchmark` | Hugging Face dataset repository (`HF_REPO` also accepted) |
| `HF_OUTPUT_REPO` | `rlacombe/glycogen-ar-50m` | Hugging Face model repository for checkpoint push/pull |
| `EVAL_SUBSET` | `all` | `all`, `seen`, or `unseen` test subset filter |
| `EVAL_USE_GRAPH_EQUALITY` | `0` | Set `1` to use `glycowork.compare_glycans` for seen/unseen subset classification |

## Outputs and companion scripts

Written to `/output/` on Hugging Face Jobs or `./output/ar-seq2seq/` locally:
- `checkpoints/best_model.pt` — checkpoint with highest validation `beam = 4` structural accuracy.
- `pareto.json` — per-beam accuracy, NFE, and batched wall-clock latency on `X_test`.

Companion diagnostic scripts:
- `beam_compare_hf.py` — A/B check comparing `train.py`'s batched beam search against a HuggingFace-style `BeamSearchScorer` implementation.
- `beam_diversity.py` — inspects decoded candidates per spectrum for string diversity and `glycowork` graph equivalence.
- `pareto_high_beams.py` — runs `beam = 16` and `beam = 32` with conservative batch caps and `enable_cudnn_sdp(False)` on H200 GPUs.
