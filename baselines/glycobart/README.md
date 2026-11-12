# GlycoBART Reproduction Baseline (207M)

From-scratch reproduction of the 207M-parameter GlycoBART model ([Abtheen et al., 2025](https://doi.org/10.1101/2025.07.02.662857)) on the leak-free CandyCrunch v2 benchmark (`378,880` train / `42,097` val / `76,774` test).

See [`BASELINE_PROTOCOL.md`](BASELINE_PROTOCOL.md) for full details on LC-MS file-level split construction, CABSEL text tokenization, antenna-to-IUPAC tree reconstruction, and single-spectrum vs. sample-level evaluation.

## Headline per-spectrum test performance (1× H200 BF16)

| Decoding setting | Per-spectrum top-1 accuracy | Top-5 accuracy | Mean NFE | Batched latency |
|---|---:|---:|---:|---:|
| `beam = 1` (greedy, 3 seeds) | `80.92% ± 0.31%` | — | `23.0` | `2.80 ms/spec` |
| `beam = 8` (seed 0) | `78.57%` | `85.78%` | `180` | `27.12 ms/spec` |
| `beam = 32` (seed 0) | `78.57%` | `86.35%` | `720` | `91.21 ms/spec` |

In single-spectrum serial streaming (`batch = 1`), greedy decoding takes `183 ± 99 ms/spec`. At `beam = 1`, GlycoBART reaches `91.24%` accuracy in compressed antenna space before losing `10.3` percentage points during lossy antenna-to-IUPAC tree reconstruction.

## Scripts

| Script | Purpose |
|---|---|
| `train_glycobart.py` | Train 207M `BartForConditionalGeneration` from scratch on `sentences_benchmark.pkl`. |
| `eval_glycobart.py` | Per-spectrum beam sweep (`beam ∈ {1, 2, 4, 8, 16, 32}`) on the `76,774`-row test split without post-processing. |
| `eval_cabsel_full_pipeline.py` | Evaluate with CABSEL's full post-processing pipeline, including reference-database snap-back (`glycowork.df_glycan`). |
| `eval_with_postprocessing.py` | Evaluate with precursor-mass, class, and fragment reranking without database snap-back. |
| `eval_top5_at_beam.py` | Top-5 candidate-pool diagnostics at fixed beam width `5`. |
| `eval_top5_per_beam.py` | Per-`K` top-5 diagnostics combining `beam = K` with `(5 - K)` nucleus-sampled candidates. |
| `post_process.py` | Memoized precursor-mass checking and CandyCrumbs theoretical fragment scoring. |

## Reproduction commands

```bash
pip install -r ../../requirements.txt

# Train 3-seed headline reproduction (~9 h per seed on 1× H200 BF16)
BATCH_SIZE=160 LR=1.414e-5 EPOCHS=30 EARLY_STOP_PATIENCE=999 INIT_SEED=0 python train_glycobart.py
BATCH_SIZE=160 LR=1.414e-5 EPOCHS=30 EARLY_STOP_PATIENCE=999 INIT_SEED=1 python train_glycobart.py
BATCH_SIZE=160 LR=1.414e-5 EPOCHS=30 EARLY_STOP_PATIENCE=999 INIT_SEED=2 python train_glycobart.py

# Per-spectrum beam sweep (uses local outputs/best_model.pt if present, else pulls from HF_OUTPUT_REPO)
python eval_glycobart.py
```

## Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `INIT_SEED` | `0` | Initialization and data-order seed (`0`, `1`, `2`) |
| `BATCH_SIZE` | `32` | Training batch size (`160` with `LR=1.414e-5` for H200 3-seed headline runs) |
| `LR` | `1e-5` | AdamW learning rate |
| `EPOCHS` | `25` | Epoch cap (`30` for headline runs) |
| `EARLY_STOP_PATIENCE` | `5` | Validation early-stop patience (`999` disables early stopping) |
| `TRAIN_SECONDS` | `50400` | Training wall-clock cap in seconds (14 h default) |
| `HF_DATASET_REPO` | `rlacombe/candycrunch-v2-benchmark` | Hugging Face dataset repository (`sentences_benchmark.pkl`, `vocab_glycobart.txt`) |
| `HF_OUTPUT_REPO` | `rlacombe/glycobart-207m-repro` | Hugging Face model repository for checkpoint push/pull |
| `EVAL_SUBSET` | `all` | `all`, `seen`, or `unseen` test subset filter in `eval_glycobart.py` |
