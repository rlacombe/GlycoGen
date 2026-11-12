# `data_utils/` — CandyCrunch v2 Benchmark Split Builders

Deterministic CPU-only pipeline that builds the leak-free CandyCrunch v2 benchmark (`378,880` train / `42,097` val / `76,774` test = `497,751` rows from `503,446` raw spectra) from the upstream CandyCrunch v2 archive ([Urban et al., *Nature Methods* 2024](https://doi.org/10.1038/s41592-024-02314-6); Zenodo [`10.5281/zenodo.15741423`](https://doi.org/10.5281/zenodo.15741423)).

All models in this repository (GlycoGen-DFM, GlycoGen-MDLM, GlycoGen-AR, and our GlycoBART reproduction) train and evaluate on these exact row identities. Pre-built outputs are hosted at [`rlacombe/candycrunch-v2-benchmark`](https://huggingface.co/datasets/rlacombe/candycrunch-v2-benchmark).

## Two-step rebuild from raw Zenodo files

Run from the repository root after placing `dataset.parquet` (or `dataset.xlsx`) and `glycans_antennae.csv` in `./data/raw/`:

```bash
# Step 1 — Build the LC-MS file-grouped split, row_manifest.csv, and CABSEL text corpus (~5 min on Apple M3)
python data_utils/make_benchmark_split.py \
    --dataset  ./data/raw/dataset.parquet \
    --antennae ./data/raw/glycans_antennae.csv \
    --out_dir  ./data

# Step 2 — Build 2,048-bin dual-channel numerical pickles aligned to row_manifest.csv (~25 min on Apple M3)
python data_utils/make_aligned_numerical_pkls.py
```

`make_numerical.py` is an optional cross-check script that calls `candycrunch.prediction.bin_intensities` from the external `candycrunch` package; `make_aligned_numerical_pkls.py` implements the identical binning directly with `numpy` and `numpy_indexed` so rebuilding requires no external `candycrunch` installation.

## Inputs and outputs

### Step 1: `make_benchmark_split.py`

**Inputs (`./data/raw/`):**
- `dataset.parquet` or `dataset.xlsx` — raw CandyCrunch v2 table (`503,446` spectra) with MS/MS `peak_d`, `reducing_mass`, `RT`, `filename`, `glycan`, and the seven experimental metadata columns (`LC_type`, `mode`, `ionization`, `modification`, `trap`, `fragmentation`, `glycan_type`).
- `glycans_antennae.csv` — glycan-to-antenna mapping table from [Abtheen et al. (2025)](https://doi.org/10.1101/2025.07.02.662857).

**Filtering (`503,446` -> `497,751` usable rows):**
1. Urban et al. quality and run-grouping filter (`RT.fillna(15) > 2`, non-null `filename`, non-empty L1-normalizable `peak_d`) retains `499,210` rows.
2. Antenna-mapping filter (drops curly-brace `{...}` ambiguous substituent strings so GlycoBART can also train on every row) retains `501,987` rows.
3. Their intersection yields `497,751` usable rows (`1.13%` excluded overall).

**Outputs (`./data/`):**
- `row_manifest.csv` — `497,751` rows with columns `orig_idx,filename,glycan,split`.
- `sentences_benchmark.pkl` — dictionary containing `{train,val,test}_ms`, `{train,val,test}_glycan` (antenna tokens), `{train,val,test}_glycan_iupac` (full IUPAC-condensed strings), `vocab_list`, and `split_metadata`.
- `vocab_glycobart.txt` — `10,071` CABSEL vocabulary tokens built from `train` only.

### Step 2: `make_aligned_numerical_pkls.py`

**Inputs:** `./data/raw/dataset.parquet` and `./data/row_manifest.csv`.

**Outputs (`./data/`):**
- `X_train.pkl`, `X_val.pkl`, `X_test.pkl` — lists of 9-tuples in exact row order of `row_manifest.csv`:
  ```text
  (binned_intensities[2048]  float32,
   mz_remainder[2048]        float32,
   reducing_mass             float32,
   glycan_type               int,
   RT_normalized             float32,
   mode                      int,
   lc                        int,
   modification              int,
   trap                      int)
  ```
- `y_train.pkl`, `y_val.pkl`, `y_test.pkl` — parallel lists of IUPAC-condensed glycan strings.

GlycoGen-DFM, GlycoGen-MDLM, and GlycoGen-AR consume only `x[0..2]` (the two 2,048-bin spectrum channels on `39.714–3000.0` Da plus scalar precursor `reducing_mass`), keeping metadata fields `x[3..8]` for analyzer-stratified evaluation (`EVAL_TRAP`) and format compatibility.

## Split statistics

| Split | Spectra | LC-MS `filename` groups | Overlap with `test` filenames |
|---|---:|---:|---:|
| `train` | `378,880` | `2,455` (shared across `train` + `val`) | `0` |
| `val` | `42,097` | subset of the `2,455` non-test files (`10%` sampled with `seed=42`) | `0` |
| `test` | `76,774` | `434` held-out LC-MS files (`GroupShuffleSplit(test_size=0.15, random_state=42)`) | — |
| **Total** | **`497,751`** | **`2,889`** | **`0`** |
