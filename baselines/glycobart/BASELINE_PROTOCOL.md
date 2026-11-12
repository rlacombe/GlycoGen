# GlycoBART Baseline — Methodology Protocol

**Model:** 207M `BartForConditionalGeneration` ([Abtheen et al., 2025](https://doi.org/10.1101/2025.07.02.662857)).  
**Goal:** Retrain GlycoBART from scratch on a leak-free, LC-MS run-grouped partition of CandyCrunch v2 ([Urban et al., 2024](https://doi.org/10.1038/s41592-024-02314-6)) for a controlled per-spectrum comparison against GlycoGen-DFM, GlycoGen-MDLM, and GlycoGen-AR on identical row identities.

---

## 1. Why a unified run-grouped protocol is required

Two preprocessing pipelines exist for the upstream CandyCrunch v2 archive (`dataset.xlsx` / `dataset.parquet` from Zenodo [`10.5281/zenodo.15741423`](https://doi.org/10.5281/zenodo.15741423)), with different split guarantees and input formats:

| Pipeline | Filtering | Train/test split | Leakage guarantee |
|---|---|---|---|
| **CABSEL / GlycoBART** (`GenerateMSCorpus.ipynb`, Abtheen et al., 2025) | Drops curly-brace `{...}` structures and maps glycans to antenna paths via `glycans_antennae.csv` | Uniform random row split (`train_test_split(test_size=0.15, random_state=0)`) | **Leakage across scans:** co-eluting MS/MS scans from the same LC-MS file appear in both train and test |
| **CandyCrunch v2** (`build_training_pickles.py`, Urban et al., 2024) | `RT.fillna(15) > 2`, non-empty normalizable `peak_d`, non-null `filename` | `GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42)` grouped by `filename` | **Zero run overlap:** every LC-MS `filename` lies wholly in `{train, val}` or wholly in `test` |

Because a single glycan eluting off a PGC or C18 column generates 5 to 50 repeated MS/MS scans across adjacent retention times, charge states, adducts, and separated reducing-end anomers within one LC-MS run, uniform spectrum-level splitting leaks near-duplicate fragmentation fingerprints, retention times, and sample matrices between train and test. Moreover, the released `CABSEL/glycobart` checkpoint is trained on the full corpus (including test rows) and cannot be evaluated without train-on-test contamination.

Meanwhile, Urban et al.'s pre-built numerical pickles omit the raw string metadata columns (`LC_type`, `mode`, `ionization`, `modification`, `trap`, `fragmentation`, `glycan_type`) required to construct GlycoBART's tokenized MS sentences. Rebuilding from the raw Zenodo table allows us to enforce Urban et al.'s LC-MS file-level `GroupShuffleSplit` while generating **both** CABSEL text sentences (for GlycoBART) and 2,048-bin dual-channel numerical spectra (for GlycoGen-DFM, GlycoGen-MDLM, and GlycoGen-AR) on the exact same `497,751` rows.

---

## 2. Deterministic CandyCrunch v2 benchmark split (`data_utils/make_benchmark_split.py`)

Starting from `dataset.parquet` (`503,446` rows) and `glycans_antennae.csv`:

1. **Run-quality filter (Urban et al., 2024):** keep rows with `pd.to_numeric(RT, errors="coerce").fillna(15) > 2`, non-null `filename`, and non-empty L1-normalizable `peak_d` -> **`499,210` rows**.
2. **Antenna-mapping filter (Abtheen et al., 2025):** keep rows whose `glycan` has a non-removed antenna decomposition in `glycans_antennae.csv` (excludes curly-brace `{...}` glycans with unlocalized substituents) -> **`501,987` rows**.
3. **Intersection:** **`497,751` usable rows** (`5,695` rows = `1.13%` excluded: `4,236` by the run-quality filter alone, `1,459` by the curly-brace filter alone).
4. **LC-MS file-grouped test split:**
   ```python
   GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42).split(
       df_use, groups=df_use["filename"].astype(str)
   )
   ```
   - **Train + Val pool:** `420,977` rows across `2,455` LC-MS files.
   - **Held-out Test:** `76,774` rows across `434` LC-MS files.
   - **Shared filenames between `{train, val}` and `test`:** `0`.
5. **Within-train validation split (`10%`, `seed=42`):**
   ```python
   rng = np.random.default_rng(seed=42)
   val_pos_in_tr = rng.choice(len(tr_pos), size=int(0.10 * len(tr_pos)), replace=False)
   ```
   - **`train`:** `378,880` spectra
   - **`val`:** `42,097` spectra
   - **`test`:** `76,774` spectra

`data_utils/make_benchmark_split.py` writes `sentences_benchmark.pkl`, `vocab_glycobart.txt` (`10,071` tokens built from `train` only), and `row_manifest.csv`. `data_utils/make_aligned_numerical_pkls.py` joins against `row_manifest.csv` to emit `X_{train,val,test}.pkl` and `y_{train,val,test}.pkl` in identical row order.

---

## 3. Tokenization and architecture (CABSEL specification verbatim)

Each input spectrum is formatted as a CABSEL MS sentence:

```text
<s> LC_type mode ionization modification trap fragmentation glycan_type
    binned_RT binned_mass mz_bin1 mz_bin2 ... </s>
```

- Peak `m/z` values above `0.001` relative intensity are sorted in descending intensity order and quantized onto `[39.714, 3000.0]` Da in `0.3` Da bins (`mz{idx}`).
- Per-file normalized retention times (`RT / max(max_file_RT, 30)`) are quantized onto `[0.0, 1.0]` in `0.01` increments (`rt{idx}`).
- Targets are antenna-decomposed token sequences (`split_antenna`), converted back to IUPAC-condensed trees at evaluation time via `to_iupac` (`convert_to_iupac`) with `.` -> `?` wildcard-linkage normalization before `glycowork.motif.graph.compare_glycans`.

Model architecture matches `CABSEL/glycobart` (`BartForConditionalGeneration`, initialized from scratch):
- `d_model = 768`, `12` encoder layers, `12` decoder layers, `16` attention heads, `ffn_dim = 3,072`, `max_position_embeddings = 512`, `dropout = 0.1`, GELU activations, `vocab_size = 10,078` (`10,071` tokens + `7` specials), **`~207M` parameters**.

---

## 4. Training and per-spectrum evaluation

For the 3-seed headline benchmark (`INIT_SEED ∈ {0, 1, 2}` on a single 141 GB H200 GPU in BF16), we train with AdamW at `BATCH_SIZE = 160`, learning rate `1.414e-5` (`1e-5` scaled by `sqrt(2)`), zero weight decay, gradient clipping at `1.0`, and a full `30`-epoch budget (`EARLY_STOP_PATIENCE = 999`), selecting the checkpoint with highest validation structural accuracy (`beam = 4`).

At test time, every model in the benchmark is evaluated **per spectrum without multi-scan aggregation or reference-database snap-back**:
- **Headline 3-seed result (`beam = 1`):** `80.92% ± 0.31%` per-spectrum top-1 structural accuracy (`2.80 ms/spec` batched at `batch = 128`, `23` NFE; `183 ± 99 ms/spec` unbatched at `batch = 1`).
- **Wider beam search (seed 0):** `85.78%` top-5 at `beam = 8` (`27.12 ms/spec`), `86.35%` top-5 at `beam = 32` (`91.21 ms/spec`), while top-1 accuracy dips slightly to `78.57%`.
- **Antenna-to-IUPAC bottleneck:** at `beam = 1`, GlycoBART achieves `91.24%` accuracy in compressed antenna notation, losing `10.3` percentage points when reconstructing full IUPAC trees due to ambiguous branch attachments. By contrast, the `89.1%` top-1 figure reported in Abtheen et al. (2025) combines a random row-level split with sample-level multi-scan consensus voting and snap-back to `glycowork.df_glycan` (see `eval_cabsel_full_pipeline.py` and `eval_with_postprocessing.py` for offline post-processing ablations).

---

## 5. End-to-end commands

```bash
# 1. Optional: rebuild split locally from raw Zenodo files (or let scripts auto-download from rlacombe/candycrunch-v2-benchmark)
python data_utils/make_benchmark_split.py \
    --dataset ./data/raw/dataset.parquet \
    --antennae ./data/raw/glycans_antennae.csv \
    --out_dir ./data

# 2. Train from scratch on H200 BF16
cd baselines/glycobart
BATCH_SIZE=160 LR=1.414e-5 EPOCHS=30 EARLY_STOP_PATIENCE=999 INIT_SEED=0 python train_glycobart.py

# 3. Run per-spectrum Pareto beam sweep on the 76,774-row test set
python eval_glycobart.py
```
