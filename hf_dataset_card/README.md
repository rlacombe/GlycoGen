---
license: cc-by-4.0
pretty_name: CandyCrunch v2 Benchmark
size_categories:
  - 100K<n<1M
task_categories:
  - other
tags:
  - glycomics
  - mass-spectrometry
  - msms
  - glycans
  - bioinformatics
  - chemistry
  - discrete-flow-matching
---

# CandyCrunch v2 Benchmark (`rlacombe/candycrunch-v2-benchmark`)

Leak-free, LC-MS run-grouped benchmark for single-spectrum *de novo* glycan structure prediction from tandem mass spectrometry (MS/MS), accompanying **"GlycoGen: Glycan Structure Prediction with Crystallizing Flows"** (NeurIPS 2026; [code repository](https://github.com/rlacombe/GlycoGen)).

Derived deterministically from the 503,446-spectrum CandyCrunch v2 corpus curated by [Urban et al. (*Nature Methods*, 2024)](https://doi.org/10.1038/s41592-024-02314-6) and archived at [Zenodo `10.5281/zenodo.15741423`](https://doi.org/10.5281/zenodo.15741423) ([Abtheen et al., 2025](https://doi.org/10.1101/2025.07.02.662857)) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Overview and motivation

In liquid chromatography-tandem mass spectrometry (LC-MS/MS) glycomics, a single glycan eluting off a porous graphitized carbon (PGC) or C18 column generates **5 to 50 repeated MS/MS scans** within one analytical run across adjacent retention times, multiple charge states, adduct forms, and chromatographically separated $\alpha/\beta$ reducing-end anomers.

Under a uniform random row-level train/test split, sibling scans of the same chromatographic peak land in both training and test sets. Models evaluated on random spectrum splits can therefore exploit memorized retention times, run-specific ionization profiles, and biological sample matrices rather than generalizing to unseen runs—inflating top-1 accuracy by 2 to 5 percentage points.

Following Urban et al. (2024), **CandyCrunch v2 Benchmark** partitions the corpus strictly at the **LC-MS source `filename` level** via `GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=42)`, guaranteeing **zero shared LC-MS runs between `{train, val}` and `test`**. Every row is shipped in **two synchronized formats** sharing identical row order (`row_manifest.csv`):

1. **Continuous dual-channel numerical spectra (`X_*.pkl`, `y_*.pkl`)** — 2,048-bin intensity + sub-bin `m/z` remainder vectors (`39.714–3000.0` Da) paired with full IUPAC-condensed glycan strings, used by **GlycoGen-DFM**, **GlycoGen-MDLM**, and **GlycoGen-AR**.
2. **Tokenized MS sentences and antenna paths (`sentences_benchmark.pkl`, `vocab_glycobart.txt`)** — quantized `m/z`, retention-time, and metadata token sequences paired with antenna-decomposed targets, enabling controlled from-scratch retraining of **GlycoBART** ([Abtheen et al., 2025](https://doi.org/10.1101/2025.07.02.662857)) on the exact same train/val/test rows.

---

## Split statistics

Starting from `503,446` raw spectra in `dataset.xlsx` / `dataset.parquet`:
1. **Run-quality filter** (`RT.fillna(15) > 2`, non-null `filename`, non-empty L1-normalizable `peak_d`) retains `499,210` rows.
2. **Antenna-mapping filter** (excludes curly-brace `{...}` glycans with unlocalized substituents via `glycans_antennae.csv` so sequence-to-antenna baselines train on identical rows) retains `501,987` rows.
3. **Intersection:** `497,751` usable spectra (`5,695` rows = `1.13%` excluded overall) spanning `3,572` unique IUPAC-condensed glycan strings across `O`-glycans, `N`-glycans, free oligosaccharides, and glycosphingolipids.

| Split | Spectra | Unique LC-MS `filename` runs | Shared filenames with `test` | Assignment rule |
|---|---:|---:|---:|---|
| `train` | `378,880` | `2,455` (across `train` + `val`) | `0` | `90%` of non-test pool (`np.random.default_rng(seed=42)`) |
| `val` | `42,097` | subset of the `2,455` non-test runs | `0` | `10%` of non-test pool (`np.random.default_rng(seed=42)`) |
| `test` | `76,774` | `434` disjoint runs | — | `GroupShuffleSplit(test_size=0.15, random_state=42)` by `filename` |
| **Total** | **`497,751`** | **`2,889`** | **`0`** | Deterministic, byte-reproducible |

Within the `76,774` held-out test spectra, `76,203` match a training string verbatim, `66` represent training chemical graphs serialized under an unseen IUPAC branch traversal order, and `505` represent glycan graph topologies absent from `train`.

---

## Repository files and data schema

| File | Format | Description |
|---|---|---|
| `X_train.pkl`, `X_val.pkl`, `X_test.pkl` | Python `list` of 9-tuples (`pickle`) | Numerical spectrum features aligned to `row_manifest.csv` (`378,880` / `42,097` / `76,774` rows). |
| `y_train.pkl`, `y_val.pkl`, `y_test.pkl` | Python `list[str]` (`pickle`) | Ground-truth IUPAC-condensed glycan strings in identical row order (`e.g. "Neu5Ac(a2-6)Gal(b1-4)GlcNAc"`). |
| `row_manifest.csv` | CSV (`497,751` rows) | Per-row provenance linking all formats back to the raw Zenodo table: columns `orig_idx,filename,glycan,split`. |
| `sentences_benchmark.pkl` | Python `dict` (`pickle`) | Tokenized MS sentences and antenna targets for GlycoBART reproduction (schema below). |
| `vocab_glycobart.txt` | Plain text (`10,071` lines) | One token per line built strictly from `train` (`mz*` bins, `rt*` bins, metadata literals, and antenna tokens). |
| `spectrum_metadata.npz` | NumPy archive (`.npz`) | Per-test-row arrays (`test_reducing_mass`, `test_mode`, `test_modification`) used only by optional precursor-mass post-processing diagnostics. |
| `test_peak_d.pkl` | Python `list[dict]` (`pickle`) | Raw peak dictionaries `{mz: intensity}` for `test` rows, used only for raw-preprocessing latency microbenchmarks and CandyCrumbs fragment-reranking ablations. |

### Schema of `X_{train,val,test}.pkl`

Each element `x = X_split[i]` is a 9-tuple following CandyCrunch's numerical format:

| Index | Field | Type / Shape | Encoding |
|---:|---|---|---|
| `x[0]` | `binned_intensities` | `np.ndarray`, `float32[2048]` | L1-normalized fragment intensities summed into 2,048 linear bins across `[39.714, 3000.0]` Da (`np.linspace(39.714, 3000.0, 2048)`). |
| `x[1]` | `mz_remainder` | `np.ndarray`, `float32[2048]` | Sub-bin mass offset `mz - frame_edge` (Da) of the most intense peak in each bin (`0.0` for empty bins), preserving exact peak mass within the `~1.45` Da bin width. |
| `x[2]` | `reducing_mass` | `float` | Observed precursor m/z value. |
| `x[3]` | `glycan_type` | `int` | `0` = O-glycan, `1` = N-glycan, `2` = free/other reducing end, `3` = fallback. |
| `x[4]` | `RT_normalized` | `float` | Per-file normalized retention time `RT / max(max_file_RT, 30.0)`. |
| `x[5]` | `mode` | `int` | Ionization polarity: `0` = `negative`, `1` = `positive`, `2` = unknown/other. |
| `x[6]` | `lc` | `int` | Liquid chromatography column: `0` = `PGC`, `1` = `C18`, `2` = unknown/other. |
| `x[7]` | `modification` | `int` | Sample derivatization: `0` = `reduced`, `1` = `permethylated`, `2` = underivatized/other. |
| `x[8]` | `trap` | `int` | Mass analyzer family: `0` = `linear` ion trap, `1` = `orbitrap`, `2` = Bruker `amazon`, `3` = other/unknown. |

> **Note:** GlycoGen-DFM, GlycoGen-MDLM, and GlycoGen-AR consume **only `x[0]`, `x[1]`, and `x[2]`** (spectrum intensities, sub-bin m/z remainders, and precursor m/z), without feeding glycan class, retention time, or instrument metadata into the model. Fields `x[3..8]` are retained for per-instrument evaluation slicing (e.g., `x[8]` mass-analyzer transfer) and compatibility with CandyCrunch pipelines.

### Schema of `sentences_benchmark.pkl`

A single Python `dict` with keys:
- `train_ms`, `val_ms`, `test_ms` (`list[str]`): space-separated CABSEL input sentences (`LC_type mode ionization modification trap fragmentation glycan_type rt{bin} mz{prec_bin} mz{peak_1} ...`).
- `train_glycan`, `val_glycan`, `test_glycan` (`list[str]`): space-separated antenna-decomposed target token strings.
- `train_glycan_iupac`, `val_glycan_iupac`, `test_glycan_iupac` (`list[str]`): full IUPAC-condensed target strings (identical to `y_{train,val,test}.pkl`).
- `vocab_list` (`list[str]`): sorted `10,071`-token training vocabulary.
- `split_metadata` (`dict`): split seeds, fractions, row counts, and `filenames_in_train_and_test = 0` verification record.

---

## Quickstart: loading splits in Python

```python
import pickle
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "rlacombe/candycrunch-v2-benchmark"

def load_split(split: str = "test"):
    x_path = hf_hub_download(repo_id=REPO_ID, filename=f"X_{split}.pkl", repo_type="dataset")
    y_path = hf_hub_download(repo_id=REPO_ID, filename=f"y_{split}.pkl", repo_type="dataset")
    X = pickle.loads(Path(x_path).read_bytes())
    y = pickle.loads(Path(y_path).read_bytes())
    return X, y

X_test, y_test = load_split("test")
print(f"Loaded {len(X_test):,} test spectra")  # 76,774

intensities, mz_remainder, precursor_mz = X_test[0][:3]
print(intensities.shape, mz_remainder.shape, precursor_mz, y_test[0])
# (2048,) (2048,) 1136.42 '...'

# Optional: inspect LC-MS filename provenance
manifest_path = hf_hub_download(repo_id=REPO_ID, filename="row_manifest.csv", repo_type="dataset")
manifest = pd.read_csv(manifest_path)
print(manifest["split"].value_counts().to_dict())
# {'train': 378880, 'test': 76774, 'val': 42097}
```

---

## Evaluation metric

Because branched glycans admit multiple valid IUPAC-condensed string serializations depending on bracket traversal order, string equality undercounts structurally exact predictions. All models on this benchmark are scored **per spectrum** via `glycowork` chemical graph isomorphism ([Thomès et al., 2021](https://doi.org/10.1093/glycob/cwab067)), without multi-scan consensus voting or reference-database snap-back:

```python
from glycowork.motif.graph import compare_glycans

def struct_equal(pred_iupac: str, true_iupac: str) -> bool:
    if not pred_iupac or not true_iupac:
        return False
    if pred_iupac == true_iupac:
        return True
    try:
        return bool(compare_glycans(pred_iupac, true_iupac))
    except Exception:
        return False
```

---

## Baseline benchmark results

All models are trained from scratch across 3 seeds (`0, 1, 2`) on a single 141 GB NVIDIA H200 GPU in BF16 and evaluated per spectrum on all `76,774` test rows:

| Model | Params | Checkpoint | Inference setting | Top-1 structural accuracy | Mean NFE | Batched latency |
|---|---:|---|---|---:|---:|---:|
| **GlycoGen-DFM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-dfm-50m`](https://huggingface.co/rlacombe/glycogen-dfm-50m) | `K = 1` | `90.19% ± 0.04%` | `1.0` | `0.046 ms/spec` (`61×`) |
| **GlycoGen-DFM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-dfm-50m`](https://huggingface.co/rlacombe/glycogen-dfm-50m) | `K = 8` | **`92.51% ± 0.06%`** | `5.0` | `0.201 ms/spec` (`13.9×`) |
| **GlycoGen-DFM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-dfm-50m`](https://huggingface.co/rlacombe/glycogen-dfm-50m) | `K = 32` | **`92.67% ± 0.05%`** | `17.4` | `0.675 ms/spec` (`4.1×`) |
| **GlycoGen-DFM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-dfm-50m`](https://huggingface.co/rlacombe/glycogen-dfm-50m) | `K = 64` | **`92.69% ± 0.06%`** | `33.9` | `1.306 ms/spec` (`2.1×`) |
| **GlycoGen-MDLM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-mdlm-50m`](https://huggingface.co/rlacombe/glycogen-mdlm-50m) | `K = 1` | `89.73% ± 0.17%` | `1.0` | `0.046 ms/spec` (`61×`) |
| **GlycoGen-MDLM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-mdlm-50m`](https://huggingface.co/rlacombe/glycogen-mdlm-50m) | `K = 8` | `92.30% ± 0.12%` | `5.0` | `0.199 ms/spec` (`14.1×`) |
| **GlycoGen-MDLM + Crystallizing Flows** | 50M | [`rlacombe/glycogen-mdlm-50m`](https://huggingface.co/rlacombe/glycogen-mdlm-50m) | `K = 32` | `92.47% ± 0.11%` | `17.4` | `0.676 ms/spec` (`4.1×`) |
| **GlycoGen-AR** | 46.4M | [`rlacombe/glycogen-ar-50m`](https://huggingface.co/rlacombe/glycogen-ar-50m) | `beam = 1` | `90.38% ± 0.05%` | `33.0` | `2.21 ms/spec` (`1.3×`) |
| **GlycoGen-AR** | 46.4M | [`rlacombe/glycogen-ar-50m`](https://huggingface.co/rlacombe/glycogen-ar-50m) | `beam = 32` (seed 0) | `90.39%` | `1,045` | `72.66 ms/spec` |
| **GlycoBART reproduction** | 207M | [`rlacombe/glycobart-207m-repro`](https://huggingface.co/rlacombe/glycobart-207m-repro) | `beam = 1` | `80.92% ± 0.31%` | `23.0` | `2.80 ms/spec` (`1.0×`) |
| **GlycoBART reproduction** | 207M | [`rlacombe/glycobart-207m-repro`](https://huggingface.co/rlacombe/glycobart-207m-repro) | `beam = 32` (seed 0) | `78.57%` (`86.35%` top-5) | `720` | `91.21 ms/spec` |

---

## Deterministic rebuild from raw Zenodo archive

You can regenerate all split files byte-for-byte on CPU (~30 minutes on an Apple M3 laptop) from the upstream Zenodo record [`10.5281/zenodo.15741423`](https://doi.org/10.5281/zenodo.15741423) using the [`rlacombe/GlycoGen`](https://github.com/rlacombe/GlycoGen) repository:

```bash
git clone https://github.com/rlacombe/GlycoGen.git
cd GlycoGen
pip install -r requirements.txt

# 1. Place dataset.xlsx (or converted dataset.parquet) and glycans_antennae.csv
#    from https://zenodo.org/records/15741423 into ./data/raw/

# 2. Generate row_manifest.csv, sentences_benchmark.pkl, and vocab_glycobart.txt
python data_utils/make_benchmark_split.py \
    --dataset  ./data/raw/dataset.parquet \
    --antennae ./data/raw/glycans_antennae.csv \
    --out_dir  ./data

# 3. Generate aligned X_{train,val,test}.pkl and y_{train,val,test}.pkl
python data_utils/make_aligned_numerical_pkls.py
```

---

## License and attribution

This benchmark dataset is distributed under the **Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))** license, matching the license of the upstream CandyCrunch v2 archive ([Zenodo `10.5281/zenodo.15741423`](https://doi.org/10.5281/zenodo.15741423)).

- **Upstream experimental MS/MS curation and run-grouped quality filtering:** James Urban, Chunsheng Jin, Kristina A. Thomsson, Niclas G. Karlsson, Callum M. Ives, Elisa Fadda, and Daniel Bojar (*Nature Methods*, 2024).
- **Raw table archive and antenna decomposition table (`glycans_antennae.csv`):** Ejas Althaf Abtheen, Sriram Neelamegham, and Rudiyanto Gunawan (Zenodo `10.5281/zenodo.15741423`, 2025).
- **Synchronized LC-MS file-grouped benchmark split and dual-channel numerical packaging:** Romain Lacombe (NeurIPS 2026).

---

## Citation

If you use this benchmark, please cite the original CandyCrunch v2 dataset paper, the upstream Zenodo archive, and the GlycoGen paper:

```bibtex
@article{candycrunch2024,
  title   = {Predicting glycan structure from tandem mass spectrometry via deep learning},
  author  = {Urban, James and Jin, Chunsheng and Thomsson, Kristina A. and
             Karlsson, Niclas G. and Ives, Callum M. and Fadda, Elisa and
             Bojar, Daniel},
  journal = {Nature Methods},
  volume  = {21},
  number  = {7},
  pages   = {1206--1215},
  year    = {2024},
  doi     = {10.1038/s41592-024-02314-6}
}

@misc{zenodo2024cc2,
  title        = {Transformer-based Deep Learning for Glycan Structure Inference from Tandem Mass Spectrometry {[Dataset]}},
  author       = {Abtheen, Ejas Althaf and Neelamegham, Sriram and Gunawan, Rudiyanto},
  howpublished = {Zenodo, \url{https://zenodo.org/records/15741423}},
  year         = {2025},
  doi          = {10.5281/zenodo.15741423}
}

@inproceedings{lacombe2026glycogen,
  title     = {GlycoGen: Glycan Structure Prediction with Crystallizing Flows},
  author    = {Lacombe, Romain},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
