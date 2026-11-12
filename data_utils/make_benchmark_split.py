"""Generate the leak-free CandyCrunch v2 split with CABSEL tokenization
for GlycoGen experiments.

Produces train/val/test pkls in both formats:
  - CABSEL text format (for GlycoBART): MS sentences + antenna-form targets
  - numerical format (for GlycoGen DFM/AR): (mz, mz_r, prec) tuples + IUPAC targets

The split is identical across formats (same row identities, same
train/val/test assignment).

Methodology (see BASELINE_PROTOCOL.md for the full justification):
  1. Source: dataset.xlsx + glycans_antennae.csv (Urban et al. 2024).
  2. Run-leakage-prevention filter (Urban et al., build_training_pickles.py):
       RT.fillna(15) > 2  AND  valid+normalisable peak_d  AND  filename non-null.
  3. CABSEL antenna filter (Abtheen et al., GenerateMSCorpus.ipynb):
       glycan has antenna mapping (drops curly-bracket structures).
  4. Both filters applied -> 497,751 usable rows.
  5. GroupShuffleSplit(test_size=0.15, random_state=42) grouped by filename
     -> 420,977 train / 76,774 test, with 0 filenames shared across
     train/test (run-stratified -> no same-MS/MS-run leakage).
  6. Within train, val = 10% with seed=42:
       rng = np.random.default_rng(seed=42); val_idx = rng.choice(len(train), 10%, replace=False)
  7. Output four pkls + a row manifest CSV.

Run:
  python data_utils/make_benchmark_split.py \\
    --dataset /path/to/dataset.parquet \\
    --antennae /path/to/glycans_antennae.csv \\
    --out_dir ./data

Default paths point to `./data/raw/`. Pure CPU; ~5 min on M3.
"""
from __future__ import annotations

import argparse
import ast
import pickle
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ---- Run-leakage-prevention recipe (from BojarLab/CandyCrunch build_training_pickles.py) -----

def safe_peak_parse(value):
    """Parse `peak_d` (dict or literal-dict string) into {float mz: float intensity}.
    Returns None on empty/unparseable values so caller can drop the row."""
    if isinstance(value, dict):
        return {float(k): float(v) for k, v in value.items()}
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, dict):
            return {float(k): float(v) for k, v in parsed.items()}
    return None

def normalise_spectrum(spec):
    """L1-normalise intensities so they sum to 1. Returns None on zero-sum spectra."""
    total = float(sum(spec.values()))
    if total <= 0:
        return None
    return {mz: v / total for mz, v in spec.items()}

# ---- CABSEL tokenization (from CABSEL/glycotrans GenerateMSCorpus.ipynb) -------

def bin_linear(min_val, max_val, size):
    return np.arange(min_val, max_val + size, size)

# Matches CABSEL's binning in GenerateMSCorpus.ipynb
MIN_MZ, MAX_MZ, SIZE_MZ = 39.714, 3000.0, 0.3
MIN_RT, MAX_RT, SIZE_RT = 0.0, 1.0, 0.01
INTENSITY_THRESHOLD = 0.001

MZ_BIN_EDGES = bin_linear(MIN_MZ, MAX_MZ, SIZE_MZ)
RT_BIN_EDGES = bin_linear(MIN_RT, MAX_RT, SIZE_RT)

def rt_to_token(rt_value):
    """Bin a normalised RT into a `rt{idx}` token; keep the raw string on non-numeric
    RT (CABSEL convention: preserves literals like "unspecified")."""
    if isinstance(rt_value, (int, float)) and not pd.isna(rt_value):
        return f"rt{int(np.digitize([rt_value], RT_BIN_EDGES)[0])}"
    return str(rt_value)

def mz_to_tokens(mz_values):
    """Bin an array of m/z values into `mz{idx}` tokens on the CABSEL grid."""
    return [f"mz{i}" for i in np.digitize(mz_values, MZ_BIN_EDGES)]

def split_antenna(antenna):
    """CABSEL's antenna decomposition: replace ()/, with spaces."""
    return antenna.replace("(", " ").replace(")", " ").replace(",", " ").strip()

def normalize_rt_per_file(group: pd.DataFrame) -> pd.DataFrame:
    """CABSEL's per-file RT normalization."""
    rt_numeric = pd.to_numeric(group["RT"], errors="coerce")
    max_rt = max(rt_numeric.max(), 30)
    out = group.copy()
    out["RT"] = out["RT"].astype(object)
    out.loc[rt_numeric.notna(), "RT"] = rt_numeric / max_rt
    return out

def build_ms_sentence(row):
    """CABSEL's MS sentence: 7 metadata + binned_RT + binned_mass + sorted-mz tokens."""
    meta = " ".join(str(row[c]) for c in
                    ["LC_type", "mode", "ionization", "modification",
                     "trap", "fragmentation", "glycan_type"])
    binned_rt = rt_to_token(row["RT_normalised"])
    binned_mass = f"mz{int(np.digitize(row['reducing_mass'], MZ_BIN_EDGES))}"
    mzs = " ".join(row["sorted_mz_tokens"])
    return f"{meta} {binned_rt} {binned_mass} {mzs}"

def process_peak_for_tokens(peak_dict_or_str):
    """Match CABSEL's process_peak_d: parse, normalise, threshold, sort by descending intensity."""
    spec = safe_peak_parse(peak_dict_or_str)
    if spec is None:
        return []
    spec = normalise_spectrum(spec)
    if spec is None:
        return []
    sorted_items = sorted(spec.items(), key=lambda kv: kv[1], reverse=True)
    sorted_items = [(mz, inten) for mz, inten in sorted_items if inten > INTENSITY_THRESHOLD]
    return mz_to_tokens([mz for mz, _ in sorted_items])

# ---- Vocabulary (CABSEL's full vocab construction) -----------------------------

MS_METADATA_COLS = ["LC_type", "mode", "ionization", "modification",
                    "trap", "fragmentation", "glycan_type"]

def build_vocab(df: pd.DataFrame) -> list[str]:
    """Match CABSEL's vocab construction: metadata vals + mz tokens + rt tokens + glycan tokens."""
    words: set[str] = set()
    for col in MS_METADATA_COLS:
        words.update(df[col].astype(str).dropna().unique())
    for i in range(len(MZ_BIN_EDGES) + 1):
        words.add(f"mz{i}")
    for i in range(len(RT_BIN_EDGES) + 1):
        words.add(f"rt{i}")
    for ant in df["antenna"].unique():
        for tok in ant.split():
            if tok:
                words.add(tok)
    return sorted(words)

# ---- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./data/raw/dataset.parquet",
                    help="dataset.xlsx or its parquet equivalent")
    ap.add_argument("--antennae", default="./data/raw/glycans_antennae.csv")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--val_seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset}")
    if str(args.dataset).endswith(".parquet"):
        df = pd.read_parquet(args.dataset)
    else:
        df = pd.read_excel(args.dataset)
    df = df.reset_index(drop=True)
    df["orig_idx"] = df.index
    print(f"  -> {len(df)} rows, cols={list(df.columns)}")

    print(f"[load] {args.antennae}")
    df_ant = pd.read_csv(args.antennae)
    g2a = dict(zip(df_ant["glycan"], df_ant["antenna"]))
    for g in list(g2a):
        if "{" in g or "}" in g:
            g2a[g] = "remove"

    print("[filter] applying run-leakage-prevention filter (RT>2, valid peak_d, non-null filename)")
    rt_proc = pd.to_numeric(df["RT"], errors="coerce").fillna(15)
    keep_rt = (rt_proc > 2) & df["filename"].notna()

    t0 = time.time()
    parsed = df["peak_d"].where(keep_rt).map(safe_peak_parse).map(
        lambda s: None if s is None else normalise_spectrum(s)
    )
    print(f"  peak parsing: {time.time()-t0:.1f}s")
    keep_leakfree = keep_rt & parsed.notnull()

    print("[filter] applying CABSEL filter (glycan has antenna mapping, not curly-bracket)")
    ant_series = df["glycan"].map(g2a)
    has_antenna = ant_series.notna() & (ant_series != "remove")
    df["antenna"] = ant_series.apply(lambda x: split_antenna(x) if isinstance(x, str) and x != "remove" else None)

    keep_both = keep_leakfree & has_antenna
    print(f"  run-leakage-prevention filter kept: {keep_leakfree.sum()}")
    print(f"  CABSEL antenna filter kept        : {has_antenna.sum()}")
    print(f"  intersection                      : {keep_both.sum()}  (= USABLE rows)")
    print(f"  dropped by run-leakage filter only (in CABSEL but filtered out) : {(has_antenna & ~keep_leakfree).sum()}")
    print(f"  dropped by CABSEL filter only (in leak-free set but no antenna) : {(keep_leakfree & ~has_antenna).sum()}")

    df_use = df.loc[keep_both].copy().reset_index(drop=True)
    df_use["normalised_peaks"] = parsed.loc[keep_both].values

    print("[tokenize] building MS sentences and antenna targets")
    df_use = df_use.groupby("filename", group_keys=False).apply(normalize_rt_per_file)
    df_use["RT_normalised"] = df_use["RT"]

    t0 = time.time()
    df_use["sorted_mz_tokens"] = df_use["normalised_peaks"].apply(
        lambda spec: mz_to_tokens([mz for mz, _ in
                                   sorted(((mz, v) for mz, v in spec.items() if v > INTENSITY_THRESHOLD),
                                          key=lambda kv: kv[1], reverse=True)])
    )
    print(f"  per-row mz tokenization: {time.time()-t0:.1f}s")

    t0 = time.time()
    df_use["ms_sentence"] = df_use.apply(build_ms_sentence, axis=1)
    print(f"  per-row sentence assembly: {time.time()-t0:.1f}s")

    print("[split] GroupShuffleSplit by filename, seed=42, test_size=0.15")
    gss = GroupShuffleSplit(test_size=args.test_frac, n_splits=1, random_state=args.split_seed)
    tr_pos, te_pos = next(gss.split(df_use, groups=df_use["filename"].astype(str)))

    train_files = set(df_use.iloc[tr_pos]["filename"])
    test_files = set(df_use.iloc[te_pos]["filename"])
    overlap = train_files & test_files
    assert not overlap, f"RUN-LEAKAGE DETECTED: {len(overlap)} filenames in both train and test"
    print(f"  train rows: {len(tr_pos)}  ({len(train_files)} files)")
    print(f"  test  rows: {len(te_pos)}  ({len(test_files)} files)")
    print(f"  filenames in both train and test: {len(overlap)}  (must be 0)")

    print(f"[split] within-train val: {args.val_frac:.0%} with seed={args.val_seed}")
    rng = np.random.default_rng(seed=args.val_seed)
    val_size = int(args.val_frac * len(tr_pos))
    val_pos_in_tr = rng.choice(len(tr_pos), size=val_size, replace=False)
    train_only_mask = np.ones(len(tr_pos), dtype=bool)
    train_only_mask[val_pos_in_tr] = False
    train_only_pos = tr_pos[train_only_mask]
    val_pos = tr_pos[val_pos_in_tr]
    print(f"  train-only: {len(train_only_pos)}")
    print(f"  val       : {len(val_pos)}")
    print(f"  test      : {len(te_pos)}")

    df_train = df_use.iloc[train_only_pos]
    df_val = df_use.iloc[val_pos]
    df_test = df_use.iloc[te_pos]

    print("[vocab] building CABSEL-format vocab from train")
    vocab_list = build_vocab(df_train)
    print(f"  vocab size: {len(vocab_list)}")
    with open(out / "vocab_glycobart.txt", "w") as f:
        for w in vocab_list:
            f.write(w + "\n")

    print("[write] sentences pkls")
    payload = {
        "train_ms": df_train["ms_sentence"].tolist(),
        "train_glycan": df_train["antenna"].tolist(),
        "train_glycan_iupac": df_train["glycan"].tolist(),
        "val_ms": df_val["ms_sentence"].tolist(),
        "val_glycan": df_val["antenna"].tolist(),
        "val_glycan_iupac": df_val["glycan"].tolist(),
        "test_ms": df_test["ms_sentence"].tolist(),
        "test_glycan": df_test["antenna"].tolist(),
        "test_glycan_iupac": df_test["glycan"].tolist(),
        "vocab_list": vocab_list,
        "split_metadata": {
            "split_seed": args.split_seed,
            "val_seed": args.val_seed,
            "test_frac": args.test_frac,
            "val_frac": args.val_frac,
            "n_train": len(train_only_pos),
            "n_val": len(val_pos),
            "n_test": len(te_pos),
            "n_filtered_leakfree": int(keep_leakfree.sum()),
            "n_filtered_cabsel": int(has_antenna.sum()),
            "n_usable": int(keep_both.sum()),
            "filenames_in_train_and_test": 0,
        },
    }
    with open(out / "sentences_benchmark.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("[write] row manifest CSV (orig_idx, split, filename, glycan)")
    rows = []
    for split, idx in [("train", train_only_pos), ("val", val_pos), ("test", te_pos)]:
        sub = df_use.iloc[idx][["orig_idx", "filename", "glycan"]].copy()
        sub["split"] = split
        rows.append(sub)
    pd.concat(rows).to_csv(out / "row_manifest.csv", index=False)

    print(f"\n[done] outputs in {out}/")
    print(f"  sentences_benchmark.pkl  ({len(train_only_pos)+len(val_pos)+len(te_pos):,} sentences)")
    print(f"  vocab_glycobart.txt      ({len(vocab_list):,} tokens)")
    print(f"  row_manifest.csv         ({len(train_only_pos)+len(val_pos)+len(te_pos):,} rows)")

if __name__ == "__main__":
    main()
