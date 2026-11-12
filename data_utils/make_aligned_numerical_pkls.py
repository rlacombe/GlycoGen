"""Generate the numerical encoder pkls for the leak-free CandyCrunch v2
split. Each X row is a 9-tuple in CandyCrunch's `build_training_pickles.py`
format:

    (binned_intensities[2048] float32,
     mz_remainder[2048] float32,
     reducing_mass float32,
     glycan_type int,
     RT_normalized float32,
     mode int, lc int, modification int, trap int)

Compatible with the publicly hosted CandyCrunch v2 `X_*_CC2_240110.pkl`
format (the AR/DFM encoder consumes only x[0..2]; the rest are kept for
compatibility).

Aligned to row_manifest.csv (the same row identities as
sentences_benchmark.pkl), so DFM/MDLM/AR/GlycoBART share the same 378,880
train / 42,097 val / 76,774 test rows.
"""
from __future__ import annotations
import ast
import pickle
import time
from pathlib import Path

import numpy as np
import numpy_indexed as npi
import pandas as pd

DATASET  = "./data/raw/dataset.parquet"
MANIFEST = "./data/row_manifest.csv"
OUT_DIR  = Path("./data")

# CandyCrunch metadata maps (verbatim from build_training_pickles.py).
MODE_MAP = {"negative": 0, "positive": 1}
LC_MAP   = {"PGC": 0, "C18": 1}
MOD_MAP  = {"reduced": 0, "permethylated": 1}
TRAP_MAP = {"linear": 0, "orbitrap": 1, "amazon": 2}

# Binning params (CC2 default: 39.714–3000 Da, 2048 bins).
MIN_MZ, MAX_MZ, BIN_NUM = 39.714, 3000.0, 2048
FRAMES = np.linspace(MIN_MZ, MAX_MZ, BIN_NUM)


def safe_peak_parse(value):
    """Parse `peak_d` (dict or literal-dict string) into {float mz: float intensity}.
    Returns None on empty/unparseable values so caller can zero the row."""
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


def bin_intensities(peak_d, frames=FRAMES):
    """Verbatim from `candycrunch.prediction.bin_intensities`."""
    num_frames = len(frames)
    binned = np.zeros(num_frames)
    mz_diff = np.zeros(num_frames)
    mzs = np.array(list(peak_d.keys()), dtype="float32")
    intensities = np.array(list(peak_d.values()))
    bin_indices = np.digitize(mzs, frames, right=True)
    mz_remainder = mzs - frames[bin_indices - 1]
    max_intensities = npi.group_by(bin_indices - 1).max(intensities)
    mz_remainder = mz_remainder * np.isin(intensities, max_intensities)
    unique_bins, summed = npi.group_by(bin_indices).sum(intensities)
    _, max_mz_remainder = npi.group_by(bin_indices).max(mz_remainder)
    binned[unique_bins - 1] = summed
    mz_diff[unique_bins - 1] = max_mz_remainder
    return binned, mz_diff


def classify_glycan_type(g: str) -> int:
    """Rule-based glycan-type integer (CandyCrunch convention):
    0 = O-glycan, 1 = N-glycan (chitobiose core), 2 = free / other reducing end, 3 = fallback."""
    if g.endswith(("GalNAc", "GalNAc6S", "GalNAcOS", "Fuc", "Man", "Gal")):
        return 0
    if "GlcNAc(b1-4)GlcNAc" in g:
        return 1
    if g.endswith(("Glc", "GlcOS", "GlcNAc", "Ins")):
        return 2
    return 3


def map_meta(v, mapping, fallback):
    """Lookup a metadata string in `mapping`, returning `fallback` on non-strings or misses."""
    if not isinstance(v, str):
        return fallback
    return mapping.get(v.strip(), fallback)


def main():
    print(f"[load] {DATASET}", flush=True)
    df = pd.read_parquet(DATASET).reset_index(drop=True)
    df["orig_idx"] = df.index
    print(f"  -> {len(df):,} rows", flush=True)

    print(f"[load] {MANIFEST}", flush=True)
    manifest = pd.read_csv(MANIFEST)
    counts = manifest["split"].value_counts().to_dict()
    print(f"  -> {len(manifest):,} rows; splits={counts}", flush=True)

    # Inner-join on orig_idx; preserve manifest's row order so each split's
    # output is in the same order as sentences_benchmark.pkl.
    manifest_pos = pd.Series(np.arange(len(manifest)), index=manifest["orig_idx"].values)
    df_use = df.merge(manifest[["orig_idx", "split"]], on="orig_idx", how="inner")
    df_use["manifest_pos"] = df_use["orig_idx"].map(manifest_pos)
    df_use = df_use.sort_values("manifest_pos").reset_index(drop=True)
    print(f"  merged: {len(df_use):,} rows", flush=True)
    assert len(df_use) == len(manifest), \
        f"row count mismatch: df_use={len(df_use)} manifest={len(manifest)}"

    # Per-file RT normalization (CandyCrunch convention: rt / max(rt.max(), 30)).
    # Rows in the manifest have already passed the RT>2 filter; we just normalize.
    print("[rt] per-file RT normalization", flush=True)
    df_use["RT"] = pd.to_numeric(df_use["RT"], errors="coerce").fillna(15)
    df_use["RT_norm"] = df_use.groupby("filename")["RT"].transform(
        lambda rt: rt / max(rt.max(), 30.0)
    )

    # Glycan type + metadata int codes
    df_use["gt_int"]   = df_use["glycan"].astype(str).map(classify_glycan_type)
    df_use["mode_int"] = df_use["mode"].apply(lambda v: map_meta(v, MODE_MAP, 2))
    df_use["lc_int"]   = df_use["LC_type"].apply(lambda v: map_meta(v, LC_MAP, 2))
    df_use["mod_int"]  = df_use["modification"].apply(lambda v: map_meta(v, MOD_MAP, 2))
    df_use["trap_int"] = df_use["trap"].apply(lambda v: map_meta(v, TRAP_MAP, 3))

    # Bin peaks per row.
    print("[peaks] binning peak_d into 2048-bin spectra", flush=True)
    n = len(df_use)
    binned_list = [None] * n
    remainder_list = [None] * n
    bad_rows = 0
    t0 = time.time()
    peak_col = df_use["peak_d"].tolist()
    for i in range(n):
        spec = safe_peak_parse(peak_col[i])
        spec = normalise_spectrum(spec) if spec else None
        if spec is None or len(spec) == 0:
            binned_list[i] = np.zeros(BIN_NUM, dtype=np.float32)
            remainder_list[i] = np.zeros(BIN_NUM, dtype=np.float32)
            bad_rows += 1
            continue
        try:
            b, r = bin_intensities(spec)
            binned_list[i] = b.astype(np.float32)
            remainder_list[i] = r.astype(np.float32)
        except Exception as e:
            binned_list[i] = np.zeros(BIN_NUM, dtype=np.float32)
            remainder_list[i] = np.zeros(BIN_NUM, dtype=np.float32)
            bad_rows += 1
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,}/{n:,} dt={time.time()-t0:.1f}s bad={bad_rows}", flush=True)
    print(f"  done dt={time.time()-t0:.1f}s bad={bad_rows}", flush=True)

    # Pull split-aware columns into numpy arrays for fast indexing.
    splits = df_use["split"].values
    rmass  = df_use["reducing_mass"].astype(np.float32).values
    gtint  = df_use["gt_int"].astype(int).values
    rtnorm = df_use["RT_norm"].astype(np.float32).values
    md_i   = df_use["mode_int"].astype(int).values
    lc_i   = df_use["lc_int"].astype(int).values
    mod_i  = df_use["mod_int"].astype(int).values
    tp_i   = df_use["trap_int"].astype(int).values
    glycans = df_use["glycan"].astype(str).values

    print("[build] assembling 9-tuples per split", flush=True)
    split_X = {"train": [], "val": [], "test": []}
    split_y = {"train": [], "val": [], "test": []}
    for i in range(n):
        s = splits[i]
        tup = (
            binned_list[i],
            remainder_list[i],
            float(rmass[i]),
            int(gtint[i]),
            float(rtnorm[i]),
            int(md_i[i]),
            int(lc_i[i]),
            int(mod_i[i]),
            int(tp_i[i]),
        )
        split_X[s].append(tup)
        split_y[s].append(glycans[i])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        xp = OUT_DIR / f"X_{s}.pkl"
        yp = OUT_DIR / f"y_{s}.pkl"
        with open(xp, "wb") as f: pickle.dump(split_X[s], f)
        with open(yp, "wb") as f: pickle.dump(split_y[s], f)
        sz = (xp.stat().st_size + yp.stat().st_size) / 1e6
        print(f"[write] {s}: X={xp.name} y={yp.name} n={len(split_X[s])} ({sz:.1f}MB total)", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
