"""Older builder: generate the numerical (DFM/AR) format pkls aligned to
the leak-free CC2 split, using `candycrunch.prediction.bin_intensities`
as an external dependency. `make_aligned_numerical_pkls.py` is the
preferred entry point (it re-implements the same binning locally so the
release does not depend on the CandyCrunch package); this file is
retained only as a cross-check reference.

Loads dataset.parquet + row_manifest.csv, applies CABSEL/Urban et al.'s
build_training_pickles.py preprocessing on the leak-free row identities,
emits X_{train,val,test}.pkl and y_{train,val,test}.pkl matching CC2's
numerical format convention.

X_*.pkl: list of tuples
    (binned_intensities[2048], mz_remainder[2048], reducing_mass,
     glycan_type, RT, mode, lc, modification, trap)
y_*.pkl: list of full IUPAC strings (same as sentences_benchmark.pkl[*_glycan_iupac])
"""
import ast
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from candycrunch.prediction import bin_intensities


MODE_MAP = {"negative": 0, "positive": 1}
LC_MAP = {"PGC": 0, "C18": 1}
MOD_MAP = {"reduced": 0, "permethylated": 1}
TRAP_MAP = {"linear": 0, "orbitrap": 1, "amazon": 2}

FEATURE_COLS = ["binned_intensities", "mz_remainder", "reducing_mass",
                "glycan_type_int", "RT", "mode_int", "lc_int",
                "modification_int", "trap_int"]


def safe_peak_parse(v):
    """Parse `peak_d` (dict or literal-dict string) into {float mz: float intensity}.
    Returns None on empty/unparseable values so caller can zero the row."""
    if isinstance(v, dict):
        return {float(k): float(x) for k, x in v.items()}
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            p = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return None
        if isinstance(p, dict):
            return {float(k): float(x) for k, x in p.items()}
    return None


def normalise(spec):
    """L1-normalise intensities so they sum to 1. Returns None on zero-sum spectra."""
    t = float(sum(spec.values()))
    return None if t <= 0 else {k: v / t for k, v in spec.items()}


def classify_glycan_type(g):
    """Rule-based glycan-type integer (CandyCrunch convention):
    0 = O-glycan, 1 = N-glycan (chitobiose core), 2 = free / other reducing end, 3 = fallback."""
    if not isinstance(g, str):
        return 3
    if g.endswith(("GalNAc", "GalNAc6S", "GalNAcOS", "Fuc", "Man", "Gal")):
        return 0
    if "GlcNAc(b1-4)GlcNAc" in g:
        return 1
    if g.endswith(("Glc", "GlcOS", "GlcNAc", "Ins")):
        return 2
    return 3


def main():
    base = Path("./")
    out = base / "data"
    df = pd.read_parquet("./data/raw/dataset.parquet").reset_index(drop=True)
    manifest = pd.read_csv(out / "row_manifest.csv")
    print(f"[load] df={len(df)}, manifest={len(manifest)}")

    # Process all rows we'll need (union of train/val/test orig_idx)
    needed_idx = manifest["orig_idx"].unique()
    print(f"[work] processing {len(needed_idx)} rows")
    sub = df.loc[needed_idx].copy()
    sub["orig_idx"] = needed_idx

    # RT processing per build_training_pickles.py
    sub["RT"] = pd.to_numeric(sub["RT"], errors="coerce").fillna(15)
    sub["RT_norm"] = sub.groupby("filename")["RT"].transform(lambda rt: rt / max(rt.max(), 30.0))

    # Glycan type integer (rule-based, from build_training_pickles.py)
    sub["glycan_type_int"] = sub["glycan"].map(classify_glycan_type)

    # Metadata integers (default fallbacks per build_training_pickles.py)
    sub["mode_int"] = sub["mode"].astype(str).str.lower().map(MODE_MAP).fillna(2).astype(int)
    sub["lc_int"] = sub["LC_type"].astype(str).str.lower().map(LC_MAP).fillna(2).astype(int)
    sub["modification_int"] = sub["modification"].astype(str).str.lower().map(MOD_MAP).fillna(2).astype(int)
    sub["trap_int"] = sub["trap"].astype(str).str.lower().map(TRAP_MAP).fillna(3).astype(int)

    # Bin intensities
    print("[work] parsing peaks...")
    t0 = time.time()
    parsed = sub["peak_d"].map(safe_peak_parse).map(lambda s: None if s is None else normalise(s))
    sub["parsed_spec"] = parsed
    print(f"  parsed in {time.time()-t0:.1f}s")

    print("[work] binning intensities (2048 bins, 39.714-3000 m/z)...")
    t0 = time.time()
    frames = np.linspace(39.714, 3000.0, 2048)
    binned_arr, remain_arr = [], []
    n_invalid = 0
    for spec in sub["parsed_spec"]:
        if spec is None:
            binned_arr.append(np.zeros(2048, dtype=np.float32))
            remain_arr.append(np.zeros(2048, dtype=np.float32))
            n_invalid += 1
        else:
            b, r = bin_intensities(spec, frames)
            binned_arr.append(np.asarray(b, dtype=np.float32))
            remain_arr.append(np.asarray(r, dtype=np.float32))
    sub["binned_intensities"] = binned_arr
    sub["mz_remainder"] = remain_arr
    print(f"  binned in {time.time()-t0:.1f}s ({n_invalid} invalid spectra zeroed)")

    sub["RT"] = sub["RT_norm"]  # overwrite RT with the normalized value
    # Index by orig_idx for fast lookup
    sub_indexed = sub.set_index("orig_idx", drop=False)

    for split in ["train", "val", "test"]:
        sm = manifest[manifest["split"] == split].reset_index(drop=True)
        sm_idx = sm["orig_idx"].values
        rows = sub_indexed.loc[sm_idx]
        X = list(rows[FEATURE_COLS].itertuples(index=False, name=None))
        y = sm["glycan"].tolist()
        with open(out / f"X_{split}.pkl", "wb") as f:
            pickle.dump(X, f)
        with open(out / f"y_{split}.pkl", "wb") as f:
            pickle.dump(y, f)
        x_size = (out / f"X_{split}.pkl").stat().st_size / 1e9
        y_size = (out / f"y_{split}.pkl").stat().st_size / 1e6
        print(f"[write] {split}: X_{split}.pkl ({x_size:.2f} GB), y_{split}.pkl ({y_size:.2f} MB), n={len(X)}")


if __name__ == "__main__":
    main()
