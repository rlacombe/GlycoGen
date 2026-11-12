"""Evaluate GlycoBART with CABSEL's published post-processing pipeline
verbatim.

Imports and calls glycotrans.utils.filter_glycans_glycobart, including
the database-snap step.

Differences from rerank-only post-processing (post_process.py):
  - Filters (drops) candidates failing class enforcement, mass check, or
    fragment-pattern matching
  - Scores remaining candidates via CandyCrumbs (theoretical fragment matching)
  - Snaps surviving predictions to the nearest entry in `df_glycan` (the
    glycowork-curated database of known glycans)

Usage:
    python eval_cabsel_full_pipeline.py --beam 8 --n_spectra 76774
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_glycobart import (
    GlycoBartTokenizer, struct_equal, to_iupac, MAX_SRC_LEN,
    DATA_DIR, OUT_DIR, SENTENCES_PKL, VOCAB_TXT, ensure_data_file,
)

# Add the local glycotrans clone to path; the eval job will copy it from /data
GLYCOTRANS_PATH = os.environ.get("GLYCOTRANS_PATH", "/workdir/glycotrans")
if Path(GLYCOTRANS_PATH).exists():
    sys.path.insert(0, GLYCOTRANS_PATH)

from transformers import BartConfig, BartForConditionalGeneration


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


def derive_glycan_class(iupac: str) -> str:
    """Return CABSEL-style glycan_class string from a truth IUPAC.
    Class is per-experiment in CC2 (a property of the sample/filename), not
    per-prediction — using the truth's glycan_type here is equivalent to using
    the experiment's class label, which CABSEL's pipeline assumes is known."""
    if not isinstance(iupac, str):
        return 'N'  # fallback
    if "GlcNAc(b1-4)GlcNAc" in iupac:
        return 'N'
    if iupac.endswith(("GalNAc", "GalNAc6S", "GalNAcOS", "Fuc", "Man", "Gal")):
        return 'O'
    if iupac.endswith(("Glc", "GlcOS", "GlcNAc", "Ins")):
        return 'free'
    return 'N'  # fallback to most common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "best_model.pt"))
    ap.add_argument("--output", default=str(OUT_DIR / "pareto_cabsel_full.json"))
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--n_spectra", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--MS1_ppm", type=float, default=10,
                    help="CABSEL paper default. Our data has ~250-500 ppm drift; "
                         "set higher (e.g., 500) if filter rejects too many.")
    ap.add_argument("--MS2_tolerance", type=float, default=0.5)
    ap.add_argument("--annotation_thresh", type=float, default=3,
                    help="CABSEL paper default")
    args = ap.parse_args()

    print(f"[setup] device={DEVICE}", flush=True)

    sent_path = ensure_data_file(SENTENCES_PKL, "sentences_benchmark.pkl")
    print(f"[load] {sent_path}", flush=True)
    with open(sent_path, "rb") as f:
        d = pickle.load(f)
    test_ms = d["test_ms"]
    test_iupac = d["test_glycan_iupac"]
    n_total = len(test_ms)
    n = n_total if args.n_spectra < 0 else min(args.n_spectra, n_total)
    print(f"[load] full test set: {n_total}; using first {n}", flush=True)

    meta_path = ensure_data_file(DATA_DIR / "spectrum_metadata.npz", "spectrum_metadata.npz")
    meta = np.load(meta_path, allow_pickle=True)
    test_mass = meta["test_reducing_mass"][:n]
    test_mode = meta["test_mode"][:n]
    test_mod = meta["test_modification"][:n]

    peak_path = ensure_data_file(DATA_DIR / "test_peak_d.pkl", "test_peak_d.pkl")
    with open(peak_path, "rb") as f:
        all_peak_d = pickle.load(f)
    test_peak_d = all_peak_d[:n]

    # Per-spectrum glycan_class derived from truth structure (= per-experiment class)
    test_class = [derive_glycan_class(g) for g in test_iupac[:n]]
    from collections import Counter
    print(f"[setup] test class distribution: {Counter(test_class)}", flush=True)

    # Load checkpoint
    print(f"[load] checkpoint: {args.checkpoint}", flush=True)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    config = BartConfig(**ck["config"])
    model = BartForConditionalGeneration(config).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tokenizer = GlycoBartTokenizer(ck["vocab_list"], max_seq_length=MAX_SRC_LEN)

    # ── Inference: top-K antenna predictions per spectrum ─────────────────
    print(f"[infer] beam={args.beam} top_k={args.top_k}", flush=True)
    pred_antenna = []
    t0 = time.time()
    for s in range(0, n, args.batch_size):
        e = min(s + args.batch_size, n)
        encoded = [tokenizer.encode(t, MAX_SRC_LEN) for t in test_ms[s:e]]
        ids = torch.tensor([x[0] for x in encoded], device=DEVICE)
        mask = torch.tensor([x[1] for x in encoded], device=DEVICE)
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out_ids = model.generate(
                    ids, attention_mask=mask,
                    num_beams=args.beam, num_return_sequences=min(args.top_k, args.beam),
                    max_length=200, early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
        bs_actual = e - s
        k = min(args.top_k, args.beam)
        for i in range(bs_actual):
            antennas = []
            for j in range(k):
                row = out_ids[i * k + j]
                ant = tokenizer.decode(row.tolist(), skip_special_tokens=True)
                antennas.append(ant)
            pred_antenna.append(antennas)
        if (s // args.batch_size) % 10 == 0:
            print(f"  [infer] {s}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[infer] complete in {time.time()-t0:.1f}s", flush=True)

    # ── CABSEL filter_glycans_glycobart: import + run per-class subset ─────
    print("[postproc] importing CABSEL filter_glycans_glycobart...", flush=True)
    try:
        from glycotrans.utils import filter_glycans_glycobart
    except ImportError as e:
        print(f"[FATAL] cannot import filter_glycans_glycobart: {e}", flush=True)
        print(f"[FATAL] sys.path: {sys.path}", flush=True)
        raise

    # CABSEL's pipeline takes per-class. Group test spectra by class, run separately.
    pp_top1 = ["" for _ in range(n)]  # final post-processed top-1 per spectrum
    pp_topk = [[] for _ in range(n)]  # final post-processed top-K per spectrum

    for cls in sorted(set(test_class)):
        # Skip classes not supported by enforce_class_modified
        if cls not in {'O', 'N', 'free', 'lipid'}:
            print(f"[postproc] skipping class={cls!r} (not supported by CABSEL pipeline)", flush=True)
            continue
        idxs = [i for i, c in enumerate(test_class) if c == cls]
        if not idxs:
            continue
        # Most spectra in our test set are negative mode + reduced; for mixed,
        # group by (class, mode, modification) tuple. For simplicity, group by
        # class only and use majority-vote (mode, modification) per class.
        from collections import Counter
        modes_in_class = Counter([str(test_mode[i]) for i in idxs])
        mods_in_class = Counter([str(test_mod[i]) for i in idxs])
        majority_mode = modes_in_class.most_common(1)[0][0]
        majority_mod = mods_in_class.most_common(1)[0][0]
        print(f"[postproc] class={cls} n={len(idxs)} mode={majority_mode} mod={majority_mod}", flush=True)

        # Build the dataframe CABSEL expects
        df_in = pd.DataFrame({
            'predictions': [pred_antenna[i] for i in idxs],
            'intensity': [0] * len(idxs),
            'reducing_mass': [float(test_mass[i]) for i in idxs],
            'peak_d': [test_peak_d[i] for i in idxs],
            'RT': [0.0] * len(idxs),
            'RT_original': [0.0] * len(idxs),
            'scan_number': [str(i) for i in idxs],
        })

        try:
            t1 = time.time()
            df_filtered, _ = filter_glycans_glycobart(
                df_in,
                glycan_class=cls,
                modification=majority_mod if majority_mod in {'reduced', '2AA', '2AB', 'permethylated'} else 'underivatized',
                mode=majority_mode if majority_mode in {'positive', 'negative'} else 'negative',
                MS1_ppm=args.MS1_ppm,
                MS2_tolerance=args.MS2_tolerance,
                annotation_thresh=args.annotation_thresh,
                taxonomy_filter='Mammalia',  # CABSEL default
            )
            print(f"[postproc] class={cls} done in {time.time()-t1:.0f}s, {len(df_filtered)} surviving rows", flush=True)
        except Exception as e:
            print(f"[postproc] class={cls} FAILED: {type(e).__name__}: {e}", flush=True)
            continue

        # Map results back: df_filtered has scan_number column corresponding to our idx
        # df_filtered['predictions_glycobart'] is the post-processed top-K list
        scan_to_preds = dict(zip(
            df_filtered.get('scan_number', []).astype(str),
            df_filtered.get('predictions_glycobart', df_filtered.get('predictions', [])),
        ))
        for i in idxs:
            preds = scan_to_preds.get(str(i), [])
            if preds and len(preds) > 0:
                pp_top1[i] = preds[0]
                pp_topk[i] = list(preds[:args.top_k])

    # ── Score raw + post-processed ────────────────────────────────────────
    raw_top1 = pp_top1_count = raw_topk = pp_topk_count = 0
    raw_predictions = [[to_iupac(a) for a in pred_antenna[i]] for i in range(n)]
    for i in range(n):
        truth = test_iupac[i]
        if struct_equal(raw_predictions[i][0], truth):
            raw_top1 += 1
        if any(struct_equal(p, truth) for p in raw_predictions[i]):
            raw_topk += 1
        if struct_equal(pp_top1[i], truth):
            pp_top1_count += 1
        if any(struct_equal(p, truth) for p in pp_topk[i]):
            pp_topk_count += 1

    raw_top1_acc = raw_top1 / n
    raw_topk_acc = raw_topk / n
    pp_top1_acc = pp_top1_count / n
    pp_topk_acc = pp_topk_count / n

    print(f"\n=== Results (n={n}, beam={args.beam}, top_k={args.top_k}) ===")
    print(f"  raw model top-1                : {raw_top1_acc:.4f}")
    print(f"  CABSEL full-pipeline top-1     : {pp_top1_acc:.4f}  (Δ {pp_top1_acc-raw_top1_acc:+.4f})")
    print(f"  raw model top-K                : {raw_topk_acc:.4f}")
    print(f"  CABSEL full-pipeline top-K     : {pp_topk_acc:.4f}  (Δ {pp_topk_acc-raw_topk_acc:+.4f})")

    out = {
        "checkpoint": args.checkpoint,
        "beam": args.beam,
        "top_k": args.top_k,
        "n": n,
        "MS1_ppm": args.MS1_ppm,
        "MS2_tolerance": args.MS2_tolerance,
        "annotation_thresh": args.annotation_thresh,
        "raw_top1": raw_top1_acc,
        "raw_topk": raw_topk_acc,
        "cabsel_full_top1": pp_top1_acc,
        "cabsel_full_topk": pp_topk_acc,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
