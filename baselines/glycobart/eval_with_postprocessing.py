"""Evaluate GlycoBART with the post-processing pipeline.

Loads the trained checkpoint, runs inference at a given beam, applies
filter_glycans_glycobart-style post-processing (mass + class + fragment,
no db-snap), and reports both raw-model and post-processed accuracy.

Raw numbers come from eval_glycobart.py; post-processed numbers come
from this script. The same post_process module can be applied to
DFM/AR predictions.

Usage (HF Jobs):
    python eval_with_postprocessing.py --beam 8 --n_spectra 76774
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_glycobart import (
    GlycoBartTokenizer, struct_equal, to_iupac, MAX_SRC_LEN,
    DATA_DIR, OUT_DIR, SENTENCES_PKL, VOCAB_TXT, ensure_data_file,
)
from post_process import FilterConfig, score_predictions

from transformers import BartConfig, BartForConditionalGeneration


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "best_model.pt"))
    ap.add_argument("--output", default=str(OUT_DIR / "pareto_post.json"))
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--n_spectra", type=int, default=-1, help="-1 = full test set")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--MS1_ppm", type=float, default=500,
                    help="Mass-check ppm threshold. Test set has ~250-500 ppm calibration drift, "
                         "so CABSEL's default 10 is too strict. Default 500 here.")
    ap.add_argument("--MS2_tolerance", type=float, default=0.5)
    ap.add_argument("--n_jobs", type=int, default=-1, help="-1 = all CPU cores")
    args = ap.parse_args()

    print(f"[setup] device={DEVICE}", flush=True)
    sent_path = ensure_data_file(SENTENCES_PKL, "sentences_benchmark.pkl")
    print(f"[load] {sent_path}", flush=True)
    with open(sent_path, "rb") as f:
        d = pickle.load(f)
    test_ms = d["test_ms"]
    test_iupac = d["test_glycan_iupac"]
    test_antenna = d["test_glycan"]
    n_total = len(test_ms)
    n = n_total if args.n_spectra < 0 else min(args.n_spectra, n_total)
    print(f"[load] full test set: {n_total}; using first {n}", flush=True)

    # Spectrum metadata (mass, mode, modification)
    meta_path = ensure_data_file(DATA_DIR / "spectrum_metadata.npz", "spectrum_metadata.npz")
    print(f"[load] {meta_path}", flush=True)
    meta = np.load(meta_path, allow_pickle=True)
    test_mass = meta["test_reducing_mass"][:n]
    test_mode = meta["test_mode"][:n]
    test_mod = meta["test_modification"][:n]

    # Peak data
    peak_path = ensure_data_file(DATA_DIR / "test_peak_d.pkl", "test_peak_d.pkl")
    print(f"[load] {peak_path}", flush=True)
    with open(peak_path, "rb") as f:
        all_peak_d = pickle.load(f)
    test_peak_d = all_peak_d[:n]

    # Glycan class is NOT used — we don't have non-leaky per-spectrum class info,
    # and post_process.process_spectrum ignores the parameter (re-rank-only mode).
    # We pass 'other' uniformly to avoid any class-dependent behavior.
    test_glycan_class = ['other'] * n

    # Load checkpoint
    print(f"[load] checkpoint: {args.checkpoint}", flush=True)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    config = BartConfig(**ck["config"])
    model = BartForConditionalGeneration(config).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tokenizer = GlycoBartTokenizer(ck["vocab_list"], max_seq_length=MAX_SRC_LEN)

    # ── Inference: top-K predictions per spectrum ─────────────────────────
    print(f"[infer] beam={args.beam} top_k={args.top_k} bs={args.batch_size}", flush=True)
    predictions_per_spectrum = []
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
            preds_iupac = []
            for j in range(k):
                row = out_ids[i * k + j]
                ant = tokenizer.decode(row.tolist(), skip_special_tokens=True)
                preds_iupac.append(to_iupac(ant))
            predictions_per_spectrum.append(preds_iupac)
        if (s // args.batch_size) % 10 == 0:
            elap = time.time() - t0
            print(f"  [infer] {s}/{n} ({elap:.0f}s, {(elap/max(s,1)*1000):.1f} ms/spec)", flush=True)
    print(f"[infer] complete in {time.time()-t0:.1f}s, {len(predictions_per_spectrum)} predictions", flush=True)

    # ── Post-processing ───────────────────────────────────────────────────
    config_pp = FilterConfig(MS1_ppm=args.MS1_ppm, MS2_tolerance=args.MS2_tolerance)
    print(f"[postproc] config: {config_pp}", flush=True)
    pp_results = score_predictions(
        predictions_per_spectrum,
        observed_masses=[float(m) for m in test_mass],
        modes=[str(m) for m in test_mode],
        modifications=[str(m) for m in test_mod],
        glycan_classes=test_glycan_class,
        peak_ds=test_peak_d,
        config=config_pp,
        n_jobs=args.n_jobs,
        verbose=True,
    )

    # ── Score raw + post-processed ─────────────────────────────────────────
    raw_top1 = 0; pp_top1 = 0; raw_topk = 0; pp_topk = 0
    for i in range(n):
        truth = test_iupac[i]
        raw_preds = predictions_per_spectrum[i]
        pp = pp_results[i]
        # Raw top-1
        if struct_equal(raw_preds[0], truth):
            raw_top1 += 1
        # Raw top-k (any of the K candidates structurally matches)
        if any(struct_equal(p, truth) for p in raw_preds):
            raw_topk += 1
        # Post-processed top-1
        if struct_equal(pp["top1_post"], truth):
            pp_top1 += 1
        # Post-processed top-k
        if any(struct_equal(p, truth) for p in pp["topk_post"]):
            pp_topk += 1

    raw_top1 /= n; pp_top1 /= n; raw_topk /= n; pp_topk /= n

    print(f"\n=== Results (n={n}, beam={args.beam}, top_k={args.top_k}) ===")
    print(f"  raw model top-1     : {raw_top1:.4f}")
    print(f"  post-proc top-1     : {pp_top1:.4f}  (Δ {pp_top1-raw_top1:+.4f})")
    print(f"  raw model top-K     : {raw_topk:.4f}")
    print(f"  post-proc top-K     : {pp_topk:.4f}  (Δ {pp_topk-raw_topk:+.4f})")

    out = {
        "checkpoint": args.checkpoint,
        "beam": args.beam,
        "top_k": args.top_k,
        "n": n,
        "MS1_ppm": args.MS1_ppm,
        "MS2_tolerance": args.MS2_tolerance,
        "raw_top1": raw_top1,
        "raw_topk": raw_topk,
        "pp_top1": pp_top1,
        "pp_topk": pp_topk,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
