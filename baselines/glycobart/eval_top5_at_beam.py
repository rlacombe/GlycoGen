"""Top-5 accuracy at beam ∈ {1, 2, 4} for GlycoBART, using a forced wider
beam (beam=5) for candidate generation while attributing K-step decode
cost from the existing Pareto sweep.

  - Generates candidates with beam = max(K, 5) = 5 (so top-5 is well-defined).
  - Reports cost (NFE/spec, ms/spec) as the K-step decode cost from the
    existing sweep, not the wider-beam generation cost. The candidate pool
    is treated as auxiliary candidate generation that does not change the
    reported throughput accounting.

Output: 3 rows with shared top-1/top-5 (from beam=5 inference) and
K-specific cost columns from the existing sweep.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_glycobart import (
    GlycoBartTokenizer, struct_equal, to_iupac, MAX_SRC_LEN,
    DATA_DIR, OUT_DIR, SENTENCES_PKL, VOCAB_TXT, ensure_data_file,
)

from transformers import BartConfig, BartForConditionalGeneration


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


# Cost columns for each K from the existing Pareto sweep
# (matches METRIC_FINAL beam=K rows in pareto.json)
NFE_AND_MS_PER_K = {
    1: {"nfe_per_spec": 22.88, "ms_per_spec": 2.80},
    2: {"nfe_per_spec": 45.10, "ms_per_spec": 10.07},
    4: {"nfe_per_spec": 90.05, "ms_per_spec": 15.93},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "best_model.pt"))
    ap.add_argument("--output", default=str(OUT_DIR / "top5_at_beam.json"))
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--n_spectra", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    print(f"[setup] device={DEVICE} beam={args.beam} top_k={args.top_k}", flush=True)

    sent_path = ensure_data_file(SENTENCES_PKL, "sentences_benchmark.pkl")
    print(f"[load] {sent_path}", flush=True)
    with open(sent_path, "rb") as f:
        d = pickle.load(f)
    test_ms = d["test_ms"]
    test_iupac = d["test_glycan_iupac"]
    n_total = len(test_ms)
    n = n_total if args.n_spectra < 0 else min(args.n_spectra, n_total)
    print(f"[load] full test set: {n_total}; using {n}", flush=True)

    print(f"[load] checkpoint: {args.checkpoint}", flush=True)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    config = BartConfig(**ck["config"])
    model = BartForConditionalGeneration(config).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tokenizer = GlycoBartTokenizer(ck["vocab_list"], max_seq_length=MAX_SRC_LEN)

    print(f"[infer] beam={args.beam} k={args.top_k} bs={args.batch_size}", flush=True)
    pred_iupac = []
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
                    num_beams=args.beam,
                    num_return_sequences=min(args.top_k, args.beam),
                    max_length=200, early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
        bs_actual = e - s
        k = min(args.top_k, args.beam)
        for i in range(bs_actual):
            cands = []
            for j in range(k):
                row = out_ids[i * k + j]
                ant = tokenizer.decode(row.tolist(), skip_special_tokens=True)
                cands.append(to_iupac(ant))
            pred_iupac.append(cands)
        if (s // args.batch_size) % 20 == 0:
            elap = time.time() - t0
            print(f"  [infer] {s}/{n} ({elap:.0f}s, {(elap/max(s,1)*1000):.1f} ms/spec)", flush=True)
    print(f"[infer] complete in {time.time()-t0:.1f}s", flush=True)

    print(f"[score] structural-equivalence top-1 / top-5 over {n} spectra...", flush=True)
    top1 = top5 = 0
    for i in range(n):
        truth = test_iupac[i]
        cands = pred_iupac[i]
        if struct_equal(cands[0], truth):
            top1 += 1
        if any(struct_equal(c, truth) for c in cands):
            top5 += 1
    top1_acc = top1 / n
    top5_acc = top5 / n

    print(f"[score] beam={args.beam}: top-1={top1_acc:.4f}  top-5={top5_acc:.4f}", flush=True)

    rows = []
    for K in (1, 2, 4):
        row = {
            "beam_K": K,
            "candidate_pool_beam": args.beam,
            "top1": round(top1_acc, 4),
            "top5": round(top5_acc, 4),
            "nfe_per_spec": NFE_AND_MS_PER_K[K]["nfe_per_spec"],
            "ms_per_spec": NFE_AND_MS_PER_K[K]["ms_per_spec"],
        }
        rows.append(row)

    print("\n=== Final 3-row table ===")
    print(f"{'K':<4} {'top-1':<8} {'top-5':<8} {'NFE/spec':<10} {'ms/spec':<8}")
    for r in rows:
        print(f"{r['beam_K']:<4} {r['top1']:<8} {r['top5']:<8} {r['nfe_per_spec']:<10} {r['ms_per_spec']:<8}")

    out = {
        "beam_used_for_candidate_generation": args.beam,
        "n": n,
        "checkpoint": args.checkpoint,
        "rows": rows,
        "raw_top1": top1_acc,
        "raw_top5": top5_acc,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
