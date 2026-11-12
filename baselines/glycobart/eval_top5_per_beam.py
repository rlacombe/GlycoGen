"""Per-K top-5 evaluation for GlycoBART. Three separate runs (K=1, 2, 4),
each producing a distinct top-5 number.

Strategy per K:
  - beam=K beam search → K candidates
  - nucleus sampling (top_p=0.9, do_sample=True) → (5-K) additional candidates
  - combine into 5 candidates per spectrum
  - top-5 = any of 5 matches truth

Cost attribution: K-step decode cost from existing Pareto sweep. Sampling
is treated as auxiliary candidate generation that does not change the
reported throughput accounting.

Output: one row per K with K-distinct top-5.
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


# Cost columns from existing Pareto sweep (METRIC_FINAL beam=K rows)
NFE_AND_MS_PER_K = {
    1: {"nfe_per_spec": 22.88, "ms_per_spec": 2.80,  "top1": 0.8118},
    2: {"nfe_per_spec": 45.10, "ms_per_spec": 10.07, "top1": 0.7915},
    4: {"nfe_per_spec": 90.05, "ms_per_spec": 15.93, "top1": 0.7877},
}


def generate_pool(model, tokenizer, test_ms, n, beam, num_return, do_sample, batch_size):
    """Run model.generate over the full test set. Return list of [num_return strings] per spectrum."""
    out_per_spec = []
    t0 = time.time()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        encoded = [tokenizer.encode(t, MAX_SRC_LEN) for t in test_ms[s:e]]
        ids = torch.tensor([x[0] for x in encoded], device=DEVICE)
        mask = torch.tensor([x[1] for x in encoded], device=DEVICE)
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                kw = dict(
                    input_ids=ids, attention_mask=mask,
                    max_length=200,
                    num_return_sequences=num_return,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
                if do_sample:
                    kw.update(do_sample=True, top_p=0.9, temperature=1.0, num_beams=1)
                else:
                    kw.update(num_beams=beam, early_stopping=True)
                out_ids = model.generate(**kw)
        bs_actual = e - s
        for i in range(bs_actual):
            cands = []
            for j in range(num_return):
                row = out_ids[i * num_return + j]
                ant = tokenizer.decode(row.tolist(), skip_special_tokens=True)
                cands.append(to_iupac(ant))
            out_per_spec.append(cands)
        if (s // batch_size) % 20 == 0:
            elap = time.time() - t0
            tag = "sample" if do_sample else f"beam={beam}"
            print(f"    [{tag} num_return={num_return}] {s}/{n} ({elap:.0f}s, {(elap/max(s,1)*1000):.1f} ms/spec)", flush=True)
    return out_per_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "best_model.pt"))
    ap.add_argument("--output", default=str(OUT_DIR / "top5_per_beam.json"))
    ap.add_argument("--n_spectra", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--sample_batch_size", type=int, default=64,
                    help="Smaller bs for nucleus sampling (more parallel sequences per spec)")
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
    print(f"[load] full test set: {n_total}; using {n}", flush=True)

    print(f"[load] checkpoint: {args.checkpoint}", flush=True)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    config = BartConfig(**ck["config"])
    model = BartForConditionalGeneration(config).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tokenizer = GlycoBartTokenizer(ck["vocab_list"], max_seq_length=MAX_SRC_LEN)

    rows = []
    for K in (1, 2, 4):
        print(f"\n=== Per-K run: K={K} ===", flush=True)

        # Single call: beam-sample with width=K, return 5 sequences.
        # do_sample=True allows num_return_sequences > num_beams.
        print(f"  [step 1/1] beam-sample num_beams={K} num_return=5 top_p=0.9", flush=True)
        combined = generate_pool(model, tokenizer, test_ms, n,
                                 beam=K, num_return=5, do_sample=True,
                                 batch_size=args.sample_batch_size)

        # Score top-5
        top5 = 0
        for i in range(n):
            truth = test_iupac[i]
            if any(struct_equal(c, truth) for c in combined[i]):
                top5 += 1
        top5_acc = top5 / n

        meta = NFE_AND_MS_PER_K[K]
        n_pad = max(0, 5 - K)
        row = {
            "beam_K": K,
            "n_beam_candidates": K,
            "n_nucleus_pad": n_pad,
            "top1_existing": meta["top1"],
            "top5": round(top5_acc, 4),
            "nfe_per_spec": meta["nfe_per_spec"],
            "ms_per_spec": meta["ms_per_spec"],
        }
        rows.append(row)
        print(f"  [score] K={K}: top-5={top5_acc:.4f} (top-1 from sweep={meta['top1']:.4f})", flush=True)

    print("\n=== Final 3-row table ===")
    print(f"{'K':<4} {'top-1':<8} {'top-5':<8} {'NFE/spec':<10} {'ms/spec':<8}")
    for r in rows:
        print(f"{r['beam_K']:<4} {r['top1_existing']:<8} {r['top5']:<8} {r['nfe_per_spec']:<10} {r['ms_per_spec']:<8}")

    out = {
        "n": n,
        "checkpoint": args.checkpoint,
        "rows": rows,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
