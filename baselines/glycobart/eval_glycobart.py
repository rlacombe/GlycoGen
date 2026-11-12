# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "numpy<2",
#   "transformers>=4.44",
#   "huggingface_hub>=0.24",
#   "glycowork",
# ]
# ///
"""Pareto sweep for GlycoBART on the leak-free CandyCrunch v2 test set.

For each beam in {1, 2, 4, 8, 16, 32}:
  1. ACCURACY: batched generate(num_beams=k, max_length=200, early_stopping=True)
     on the FULL test set, post-processed via convert_to_iupac and
     scored via glycowork.compare_glycans.
  2. WALL-CLOCK: time one batched generate(...) call on the largest batch that
     fits at this beam value, after a 5-pass GPU warmup; divide by batch size.
  3. NFE: nfe_per_spectrum = mean output sequence length × k.

Outputs:
  /output/pareto.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from transformers import BartConfig, BartForConditionalGeneration

# Try to import shared helpers from the sibling train_glycobart.py (local dev
# convenience). If the sibling isn't there (e.g. on HF Jobs, where uv only
# uploads this single script), fall back to the inlined copies below.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from train_glycobart import (
        GlycoBartTokenizer, struct_equal, to_iupac, MAX_SRC_LEN,
        DATA_DIR, OUT_DIR, SENTENCES_PKL, VOCAB_TXT,
        HF_DATASET_REPO, ensure_data_file,
    )
except ImportError:
    # ---- Inlined from train_glycobart.py (unchanged, so numbers match) ----
    if os.getenv("COMPUTE_BACKEND") == "hfjobs" or os.getenv("HF_JOB") == "1":
        DATA_DIR = Path("/data")
        OUT_DIR = Path("/output")
    else:
        BASE = Path(__file__).resolve().parent
        DATA_DIR = BASE / "data"
        OUT_DIR = BASE / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_DATASET_REPO = os.environ.get(
        "HF_DATASET_REPO",
        os.environ.get("HF_REPO", "rlacombe/candycrunch-v2-benchmark"),
    )
    SENTENCES_PKL = Path(os.environ.get("SENTENCES_PKL", str(DATA_DIR / "sentences_benchmark.pkl")))
    VOCAB_TXT = Path(os.environ.get("VOCAB_TXT", str(DATA_DIR / "vocab_glycobart.txt")))
    MAX_SRC_LEN = 512

    def ensure_data_file(local_path: Path, filename: str) -> Path:
        if local_path.exists():
            return local_path
        from huggingface_hub import hf_hub_download
        print(f"[download] {filename} <- {HF_DATASET_REPO}", flush=True)
        return Path(
            hf_hub_download(repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset")
        )

    class GlycoBartTokenizer:
        SPECIALS = {
            "pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>",
            "sep_token": "<sep>", "cls_token": "<cls>", "unk_token": "<unk>",
            "mask_token": "<mask>",
        }
        def __init__(self, vocab_list, max_seq_length=512):
            special_syms = list(self.SPECIALS.values())
            body = [w for w in vocab_list if w not in special_syms]
            full = special_syms + body
            self.vocab = {w: i for i, w in enumerate(full)}
            self.inv = {i: w for w, i in self.vocab.items()}
            self.max_seq_length = max_seq_length
            self.pad_id = self.vocab[self.SPECIALS["pad_token"]]
            self.bos_id = self.vocab[self.SPECIALS["bos_token"]]
            self.eos_id = self.vocab[self.SPECIALS["eos_token"]]
            self.unk_id = self.vocab[self.SPECIALS["unk_token"]]
        def __len__(self): return len(self.vocab)
        def encode(self, text, max_len=None):
            max_len = max_len or self.max_seq_length
            toks = text.split()
            ids = [self.vocab.get(t, self.unk_id) for t in toks]
            ids = [self.bos_id] + ids + [self.eos_id]
            attn = [1] * len(ids)
            if len(ids) < max_len:
                pad_n = max_len - len(ids)
                ids += [self.pad_id] * pad_n
                attn += [0] * pad_n
            else:
                ids = ids[:max_len]; attn = attn[:max_len]
                ids[-1] = self.eos_id
            return ids, attn
        def decode(self, ids, skip_special_tokens=True):
            special_ids = {self.vocab[v] for v in self.SPECIALS.values()}
            toks = []
            for i in ids:
                if isinstance(i, torch.Tensor): i = i.item()
                if i in special_ids and skip_special_tokens: continue
                toks.append(self.inv.get(i, ""))
            return " ".join(toks).strip()

    def _convert_to_antenna(text):
        words = text.split()
        if not words: return ""
        def is_linkage(word):
            return (word[0] in ["a", "b", "."] or word[0].isdigit() or
                    word[-1] in ["a", "b", "."] or word[-1].isdigit())
        reformatted = ["(" + w + ")" if is_linkage(w) else w for w in words]
        out = reformatted[0]
        for i in range(1, len(reformatted)):
            prev_w, cur_w = reformatted[i - 1], reformatted[i]
            if not prev_w.endswith(")") and not cur_w.startswith("("):
                out += ". "
            out += cur_w
        out += "."
        return out
    def _antennae_to_paths(antennae):
        sentences = antennae.replace(")", ") ")
        sentences = sentences[:-1]
        return [s.split() for s in sentences.split(". ") if s]
    def _build_glycan_tree(paths):
        def add_path(tree, path):
            if not path: return
            if path[0] not in tree: tree[path[0]] = {}
            add_path(tree[path[0]], path[1:])
        tree = {}
        for path in sorted(paths, key=len, reverse=True):
            add_path(tree, list(reversed(path)))
        return tree
    def _dict_to_iupac(d):
        if not isinstance(d, dict) or not d: return ""
        parts = [_dict_to_iupac(v) + k for k, v in d.items()]
        if len(parts) > 1:
            return parts[0] + "[" + "][".join(parts[1:]) + "]"
        return "".join(parts)
    def to_iupac(antenna_text):
        if not antenna_text: return ""
        try:
            return _dict_to_iupac(_build_glycan_tree(
                _antennae_to_paths(_convert_to_antenna(antenna_text))))
        except Exception:
            return antenna_text

    try:
        from glycowork.motif.graph import compare_glycans
        _HAS_GLYCOWORK = True
    except ImportError:
        _HAS_GLYCOWORK = False
    def _normalize_linkages(s):
        if not s: return s
        return (s.replace("a1-.", "a1-?").replace("b1-.", "b1-?")
                 .replace("a2-.", "a2-?").replace("b2-.", "b2-?")
                 .replace(".1-3", "?1-3").replace(".1-4", "?1-4").replace(".1-6", "?1-6")
                 .replace(".1-?", "?1-?").replace(".1-.", "?1-?")
                 .replace("(.", "(?"))
    def struct_equal(pred, truth):
        if not pred or not truth: return False
        pred_n = _normalize_linkages(pred); truth_n = _normalize_linkages(truth)
        if pred_n == truth_n: return True
        if not _HAS_GLYCOWORK: return False
        try:
            return bool(compare_glycans(pred_n, truth_n))
        except Exception:
            return False
    print("[setup] Using inlined helpers (sibling train_glycobart.py not found)",
          flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

BEAMS = tuple(int(b) for b in os.environ.get("BEAMS", "1,2,4,8,16,32").split(","))
EVAL_MAX_LEN = int(os.environ.get("EVAL_MAX_LEN", 200))
WALLCLOCK_BATCH_TARGET = int(os.environ.get("WALLCLOCK_BATCH_TARGET", 256))
WALLCLOCK_WARMUP = int(os.environ.get("WALLCLOCK_WARMUP", 5))

# ---- Optional eval-only hooks (set by callers running structure-generalization
# experiments; no-op by default) --------------------------------------------
EVAL_SUBSET = os.environ.get("EVAL_SUBSET", "all").strip().lower()
if EVAL_SUBSET not in ("all", "seen", "unseen"):
    raise ValueError(f"EVAL_SUBSET must be all/seen/unseen, got {EVAL_SUBSET!r}")
EVAL_USE_GRAPH_EQUALITY = os.environ.get("EVAL_USE_GRAPH_EQUALITY", "0") == "1"
EVAL_OUTPUT = os.environ.get("EVAL_OUTPUT")  # override output path
# y_train (labels only, ~2 MB) is used to compute the seen/unseen mask over
# the test set. Defaults to the same preprocessed data repo the DFM eval uses so
# the mask definition is identical across DFM and GlycoBART evals.
Y_TRAIN_HF_REPO = os.environ.get("Y_TRAIN_HF_REPO", HF_DATASET_REPO)
Y_TRAIN_PATH_IN_MOUNT = Path("/data/y_train.pkl")
# Fetch best_model.pt from HF model repo when no local checkpoint exists.
HF_CHECKPOINT_REPO = os.environ.get(
    "HF_CHECKPOINT_REPO",
    os.environ.get("HF_OUTPUT_REPO", "rlacombe/glycobart-207m-repro"),
)

def find_max_batch(model, tokenizer, src_ids_pool, src_mask_pool, beam):
    """Binary-search the largest batch that fits at this beam, capped at TARGET."""
    lo, hi = 1, min(WALLCLOCK_BATCH_TARGET, len(src_ids_pool))
    last_ok = 1
    while lo <= hi:
        mid = (lo + hi + 1) // 2
        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                    dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    _ = model.generate(
                        src_ids_pool[:mid].to(DEVICE),
                        attention_mask=src_mask_pool[:mid].to(DEVICE),
                        num_beams=beam, num_return_sequences=1,
                        max_length=EVAL_MAX_LEN, early_stopping=True,
                        pad_token_id=tokenizer.pad_id,
                        bos_token_id=tokenizer.bos_id,
                        eos_token_id=tokenizer.eos_id,
                    )
            torch.cuda.synchronize() if DEVICE.type == "cuda" else None
            last_ok = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid - 1
    return last_ok

# --- Mass check ported VERBATIM from CABSEL/glycotrans/glycotrans/utils.py ----
# This is mass_check_modified_ppm (line 591 of utils.py). Returns the list of
# candidate m/z values consistent with the observed mass within mass_thresh (ppm).
# Empty list = no consistent adduct/charge state found.

_ADDUCT_C1 = {'M+H': 1.00728, 'M+NH4': 18.03383, 'M+Na': 22.98922, 'M+K': 38.96316,
              'M+2Na-H': 44.97116, 'M+2K-H': 76.91904, 'M+H-H20': -17.00273,
              'M-H20-H': -19.01894, 'M-H': -1.00728, 'M+NA-2H': 20.97467,
              'M+K-2H': 36.94861}
_ADDUCT_C2 = {'M+2H': 1.00728, 'M+H+NH4': 9.52055, 'M+H+Na': 11.99825,
              'M+H+K': 19.98522, 'M+2Na': 22.98922, 'M+2K': 38.96316,
              'M-2H': -1.00728, 'M+Na-3H': 9.98369}
_ADDUCT_C3 = {'M+3H': 1.00728, 'M-3H': -1.00728}
_ADDUCT_C4 = {'M+4H': 1.00728}
_MOD_MASS = {'reduced': 20.02621, '2AA': 139.06333, '2AB': 138.07931,
             'permethylated': 46.04186, 'permethylated and reduced': 47.04969,
             'spacer': 90.07931}

def _glycan_neutral_mass(iupac):
    """glycowork.glycan_to_mass(glycan, sample_prep='underivatized'). Returns None on failure."""
    if not iupac:
        return None
    try:
        from glycowork.motif.tokenization import glycan_to_mass
        return float(glycan_to_mass(iupac, sample_prep='underivatized'))
    except Exception:
        return None

def _mass_check_modified_ppm(observed_mass, glycan_iupac, mode='negative',
                             modification='underivatized',
                             double_thresh=900, triple_thresh=1500, quadruple_thresh=3500,
                             mass_thresh=10):
    """Direct port of CABSEL's mass_check_modified_ppm. Returns list of consistent m/z."""
    mz = _glycan_neutral_mass(glycan_iupac)
    if mz is None:
        return []
    if modification != 'underivatized':
        mz = mz - 18.01056  # remove H2O
        mz += _MOD_MASS.get(modification, 0)
    thresholds = [double_thresh, triple_thresh, quadruple_thresh]
    # Charge 1
    adducts1 = (['M+H', 'M+NH4', 'M+Na', 'M+K', 'M+2Na-H', 'M+2K-H', 'M+H-H20']
                if mode == 'positive' else ['M-H'])
    if mode == 'positive':
        og_list = ([mz] + [mz + _ADDUCT_C1.get(a, 999) for a in adducts1]
                   + [(2 * mz + 22.98922), (2 * mz + 1.00728)])
    else:
        og_list = [mz] + [mz + _ADDUCT_C1.get(a, 999) for a in adducts1]
    # Charge 2/3/4 lists
    adducts2 = (['M+2H', 'M+H+NH4', 'M+H+Na', 'M+H+K', 'M+2Na', 'M+2K']
                if mode == 'positive' else ['M-2H'])
    am2 = [_ADDUCT_C2.get(a, 999) for a in adducts2]
    adducts3 = ['M+3H'] if mode == 'positive' else ['M-3H']
    am3 = [_ADDUCT_C3.get(a, 999) for a in adducts3]
    adducts4 = ['M+4H'] if mode == 'positive' else []
    am4 = [_ADDUCT_C4.get(a, 999) for a in adducts4]
    am = [am2, am3, am4]
    mz_list = og_list + [
        (m / z + adduct_adjust)
        for z, threshold, alist in zip([2, 3, 4], thresholds, am)
        for adduct_adjust in alist
        for m in og_list
    ]
    return [m for m in mz_list
            if (mz > 0 and abs(observed_mass - m) * 1e6 / max(mz, 1.0) < mass_thresh)]

def _mass_aware_select(preds_iupac, spectrum_mass, mode, modification, mass_thresh=10):
    """Pick the top-K candidate that's mass-consistent with the observed spectrum.

    Strategy:
      1. For each candidate, run mass_check_modified_ppm (CABSEL recipe).
      2. Prefer candidates with at least one consistent m/z (non-empty result).
      3. Among consistent candidates: pick the one with smallest min |observed - candidate_mz|
         (i.e., tightest mass match).
      4. If no candidate is consistent: fall back to the highest-score top-1.
      5. Tie-break: keep model rank (preds_iupac[0] is the highest-score candidate).
    """
    if not preds_iupac:
        return "", 0, float("inf")
    if spectrum_mass is None:
        return preds_iupac[0], 0, float("inf")
    consistent = []  # (idx, min_err, pred)
    for j, pred in enumerate(preds_iupac):
        candidates_mz = _mass_check_modified_ppm(spectrum_mass, pred, mode=mode,
                                                 modification=modification,
                                                 mass_thresh=mass_thresh)
        if candidates_mz:
            min_err = min(abs(spectrum_mass - m) for m in candidates_mz)
            consistent.append((j, min_err, pred))
    if not consistent:
        # No candidate is mass-consistent at the threshold; fall back to top-1.
        return preds_iupac[0], 0, float("inf")
    # Sort by (min_err, j) — tightest mass first, then by model rank.
    consistent.sort(key=lambda x: (x[1], x[0]))
    j, err, pred = consistent[0]
    return pred, j, err

@torch.no_grad()
def run_full_accuracy(model, tokenizer, ms_list, target_iupac, target_antenna,
                      reducing_mass, modes, modifications, *,
                      beam, batch_size, top_k=5):
    """Compute five metrics per spectrum:
        iupac_top1            : compare_glycans(to_iupac(top-1 prediction), truth_iupac)
        iupac_topk            : any of top-k predictions struct-matches truth_iupac
        antenna_top1          : exact-string match on antenna form (top-1 only)
        antenna_topk          : exact-string match anywhere in top-k
        iupac_top1_mass_aware : among top-k, pick the candidate whose computed mass best matches
                                the spectrum's reducing_mass; score that one via compare_glycans
    The k value used for top-K is min(top_k, beam).
    """
    n = len(ms_list)
    k = min(top_k, beam)
    iupac_top1, iupac_topk = 0, 0
    antenna_top1, antenna_topk = 0, 0
    iupac_top1_mass_aware = 0
    out_lengths = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        encoded = [tokenizer.encode(t, MAX_SRC_LEN) for t in ms_list[s:e]]
        ids = torch.tensor([x[0] for x in encoded], device=DEVICE)
        mask = torch.tensor([x[1] for x in encoded], device=DEVICE)
        with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            out_ids = model.generate(
                ids, attention_mask=mask,
                num_beams=beam, num_return_sequences=k,
                max_length=EVAL_MAX_LEN, early_stopping=True,
                pad_token_id=tokenizer.pad_id,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
            )
        bs_actual = e - s
        for i in range(bs_actual):
            preds_iupac, preds_antenna = [], []
            for j in range(k):
                row = out_ids[i * k + j]
                decoded_ids = row.tolist()
                if j == 0:
                    gen_len = len(decoded_ids)
                    for pos in range(1, len(decoded_ids)):
                        if decoded_ids[pos] == tokenizer.eos_id:
                            gen_len = pos
                            break
                    out_lengths.append(max(1, gen_len))
                pred_antenna = tokenizer.decode(decoded_ids, skip_special_tokens=True)
                preds_antenna.append(pred_antenna)
                preds_iupac.append(to_iupac(pred_antenna))
            truth_iupac = target_iupac[s + i]
            truth_antenna = target_antenna[s + i] if target_antenna is not None else ""
            spectrum_mass = float(reducing_mass[s + i]) if reducing_mass is not None else None
            # IUPAC top-1
            if struct_equal(preds_iupac[0], truth_iupac):
                iupac_top1 += 1
            # IUPAC top-k (any match)
            if any(struct_equal(p, truth_iupac) for p in preds_iupac):
                iupac_topk += 1
            # Antenna top-1
            if preds_antenna[0].strip() == truth_antenna.strip():
                antenna_top1 += 1
            # Antenna top-k (any exact match)
            if any(p.strip() == truth_antenna.strip() for p in preds_antenna):
                antenna_topk += 1
            # Mass-aware top-1: select among top-K via CABSEL's mass_check_modified_ppm
            if spectrum_mass is not None and k > 1:
                spec_mode = modes[s + i] if modes is not None else "negative"
                spec_mod = modifications[s + i] if modifications is not None else "underivatized"
                pred_ma, _, _ = _mass_aware_select(preds_iupac, spectrum_mass, spec_mode, spec_mod)
                if struct_equal(pred_ma, truth_iupac):
                    iupac_top1_mass_aware += 1
            elif spectrum_mass is not None:
                # k=1: mass-aware == top-1
                if struct_equal(preds_iupac[0], truth_iupac):
                    iupac_top1_mass_aware += 1
        if (s // batch_size) % 50 == 0:
            print(f"    [acc] beam={beam} k={k} progress {s}/{n}", flush=True)
    return {
        "iupac_top1": iupac_top1 / n,
        "iupac_topk": iupac_topk / n,
        "antenna_top1": antenna_top1 / n,
        "antenna_topk": antenna_topk / n,
        "iupac_top1_mass_aware": iupac_top1_mass_aware / n,
        "k": k,
        "mean_out_len": sum(out_lengths) / max(len(out_lengths), 1),
    }

def time_wallclock(model, tokenizer, ids_pool, mask_pool, *, beam, batch_size, n_trials=3):
    """Batched wallclock per spectrum. Warmup, then n_trials timed runs.
    Returns mean and std (ms/spec)."""
    with torch.no_grad():
        for _ in range(WALLCLOCK_WARMUP):
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                _ = model.generate(
                    ids_pool[:batch_size].to(DEVICE),
                    attention_mask=mask_pool[:batch_size].to(DEVICE),
                    num_beams=beam, num_return_sequences=1,
                    max_length=EVAL_MAX_LEN, early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
    times = []
    for _ in range(n_trials):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                _ = model.generate(
                    ids_pool[:batch_size].to(DEVICE),
                    attention_mask=mask_pool[:batch_size].to(DEVICE),
                    num_beams=beam, num_return_sequences=1,
                    max_length=EVAL_MAX_LEN, early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) / batch_size * 1000)  # ms/spec
    import statistics
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)

def time_wallclock_unbatched(model, tokenizer, ids_pool, mask_pool, *, beam, n_spectra=100):
    """Unbatched wallclock per spectrum: time single-spectrum forward over n_spectra
    with 5-pass warmup. Reports mean and std. Reflects streaming latency (the
    actual per-spectrum cost when MS instruments produce one spectrum at a time)."""
    n = min(n_spectra, len(ids_pool))
    # Warmup
    with torch.no_grad():
        for i in range(min(WALLCLOCK_WARMUP, n)):
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                _ = model.generate(
                    ids_pool[i:i + 1].to(DEVICE),
                    attention_mask=mask_pool[i:i + 1].to(DEVICE),
                    num_beams=beam, num_return_sequences=1,
                    max_length=EVAL_MAX_LEN, early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
    times = []
    for i in range(n):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                _ = model.generate(
                    ids_pool[i:i + 1].to(DEVICE),
                    attention_mask=mask_pool[i:i + 1].to(DEVICE),
                    num_beams=beam, num_return_sequences=1,
                    max_length=EVAL_MAX_LEN, early_stopping=True,
                    pad_token_id=tokenizer.pad_id,
                    bos_token_id=tokenizer.bos_id,
                    eos_token_id=tokenizer.eos_id,
                )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)  # ms
    import statistics
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUT_DIR / "best_model.pt"))
    ap.add_argument("--output", default=str(OUT_DIR / "pareto.json"))
    ap.add_argument("--from_hub", default=None,
                    help="If set, load BartForConditionalGeneration from this HF hub id "
                         "(e.g. CABSEL/glycobart). Use ONLY for wallclock+NFE benchmarking; "
                         "accuracy on the leak-free test set would be contaminated.")
    args = ap.parse_args()

    sent_path = ensure_data_file(SENTENCES_PKL, "sentences_benchmark.pkl")
    print(f"[load] {sent_path}", flush=True)
    with open(sent_path, "rb") as f:
        d = pickle.load(f)
    test_ms = d["test_ms"]; test_iupac = d["test_glycan_iupac"]
    test_antenna = d.get("test_glycan", [""] * len(test_ms))
    print(f"[load] test set size: {len(test_ms)}", flush=True)

    # Load spectrum metadata for mass-aware metric: reducing_mass + mode + modification
    import numpy as np
    test_reducing_mass = None
    test_modes = None
    test_modifications = None
    meta_path = DATA_DIR / "spectrum_metadata.npz"
    rm_path = DATA_DIR / "reducing_mass.npz"
    if meta_path.exists():
        meta = np.load(meta_path, allow_pickle=True)
        if "test_reducing_mass" in meta.files and len(meta["test_reducing_mass"]) == len(test_ms):
            test_reducing_mass = meta["test_reducing_mass"]
            test_modes = meta["test_mode"]
            test_modifications = meta["test_modification"]
            print(f"[load] spectrum_metadata.npz: {len(test_reducing_mass)} test rows, "
                  f"mass range [{test_reducing_mass.min():.1f}, {test_reducing_mass.max():.1f}], "
                  f"modes={set(test_modes.tolist()[:1000])}, "
                  f"mods={set(test_modifications.tolist()[:1000])}", flush=True)
        else:
            print(f"[load] spectrum_metadata.npz exists but size mismatch; mass-aware degraded", flush=True)
    elif rm_path.exists():
        rm = np.load(rm_path)
        if "test" in rm.files and len(rm["test"]) == len(test_ms):
            test_reducing_mass = rm["test"]
            print(f"[load] reducing_mass.npz only (no mode/mod info); mass-aware uses defaults", flush=True)
    else:
        print(f"[load] no spectrum metadata; mass-aware metric will degenerate to top-1", flush=True)

    # Optional: fetch checkpoint from an HF model repo (eval-only convenience,
    # mirrors train.py PARETO_ONLY). Fires when no local default checkpoint is
    # present on disk and user hasn't overridden --checkpoint or --from_hub.
    default_ckpt_path = str(OUT_DIR / "best_model.pt")
    if (
        HF_CHECKPOINT_REPO
        and args.checkpoint == default_ckpt_path
        and not Path(default_ckpt_path).exists()
        and not args.from_hub
    ):
        from huggingface_hub import hf_hub_download
        print(f"[hub-fetch] pulling best_model.pt from {HF_CHECKPOINT_REPO}", flush=True)
        p = hf_hub_download(
            repo_id=HF_CHECKPOINT_REPO, filename="best_model.pt",
            repo_type="model", cache_dir="/tmp/hf-ckpt",
        )
        args.checkpoint = p
        print(f"[hub-fetch] using {args.checkpoint}", flush=True)

    if args.from_hub:
        print(f"[load] from HF hub: {args.from_hub} (CONTAMINATED for accuracy)", flush=True)
        model = BartForConditionalGeneration.from_pretrained(args.from_hub).to(DEVICE)
        vocab_path = ensure_data_file(VOCAB_TXT, "vocab_glycobart.txt")
        with open(vocab_path) as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        tokenizer = GlycoBartTokenizer(vocab_list, max_seq_length=MAX_SRC_LEN)
        report_status = "WALLCLOCK_NFE_ONLY_FROM_PUBLIC_CKPT"
    else:
        print(f"[load] checkpoint: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        config = BartConfig(**ckpt["config"])
        model = BartForConditionalGeneration(config).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        tokenizer = GlycoBartTokenizer(ckpt["vocab_list"], max_seq_length=MAX_SRC_LEN)
        report_status = "FROM_SCRATCH_RETRAIN"
    model.eval()

    # ---- Optional eval-only subset filtering (EVAL_SUBSET=seen|unseen) ----
    # No-op when EVAL_SUBSET=all. When set, filters test_ms / test_iupac
    # / test_antenna (and metadata arrays) so accuracy is computed only on
    # the seen or unseen subset. Uses the same y_train + mask definition
    # as the DFM eval (train.py).
    if EVAL_SUBSET in ("seen", "unseen"):
        # Prefer the /data mount if it contains y_train.pkl; else HF download.
        if Y_TRAIN_PATH_IN_MOUNT.exists():
            print(f"[subset] loading y_train from mount: {Y_TRAIN_PATH_IN_MOUNT}", flush=True)
            with open(Y_TRAIN_PATH_IN_MOUNT, "rb") as f:
                y_train = pickle.load(f)
        else:
            from huggingface_hub import hf_hub_download
            print(f"[subset] downloading y_train.pkl from {Y_TRAIN_HF_REPO}", flush=True)
            _p = hf_hub_download(repo_id=Y_TRAIN_HF_REPO, filename="y_train.pkl",
                                 repo_type="dataset", cache_dir="/tmp/hf-data")
            with open(_p, "rb") as f:
                y_train = pickle.load(f)
        y_train_strs = [y if isinstance(y, str) else str(y) for y in y_train]
        train_set = set(y_train_strs)
        is_seen = [y in train_set for y in test_iupac]
        if EVAL_USE_GRAPH_EQUALITY:
            from glycowork.motif.graph import compare_glycans
            train_list = sorted(train_set)
            resolved = {}
            for i, y_te in enumerate(test_iupac):
                if is_seen[i]:
                    continue
                if y_te in resolved:
                    is_seen[i] = resolved[y_te]
                    continue
                found = False
                for y_tr in train_list:
                    try:
                        if compare_glycans(y_te, y_tr):
                            found = True
                            break
                    except Exception:
                        continue
                resolved[y_te] = found
                is_seen[i] = found
        keep = is_seen if EVAL_SUBSET == "seen" else [not s for s in is_seen]
        n_kept = sum(keep)
        n_total = len(test_ms)
        print(f"[subset] EVAL_SUBSET={EVAL_SUBSET} graph={EVAL_USE_GRAPH_EQUALITY}: "
              f"{n_kept}/{n_total} kept", flush=True)
        test_ms = [x for x, k in zip(test_ms, keep) if k]
        test_iupac = [x for x, k in zip(test_iupac, keep) if k]
        test_antenna = [x for x, k in zip(test_antenna, keep) if k]
        if test_reducing_mass is not None:
            test_reducing_mass = np.array([x for x, k in zip(test_reducing_mass, keep) if k])
            test_modes = np.array([x for x, k in zip(test_modes, keep) if k])
            test_modifications = np.array([x for x, k in zip(test_modifications, keep) if k])
        print(f"[subset] filtered test set size: {len(test_ms)}", flush=True)

    # ---- Optional: one-time preprocessing benchmark (raw peak_d -> sentence).
    # BENCH_E2E=1 or BENCH_PREPROCESS=1 turns this on. Loads raw peak_d dicts
    # from /data/test_peak_d.pkl and times the CABSEL preprocessing path
    # (parse -> normalise -> filter -> sort by intensity -> mz-token digitize
    # -> sentence-string join). Result is beam-independent, copied to every
    # pareto row. Metadata uses placeholders since sentence assembly is O(1).
    preprocess_ms = None
    preprocess_ms_std = None
    preprocess_n = 0
    if os.environ.get("BENCH_E2E", "0") in ("1", "true", "yes") or \
       os.environ.get("BENCH_PREPROCESS", "0") in ("1", "true", "yes"):
        try:
            import statistics as _stats
            import ast as _ast
            _pkd_path = os.environ.get("BENCH_PEAK_D", "/data/test_peak_d.pkl")
            print(f"[bench-preprocess] loading raw peak_d from {_pkd_path}", flush=True)
            with open(_pkd_path, "rb") as _f:
                _peak_d_list = pickle.load(_f)
            print(f"[bench-preprocess] loaded {len(_peak_d_list):,} peak dicts", flush=True)
            _MIN_MZ, _MAX_MZ, _SIZE_MZ = 39.714, 3000.0, 0.3
            _MIN_RT, _MAX_RT, _SIZE_RT = 0.0, 1.0, 0.01
            _THRESH = 0.001
            _MZ_EDGES = np.arange(_MIN_MZ, _MAX_MZ + _SIZE_MZ, _SIZE_MZ)
            _RT_EDGES = np.arange(_MIN_RT, _MAX_RT + _SIZE_RT, _SIZE_RT)
            _META_COLS = ["LC_type", "mode", "ionization", "modification",
                          "trap", "fragmentation", "glycan_type"]
            _placeholder_meta = dict(LC_type="C18", mode="negative",
                                     ionization="ESI", modification="reduced",
                                     trap="orbitrap", fragmentation="CID",
                                     glycan_type="Nglycan",
                                     RT_normalised=0.35, reducing_mass=800.0)
            def _safe_peak_parse(v):
                if isinstance(v, dict):
                    return {float(k): float(x) for k, x in v.items()}
                if isinstance(v, str):
                    s = v.strip()
                    if not s: return None
                    try:
                        p = _ast.literal_eval(s)
                    except (SyntaxError, ValueError):
                        return None
                    if isinstance(p, dict):
                        return {float(k): float(x) for k, x in p.items()}
                return None
            def _normalise(spec):
                t = float(sum(spec.values()))
                if t <= 0: return None
                return {mz: v/t for mz, v in spec.items()}
            def _rt_tok(rtv):
                if isinstance(rtv, (int, float)) and not np.isnan(rtv):
                    return f"rt{int(np.digitize([rtv], _RT_EDGES)[0])}"
                return str(rtv)
            def _preprocess(pd_raw, meta):
                sp = _safe_peak_parse(pd_raw)
                if sp is None or len(sp) == 0: return None
                sp = _normalise(sp)
                if sp is None: return None
                items = sorted(sp.items(), key=lambda kv: kv[1], reverse=True)
                items = [(mz, i) for mz, i in items if i > _THRESH]
                if not items: return ""
                toks = [f"mz{i_}" for i_ in np.digitize(
                    [mz for mz, _ in items], _MZ_EDGES)]
                m = " ".join(str(meta[c]) for c in _META_COLS)
                brt = _rt_tok(meta["RT_normalised"])
                bmass = f"mz{int(np.digitize(meta['reducing_mass'], _MZ_EDGES))}"
                return f"{m} {brt} {bmass} {' '.join(toks)}"
            _n_pp = min(int(os.environ.get("BENCH_PREPROCESS_N", "200")),
                        len(_peak_d_list))
            _warmup = 10
            _rng = np.random.default_rng(0)
            _idxs = _rng.choice(len(_peak_d_list),
                                size=min(_n_pp + _warmup, len(_peak_d_list)),
                                replace=False)
            for i in _idxs[:_warmup]:
                _ = _preprocess(_peak_d_list[i], _placeholder_meta)
            _pp_times = []
            for i in _idxs[_warmup:_warmup + _n_pp]:
                _t0 = time.perf_counter()
                _ = _preprocess(_peak_d_list[i], _placeholder_meta)
                _pp_times.append((time.perf_counter() - _t0) * 1000.0)
            preprocess_ms = float(_stats.mean(_pp_times))
            preprocess_ms_std = float(_stats.stdev(_pp_times)) if len(_pp_times) > 1 else 0.0
            preprocess_n = len(_pp_times)
            print(f"[bench-preprocess] GlycoBART preprocess ms/spec = "
                  f"{preprocess_ms:.3f}±{preprocess_ms_std:.3f} (n={preprocess_n})",
                  flush=True)
        except Exception as _e:
            print(f"[bench-preprocess] SKIPPED (error: {_e})", flush=True)

    # Pre-tokenize a wallclock pool (first WALLCLOCK_BATCH_TARGET test samples)
    pool_size = WALLCLOCK_BATCH_TARGET
    pool_encoded = [tokenizer.encode(t, MAX_SRC_LEN) for t in test_ms[:pool_size]]
    ids_pool = torch.tensor([x[0] for x in pool_encoded])
    mask_pool = torch.tensor([x[1] for x in pool_encoded])

    pareto = []
    for beam in BEAMS:
        print(f"\n=== beam={beam} ===", flush=True)
        bs = find_max_batch(model, tokenizer, ids_pool, mask_pool, beam)
        print(f"[batch] beam={beam} max-batch-that-fits = {bs}", flush=True)

        ms_per_spec, ms_per_spec_std = time_wallclock(model, tokenizer, ids_pool, mask_pool,
                                                       beam=beam, batch_size=bs, n_trials=3)
        print(f"[wall] beam={beam} batched ms/spec = {ms_per_spec:.2f} ± {ms_per_spec_std:.2f} (batch={bs}, 3 trials)", flush=True)
        # Unbatched: per-spectrum streaming latency
        ms_per_spec_unbatched, ms_per_spec_unbatched_std = time_wallclock_unbatched(
            model, tokenizer, ids_pool, mask_pool, beam=beam, n_spectra=100)
        print(f"[wall] beam={beam} unbatched ms/spec = {ms_per_spec_unbatched:.2f} ± {ms_per_spec_unbatched_std:.2f} (n=100)", flush=True)

        _skip_acc = os.environ.get("SKIP_ACCURACY", "0") in ("1", "true", "yes")
        if not args.from_hub and not _skip_acc:
            top_k = min(5, beam) if beam > 1 else 1
            metrics = run_full_accuracy(model, tokenizer, test_ms, test_iupac, test_antenna,
                                        test_reducing_mass, test_modes, test_modifications,
                                        beam=beam, batch_size=bs, top_k=top_k)
            mean_len = metrics["mean_out_len"]
        else:
            metrics = {"iupac_top1": float("nan"), "iupac_topk": float("nan"),
                       "antenna_top1": float("nan"), "antenna_topk": float("nan"),
                       "iupac_top1_mass_aware": float("nan"),
                       "k": 0, "mean_out_len": float("nan")}
            mean_len = float("nan")
        nfe = mean_len * beam if not math.isnan(mean_len) else float("nan")
        print(f"[acc] beam={beam} k={metrics['k']} "
              f"iupac_top1={metrics['iupac_top1']:.4f} iupac_topk={metrics['iupac_topk']:.4f} "
              f"iupac_top1_mass={metrics['iupac_top1_mass_aware']:.4f} "
              f"antenna_top1={metrics['antenna_top1']:.4f} antenna_topk={metrics['antenna_topk']:.4f} "
              f"mean_out_len={mean_len:.1f}  nfe={nfe:.1f}", flush=True)

        def _safe(x):
            return None if (isinstance(x, float) and math.isnan(x)) else x

        _row = {
            "sweep_value": beam,
            "metric_name": "beam",
            "accuracy": _safe(metrics["iupac_top1"]),
            "iupac_top1": _safe(metrics["iupac_top1"]),
            "iupac_topk": _safe(metrics["iupac_topk"]),
            "iupac_top1_mass_aware": _safe(metrics["iupac_top1_mass_aware"]),
            "antenna_top1": _safe(metrics["antenna_top1"]),
            "antenna_topk": _safe(metrics["antenna_topk"]),
            "k": metrics["k"],
            "nfe_per_spectrum": _safe(nfe),
            "mean_out_len": _safe(mean_len),
            "wallclock_ms_per_spectrum": ms_per_spec,
            "wallclock_ms_per_spectrum_std": ms_per_spec_std,
            "wallclock_ms_per_spectrum_unbatched": ms_per_spec_unbatched,
            "wallclock_ms_per_spectrum_unbatched_std": ms_per_spec_unbatched_std,
            "batch_size_eval": bs,
        }
        if preprocess_ms is not None:
            _row["preprocess_ms_per_spectrum"] = float(preprocess_ms)
            _row["preprocess_ms_per_spectrum_std"] = float(preprocess_ms_std)
            _row["preprocess_n"] = int(preprocess_n)
        pareto.append(_row)
        print(f"METRIC_FINAL beam={beam} k={metrics['k']} "
              f"iupac_top1={metrics['iupac_top1']:.4f} iupac_topk={metrics['iupac_topk']:.4f} "
              f"iupac_top1_mass={metrics['iupac_top1_mass_aware']:.4f} "
              f"antenna_top1={metrics['antenna_top1']:.4f} antenna_topk={metrics['antenna_topk']:.4f} "
              f"nfe={nfe:.2f} ms_per_spec={ms_per_spec:.2f} batch={bs}", flush=True)

    out = {
        "cell": "xp-glycobart-207m",
        "checkpoint": args.checkpoint if not args.from_hub else f"hub:{args.from_hub}",
        "init_seed": int(os.environ.get("INIT_SEED", 0)),
        "hardware": "H200 141GB BF16" if DEVICE.type == "cuda" else f"{DEVICE.type}",
        "status": report_status,
        "test_set_n": len(test_ms),
        "split_metadata": d["split_metadata"],
        "pareto": pareto,
    }
    # Tag output with eval-only provenance if we filtered a subset.
    if EVAL_SUBSET != "all":
        out["eval_subset"] = EVAL_SUBSET
        out["eval_use_graph_equality"] = EVAL_USE_GRAPH_EQUALITY
    # Override output path via EVAL_OUTPUT if set (matches DFM eval convention).
    write_path = EVAL_OUTPUT if EVAL_OUTPUT else args.output
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    with open(write_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {write_path}", flush=True)

    # Push pareto.json to HF model repo for persistence
    HF_OUTPUT_REPO = os.environ.get("HF_OUTPUT_REPO", "rlacombe/glycobart-207m-repro")
    if HF_OUTPUT_REPO:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(HF_OUTPUT_REPO, repo_type="model", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=write_path,
                path_in_repo=Path(write_path).name,
                repo_id=HF_OUTPUT_REPO,
                repo_type="model",
                commit_message="pareto sweep results",
            )
            print(f"[hub] pushed pareto.json -> {HF_OUTPUT_REPO}", flush=True)
        except Exception as e:
            print(f"[hub] pareto push failed (non-fatal): {e}", flush=True)

if __name__ == "__main__":
    main()
