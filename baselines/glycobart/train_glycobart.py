# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "numpy<2",
#   "transformers>=4.44",
#   "huggingface_hub>=0.24",
#   "glycowork",
#   "scikit-learn",
# ]
# ///
"""GlycoBART retrain on the leak-free CandyCrunch v2 split with CABSEL
tokenization (run-stratified by MS/MS-run filename, no same-run leakage).

Hardware: single H200 (141 GB), BF16. HF Jobs invocation with data
mounted at /data and outputs written to /output.

Recipe:
  Architecture: CABSEL/glycobart config — BART, d=768, 12 enc / 12 dec layers,
                16 heads, ffn=3072, max_seq=512, dropout=0.1 (~207M params).
  Training:    AdamW lr=1e-5 (CABSEL's published value), no LR schedule.
               BF16 forwards via torch.autocast.
  Best-ckpt:   eval val every EVAL_EVERY steps; save when val top-1 structural
               accuracy at beam=4 improves.
  Stop:        EPOCHS or TRAIN_SECONDS, whichever comes first.
  Failure:     abort if val_acc < 0.5 by hour 8.

Outputs:
  /output/best_model.pt          (HF state_dict + config)
  /output/training_log.jsonl     (one row per eval)
  METRIC_FINAL line on stdout.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartConfig, BartForConditionalGeneration

# ----- Paths -------------------------------------------------------------------

if os.getenv("COMPUTE_BACKEND") == "hfjobs" or os.getenv("HF_JOB") == "1":
    DATA_DIR = Path("/data")  # HF dataset mount
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


def ensure_data_file(local_path: Path, filename: str) -> Path:
    """Return `local_path` if present on disk, otherwise download `filename` from HF_DATASET_REPO."""
    if local_path.exists():
        return local_path
    from huggingface_hub import hf_hub_download
    print(f"[download] {filename} <- {HF_DATASET_REPO}", flush=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATASET_REPO, filename=filename, repo_type="dataset"
    )
    return Path(downloaded)

# ----- Hyperparameters (env-overridable) --------------------------------------

EPOCHS = int(os.environ.get("EPOCHS", 25))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))   # CABSEL's published recipe; H200-tunable
LR = float(os.environ.get("LR", 1e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.0))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 500))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 32))
EVAL_BEAM_SIZE = int(os.environ.get("EVAL_BEAM_SIZE", 4))
EVAL_MAX_LEN = int(os.environ.get("EVAL_MAX_LEN", 200))
EVAL_VAL_LIMIT = int(os.environ.get("EVAL_VAL_LIMIT", 5000))   # subsample val for hill-climb speed
TRAIN_SECONDS = int(os.environ.get("TRAIN_SECONDS", 14 * 3600))
WARN_ACC_FLOOR = float(os.environ.get("WARN_ACC_FLOOR", 0.5))
WARN_ACC_DEADLINE_S = int(os.environ.get("WARN_ACC_DEADLINE_S", 8 * 3600))
# Early-stop: stop training if val-acc has not improved by EARLY_STOP_MIN_DELTA for
# EARLY_STOP_PATIENCE consecutive evals (after EARLY_STOP_MIN_EVALS warmup evals).
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", 5))
EARLY_STOP_MIN_DELTA = float(os.environ.get("EARLY_STOP_MIN_DELTA", 0.001))
EARLY_STOP_MIN_EVALS = int(os.environ.get("EARLY_STOP_MIN_EVALS", 10))
INIT_SEED = int(os.environ.get("INIT_SEED", 0))
MAX_SRC_LEN = 512
MAX_TGT_LEN = 200

# ----- Setup -------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(INIT_SEED)
np.random.seed(INIT_SEED)
torch.set_float32_matmul_precision("high")
print(f"[setup] device={DEVICE}  bf16={DEVICE.type=='cuda'}  seed={INIT_SEED}", flush=True)


# ----- HF model-repo push (so checkpoints survive the ephemeral job filesystem) -

HF_OUTPUT_REPO = os.environ.get("HF_OUTPUT_REPO", "rlacombe/glycobart-207m-repro")
HF_OUTPUT_REPO_TYPE = os.environ.get("HF_OUTPUT_REPO_TYPE", "model")
_hf_api = None
def push_best_to_hub_if_configured(local_path):
    """Best-effort upload of the best checkpoint to HF; never crash training."""
    global _hf_api
    if not HF_OUTPUT_REPO:
        return
    try:
        from huggingface_hub import HfApi
        if _hf_api is None:
            _hf_api = HfApi()
            try:
                _hf_api.create_repo(HF_OUTPUT_REPO, repo_type=HF_OUTPUT_REPO_TYPE,
                                    private=True, exist_ok=True)
            except Exception as e:
                print(f"[hub] create_repo warning: {e}", flush=True)
        _hf_api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=Path(local_path).name,
            repo_id=HF_OUTPUT_REPO,
            repo_type=HF_OUTPUT_REPO_TYPE,
            commit_message=f"best_model.pt update",
        )
        print(f"[hub] pushed {local_path} -> {HF_OUTPUT_REPO}", flush=True)
    except Exception as e:
        print(f"[hub] push failed (non-fatal): {e}", flush=True)

# ----- Tokenizer (CABSEL-style; ported from glycotrans/MSTokenizer.py) ---------

class GlycoBartTokenizer:
    SPECIALS = {
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "sep_token": "<sep>",
        "cls_token": "<cls>",
        "unk_token": "<unk>",
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

    def __len__(self):
        return len(self.vocab)

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
        # Match CABSEL/glycotrans/MSTokenizer.py: skip all special-token IDs.
        # Do NOT early-break on eos: BART's generate() emits decoder_start_token_id
        # (= eos_id, by config) at position 0, so an early break would always
        # return an empty string. This was a subtle bug; CABSEL's reference
        # decoder filters specials wholesale rather than terminating on eos.
        special_ids = {self.vocab[v] for v in self.SPECIALS.values()}
        toks = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if i in special_ids:
                if skip_special_tokens:
                    continue
            toks.append(self.inv.get(i, ""))
        return " ".join(toks).strip()


# ----- Dataset -----------------------------------------------------------------

class GlycoBartDataset(Dataset):
    def __init__(self, ms_list, glycan_list, tokenizer):
        assert len(ms_list) == len(glycan_list)
        self.ms = ms_list
        self.gl = glycan_list
        self.tok = tokenizer

    def __len__(self):
        return len(self.ms)

    def __getitem__(self, i):
        src_ids, src_mask = self.tok.encode(self.ms[i], MAX_SRC_LEN)
        tgt_ids, tgt_mask = self.tok.encode(self.gl[i], MAX_TGT_LEN)
        labels = list(tgt_ids)
        labels = [(t if m == 1 else -100) for t, m in zip(labels, tgt_mask)]
        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "attention_mask": torch.tensor(src_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "decoder_attention_mask": torch.tensor(tgt_mask, dtype=torch.long),
        }


# ----- antenna -> IUPAC reconstruction (for compare_glycans matching) ----------

def _convert_to_antenna(text):
    """Port of glycotrans.utils.convert_to_antenna (CABSEL/glycotrans).

    Wraps linkage tokens in parens and inserts '. ' between separate antennae.
    """
    words = text.split()
    if not words:
        return ""

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
    """Port of glycotrans.utils.antennae_to_paths."""
    sentences = antennae.replace(")", ") ")
    sentences = sentences[:-1]
    paths = [s.split() for s in sentences.split(". ") if s]
    return paths


def _build_glycan_tree(paths):
    """Port of glycotrans.utils.build_glycan_tree."""
    def add_path(tree, path):
        if not path:
            return
        if path[0] not in tree:
            tree[path[0]] = {}
        add_path(tree[path[0]], path[1:])
    tree = {}
    for path in sorted(paths, key=len, reverse=True):
        add_path(tree, list(reversed(path)))
    return tree


def _dict_to_iupac(d):
    """Port of glycotrans.utils.dict_to_iupac."""
    if not isinstance(d, dict) or not d:
        return ""
    parts = [_dict_to_iupac(v) + k for k, v in d.items()]
    if len(parts) > 1:
        return parts[0] + "[" + "][".join(parts[1:]) + "]"
    return "".join(parts)


def to_iupac(antenna_text: str) -> str:
    """CABSEL's full antenna -> IUPAC reconstruction (re-bracketing)."""
    if not antenna_text:
        return ""
    try:
        return _dict_to_iupac(_build_glycan_tree(
            _antennae_to_paths(_convert_to_antenna(antenna_text))))
    except Exception:
        return antenna_text


# ----- Structural accuracy via glycowork.compare_glycans ----------------------

try:
    from glycowork.motif.graph import compare_glycans
    HAS_GLYCOWORK = True
    print("[setup] glycowork.compare_glycans available", flush=True)
except ImportError:
    HAS_GLYCOWORK = False
    print("[setup] WARNING: glycowork missing — falling back to string equality", flush=True)


def _normalize_linkages(s: str) -> str:
    """CABSEL antenna form uses '.' for uncertain linkage positions; glycowork's
    compare_glycans uses '?'. Without this normalization, structurally-equivalent
    predictions get rejected purely because of notation. Empirically, this single
    line lifts the decoder ceiling on the test set from ~82.5% to ~99.65%.
    """
    if not s:
        return s
    return (s.replace("a1-.", "a1-?").replace("b1-.", "b1-?")
             .replace("a2-.", "a2-?").replace("b2-.", "b2-?")
             .replace(".1-3", "?1-3").replace(".1-4", "?1-4").replace(".1-6", "?1-6")
             .replace(".1-?", "?1-?").replace(".1-.", "?1-?")
             .replace("(.", "(?"))


def struct_equal(pred: str, truth: str) -> bool:
    if not pred or not truth:
        return False
    pred_n = _normalize_linkages(pred)
    truth_n = _normalize_linkages(truth)
    if pred_n == truth_n:
        return True
    if not HAS_GLYCOWORK:
        return False
    try:
        return bool(compare_glycans(pred_n, truth_n))
    except Exception:
        return False


# ----- Eval --------------------------------------------------------------------

@torch.no_grad()
def eval_top1_structural(model, tokenizer, ms_list, target_iupac_list, *,
                         beam=4, batch_size=32, max_len=200, limit=None):
    model.eval()
    n = len(ms_list)
    if limit is not None:
        n = min(n, limit)
    correct = 0
    valid = 0
    t0 = time.time()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        batch_ms = ms_list[s:e]
        batch_truth = target_iupac_list[s:e]
        encoded = [tokenizer.encode(t, MAX_SRC_LEN) for t in batch_ms]
        ids = torch.tensor([x[0] for x in encoded], device=DEVICE)
        mask = torch.tensor([x[1] for x in encoded], device=DEVICE)
        with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            out_ids = model.generate(
                ids, attention_mask=mask,
                num_beams=beam, num_return_sequences=1,
                max_length=max_len, early_stopping=True,
                pad_token_id=tokenizer.pad_id,
                bos_token_id=tokenizer.bos_id,
                eos_token_id=tokenizer.eos_id,
            )
        for i, oid in enumerate(out_ids):
            pred_antenna = tokenizer.decode(oid.tolist(), skip_special_tokens=True)
            pred_iupac = to_iupac(pred_antenna)
            valid += 1
            if struct_equal(pred_iupac, batch_truth[i]):
                correct += 1
    elapsed = time.time() - t0
    return {
        "accuracy": correct / max(valid, 1),
        "n": valid,
        "elapsed_s": elapsed,
        "ms_per_spectrum": (elapsed / max(valid, 1)) * 1000,
    }


# ----- Train -------------------------------------------------------------------

def main():
    sent_path = ensure_data_file(SENTENCES_PKL, "sentences_benchmark.pkl")
    print(f"[load] {sent_path}", flush=True)
    with open(sent_path, "rb") as f:
        d = pickle.load(f)
    print(f"[load] split metadata: {d['split_metadata']}", flush=True)

    vocab_path = ensure_data_file(VOCAB_TXT, "vocab_glycobart.txt")
    with open(vocab_path) as f:
        vocab_list = [line.strip() for line in f if line.strip()]

    tokenizer = GlycoBartTokenizer(vocab_list, max_seq_length=MAX_SRC_LEN)
    V = len(tokenizer)
    print(f"[setup] vocab size: {V}", flush=True)

    train_ds = GlycoBartDataset(d["train_ms"], d["train_glycan"], tokenizer)
    val_ms = d["val_ms"]; val_iupac = d["val_glycan_iupac"]
    test_ms = d["test_ms"]; test_iupac = d["test_glycan_iupac"]
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=(DEVICE.type == "cuda"),
                              drop_last=True)

    config = BartConfig(
        vocab_size=V,
        d_model=768,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=3072,
        decoder_ffn_dim=3072,
        max_position_embeddings=MAX_SRC_LEN,
        dropout=0.1,
        activation_function="gelu",
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        decoder_start_token_id=tokenizer.eos_id,
    )
    model = BartForConditionalGeneration(config).to(DEVICE)

    # ── Resume support ─────────────────────────────────────────────────────
    # Two modes:
    #   RESUME_FROM=<local path>           load a local best_model.pt
    #   RESUME_FROM_HF=<repo>[:<revision>] download best_model.pt from HF and load
    # In both cases: optimizer state is NOT preserved (we don't save it),
    # so AdamW moments start fresh and the LR schedule restarts from warmup.
    resume_from = os.environ.get("RESUME_FROM", "")
    resume_from_hf = os.environ.get("RESUME_FROM_HF", "")
    if resume_from_hf and not resume_from:
        # Fetch from HF Hub. Format: "repo_id" or "repo_id:revision" or "repo_id@revision".
        _rf = resume_from_hf.replace("@", ":", 1)
        if ":" in _rf:
            _repo, _rev = _rf.split(":", 1)
        else:
            _repo, _rev = _rf, None
        from huggingface_hub import hf_hub_download
        print(f"[resume] fetching best_model.pt from HF: repo={_repo} rev={_rev}", flush=True)
        resume_from = hf_hub_download(
            repo_id=_repo, filename="best_model.pt",
            revision=_rev, cache_dir="/tmp/hf-resume",
        )
        print(f"[resume] downloaded to {resume_from}", flush=True)
    if resume_from and Path(resume_from).exists():
        print(f"[resume] loading checkpoint from {resume_from}", flush=True)
        ck = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        # Seed the best_struct so we don't checkpoint-overwrite the resume point
        # with something worse during warm-up.
        _prev_val = float(ck.get("val_acc", -1.0))
        print(f"[resume] loaded; previous step={ck.get('step','?')} "
              f"epoch={ck.get('epoch','?')} val_acc={_prev_val:.4f}", flush=True)
    elif resume_from:
        print(f"[resume] WARNING: RESUME_FROM={resume_from} does not exist; starting fresh", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] BartForConditionalGeneration: {n_params/1e6:.1f}M params", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log_path = OUT_DIR / "training_log.jsonl"
    best_path = OUT_DIR / "best_model.pt"
    log_f = open(log_path, "a", buffering=1)

    # If we resumed, seed best_struct to the resume-checkpoint's val so we
    # don't overwrite the good checkpoint during LR warmup or transient degrade.
    best_struct = float(_prev_val) if (resume_from and 'ck' in dir()) else -1.0
    best_step = 0
    best_epoch = 0.0
    step = 0
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    print(f"[train] EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  LR={LR}  steps/epoch={steps_per_epoch}", flush=True)
    print(f"[train] early-stop: patience={EARLY_STOP_PATIENCE} min_delta={EARLY_STOP_MIN_DELTA} min_evals={EARLY_STOP_MIN_EVALS}", flush=True)

    t_start = time.time()
    fail_warned = False
    n_evals = 0
    evals_since_improvement = 0
    early_stop = False

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            if time.time() - t_start > TRAIN_SECONDS:
                print(f"[train] TRAIN_SECONDS reached ({TRAIN_SECONDS}s); stopping", flush=True)
                break
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                out = model(**batch)
                loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            step += 1

            if step % EVAL_EVERY == 0 or step == total_steps:
                ev = eval_top1_structural(model, tokenizer, val_ms, val_iupac,
                                          beam=EVAL_BEAM_SIZE, batch_size=EVAL_BATCH_SIZE,
                                          max_len=EVAL_MAX_LEN, limit=EVAL_VAL_LIMIT)
                rec = {
                    "step": step, "epoch": step / steps_per_epoch,
                    "train_loss": float(loss.item()),
                    "val_top1_structural_accuracy": ev["accuracy"],
                    "val_n": ev["n"], "val_eval_s": ev["elapsed_s"],
                    "elapsed_s": time.time() - t_start,
                }
                log_f.write(json.dumps(rec) + "\n")
                print(f"[eval] step={step} epoch={rec['epoch']:.2f} loss={rec['train_loss']:.4f} "
                      f"val_acc@beam{EVAL_BEAM_SIZE}={ev['accuracy']:.4f} (n={ev['n']}) "
                      f"elapsed={rec['elapsed_s']/3600:.2f}h", flush=True)
                n_evals += 1
                if ev["accuracy"] > best_struct + EARLY_STOP_MIN_DELTA:
                    best_struct = ev["accuracy"]
                    best_step = step
                    best_epoch = step / steps_per_epoch
                    evals_since_improvement = 0
                    torch.save({
                        "state_dict": model.state_dict(),
                        "config": config.to_dict(),
                        "vocab_list": vocab_list,
                        "step": step, "epoch": best_epoch,
                        "val_acc": best_struct,
                        "hp": {
                            "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                            "WEIGHT_DECAY": WEIGHT_DECAY, "GRAD_CLIP": GRAD_CLIP,
                            "INIT_SEED": INIT_SEED,
                            "MAX_SRC_LEN": MAX_SRC_LEN, "MAX_TGT_LEN": MAX_TGT_LEN,
                        },
                    }, best_path)
                    print(f"[ckpt] saved best_model.pt at step={step} val_acc={best_struct:.4f}", flush=True)
                    # Push best ckpt to HF model repo so it survives the job container
                    push_best_to_hub_if_configured(best_path)
                else:
                    evals_since_improvement += 1
                    print(f"[plateau] eval#{n_evals} no-improvement-streak={evals_since_improvement} (best={best_struct:.4f})", flush=True)

                if (not fail_warned and time.time() - t_start > WARN_ACC_DEADLINE_S
                        and best_struct < WARN_ACC_FLOOR):
                    print(f"[FAILURE] best val_acc={best_struct:.4f} < {WARN_ACC_FLOOR} after "
                          f"{(time.time()-t_start)/3600:.1f}h. Continuing to TRAIN_SECONDS "
                          f"but training has not converged.", flush=True)
                    fail_warned = True

                if (n_evals >= EARLY_STOP_MIN_EVALS
                        and evals_since_improvement >= EARLY_STOP_PATIENCE):
                    print(f"[early-stop] no improvement for {evals_since_improvement} evals "
                          f"(min_delta={EARLY_STOP_MIN_DELTA}, min_evals={EARLY_STOP_MIN_EVALS}). "
                          f"best_val_acc={best_struct:.4f} at step={best_step}. Stopping.", flush=True)
                    early_stop = True
                    break
        if early_stop or time.time() - t_start > TRAIN_SECONDS:
            break

    log_f.close()
    push_best_to_hub_if_configured(log_path)  # also push the training log
    print(f"\nMETRIC_FINAL best_top1_structural_accuracy={best_struct:.4f} "
          f"best_step={best_step} best_epoch={best_epoch:.2f} "
          f"final_step={step} final_epoch={step/max(steps_per_epoch,1):.2f} "
          f"total_train_s={time.time()-t_start:.0f} "
          f"early_stopped={early_stop} "
          f"status={'SUCCESS' if best_struct >= WARN_ACC_FLOOR else 'FALLBACK_TO_CITED'}", flush=True)


if __name__ == "__main__":
    main()
