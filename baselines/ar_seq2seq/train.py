# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "numpy<2",
#   "huggingface_hub>=0.24",
#   "glycowork",
# ]
# ///
"""
Autoregressive seq2seq baseline (46M) with architecture matched to
GlycoGen DFM (50M): same encoder, same decoder shape (d=512, 4L enc,
8L dec, 8 heads). Deltas vs DFM:
  1. Decoder self-attention is causal (upper-tri -inf tgt_mask).
  2. No time embedding.
  3. Teacher-forced next-token CE loss.

Trains on the leak-free CandyCrunch v2 split, selects on val with
beam=4 decode, then sweeps beam in {1,2,4,8,16,32} on the test set.

The test set is only touched in the final Pareto sweep, never during
training or model selection.
"""
import os, time, math, pickle, re, json, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16

# ----- output paths --------------------------------------------------------
ON_HF_JOB = os.getenv("COMPUTE_BACKEND") == "hfjobs" or Path("/output").is_dir()
if ON_HF_JOB:
    OUT_DIR = Path("/output")
else:
    OUT_DIR = Path("./output/ar-seq2seq")
CKPT_DIR = OUT_DIR / "checkpoints"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
PARETO_PATH = OUT_DIR / "pareto.json"

# ----- data download (leak-free CC2 split, preprocessed HF dataset) --------
# The preprocessed HF dataset ships numerical (mz, mz_r, prec, ...) pkls
# aligned to row identities in row_manifest.csv: 378,880 train / 42,097 val
# / 76,774 test. DFM, MDLM, AR, and GlycoBART all train on these identical rows.
from huggingface_hub import hf_hub_download

HF_REPO = os.environ.get(
    "HF_DATASET_REPO",
    os.environ.get("HF_REPO", "rlacombe/candycrunch-v2-benchmark"),
)

# --- HF Hub push for checkpoints + pareto (mirrors DFM's push_to_hub_if_configured) ---
HF_OUTPUT_REPO = os.environ.get("HF_OUTPUT_REPO", "rlacombe/glycogen-ar-50m")
HF_OUTPUT_REPO_TYPE = os.environ.get("HF_OUTPUT_REPO_TYPE", "model")
_hf_api = None
def push_to_hub_if_configured(local_path):
    """Best-effort upload of an artifact to the HF model repo; never crash."""
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
            commit_message=f"{Path(local_path).name} update",
        )
        print(f"[hub] pushed {local_path} -> {HF_OUTPUT_REPO}", flush=True)
    except Exception as e:
        print(f"[hub] push failed (non-fatal): {e}", flush=True)

def hf_pickle(name):
    p = hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="dataset")
    print(f"HF_LOADED {name}", flush=True)
    return pickle.loads(Path(p).read_bytes())

T0 = time.time()
X_tr_raw = hf_pickle("X_train.pkl")
y_tr_raw = hf_pickle("y_train.pkl")
X_va_raw = hf_pickle("X_val.pkl")
y_va_raw = hf_pickle("y_val.pkl")
print(f"FETCH_DONE X_train={len(X_tr_raw)} X_val={len(X_va_raw)} "
      f"dt={time.time()-T0:.1f}s", flush=True)

# ----- glycowork structural metric -----------------------------------------
try:
    from glycowork.motif.graph import compare_glycans
    HAS_GLYCOWORK = True
    print("GLYCOWORK_OK", flush=True)
except ImportError:
    HAS_GLYCOWORK = False
    print("GLYCOWORK_MISSING — falling back to string equality", flush=True)

def struct_equal(a, b):
    if a == b: return True
    if not HAS_GLYCOWORK: return False
    try: return bool(compare_glycans(a, b))
    except Exception: return False

# ----- dataset shaping ------------------------------------------------------
def to_list(y):
    if isinstance(y, list): return y
    if isinstance(y, tuple): return list(y)
    if hasattr(y, "tolist"): return y.tolist()
    return list(y)

def as_str(y):
    if isinstance(y, str): return y
    if isinstance(y, (list, tuple)) and y and isinstance(y[0], str): return y[0]
    return str(y)

X_tr_list = to_list(X_tr_raw); y_tr_list = to_list(y_tr_raw)
X_va_list = to_list(X_va_raw); y_va_list = to_list(y_va_raw)
y_tr_kept = [as_str(y) for y in y_tr_list]
y_va_kept = [as_str(y) for y in y_va_list]
# Per BASELINE_PROTOCOL.md, all rows in the preprocessed pkls have non-empty
# IUPAC labels; assert it so a silent format change doesn't slip through.
assert all(y_tr_kept), "X_train rows with empty IUPAC label — check dataset version"
assert all(y_va_kept), "X_val rows with empty IUPAC label — check dataset version"
VAL_SPLIT_SEED = 42  # documented in dataset's split_metadata; kept for pareto.json provenance
print(f"SPLIT train={len(X_tr_list)} val={len(X_va_list)}", flush=True)

# ----- spectrum featurization ----------------------------------------------
def get_feats(x):
    mz   = np.asarray(x[0], dtype=np.float32).flatten()
    mz_r = np.asarray(x[1], dtype=np.float32).flatten() if len(x) > 1 else np.zeros_like(mz)
    try: prec = float(x[2]) if len(x) > 2 else 0.0
    except (TypeError, ValueError): prec = 0.0
    return mz, mz_r, prec

MZ_LEN = get_feats(X_tr_list[0])[0].shape[0]

def stack_feats(source):
    N = len(source)
    MZ   = np.zeros((N, MZ_LEN), dtype=np.float32)
    MZR  = np.zeros((N, MZ_LEN), dtype=np.float32)
    PREC = np.zeros((N,),         dtype=np.float32)
    for i in range(N):
        mz, mz_r, prec = get_feats(source[i])
        n = min(mz.shape[0], MZ_LEN); MZ[i, :n]  = mz[:n]
        n = min(mz_r.shape[0], MZ_LEN); MZR[i, :n] = mz_r[:n]
        PREC[i] = prec
    return MZ, MZR, PREC

print("STACK train features...", flush=True); ts = time.time()
MZ_tr, MZR_tr, PREC_tr = stack_feats(X_tr_list)
print(f"STACK train done dt={time.time()-ts:.1f}s", flush=True)
print("STACK val features...", flush=True); ts = time.time()
MZ_va, MZR_va, PREC_va = stack_feats(X_va_list)
print(f"STACK val done dt={time.time()-ts:.1f}s", flush=True)

# ----- IUPAC tokenization (68 monosaccharide tokens + specials) ------------
TOKEN_RE = re.compile(r"([\(\)\[\]]|a\d-\d|b\d-\d|a\?-\?|b\?-\?|a\d-\?|b\d-\?)")
def tokenize(s): return [p for p in TOKEN_RE.split(s) if p]

PAD, MASK, BOS, EOS = "<pad>", "<mask>", "<bos>", "<eos>"
specials = [PAD, MASK, BOS, EOS]
vocab = {t: i for i, t in enumerate(specials)}
for s in y_tr_kept:
    for t in tokenize(s):
        if t not in vocab: vocab[t] = len(vocab)
V = len(vocab)
PAD_ID, MASK_ID, BOS_ID, EOS_ID = vocab[PAD], vocab[MASK], vocab[BOS], vocab[EOS]
id2tok = {v: k for k, v in vocab.items()}
print(f"VOCAB V={V}", flush=True)

MAX_LEN = 96
def encode(s):
    ids = [BOS_ID] + [vocab.get(t, MASK_ID) for t in tokenize(s)][:MAX_LEN-2] + [EOS_ID]
    return ids + [PAD_ID]*(MAX_LEN-len(ids))

def decode_ids(ids):
    out = []
    for i in ids:
        if i in (BOS_ID, PAD_ID): continue
        if i == EOS_ID: break
        out.append(id2tok.get(int(i), "?"))
    return "".join(out)

Y_tr = torch.tensor([encode(s) for s in y_tr_kept], dtype=torch.long)
Y_va = torch.tensor([encode(s) for s in y_va_kept], dtype=torch.long)

def norm_spec(x):
    x = torch.log1p(x.clamp_min(0))
    return x / (x.max(-1, keepdim=True).values + 1e-6)

MZ_tr_t  = norm_spec(torch.tensor(MZ_tr));   MZR_tr_t = norm_spec(torch.tensor(MZR_tr))
MZ_va_t  = norm_spec(torch.tensor(MZ_va));   MZR_va_t = norm_spec(torch.tensor(MZR_va))
pt = torch.tensor(PREC_tr); pm, ps = pt.mean(), pt.std().clamp_min(1.0)
PREC_tr_t = (pt - pm) / ps
PREC_va_t = (torch.tensor(PREC_va) - pm) / ps
print(f"TENSORS Y_tr={tuple(Y_tr.shape)} Y_va={tuple(Y_va.shape)}", flush=True)

# ----- architecture (matched to xp-dfm-50m) --------------------------------
D_MODEL = 512; N_HEADS = 8; N_ENC = 4; N_DEC = 8; N_SPEC_TOKENS = 64

class SpecEncoder(nn.Module):
    def __init__(self, mz_len, d=D_MODEL, n_layers=N_ENC, n_heads=N_HEADS, n_tokens=N_SPEC_TOKENS):
        super().__init__()
        self.mz_len = mz_len; self.n_tokens = n_tokens
        self.chunk = mz_len // n_tokens + (1 if mz_len % n_tokens else 0)
        self.proj = nn.Linear(2 * self.chunk, d)
        self.pos  = nn.Embedding(n_tokens + 1, d)
        self.prec_proj = nn.Linear(1, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                           dropout=0.1, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("ids", torch.arange(n_tokens + 1).unsqueeze(0))

    def forward(self, mz, mz_r, prec):
        B = mz.shape[0]
        pad = self.chunk * self.n_tokens - self.mz_len
        mz_p  = F.pad(mz,  (0, pad)).view(B, self.n_tokens, self.chunk)
        mzr_p = F.pad(mz_r,(0, pad)).view(B, self.n_tokens, self.chunk)
        h = self.proj(torch.cat([mz_p, mzr_p], dim=-1))
        p = self.prec_proj(prec.view(B, 1, 1)).expand(B, 1, -1)
        h = torch.cat([h, p], dim=1) + self.pos(self.ids.expand(B, -1))
        return self.trunk(h)

class ARDecoder(nn.Module):
    """Autoregressive decoder: identical layout to DFM CondDFM but with
    causal self-attention and no time embedding."""
    def __init__(self, V, d=D_MODEL, n_layers=N_DEC, n_heads=N_HEADS, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                           dropout=0.1, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.trunk = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, V)
        self.register_buffer("ids", torch.arange(max_len).unsqueeze(0))
        # Causal masks cached per length (recomputed if length unseen)
        self._causal_cache = {}

    def causal_mask(self, L, device):
        # Bool mask: True = position cannot be attended to. Bool matches the
        # bool tgt_key_padding_mask, avoiding torch's deprecated mixed-dtype
        # mask path which fails with cudaErrorIllegalInstruction under bf16
        # autocast on H200.
        key = (L, device)
        m = self._causal_cache.get(key)
        if m is None:
            m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                           diagonal=1)
            self._causal_cache[key] = m
        return m

    def forward(self, x, mem):
        B, L = x.shape
        h = self.emb(x) + self.pos(self.ids[:, :L])
        m = self.causal_mask(L, h.device)
        return self.head(self.trunk(h, mem, tgt_mask=m,
                                    tgt_key_padding_mask=(x == PAD_ID)))

enc = SpecEncoder(MZ_LEN).to(DEVICE)
dec = ARDecoder(V).to(DEVICE)
n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
print(f"MODEL n_params={n_params}", flush=True)

# ----- hyperparameters -----------------------------------------------------
EPOCHS    = int(os.environ.get("EPOCHS", 30))
BATCH     = int(os.environ.get("BATCH", 1024))
BASE_LR   = float(os.environ.get("BASE_LR", 3e-4))
INIT_SEED = int(os.environ.get("INIT_SEED", 0))
LR_MAX    = BASE_LR * math.sqrt(BATCH / 128)
LR_FLOOR  = 0.03 * LR_MAX
WARMUP    = 200
WD        = 0.01
GRAD_CLIP = 1.0
EVAL_FREQ = 500
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", 24 * 3600))  # 24h hard backstop

# Init seed (controls weight init + per-step torch RNG)
torch.manual_seed(INIT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(INIT_SEED)

# Re-init now that the seed is set (modules above were built before seed)
# To be deterministic w.r.t. INIT_SEED, re-instantiate the modules:
torch.manual_seed(INIT_SEED)
enc = SpecEncoder(MZ_LEN).to(DEVICE)
dec = ARDecoder(V).to(DEVICE)
n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
print(f"MODEL re-init INIT_SEED={INIT_SEED} n_params={n_params}", flush=True)

steps_per_epoch = max(len(Y_tr) // BATCH, 1)
TOTAL_STEPS = EPOCHS * steps_per_epoch
print(f"SCHEDULE EPOCHS={EPOCHS} BATCH={BATCH} steps/epoch={steps_per_epoch} "
      f"TOTAL_STEPS={TOTAL_STEPS} LR_MAX={LR_MAX:.2e} LR_FLOOR={LR_FLOOR:.2e}",
      flush=True)

opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                        lr=LR_MAX, weight_decay=WD)

def get_lr(step):
    if step < WARMUP:
        return LR_MAX * step / WARMUP
    progress = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    return LR_MAX * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

# ----- batched beam search -------------------------------------------------
@torch.no_grad()
def beam_search(mz, mzr, prec, beam, max_len=MAX_LEN):
    """Returns (B, max_len) best beams. Naïve no-KV-cache decode.
    Finished beams freeze (produce PAD with zero score change)."""
    B = mz.shape[0]; K = beam
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        mem = enc(mz, mzr, prec)  # (B, T_enc, d)
    T_enc, d = mem.shape[1], mem.shape[2]

    # Expand mem to (B*K, T_enc, d)
    mem_exp = mem.unsqueeze(1).expand(B, K, T_enc, d).reshape(B * K, T_enc, d)

    tokens = torch.full((B, K, max_len), PAD_ID, dtype=torch.long, device=DEVICE)
    tokens[:, :, 0] = BOS_ID
    scores = torch.zeros((B, K), device=DEVICE)
    if K > 1:
        scores[:, 1:] = float('-inf')   # dedupe at step 1
    finished = torch.zeros((B, K), dtype=torch.bool, device=DEVICE)

    for t in range(1, max_len):
        if bool(finished.all()): break

        prefix = tokens[:, :, :t].reshape(B * K, t)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits = dec(prefix, mem_exp)
        next_logits = logits[:, -1, :].float()
        log_probs = F.log_softmax(next_logits, dim=-1)  # (B*K, V)

        finished_flat = finished.reshape(B * K)
        if finished_flat.any():
            log_probs[finished_flat, :] = float('-inf')
            log_probs[finished_flat, PAD_ID] = 0.0  # frozen-score continuation

        cand = scores.reshape(B * K, 1) + log_probs        # (B*K, V)
        cand = cand.reshape(B, K * V)
        top_scores, top_idx = cand.topk(K, dim=1)

        beam_idx  = top_idx // V                            # (B, K)
        token_idx = top_idx %  V                            # (B, K)

        new_tokens = torch.gather(tokens, 1,
                                  beam_idx.unsqueeze(-1).expand(B, K, max_len)).clone()
        new_tokens[:, :, t] = token_idx
        tokens = new_tokens
        finished = torch.gather(finished, 1, beam_idx) | (token_idx == EOS_ID)
        scores = top_scores

    best_idx = scores.argmax(dim=1)                         # (B,)
    return tokens[torch.arange(B), best_idx]                # (B, max_len)

# Periodic val eval uses a fixed deterministic subsample. Final eval uses
# the full val set. Naive beam search has no KV cache; full-val every 500
# steps would take hours of H200 time per training run (subsample CI
# half-width at p≈0.85, n=3000 ≈ ±1.3pp; eval-to-eval gaps during training
# are typically larger).
VAL_SUBSAMPLE_N = 3000
_val_subsample_idx = np.random.default_rng(seed=VAL_SPLIT_SEED).choice(
    len(Y_va), size=min(VAL_SUBSAMPLE_N, len(Y_va)), replace=False
)
_val_subsample_idx = torch.tensor(np.sort(_val_subsample_idx), dtype=torch.long)

# ----- val evaluation ------------------------------------------------------
def _val_struct_acc(beam, eval_batch, full=False):
    """Beam search val accuracy. full=False uses the fixed subsample."""
    enc.eval(); dec.eval()
    if full:
        idx_pool = torch.arange(len(Y_va))
    else:
        idx_pool = _val_subsample_idx
    correct = 0; total = 0
    for i in range(0, len(idx_pool), eval_batch):
        idx = idx_pool[i:i+eval_batch]
        yb  = Y_va[idx].to(DEVICE)
        mb  = MZ_va_t[idx].to(DEVICE)
        mrb = MZR_va_t[idx].to(DEVICE)
        pb  = PREC_va_t[idx].to(DEVICE)
        pred = beam_search(mb, mrb, pb, beam=beam)
        for b in range(yb.size(0)):
            if struct_equal(decode_ids(pred[b].tolist()), decode_ids(yb[b].tolist())):
                correct += 1
        total += yb.size(0)
    enc.train(); dec.train()
    return correct / max(total, 1)

@torch.no_grad()
def _val_token_metrics(eval_batch=256):
    """Cheap teacher-forced token-level loss/accuracy on val."""
    enc.eval(); dec.eval()
    tot_loss = 0.0; tot_correct = 0; tot_tokens = 0
    for i in range(0, len(Y_va), eval_batch):
        sl = slice(i, min(i + eval_batch, len(Y_va)))
        yb  = Y_va[sl].to(DEVICE)
        mb  = MZ_va_t[sl].to(DEVICE)
        mrb = MZR_va_t[sl].to(DEVICE)
        pb  = PREC_va_t[sl].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            mem = enc(mb, mrb, pb)
            y_in  = yb[:, :-1]
            y_out = yb[:, 1:]
            logits = dec(y_in, mem)
        logits = logits.float()
        mask = (y_out != PAD_ID)
        if mask.sum().item() == 0: continue
        loss = F.cross_entropy(logits.reshape(-1, V), y_out.reshape(-1),
                               ignore_index=PAD_ID, reduction="sum").item()
        tot_loss += loss
        tot_correct += ((logits.argmax(-1) == y_out) & mask).sum().item()
        tot_tokens += int(mask.sum().item())
    enc.train(); dec.train()
    n = max(tot_tokens, 1)
    return tot_correct / n, math.exp(tot_loss / n)

# ============================================================================
# PARETO_ONLY mode: skip training, download best_model.pt from HF, jump to
# Pareto sweep. Uses HF_OUTPUT_REPO env var pointing at the HF checkpoint
# repo (default: rlacombe/glycogen-ar-50m).
# ============================================================================
PARETO_ONLY = os.environ.get("PARETO_ONLY", "").lower() in ("1", "true", "yes")
if not PARETO_ONLY:
    # ----- training loop -------------------------------------------------------
    print(f"TRAIN_START n_train={len(Y_tr)} n_val={len(Y_va)} budget={MAX_SECONDS}s",
          flush=True)
    t_start = time.time()
    step = 0; epoch = 0
    best_struct = 0.0; best_step = 0; best_epoch = 0.0
    best_token_acc = 0.0
    val_eval_batch = int(os.environ.get("VAL_EVAL_BATCH", 128))

    stop = False
    for epoch in range(1, EPOCHS + 1):
        if stop: break
        perm = torch.randperm(len(Y_tr))
        for i in range(0, len(Y_tr), BATCH):
            if time.time() - t_start > MAX_SECONDS:
                print(f"TIME_BUDGET_HIT step={step} epoch={epoch}", flush=True)
                stop = True; break

            idx = perm[i:i+BATCH]
            if idx.shape[0] < BATCH:  # drop last partial batch for clean step accounting
                continue
            yb  = Y_tr[idx].to(DEVICE)
            mb  = MZ_tr_t[idx].to(DEVICE)
            mrb = MZR_tr_t[idx].to(DEVICE)
            pb  = PREC_tr_t[idx].to(DEVICE)

            lr = get_lr(step)
            for pg in opt.param_groups: pg["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                mem = enc(mb, mrb, pb)
                y_in  = yb[:, :-1]
                y_out = yb[:, 1:]
                logits = dec(y_in, mem)
                loss = F.cross_entropy(logits.reshape(-1, V), y_out.reshape(-1),
                                       ignore_index=PAD_ID)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()),
                                           GRAD_CLIP)
            opt.step()
            step += 1

            if step % EVAL_FREQ == 0:
                tok_acc, ppl = _val_token_metrics()
                t_eval = time.time()
                struct = _val_struct_acc(beam=4, eval_batch=val_eval_batch)
                eval_dt = time.time() - t_eval
                cur_epoch = step / steps_per_epoch
                print(f"METRIC step={step} epoch={cur_epoch:.2f} lr={lr:.2e} "
                      f"train_loss={loss.item():.4f} val_token_acc={tok_acc:.4f} "
                      f"val_ppl={ppl:.3f} top1_structural_accuracy={struct:.4f} "
                      f"beam=4 eval_dt={eval_dt:.1f} elapsed={time.time()-t_start:.1f}",
                      flush=True)

                if struct > best_struct:
                    best_struct = struct
                    best_step = step
                    best_epoch = cur_epoch
                    torch.save({
                        "enc": enc.state_dict(),
                        "dec": dec.state_dict(),
                        "vocab": vocab, "V": V, "MAX_LEN": MAX_LEN, "MZ_LEN": MZ_LEN,
                        "step": step, "epoch": best_epoch, "val_acc": struct,
                        "config": {"BATCH": BATCH, "BASE_LR": BASE_LR, "EPOCHS": EPOCHS,
                                   "VAL_SPLIT_SEED": VAL_SPLIT_SEED,
                                   "INIT_SEED": INIT_SEED, "LR_MAX": LR_MAX,
                                   "LR_FLOOR": LR_FLOOR},
                        "spec_norm": {"prec_mean": float(pm), "prec_std": float(ps)},
                    }, CKPT_DIR / "best_model.pt")
                    print(f"CHECKPOINT best_struct={best_struct:.4f} step={step}",
                          flush=True)
                    push_to_hub_if_configured(CKPT_DIR / "best_model.pt")
                if tok_acc > best_token_acc: best_token_acc = tok_acc

    # Final val eval (always re-run, even if last train step crossed eval boundary)
    final_tok_acc, final_ppl = _val_token_metrics()
    final_struct = _val_struct_acc(beam=4, eval_batch=val_eval_batch, full=True)
    if final_struct > best_struct:
        best_struct = final_struct; best_step = step; best_epoch = step / steps_per_epoch
        torch.save({
            "enc": enc.state_dict(), "dec": dec.state_dict(),
            "vocab": vocab, "V": V, "MAX_LEN": MAX_LEN, "MZ_LEN": MZ_LEN,
            "step": step, "epoch": best_epoch, "val_acc": final_struct,
            "config": {"BATCH": BATCH, "BASE_LR": BASE_LR, "EPOCHS": EPOCHS,
                       "VAL_SPLIT_SEED": VAL_SPLIT_SEED, "INIT_SEED": INIT_SEED,
                       "LR_MAX": LR_MAX, "LR_FLOOR": LR_FLOOR},
            "spec_norm": {"prec_mean": float(pm), "prec_std": float(ps)},
        }, CKPT_DIR / "best_model.pt")
        push_to_hub_if_configured(CKPT_DIR / "best_model.pt")

    print(f"METRIC_FINAL step={step} val_token_acc={final_tok_acc:.4f} "
          f"val_ppl={final_ppl:.3f} top1_structural_accuracy={final_struct:.4f} "
          f"best_top1_structural_accuracy={best_struct:.4f} "
          f"best_epoch={best_epoch:.2f} best_step={best_step}", flush=True)
    print(f"TRAIN_DONE elapsed={time.time()-t_start:.1f}s", flush=True)
else:
    from huggingface_hub import hf_hub_download
    import shutil
    print(f"PARETO_ONLY: downloading best_model.pt from {HF_OUTPUT_REPO}", flush=True)
    _p = hf_hub_download(repo_id=HF_OUTPUT_REPO, filename="best_model.pt",
                         repo_type=HF_OUTPUT_REPO_TYPE, cache_dir="/tmp/hf-ckpt")
    shutil.copy(_p, CKPT_DIR / "best_model.pt")
    print(f"PARETO_ONLY: copied to {CKPT_DIR}/best_model.pt", flush=True)
    best_struct = 0.0; best_step = 0; best_epoch = 0.0
    best_token_acc = 0.0
    t_start = time.time()

# ============================================================================
# Pareto sweep: load best checkpoint, beam ∈ {1,2,4,8,16,32} on the test set
# ============================================================================
print("\n=== PARETO SWEEP ===", flush=True)

# X_test is loaded here for the first time. Never loaded during training
# or model selection.
print("HF_LOADING X_test, y_test (Pareto sweep only)...", flush=True)
X_te_raw = hf_pickle("X_test.pkl")
y_te_raw = hf_pickle("y_test.pkl")
X_te = to_list(X_te_raw); y_te = to_list(y_te_raw)
y_te_kept = [as_str(y) for y in y_te]
assert all(y_te_kept), "X_test rows with empty IUPAC label — check dataset version"
print(f"TEST_SET n={len(X_te)} (before eval filters)", flush=True)

# ---- Optional structure-generalization subset filter (mirrors DFM/MDLM
# EVAL_SUBSET). Values: all (default) / seen / unseen. Uses y_train (loaded
# for training) to classify each test structure. EVAL_USE_GRAPH_EQUALITY=1
# switches from string equality to glycowork.compare_glycans (slow but exact).
_eval_subset = os.environ.get("EVAL_SUBSET", "all").strip().lower()
if _eval_subset not in ("all", "seen", "unseen"):
    raise ValueError(f"EVAL_SUBSET must be all/seen/unseen, got {_eval_subset!r}")
if _eval_subset in ("seen", "unseen"):
    _use_graph = os.environ.get("EVAL_USE_GRAPH_EQUALITY", "0") == "1"
    # y_train / y_tr was loaded at module scope; use its IUPAC strings.
    y_train_strs = [as_str(y) for y in y_tr_raw]
    train_strs_set = set(y_train_strs)
    is_seen = np.array([y in train_strs_set for y in y_te_kept], dtype=bool)
    if _use_graph:
        from glycowork.motif.graph import compare_glycans
        train_list = sorted(train_strs_set)
        resolved = {}
        for i, y_te_i in enumerate(y_te_kept):
            if is_seen[i]:
                continue
            if y_te_i in resolved:
                is_seen[i] = resolved[y_te_i]
                continue
            found = False
            for y_tr_i in train_list:
                try:
                    if compare_glycans(y_te_i, y_tr_i):
                        found = True; break
                except Exception:
                    continue
            resolved[y_te_i] = found
            is_seen[i] = found
    keep = is_seen if _eval_subset == "seen" else ~is_seen
    n_kept = int(keep.sum())
    print(f"EVAL_SUBSET={_eval_subset} use_graph_equality={_use_graph}: "
          f"{n_kept}/{len(X_te)} test spectra kept", flush=True)
    X_te = [x for x, k in zip(X_te, keep) if bool(k)]
    y_te = [y for y, k in zip(y_te, keep) if bool(k)]
    y_te_kept = [y for y, k in zip(y_te_kept, keep) if bool(k)]

print(f"TEST_SET n={len(X_te)} (after eval filters)", flush=True)

MZ_te, MZR_te, PREC_te = stack_feats(X_te)
MZ_te_t  = norm_spec(torch.tensor(MZ_te))
MZR_te_t = norm_spec(torch.tensor(MZR_te))
PREC_te_t = (torch.tensor(PREC_te) - pm) / ps

# Load best model
print("LOAD best_model.pt for Pareto sweep", flush=True)
state = torch.load(CKPT_DIR / "best_model.pt", map_location=DEVICE)
enc.load_state_dict(state["enc"]); dec.load_state_dict(state["dec"])
enc.eval(); dec.eval()

def _try_beam_batch(beam, batch_size):
    """Return True if a single beam_search call at (beam, batch_size) fits."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sl = slice(0, batch_size)
        mb  = MZ_te_t[sl].to(DEVICE)
        mrb = MZR_te_t[sl].to(DEVICE)
        pb  = PREC_te_t[sl].to(DEVICE)
        _ = beam_search(mb, mrb, pb, beam=beam)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return False

def find_max_batch(beam, lo=8, hi=4096):
    """Binary search for largest batch that fits at this beam value."""
    print(f"  OOM-search beam={beam} ...", flush=True)
    if not _try_beam_batch(beam, lo):
        print(f"  OOM at min batch lo={lo} for beam={beam}; using lo=4", flush=True)
        return max(lo // 2, 1)
    while lo < hi and _try_beam_batch(beam, min(hi, lo * 2)):
        lo = min(hi, lo * 2)
    if lo >= hi:
        print(f"  capped at hi={hi} for beam={beam}", flush=True)
        return hi
    # Now lo fits but 2*lo doesn't; refine between lo and 2*lo
    a, b = lo, min(hi, lo * 2)
    while a + 1 < b:
        m = (a + b) // 2
        if _try_beam_batch(beam, m): a = m
        else: b = m
    print(f"  max-fit batch beam={beam} -> {a}", flush=True)
    return a

def measure_one_batch(beam, batch_size, n_warmup=5):
    """Time a single beam_search call at (beam, batch_size) after n_warmup runs.
    Returns ms_total; divide by batch_size for ms_per_spectrum."""
    sl = slice(0, batch_size)
    mb  = MZ_te_t[sl].to(DEVICE)
    mrb = MZR_te_t[sl].to(DEVICE)
    pb  = PREC_te_t[sl].to(DEVICE)
    for _ in range(n_warmup):
        _ = beam_search(mb, mrb, pb, beam=beam)
    torch.cuda.synchronize()
    t0 = time.time()
    _ = beam_search(mb, mrb, pb, beam=beam)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0

def run_pareto_for_beam(beam):
    print(f"\n--- BEAM={beam} ---", flush=True)
    bs = find_max_batch(beam)
    # Use a slightly safer batch (half of max-fit) for the full accuracy sweep so
    # we don't OOM mid-run from minor allocator drift.
    acc_bs = max(bs // 2, 1)

    # Wall-clock at MAX-fit batch
    wall_ms_total = measure_one_batch(beam, bs)
    ms_per_spec = wall_ms_total / bs
    print(f"  WALLCLOCK beam={beam} batch={bs} total_ms={wall_ms_total:.1f} "
          f"per_spec={ms_per_spec:.3f}ms", flush=True)

    # Accuracy + mean output length (for NFE) on the full test set
    correct = 0; total = 0; len_sum = 0
    t0 = time.time()
    for i in range(0, len(y_te_kept), acc_bs):
        sl = slice(i, min(i + acc_bs, len(y_te_kept)))
        mb  = MZ_te_t[sl].to(DEVICE)
        mrb = MZR_te_t[sl].to(DEVICE)
        pb  = PREC_te_t[sl].to(DEVICE)
        pred = beam_search(mb, mrb, pb, beam=beam)
        # mean output length: tokens emitted up to and incl. EOS, excl. BOS;
        # if no EOS, count up to MAX_LEN.
        for b in range(pred.shape[0]):
            seq = pred[b].tolist()
            # length of generated content excluding BOS, including EOS if present
            if EOS_ID in seq:
                L = seq.index(EOS_ID) + 1   # incl. EOS step
            else:
                L = MAX_LEN - 1             # exclude BOS slot
            len_sum += L
            truth = y_te_kept[i + b]
            pred_str = decode_ids(seq)
            if struct_equal(pred_str, truth):
                correct += 1
            total += 1
        if (i // acc_bs) % 50 == 0:
            print(f"  PROGRESS beam={beam} {total}/{len(y_te_kept)} "
                  f"acc_so_far={correct/max(total,1):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    acc = correct / max(total, 1)
    mean_out_len = len_sum / max(total, 1)
    nfe = mean_out_len * beam
    print(f"  RESULT beam={beam} acc={acc:.4f} mean_out_len={mean_out_len:.2f} "
          f"nfe={nfe:.2f} dt={time.time()-t0:.1f}s", flush=True)
    return {
        "sweep_value": beam,
        "metric_name": "beam",
        "accuracy": acc,
        "nfe_per_spectrum": nfe,
        "wallclock_ms_per_spectrum": ms_per_spec,
        "batch_size_eval": bs,
        "mean_output_length": mean_out_len,
        "test_set_size": total,
    }

pareto_rows = []
_beam_env = os.environ.get("BEAM_LIST", "1,2,4,8,16,32")
_beam_list = [int(b.strip()) for b in _beam_env.split(",") if b.strip()]
print(f"BEAM_LIST={_beam_list}", flush=True)
for beam in _beam_list:
    try:
        pareto_rows.append(run_pareto_for_beam(beam))
    except Exception as e:
        print(f"  ERROR beam={beam}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache(); gc.collect()

pareto = {
    "cell": "xp-ar-50m",
    "checkpoint": "best_model.pt",
    "best_epoch": best_epoch,
    "best_step": best_step,
    "best_val_top1_structural_accuracy_beam4": best_struct,
    "init_seed": INIT_SEED,
    "hardware": "H200 141GB BF16",
    "training_config": {
        "BATCH": BATCH, "BASE_LR": BASE_LR, "EPOCHS": EPOCHS,
        "LR_MAX": LR_MAX, "LR_FLOOR": LR_FLOOR, "WARMUP": WARMUP,
        "WEIGHT_DECAY": WD, "GRAD_CLIP": GRAD_CLIP,
        "VAL_SPLIT_SEED": VAL_SPLIT_SEED, "MAX_LEN": MAX_LEN,
        "vocab_size": V, "n_params": n_params,
    },
    "data": {
        "source": HF_REPO,
        "train_size": len(X_tr_list),
        "val_size": len(X_va_list),
        "test_size": len(X_te),
    },
    "timestamp": datetime.now().isoformat(),
    "pareto": pareto_rows,
}

with open(PARETO_PATH, "w") as f:
    json.dump(pareto, f, indent=2)
print(f"PARETO_SAVED {PARETO_PATH}", flush=True)
push_to_hub_if_configured(PARETO_PATH)
print("DONE", flush=True)
