# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4",
#   "numpy<2",
#   "huggingface_hub>=0.24",
#   "glycowork",
#   "numpy_indexed",
# ]
# ///
"""
GlycoGen-DFM (50M): trains on the leak-free CandyCrunch v2 benchmark split,
then runs a Pareto sweep on the test set.

Follows the DFM convention of Gat et al. (NeurIPS 2024): t=0 is fully
masked source, t=1 is clean target, kappa_t = t. At training, each
non-special position is masked with probability (1 - t). Inference uses
Crystallizing Flows (`SAMPLER=crystallizing_flows`, default) or standard
CTMC Euler unmasking (`SAMPLER=ctmc_euler`).

Steps:
  1. Trains the 50M model for 30 epochs at BATCH=1024.
  2. Selects best checkpoint by val K=1 top-1 structural accuracy
     (glycowork.compare_glycans on IUPAC).
  3. After convergence, runs the Pareto sweep on the test set for
     K in {1, 2, 4, 8, 16, 32, 64}; emits pareto_*.json + METRIC_FINAL.
"""
import os, time, math, pickle, re, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------- I/O paths ----------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On HF Jobs/Endpoints
# the runtime mounts /data (input) and /output (artifact). Locally we fall back
# to a sibling output/ dir next to train.py.
if os.getenv("COMPUTE_BACKEND") == "hfjobs" or os.getenv("HF_JOB") == "1":
    OUT = Path("/output")
else:
    OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_PATH = OUT / "best_model.pt"
# Distinguish the output filename per sampler so a velocity-form sanity-check
# sweep doesn't overwrite the default Algorithm 1 pareto.json. Boltzmann
# Flow Sampling additionally tags by temperature. OUTPUT_SUFFIX env var
# appends an extra tag (e.g. "_K3") for partial K_LIST sweeps so they don't
# clobber an existing complete sweep.
_SAMPLER_TAG = os.environ.get("SAMPLER", "crystallizing_flows").lower()
_PARETO_MODE_TAG = os.environ.get("PARETO_MODE", "k_sweep").lower()
_TEMP_TAG = os.environ.get("TEMPERATURE", "1.0")
_OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "")
if _OUTPUT_SUFFIX and not _OUTPUT_SUFFIX.startswith("_"):
    _OUTPUT_SUFFIX = "_" + _OUTPUT_SUFFIX
if _PARETO_MODE_TAG == "adaptive_t0":
    PARETO_PATH = OUT / f"pareto_adaptive_t0{_OUTPUT_SUFFIX}.json"
elif _SAMPLER_TAG in ("algorithm1", "ctmc_euler"):
    PARETO_PATH = OUT / f"pareto{_OUTPUT_SUFFIX}.json"
elif _SAMPLER_TAG == "boltzmann":
    # Replace dots in the temperature tag for filename safety
    _t = _TEMP_TAG.replace(".", "p")
    PARETO_PATH = OUT / f"pareto_boltzmann_T{_t}{_OUTPUT_SUFFIX}.json"
elif _SAMPLER_TAG == "crystallizing_flows":
    PARETO_PATH = OUT / f"pareto_crystallizing_flows{_OUTPUT_SUFFIX}.json"
elif _SAMPLER_TAG == "soft_auto_commit_ee":
    PARETO_PATH = OUT / f"pareto_soft_auto_commit_ee{_OUTPUT_SUFFIX}.json"
else:
    PARETO_PATH = OUT / f"pareto_{_SAMPLER_TAG}{_OUTPUT_SUFFIX}.json"
PROGRESS_LOG = OUT / "training_log.jsonl"

CACHE_DIR = Path("/data") if Path("/data").is_dir() else Path("/tmp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ----- HF model-repo push (so checkpoints survive ephemeral job filesystems) -
# Best-effort: failures are logged but never abort training. The local copy
# in OUT/ is the source of truth on disk; HF is the durable backup.
HF_OUTPUT_REPO = os.environ.get("HF_OUTPUT_REPO", "rlacombe/glycogen-dfm-50m")
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

# ------------------------------ Reproducibility ------------------------------
# val split comes from the dataset itself (X_val.pkl); we record the
# upstream split seed for provenance only — it is not used at runtime.
VAL_SPLIT_SEED = 42                                   # provenance metadata
INIT_SEED = int(os.environ.get("INIT_SEED", 0))       # default 0; sweep {1,2} for seed-variance
torch.manual_seed(INIT_SEED)
np.random.seed(INIT_SEED)
torch.set_float32_matmul_precision("high")

# ------------------------------ Data download --------------------------------
# The preprocessed HF dataset ships pre-built X_train / X_val / X_test splits;
# we use them directly. When the dataset is mounted at /data (e.g. HF Jobs
# --volume hf://datasets/...:/data), files are read directly from the mount;
# otherwise we fall back to hf_hub_download. Override via HF_DATASET_REPO or
# HF_REPO env var, or rebuild locally from raw CC2 via data_utils/*.
HF_REPO = os.environ.get(
    "HF_DATASET_REPO",
    os.environ.get("HF_REPO", "rlacombe/candycrunch-v2-benchmark"),
)
FILES = {
    "X_train": "X_train.pkl",
    "y_train": "y_train.pkl",
    "X_val":   "X_val.pkl",
    "y_val":   "y_val.pkl",
    "X_test":  "X_test.pkl",
    "y_test":  "y_test.pkl",
}

def hf_load(name):
    """Read a dataset file. Prefer mounted /data; else hf_hub_download + cache."""
    fname = FILES[name]
    mounted = Path("/data") / fname
    if mounted.exists():
        print(f"MOUNT_HIT {name} <- {mounted}", flush=True)
        t0 = time.time()
        obj = pickle.loads(mounted.read_bytes())
        print(f"UNPICKLE_OK {name} n={len(obj)} dt={time.time()-t0:.1f}s", flush=True)
        return obj
    from huggingface_hub import hf_hub_download
    print(f"DOWNLOAD {name} <- {HF_REPO}/{fname}", flush=True)
    t0 = time.time()
    p = hf_hub_download(repo_id=HF_REPO, filename=fname, repo_type="dataset",
                        cache_dir=str(CACHE_DIR))
    print(f"DOWNLOAD_OK {name} dt={time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    obj = pickle.loads(Path(p).read_bytes())
    print(f"UNPICKLE_OK {name} n={len(obj)} dt={time.time()-t0:.1f}s", flush=True)
    return obj

# Train + val loaded now. X_test deferred to the Pareto sweep at end of run.
T0 = time.time()
X_train = hf_load("X_train")
y_train = hf_load("y_train")
X_val   = hf_load("X_val")
y_val   = hf_load("y_val")
print(f"FETCH_TRAIN+VAL_DONE n_train={len(X_train)} n_val={len(X_val)} "
      f"dt={time.time()-T0:.1f}s", flush=True)

# ------------------------------ Glycowork metric -----------------------------
try:
    from glycowork.motif.graph import compare_glycans
    HAS_GLYCOWORK = True
    print("GLYCOWORK_OK", flush=True)
except ImportError:
    HAS_GLYCOWORK = False
    print("GLYCOWORK_MISSING — falling back to string equality (REGRESSION RISK)", flush=True)

def struct_equal(a, b):
    if a == b:
        return True
    if not HAS_GLYCOWORK:
        return False
    try:
        return bool(compare_glycans(a, b))
    except Exception:
        return False

# ------------------------------ Label normalization --------------------------
def to_list(y):
    if isinstance(y, list): return y
    if isinstance(y, tuple): return list(y)
    if hasattr(y, "tolist"): return y.tolist()
    return list(y)

def as_str(y):
    if isinstance(y, str): return y
    if isinstance(y, (list, tuple)) and y and isinstance(y[0], str): return y[0]
    return str(y)

X_train = to_list(X_train); y_train = to_list(y_train)
X_val   = to_list(X_val);   y_val   = to_list(y_val)
y_train_strs = [as_str(y) for y in y_train]
y_val_strs   = [as_str(y) for y in y_val]

# Defensive guard: drop any rows with empty labels (the preprocessed dataset
# ships fully labeled rows so this should be a no-op, but we keep it so the
# script is robust to future schema changes).
def keep_labeled(X, y_strs):
    idx = np.array([i for i, s in enumerate(y_strs) if s], dtype=np.int64)
    return idx

train_idx = keep_labeled(X_train, y_train_strs)
val_idx   = keep_labeled(X_val,   y_val_strs)
print(f"SPLIT dataset-built train={len(train_idx)} val={len(val_idx)} "
      f"(dropped_empty train={len(X_train)-len(train_idx)} "
      f"val={len(X_val)-len(val_idx)})", flush=True)

# ---- Optional held-out-instrument filter for cross-instrument generalization
# experiments. TRAIN_TRAP_EXCLUDE=<int> removes all training + val spectra
# whose trap metadata (X tuple position 8) equals this value. Test set is
# NOT filtered here; use EVAL_TRAP=<same int> at eval time to test only
# on the held-out instrument class. Codes: 0=linear, 1=orbitrap, 2=amazon,
# 3=other.
_train_trap_exclude = os.environ.get("TRAIN_TRAP_EXCLUDE", "").strip()
if _train_trap_exclude:
    _excl = int(_train_trap_exclude)
    _trap_names = {0: "linear", 1: "orbitrap", 2: "amazon", 3: "other"}
    _label = _trap_names.get(_excl, f"trap_{_excl}")
    _tr_before = len(train_idx)
    _va_before = len(val_idx)
    train_idx = [i for i in train_idx
                 if len(X_train[i]) > 8 and int(X_train[i][8]) != _excl]
    val_idx   = [i for i in val_idx
                 if len(X_val[i])   > 8 and int(X_val[i][8])   != _excl]
    print(f"TRAIN_TRAP_EXCLUDE={_excl} ({_label}): "
          f"train {_tr_before} -> {len(train_idx)}, "
          f"val {_va_before} -> {len(val_idx)}", flush=True)

# ------------------------------ Spectrum features ----------------------------
def get_feats(x):
    mz   = np.asarray(x[0], dtype=np.float32).flatten()
    mz_r = np.asarray(x[1], dtype=np.float32).flatten() if len(x) > 1 else np.zeros_like(mz)
    try:    prec = float(x[2]) if len(x) > 2 else 0.0
    except (TypeError, ValueError): prec = 0.0
    return mz, mz_r, prec

MZ_LEN = get_feats(X_train[0])[0].shape[0]

def stack_feats(idx, X):
    N = len(idx)
    MZ   = np.zeros((N, MZ_LEN), dtype=np.float32)
    MZR  = np.zeros((N, MZ_LEN), dtype=np.float32)
    PREC = np.zeros((N,),         dtype=np.float32)
    for i, j in enumerate(idx):
        mz, mz_r, prec = get_feats(X[int(j)])
        n = min(mz.shape[0], MZ_LEN); MZ[i, :n] = mz[:n]
        n = min(mz_r.shape[0], MZ_LEN); MZR[i, :n] = mz_r[:n]
        PREC[i] = prec
    return MZ, MZR, PREC

print(f"STACK train features...", flush=True)
MZ_tr, MZR_tr, PREC_tr = stack_feats(train_idx, X_train)
print(f"STACK val features...", flush=True)
MZ_va, MZR_va, PREC_va = stack_feats(val_idx, X_val)
y_train_kept = [y_train_strs[int(j)] for j in train_idx]
y_val_kept   = [y_val_strs[int(j)]   for j in val_idx]

# ------------------------------ Tokenization ---------------------------------
TOKEN_RE = re.compile(r"([\(\)\[\]]|a\d-\d|b\d-\d|a\?-\?|b\?-\?|a\d-\?|b\d-\?)")
def tokenize(s): return [p for p in TOKEN_RE.split(s) if p]

PAD, MASK, BOS, EOS = "<pad>", "<mask>", "<bos>", "<eos>"
specials = [PAD, MASK, BOS, EOS]
vocab = {t: i for i, t in enumerate(specials)}
for s in y_train_kept:
    for t in tokenize(s):
        if t not in vocab: vocab[t] = len(vocab)
V = len(vocab)
PAD_ID, MASK_ID, BOS_ID, EOS_ID = vocab[PAD], vocab[MASK], vocab[BOS], vocab[EOS]
id2tok = {v: k for k, v in vocab.items()}
print(f"VOCAB V={V} (target ~72: 68 IUPAC tokens + 4 specials)", flush=True)

MAX_LEN = 96
def encode(s):
    ids = [BOS_ID] + [vocab.get(t, MASK_ID) for t in tokenize(s)][:MAX_LEN-2] + [EOS_ID]
    return ids + [PAD_ID]*(MAX_LEN-len(ids))

def decode_ids(ids):
    """Strip BOS/EOS/PAD; concatenate tokens to IUPAC."""
    out = []
    for i in ids:
        if i == BOS_ID or i == PAD_ID: continue
        if i == EOS_ID: break
        out.append(id2tok.get(int(i), "?"))
    return "".join(out)

Y_tr = torch.tensor([encode(s) for s in y_train_kept], dtype=torch.long)
Y_va = torch.tensor([encode(s) for s in y_val_kept], dtype=torch.long)

def norm_spec(x):
    x = torch.log1p(x.clamp_min(0))
    return x / (x.max(-1, keepdim=True).values + 1e-6)

MZ_tr_t  = norm_spec(torch.tensor(MZ_tr));  MZR_tr_t = norm_spec(torch.tensor(MZR_tr))
MZ_va_t  = norm_spec(torch.tensor(MZ_va));  MZR_va_t = norm_spec(torch.tensor(MZR_va))
pt = torch.tensor(PREC_tr); pm, ps = pt.mean(), pt.std().clamp_min(1.0)
PREC_tr_t = (pt - pm) / ps
PREC_va_t = (torch.tensor(PREC_va) - pm) / ps
print(f"TENSORS Y_tr={tuple(Y_tr.shape)} Y_va={tuple(Y_va.shape)} MZ_LEN={MZ_LEN}", flush=True)

# ============================== ARCHITECTURE =================================
# 50M scale: d=512, enc_layers=4, dec_layers=8, heads=8. Time conditioning
# is a single additive-broadcast embedding (nn.Linear(1, d) added to the
# input embeddings).

# Architecture is env-overridable so the same script can run at BART-base
# shape (d=768, enc=12, dec=12, heads=16).
D_MODEL  = int(os.environ.get("D_MODEL", 512))
N_HEADS  = int(os.environ.get("N_HEADS", 8))
N_ENC    = int(os.environ.get("N_ENC", 4))
N_DEC    = int(os.environ.get("N_DEC", 8))
N_SPEC_TOKENS = int(os.environ.get("N_SPEC_TOKENS", 64))

class SpecEncoder(nn.Module):
    def __init__(self, mz_len, d=D_MODEL, n_layers=N_ENC, n_heads=N_HEADS, n_tokens=N_SPEC_TOKENS):
        super().__init__()
        self.mz_len = mz_len; self.n_tokens = n_tokens
        self.chunk = mz_len // n_tokens + (1 if mz_len % n_tokens else 0)
        self.proj = nn.Linear(2 * self.chunk, d)
        self.pos  = nn.Embedding(n_tokens + 1, d)
        self.prec_proj = nn.Linear(1, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                          dropout=0.1, batch_first=True, activation="gelu",
                                          norm_first=True)
        self.trunk = nn.TransformerEncoder(enc, num_layers=n_layers)
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

class CondDFM(nn.Module):
    def __init__(self, V, d=D_MODEL, n_layers=N_DEC, n_heads=N_HEADS, max_len=MAX_LEN):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.pos = nn.Embedding(max_len, d)
        self.time_emb = nn.Linear(1, d)
        dec = nn.TransformerDecoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                          dropout=0.1, batch_first=True, activation="gelu",
                                          norm_first=True)
        self.trunk = nn.TransformerDecoder(dec, num_layers=n_layers)
        self.head = nn.Linear(d, V)
        self.register_buffer("ids", torch.arange(max_len).unsqueeze(0))

    def forward(self, x, t, mem):
        B, L = x.shape
        h = self.emb(x) + self.pos(self.ids[:, :L])
        h = h + self.time_emb(t.view(B, 1, 1).float()).expand(-1, L, -1)
        return self.head(self.trunk(h, mem, tgt_key_padding_mask=(x == PAD_ID)))

    def forward_soft(self, soft_x, t, mem, kpm=None):
        """Soft-input forward: soft_x is [B, L, V] of probabilities/one-hots.
        Equivalent to forward(x, t, mem) when soft_x is one-hot, generalises
        to mixture embeddings via soft_x @ self.emb.weight."""
        B, L, _ = soft_x.shape
        h = soft_x @ self.emb.weight + self.pos(self.ids[:, :L])
        h = h + self.time_emb(t.view(B, 1, 1).float()).expand(-1, L, -1)
        if kpm is None:
            kpm = (soft_x.argmax(-1) == PAD_ID)
        return self.head(self.trunk(h, mem, tgt_key_padding_mask=kpm))

enc = SpecEncoder(MZ_LEN).to(DEVICE)
dec = CondDFM(V).to(DEVICE)
n_params = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in dec.parameters())
n_dec_params = sum(p.numel() for p in dec.parameters())
print(f"MODEL n_params={n_params} (enc={n_params-n_dec_params} dec={n_dec_params})", flush=True)

# ============================== HYPERPARAMETERS ==============================
EPOCHS  = int(os.environ.get("EPOCHS", 30))
BATCH   = int(os.environ.get("BATCH", 1024))
BASE_LR = float(os.environ.get("BASE_LR", 3e-4))
WARMUP  = int(os.environ.get("WARMUP", 200))
WD      = float(os.environ.get("WD", 0.01))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))

SCALE    = math.sqrt(BATCH / 128)                         # batch=128 reference
LR_MAX   = BASE_LR * SCALE                                # ≈ 8.5e-4 at BATCH=1024
LR_FLOOR = 0.03 * LR_MAX                                  # ≈ 2.5e-5

steps_per_epoch = max(1, len(train_idx) // BATCH)
TOTAL_STEPS = steps_per_epoch * EPOCHS
EVAL_EVERY  = 500

print(f"HPARAMS BATCH={BATCH} BASE_LR={BASE_LR} EPOCHS={EPOCHS} "
      f"LR_MAX={LR_MAX:.2e} LR_FLOOR={LR_FLOOR:.2e} "
      f"steps_per_epoch={steps_per_epoch} TOTAL_STEPS={TOTAL_STEPS}", flush=True)

opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                        lr=LR_MAX, weight_decay=WD)

def get_lr(step):
    if step < WARMUP:
        return LR_MAX * step / WARMUP
    progress = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    progress = min(progress, 1.0)
    return LR_FLOOR + (LR_MAX - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * progress))

# ============================== DFM MASKING ==================================
# Convention: t=0 source (fully masked), t=1 target (clean).
# At time t each non-special position is MASKED with prob (1 - t),
# CLEAN with prob t.

def corrupt(x):
    """Return (x_t, mask, t) where mask[i,j]=True iff position (i,j) is masked.
    t ~ U(0,1) per example; p(masked) = 1 - t.
    """
    B = x.shape[0]
    t = torch.rand(B, device=x.device)                            # [B] in [0,1]
    non_special = (x != PAD_ID) & (x != BOS_ID) & (x != EOS_ID)
    p_mask = (1.0 - t).unsqueeze(1)                                # [B, 1]
    mask = (torch.rand_like(x, dtype=torch.float) < p_mask) & non_special
    x_t = torch.where(mask, torch.full_like(x, MASK_ID), x)
    return x_t, mask, t

# ============================== EULER UNMASKING SAMPLER ======================
# Algorithm 1. Two modes:
#   K=1     : one-shot argmax at t=eps.
#   K>=2    : K-step sampling per Algorithm 1, with a final argmax on any
#             positions still masked at t≈1.
# Operates batched over B spectra; per-position shape is [B, L].

@torch.no_grad()
def euler_sample(mem, kpm_template, K, eps=1e-3, sample=True, generator=None):
    """Run the Euler unmasking sampler on a batch.

    Args:
      mem: encoder output [B, T, d].
      kpm_template: [B, L] long tensor of the **structural** template
        (BOS/EOS/PAD positions for each example), with non-special positions
        set to MASK_ID. We hold BOS/EOS/PAD fixed and unmask only the
        non-special positions.
      K: number of Euler steps.
      sample: if True, sample from p; if False, take argmax.
    Returns:
      Predicted ids [B, L] with BOS/EOS/PAD preserved.
    """
    X = kpm_template.clone()                                       # [B, L]
    is_non_special = (X == MASK_ID)                                # positions to fill

    if K == 1 and not sample:                                       # one-shot argmax
        t1 = torch.full((X.size(0),), 1.0 - eps, device=X.device)
        logits = dec(X, t1, mem)
        pred = logits.argmax(-1)
        X = torch.where(is_non_special, pred, X)
        return X

    t = torch.full((X.size(0),), eps, device=X.device)
    h = (1.0 - eps) / K
    g = generator
    for _k in range(K):
        logits = dec(X, t, mem)                                    # [B, L, V]
        probs = F.softmax(logits, dim=-1)
        commit_p = h / (1.0 - t).clamp_min(1e-6)                   # [B], scalar per example
        # Decide which currently-masked positions commit this step
        masked_now = (X == MASK_ID) & is_non_special
        rand_keep = torch.rand_like(X, dtype=torch.float)          # [B, L]
        commit = masked_now & (rand_keep < commit_p.view(-1, 1))
        if commit.any():
            if sample:
                # multinomial requires float32 input on most torch builds
                flat = probs.view(-1, probs.size(-1)).float()
                draws = torch.multinomial(flat, num_samples=1,
                                          generator=g).view(X.shape)
            else:
                draws = probs.argmax(-1)
            X = torch.where(commit, draws, X)
        t = t + h
    # Final commit on remaining masks via argmax at t≈1
    still_masked = (X == MASK_ID) & is_non_special
    if still_masked.any():
        t1 = torch.full((X.size(0),), 1.0 - eps, device=X.device)
        logits = dec(X, t1, mem)
        X = torch.where(still_masked, logits.argmax(-1), X)
    return X

@torch.no_grad()
def euler_sample_velocity(mem, kpm_template, K, eps=1e-3, sample=True,
                          last_step_argmax=False, generator=None):
    """Velocity-explicit Euler sampler — materialises the velocity per
    Gat et al. 2024 Table 1 and applies the per-position Euler update
    WITHOUT the "committed positions never revert" simplification of
    Algorithm 1.

    For mask-source DFM with linear schedule κ_t = t and ẋκ_t = 1, the
    per-position next-state distribution is

        P[X_{t+h}^i = x | X_t^i = z^i]
          = δ_{z^i}(x) + h · u_t^i(x, z)
          = (1 - h/(1-t)) · δ_{z^i}(x) + (h/(1-t)) · p_{1|t}^θ(x | z).

    We sample directly from this mixture per non-special position.

    last_step_argmax: if True, replace multinomial sampling with argmax
      at the final Euler step (k = K-1, where coeff = h/(1-t) = 1 by
      construction with the linear schedule).
    """
    X = kpm_template.clone()
    is_non_special = (X == MASK_ID)

    if K == 1 and not sample:
        t1 = torch.full((X.size(0),), 1.0 - eps, device=X.device)
        logits = dec(X, t1, mem)
        X = torch.where(is_non_special, logits.argmax(-1), X)
        return X

    t = torch.full((X.size(0),), eps, device=X.device)
    h = (1.0 - eps) / K
    g = generator

    for _k in range(K):
        logits = dec(X, t, mem)
        probs = F.softmax(logits, dim=-1)                              # [B, L, V]
        one_hot = F.one_hot(X, num_classes=probs.size(-1)).to(probs.dtype)
        # coeff = h/(1-t) per example, broadcast over [L, V]; clamped to [0,1]
        # for numerical safety (it should equal 1 at the final step exactly).
        coeff = (h / (1.0 - t).clamp_min(1e-6)).view(-1, 1, 1).clamp(0.0, 1.0)
        next_p = (1.0 - coeff) * one_hot + coeff * probs               # [B, L, V]
        next_p = next_p.clamp_min(0.0)
        next_p = next_p / next_p.sum(-1, keepdim=True).clamp_min(1e-12)
        is_last = (_k == K - 1)
        if sample and not (is_last and last_step_argmax):
            flat = next_p.view(-1, next_p.size(-1)).float()
            draws = torch.multinomial(flat, num_samples=1,
                                      generator=g).view(X.shape)
        else:
            draws = next_p.argmax(-1)
        X = torch.where(is_non_special, draws, X)
        t = t + h

    still_masked = (X == MASK_ID) & is_non_special
    if still_masked.any():
        t1 = torch.full((X.size(0),), 1.0 - eps, device=X.device)
        logits = dec(X, t1, mem)
        X = torch.where(still_masked, logits.argmax(-1), X)
    return X

# =================== BOLTZMANN FLOW SAMPLING (continuous-state) ==============
# A temperature-parameterised continuous-state Euler sampler. State is a
# [B, L, V] probability distribution over the vocab; the decoder consumes a
# mixture embedding via forward_soft. At each Euler step the per-position
# softmax-with-temperature(logits/T) is mixed into the state with weight
# h/(1-t). Final argmax decodes the soft state to discrete tokens.
#
#   T = 1.0  →  standard softmax propagation (fka "soft_propagation")
#   T → 0    →  one-hot(argmax) propagation
#   T → ∞    →  uniform distribution propagation (state never moves)
#
# At T=0 we use F.one_hot(argmax) directly to avoid 1/T → ∞ numerics.

@torch.no_grad()
def euler_sample_boltzmann(mem, kpm_template, K, eps=1e-3, T=1.0, **_unused):
    B, _ = kpm_template.shape
    Vsz = dec.head.out_features
    is_non_special = (kpm_template == MASK_ID)
    state = F.one_hot(kpm_template, num_classes=Vsz).to(mem.dtype)
    kpm_fixed = (kpm_template == PAD_ID)

    if K == 1:
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec.forward_soft(state, t1, mem, kpm=kpm_fixed)
        # K=1 is T-independent: argmax of the model's predictive at t=1-eps
        pred = logits.argmax(-1)
        return torch.where(is_non_special, pred, kpm_template)

    t = torch.full((B,), eps, device=mem.device)
    h = (1.0 - eps) / K
    is_non_special_3d = is_non_special.unsqueeze(-1)
    T_eps = 1e-6
    for _k in range(K):
        logits = dec.forward_soft(state, t, mem, kpm=kpm_fixed)
        if T <= T_eps:
            # T=0 limit: one-hot of argmax
            argmax = logits.argmax(-1)
            sharp_p = F.one_hot(argmax, num_classes=Vsz).to(state.dtype)
        else:
            sharp_p = F.softmax(logits / T, dim=-1)
        coeff = (h / (1.0 - t).clamp_min(1e-6)).view(-1, 1, 1).clamp(0.0, 1.0)
        new_state = (1.0 - coeff) * state + coeff * sharp_p
        state = torch.where(is_non_special_3d, new_state, state)
        t = t + h

    X = state.argmax(-1)
    return torch.where(is_non_special, X, kpm_template)

# Backward-compat alias: soft propagation is BFS at T=1.
def euler_sample_soft(mem, kpm_template, K, eps=1e-3, **_unused):
    return euler_sample_boltzmann(mem, kpm_template, K=K, eps=eps, T=1.0)

# ===================== SOFT PROPAGATION WITH AUTO-COMMIT =====================
# Hybrid of soft_propagation and greedy_confidence: the per-position state
# starts as one-hot at <mask> and drifts under the soft Euler update; as soon
# as the state's argmax at a position is no longer <mask>, snap the position
# to a one-hot at that argmax and freeze it. Subsequent steps see committed
# positions as clean discrete tokens (good model conditioning) while
# uncommitted positions retain their soft mixture (preserves uncertainty).
#
# Selection rule is data-driven (state has decided) rather than schedule-driven
# (top-N per step). NFE = K (same as soft_propagation).

# Module-level buffer for diagnostic stats from soft_auto_commit (filled in by
# the sampler; consumed by run_pareto_on_test if SAMPLER == "soft_auto_commit").
_AUTO_COMMIT_STATS = {"sum_first_all_committed_step": 0,
                      "n_spectra": 0,
                      "n_never_fully_committed": 0,
                      "K_total": 0}

# Confidence-calibration logs (populated by soft_auto_commit_ee when
# LOG_COMMIT_CONF=1). See dump at end of run_pareto_on_test.
_CALIBRATION_LOG_K1 = []      # list of floats: max-softmax at K=1 argmax positions
_CALIBRATION_LOG_MULTI = []   # list of dicts: {step, K_total, confidence} at commit

@torch.no_grad()
def euler_sample_soft_auto_commit(mem, kpm_template, K, eps=1e-3, **_unused):
    B, _ = kpm_template.shape
    Vsz = dec.head.out_features
    is_non_special = (kpm_template == MASK_ID)
    state = F.one_hot(kpm_template, num_classes=Vsz).to(mem.dtype)
    kpm_fixed = (kpm_template == PAD_ID)
    # Special positions (BOS/EOS/PAD) are committed by definition.
    committed = ~is_non_special                                            # [B, L]

    if K == 1:
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec.forward_soft(state, t1, mem, kpm=kpm_fixed)
        pred = logits.argmax(-1)
        return torch.where(is_non_special, pred, kpm_template)

    # Diagnostic: per-spectrum step at which all non-special positions first
    # become committed. Used to estimate "effective K" under hypothetical
    # per-spectrum early exit (no compute saving in this run, just measurement).
    n_non_special_per_spec = is_non_special.sum(-1)                         # [B]
    first_all_committed = torch.full((B,), -1, dtype=torch.long, device=committed.device)

    t = torch.full((B,), eps, device=mem.device)
    h = (1.0 - eps) / K
    for _k in range(K):
        logits = dec.forward_soft(state, t, mem, kpm=kpm_fixed)
        probs = F.softmax(logits, dim=-1)
        coeff = (h / (1.0 - t).clamp_min(1e-6)).view(-1, 1, 1).clamp(0.0, 1.0)
        # Velocity update at uncommitted positions only; committed positions hold
        committed_3d = committed.unsqueeze(-1)
        soft_step = (1.0 - coeff) * state + coeff * probs
        new_state = torch.where(committed_3d, state, soft_step)
        # Auto-commit: positions whose argmax is not <mask> AND not yet committed
        argmax_now = new_state.argmax(-1)
        newly = (argmax_now != MASK_ID) & (~committed)
        if newly.any():
            new_one_hot = F.one_hot(argmax_now, num_classes=Vsz).to(state.dtype)
            new_state = torch.where(newly.unsqueeze(-1), new_one_hot, new_state)
            committed = committed | newly
        # Has this spectrum reached all-committed for the first time?
        committed_per_spec_count = (committed & is_non_special).sum(-1)     # [B]
        all_committed_now = (committed_per_spec_count == n_non_special_per_spec)
        first_hit = all_committed_now & (first_all_committed == -1)
        first_all_committed = torch.where(first_hit,
                                          torch.full_like(first_all_committed, _k + 1),
                                          first_all_committed)
        state = new_state
        t = t + h

    # Aggregate diagnostic stats globally
    fully = (first_all_committed >= 0)
    _AUTO_COMMIT_STATS["sum_first_all_committed_step"] += int(
        first_all_committed[fully].sum().item()) if fully.any() else 0
    # For never-committed examples, count their effective NFE as K
    _AUTO_COMMIT_STATS["sum_first_all_committed_step"] += int((~fully).sum().item()) * K
    _AUTO_COMMIT_STATS["n_spectra"] += int(B)
    _AUTO_COMMIT_STATS["n_never_fully_committed"] += int((~fully).sum().item())
    _AUTO_COMMIT_STATS["K_total"] = K  # last seen K

    X = state.argmax(-1)
    return torch.where(is_non_special, X, kpm_template)

# ===== SOFT AUTO-COMMIT WITH PER-SPECTRUM EARLY EXIT (sub-batch shrinking) ====
# Same logic as euler_sample_soft_auto_commit, but at every step the forward
# pass runs ONLY on spectra whose all non-special positions have not yet
# committed. As individual spectra reach all-committed, they exit the active
# batch and contribute no further forward-pass cost. The wall-clock measured
# under this sampler reflects realistic per-spectrum cost under deployment
# with dynamic-batch early exit (matches what a production system would do).

@torch.no_grad()
def euler_sample_soft_auto_commit_ee(mem, kpm_template, K, eps=1e-3, **_unused):
    B, _ = kpm_template.shape
    Vsz = dec.head.out_features
    is_non_special = (kpm_template == MASK_ID)
    state = F.one_hot(kpm_template, num_classes=Vsz).to(mem.dtype)
    kpm_fixed = (kpm_template == PAD_ID)
    committed = ~is_non_special
    n_non_special_per_spec = is_non_special.sum(-1)

    # Optional: log per-commit-event max-softmax confidence for calibration diagnostic.
    # LOG_COMMIT_CONF=1 collects confidence at each argmax-flip commit; written to
    # /output/commit_confidences.jsonl (or LOG_COMMIT_CONF_PATH). K=1 has no
    # per-step commits (single-shot argmax), so this only records for K>1.
    _log_conf = os.environ.get("LOG_COMMIT_CONF", "0") in ("1", "true", "yes")

    if K == 1:
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec.forward_soft(state, t1, mem, kpm=kpm_fixed)
        pred = logits.argmax(-1)
        # For K=1, log the max-softmax confidence at the argmax positions
        # (analogous to "commit confidence" — this is the K=1 baseline).
        if _log_conf:
            probs1 = F.softmax(logits, dim=-1)
            max_probs = probs1.max(-1).values  # [B, L]
            conf_vals = max_probs[is_non_special].cpu().tolist()
            _CALIBRATION_LOG_K1.extend(conf_vals)
        return torch.where(is_non_special, pred, kpm_template)

    h = (1.0 - eps) / K
    t = torch.full((B,), eps, device=mem.device)
    spec_done = (n_non_special_per_spec == 0)                         # edge case
    n_steps_actual = 0
    sum_active_per_step = 0
    for _k in range(K):
        active = ~spec_done
        if not active.any():
            break
        idx = torch.nonzero(active, as_tuple=True)[0]
        state_act = state[idx]
        mem_act = mem[idx]
        kpm_act = kpm_fixed[idx]
        t_act = t[idx]
        committed_act = committed[idx]

        logits = dec.forward_soft(state_act, t_act, mem_act, kpm=kpm_act)
        probs = F.softmax(logits, dim=-1)
        coeff = (h / (1.0 - t_act).clamp_min(1e-6)).view(-1, 1, 1).clamp(0.0, 1.0)
        committed_3d_act = committed_act.unsqueeze(-1)
        soft_step = (1.0 - coeff) * state_act + coeff * probs
        new_state_act = torch.where(committed_3d_act, state_act, soft_step)
        argmax_now = new_state_act.argmax(-1)
        newly = (argmax_now != MASK_ID) & (~committed_act)
        if newly.any():
            if _log_conf:
                # BEFORE the one-hot replacement: max-value of the pre-commit
                # state gives the "confidence" that argmax flipped off MASK.
                # (Reading AFTER the one-hot below would trivially give 1.0.)
                max_probs_pre = new_state_act.max(-1).values  # [B_act, L]
                commit_conf_vals = max_probs_pre[newly].detach().cpu().tolist()
                _CALIBRATION_LOG_MULTI.extend([
                    {"step": _k, "K_total": K, "confidence": float(v)}
                    for v in commit_conf_vals
                ])
            new_one_hot = F.one_hot(argmax_now, num_classes=Vsz).to(state.dtype)
            new_state_act = torch.where(newly.unsqueeze(-1), new_one_hot, new_state_act)
        state[idx] = new_state_act
        new_committed = committed_act | newly
        committed[idx] = new_committed
        committed_per_spec_act = (new_committed & is_non_special[idx]).sum(-1)
        all_committed_act = (committed_per_spec_act == n_non_special_per_spec[idx])
        spec_done[idx] = spec_done[idx] | all_committed_act
        t = t + h
        n_steps_actual += 1
        sum_active_per_step += int(active.sum().item())

    # Diagnostic: sum of (active_count) across actually-executed steps,
    # divided by B, gives the average per-spectrum effective NFE under
    # dynamic-batch early exit. This is the realistic NFE we report.
    _AUTO_COMMIT_STATS["sum_active_per_step"] = (
        _AUTO_COMMIT_STATS.get("sum_active_per_step", 0) + sum_active_per_step)
    _AUTO_COMMIT_STATS["n_spectra_ee"] = (
        _AUTO_COMMIT_STATS.get("n_spectra_ee", 0) + B)
    _AUTO_COMMIT_STATS["n_steps_actual"] = n_steps_actual                   # last K's value

    X = state.argmax(-1)
    return torch.where(is_non_special, X, kpm_template)

# ============================== GREEDY CONFIDENCE ============================
@torch.no_grad()
def euler_sample_greedy_confidence(mem, kpm_template, K, eps=1e-3, **_unused):
    """MaskGIT-style: every step, commit the top-N most-confident still-masked
    positions (deterministic argmax tokens), where N is the fair-share of the
    remaining masks across remaining steps. NFE = K, fully deterministic."""
    X = kpm_template.clone()
    is_non_special = (X == MASK_ID)
    B, L = X.shape
    if K == 1:
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec(X, t1, mem)
        return torch.where(is_non_special, logits.argmax(-1), X)

    t = torch.full((B,), eps, device=mem.device)
    h = (1.0 - eps) / K
    NEG_INF = float("-inf")
    for k_idx in range(K):
        logits = dec(X, t, mem)                                  # [B, L, V]
        probs = F.softmax(logits, dim=-1)
        masked_now = (X == MASK_ID) & is_non_special             # [B, L]
        confidence = probs.max(-1).values                        # [B, L]
        confidence = torch.where(masked_now, confidence,
                                 torch.full_like(confidence, NEG_INF))
        # Fair-share schedule: at step k_idx (0-indexed), commit
        # ceil(remaining / remaining_steps) masks per example.
        remaining = masked_now.sum(-1)                            # [B]
        remaining_steps = K - k_idx
        n_commit = torch.clamp(
            ((remaining.float() + remaining_steps - 1) // remaining_steps).long(),
            min=0)                                                # [B]
        # Rank positions by confidence (descending); commit_mask = (rank < n_commit)
        rank = confidence.argsort(-1, descending=True).argsort(-1)  # [B, L]
        commit_mask = (rank < n_commit.unsqueeze(-1)) & masked_now
        argmax_tok = probs.argmax(-1)
        X = torch.where(commit_mask, argmax_tok, X)
        t = t + h

    # Final cleanup if anything remains masked (shouldn't with fair-share)
    still_masked = (X == MASK_ID) & is_non_special
    if still_masked.any():
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec(X, t1, mem)
        X = torch.where(still_masked, logits.argmax(-1), X)
    return X

# =============================== GREEDY × SOFT ===============================
@torch.no_grad()
def euler_sample_greedy_soft(mem, kpm_template, K, eps=1e-3, **_unused):
    """Fills the (selection=greedy top-N, output=soft multinomial) corner of
    the (output × selection) 2x2 matrix. Same fair-share scheduler as
    greedy_confidence; commits by multinomial sampling from the per-position
    softmax (T=1) instead of argmax (T=0)."""
    X = kpm_template.clone()
    is_non_special = (X == MASK_ID)
    B, _ = X.shape
    Vsz = dec.head.out_features
    if K == 1:
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec(X, t1, mem)
        return torch.where(is_non_special, logits.argmax(-1), X)

    t = torch.full((B,), eps, device=mem.device)
    h = (1.0 - eps) / K
    NEG_INF = float("-inf")
    for k_idx in range(K):
        logits = dec(X, t, mem)
        probs = F.softmax(logits, dim=-1)
        masked_now = (X == MASK_ID) & is_non_special
        confidence = probs.max(-1).values
        confidence = torch.where(masked_now, confidence,
                                 torch.full_like(confidence, NEG_INF))
        remaining = masked_now.sum(-1)
        remaining_steps = K - k_idx
        n_commit = torch.clamp(
            ((remaining.float() + remaining_steps - 1) // remaining_steps).long(),
            min=0)
        rank = confidence.argsort(-1, descending=True).argsort(-1)
        commit_mask = (rank < n_commit.unsqueeze(-1)) & masked_now
        # Soft commit: sample from the full softmax instead of taking argmax
        flat = probs.view(-1, Vsz).float()
        sampled = torch.multinomial(flat, num_samples=1).view(X.shape)
        X = torch.where(commit_mask, sampled, X)
        t = t + h

    still_masked = (X == MASK_ID) & is_non_special
    if still_masked.any():
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec(X, t1, mem)
        X = torch.where(still_masked, logits.argmax(-1), X)
    return X

# ============================== HEUN'S METHOD ================================
@torch.no_grad()
def euler_sample_heun(mem, kpm_template, K, eps=1e-3, sample=True,
                      last_step_argmax=True, generator=None):
    """Second-order Heun method: per Euler step, do a predictor pass at t and
    a corrector pass at t+h, then average the two predicted distributions
    before applying the velocity update. NFE = 2K. Final step uses argmax to
    avoid the same boundary noise that velocity_argmax_last fixes."""
    X = kpm_template.clone()
    is_non_special = (X == MASK_ID)
    B, L = X.shape
    if K == 1 and not sample:
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec(X, t1, mem)
        return torch.where(is_non_special, logits.argmax(-1), X)

    t = torch.full((B,), eps, device=mem.device)
    h = (1.0 - eps) / K
    Vsz = dec.head.out_features
    g = generator
    for _k in range(K):
        is_last = (_k == K - 1)
        # ---- Predictor at t ----
        logits1 = dec(X, t, mem)
        probs1 = F.softmax(logits1, dim=-1)
        one_hot_X = F.one_hot(X, num_classes=Vsz).to(probs1.dtype)
        coeff = (h / (1.0 - t).clamp_min(1e-6)).view(-1, 1, 1).clamp(0.0, 1.0)
        pred_p = (1.0 - coeff) * one_hot_X + coeff * probs1
        pred_p = pred_p.clamp_min(0.0)
        pred_p = pred_p / pred_p.sum(-1, keepdim=True).clamp_min(1e-12)
        if sample:
            X_pred = torch.multinomial(pred_p.view(-1, Vsz).float(),
                                       num_samples=1,
                                       generator=g).view(X.shape)
        else:
            X_pred = pred_p.argmax(-1)
        X_pred = torch.where(is_non_special, X_pred, X)

        # ---- Corrector at t+h on the predicted state ----
        t_next = (t + h).clamp(max=1.0 - eps)
        logits2 = dec(X_pred, t_next, mem)
        probs2 = F.softmax(logits2, dim=-1)

        # Average the two probability fields and apply the velocity update
        probs_avg = 0.5 * (probs1 + probs2)
        next_p = (1.0 - coeff) * one_hot_X + coeff * probs_avg
        next_p = next_p.clamp_min(0.0)
        next_p = next_p / next_p.sum(-1, keepdim=True).clamp_min(1e-12)

        if sample and not (is_last and last_step_argmax):
            draws = torch.multinomial(next_p.view(-1, Vsz).float(),
                                      num_samples=1, generator=g).view(X.shape)
        else:
            draws = next_p.argmax(-1)
        X = torch.where(is_non_special, draws, X)
        t = t + h

    still_masked = (X == MASK_ID) & is_non_special
    if still_masked.any():
        t1 = torch.full((B,), 1.0 - eps, device=mem.device)
        logits = dec(X, t1, mem)
        X = torch.where(still_masked, logits.argmax(-1), X)
    return X

# Sampler dispatch: SAMPLER env var picks which sampler to use in the
# Pareto sweep. Defaults to Crystallizing Flows ("crystallizing_flows";
# legacy alias "soft_auto_commit_ee" is preserved for compatibility, as is
# "algorithm1" for "ctmc_euler").
SAMPLER = os.environ.get("SAMPLER", "crystallizing_flows").lower()
# Temperature for Boltzmann Flow Sampling (only used when SAMPLER=boltzmann).
# T=1.0 is the standard softmax propagation (alias: soft_propagation);
# T=0.0 is one-hot(argmax) propagation.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))
_VALID_SAMPLERS = ("crystallizing_flows", "ctmc_euler",
                   "algorithm1", "velocity", "velocity_argmax_last",
                   "soft_propagation", "soft_auto_commit",
                   "soft_auto_commit_ee",
                   "greedy_confidence", "greedy_soft",
                   "all_argmax", "heun", "boltzmann")
if SAMPLER not in _VALID_SAMPLERS:
    raise ValueError(f"Unknown SAMPLER={SAMPLER!r}; choose one of {_VALID_SAMPLERS}")

def sample_dispatch(mem, tmpl, K, sample_mode):
    if SAMPLER == "velocity":
        return euler_sample_velocity(mem, tmpl, K=K, sample=sample_mode,
                                     last_step_argmax=False)
    if SAMPLER == "velocity_argmax_last":
        return euler_sample_velocity(mem, tmpl, K=K, sample=sample_mode,
                                     last_step_argmax=True)
    if SAMPLER == "soft_propagation":
        return euler_sample_boltzmann(mem, tmpl, K=K, T=1.0)
    if SAMPLER == "soft_auto_commit":
        return euler_sample_soft_auto_commit(mem, tmpl, K=K)
    if SAMPLER in ("crystallizing_flows", "soft_auto_commit_ee"):
        return euler_sample_soft_auto_commit_ee(mem, tmpl, K=K)
    if SAMPLER == "boltzmann":
        return euler_sample_boltzmann(mem, tmpl, K=K, T=TEMPERATURE)
    if SAMPLER == "greedy_confidence":
        return euler_sample_greedy_confidence(mem, tmpl, K=K)
    if SAMPLER == "greedy_soft":
        return euler_sample_greedy_soft(mem, tmpl, K=K)
    if SAMPLER == "all_argmax":
        return euler_sample(mem, tmpl, K=K, sample=False)
    if SAMPLER == "heun":
        return euler_sample_heun(mem, tmpl, K=K, sample=sample_mode,
                                 last_step_argmax=True)
    # "ctmc_euler" or legacy alias "algorithm1"
    return euler_sample(mem, tmpl, K=K, sample=sample_mode)

def make_template(y):
    """Convert a ground-truth Y batch [B, L] into the inference template:
    BOS / EOS / PAD positions kept; all other positions replaced with MASK.
    The model does NOT see y_truth at non-special positions."""
    non_special = (y != PAD_ID) & (y != BOS_ID) & (y != EOS_ID)
    return torch.where(non_special, torch.full_like(y, MASK_ID), y)

# ============================== EVAL (val, K=1 argmax) =======================
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", 256))

@torch.no_grad()
def eval_val_k1(max_n=None):
    """K=1 one-shot val accuracy (top-1 structural).
    Also reports val token accuracy and ppl on the masked-CE loss."""
    enc.eval(); dec.eval()
    n_examples = len(Y_va) if max_n is None else min(len(Y_va), max_n)

    tc = 0; tt = 0; tl = 0.0
    n_struct = 0; n_total = 0
    for i in range(0, n_examples, EVAL_BATCH):
        yb  = Y_va[i:i+EVAL_BATCH].to(DEVICE)
        mb  = MZ_va_t[i:i+EVAL_BATCH].to(DEVICE)
        mrb = MZR_va_t[i:i+EVAL_BATCH].to(DEVICE)
        pb  = PREC_va_t[i:i+EVAL_BATCH].to(DEVICE)

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
            mem = enc(mb, mrb, pb)
            xc, m, t = corrupt(yb)
            logits = dec(xc, t, mem)
        if m.sum().item() > 0:
            tl += F.cross_entropy(logits[m].float(), yb[m]).item() * m.sum().item()
            tc += (logits.argmax(-1)[m] == yb[m]).sum().item()
            tt += m.sum().item()

        # K=1 one-shot argmax
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
            tmpl = make_template(yb)
            pred = euler_sample(mem, tmpl, K=1, sample=False)

        for b in range(yb.size(0)):
            pred_str = decode_ids(pred[b].tolist())
            true_str = decode_ids(yb[b].tolist())
            if struct_equal(pred_str, true_str):
                n_struct += 1
        n_total += yb.size(0)

    enc.train(); dec.train()
    return (tc/max(tt,1), math.exp(tl/max(tt,1)), n_struct/max(n_total,1))

# ============================== TRAINING LOOP ================================
def save_checkpoint(step, epoch, top1_struct_acc):
    # primary metric: top-1 structural accuracy via glycowork.compare_glycans
    torch.save({
        "enc": enc.state_dict(),
        "dec": dec.state_dict(),
        "vocab": vocab, "V": V,
        "MAX_LEN": MAX_LEN, "MZ_LEN": MZ_LEN,
        "step": step, "epoch": epoch,
        "top1_structural_accuracy": top1_struct_acc,
        "primary_metric": "top1_structural_accuracy",
        "config": {"BATCH": BATCH, "BASE_LR": BASE_LR, "EPOCHS": EPOCHS,
                   "VAL_SPLIT_SEED": VAL_SPLIT_SEED, "INIT_SEED": INIT_SEED,
                   "D_MODEL": D_MODEL, "N_HEADS": N_HEADS,
                   "N_ENC": N_ENC, "N_DEC": N_DEC,
                   "LR_MAX": LR_MAX, "LR_FLOOR": LR_FLOOR,
                   "WARMUP": WARMUP, "WD": WD, "GRAD_CLIP": GRAD_CLIP},
    }, CKPT_PATH)
    push_to_hub_if_configured(CKPT_PATH)

def run_training():
    print(f"TRAIN_START n_train={len(Y_tr)} n_val={len(Y_va)} "
          f"steps_per_epoch={steps_per_epoch}", flush=True)
    t_start = time.time()
    best_struct = 0.0
    best_step = 0
    best_epoch = 0.0
    enc.train(); dec.train()
    step = 0
    log_f = open(PROGRESS_LOG, "w", buffering=1)
    try:
        for epoch in range(EPOCHS):
            # Fresh per-epoch shuffle
            order = torch.randperm(len(Y_tr))
            for s_in_e in range(steps_per_epoch):
                idx = order[s_in_e*BATCH : (s_in_e+1)*BATCH]
                yb  = Y_tr[idx].to(DEVICE, non_blocking=True)
                mb  = MZ_tr_t[idx].to(DEVICE, non_blocking=True)
                mrb = MZR_tr_t[idx].to(DEVICE, non_blocking=True)
                pb  = PREC_tr_t[idx].to(DEVICE, non_blocking=True)

                lr = get_lr(step)
                for pg in opt.param_groups: pg["lr"] = lr

                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                    enabled=(DEVICE == "cuda")):
                    mem = enc(mb, mrb, pb)
                    xc, m, t = corrupt(yb)
                    if m.sum().item() == 0:
                        step += 1
                        continue
                    logits = dec(xc, t, mem)
                    loss = F.cross_entropy(logits[m].float(), yb[m])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(enc.parameters()) + list(dec.parameters()), GRAD_CLIP)
                opt.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    val_tok_acc, val_ppl, struct = eval_val_k1()
                    cur_epoch = step / steps_per_epoch
                    log = {"step": step, "epoch": cur_epoch, "lr": lr,
                           "train_loss": float(loss.item()),
                           "val_token_acc": val_tok_acc, "val_ppl": val_ppl,
                           "top1_structural_accuracy": struct,
                           "elapsed": time.time() - t_start}
                    log_f.write(json.dumps(log) + "\n")
                    print("METRIC " + " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                               else f"{k}={v}"
                                               for k, v in log.items()), flush=True)
                    if struct > best_struct:
                        best_struct = struct
                        best_step = step
                        best_epoch = cur_epoch
                        save_checkpoint(step, cur_epoch, struct)
                        print(f"BEST_NEW step={step} epoch={cur_epoch:.2f} "
                              f"val_struct_acc={struct:.4f} -> {CKPT_PATH}", flush=True)
            print(f"EPOCH_DONE epoch={epoch+1}/{EPOCHS} step={step} "
                  f"elapsed={time.time()-t_start:.1f}s", flush=True)
    finally:
        log_f.close()

    # Final eval, in case the last 500-step boundary did not land at end of run
    val_tok_acc, val_ppl, struct = eval_val_k1()
    if struct > best_struct:
        best_struct = struct
        best_step = step
        best_epoch = step / steps_per_epoch
        save_checkpoint(step, best_epoch, struct)
        print(f"BEST_NEW (final) step={step} val_struct_acc={struct:.4f}", flush=True)

    print(f"TRAIN_DONE elapsed={time.time()-t_start:.1f}s "
          f"best_step={best_step} best_epoch={best_epoch:.2f} "
          f"best_top1_structural_accuracy={best_struct:.4f}", flush=True)
    return best_step, best_epoch, best_struct, val_tok_acc, val_ppl

# ============================== PARETO SWEEP =================================
def load_best_into_models():
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    enc.load_state_dict(ck["enc"])
    dec.load_state_dict(ck["dec"])
    enc.eval(); dec.eval()
    # accept both the new "top1_structural_accuracy" key and the legacy "val_acc"
    # name so checkpoints from earlier runs still load cleanly.
    top1 = ck.get("top1_structural_accuracy", ck.get("val_acc"))
    print(f"LOADED_BEST step={ck.get('step')} epoch={ck.get('epoch')} "
          f"top1_structural_accuracy={top1}", flush=True)
    return ck

@torch.no_grad()
def find_max_batch(K, mz_len, vocab_size, max_attempt=4096, min_attempt=1):
    """Binary search for the largest batch size that fits at this K."""
    lo, hi = min_attempt, max_attempt
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            # Build a dummy batch of all-masked spectra and run one full sampler call
            mz_dummy   = torch.zeros(mid, mz_len, device=DEVICE)
            mzr_dummy  = torch.zeros(mid, mz_len, device=DEVICE)
            prec_dummy = torch.zeros(mid, device=DEVICE)
            tmpl       = torch.full((mid, MAX_LEN), MASK_ID, device=DEVICE, dtype=torch.long)
            tmpl[:, 0] = BOS_ID
            tmpl[:, -1] = EOS_ID  # one EOS at the end is enough to test memory
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                enabled=(DEVICE == "cuda")):
                mem = enc(mz_dummy, mzr_dummy, prec_dummy)
                _ = sample_dispatch(mem, tmpl, K, (K > 1))
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid - 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                hi = mid - 1
            else:
                raise
    return max(best, 1)

@torch.no_grad()
def run_pareto_on_test(X_te_raw, y_te_raw):
    """Sweep K∈{1,2,4,8,16,32,64} on the FULL test set.

    Optional eval-only subset filtering, controlled via env vars set by
    eval.py (all default no-op):
      EVAL_SUBSET=all|seen|unseen    filter test to structures seen/unseen in y_train
      EVAL_USE_GRAPH_EQUALITY=0|1    use glycowork.compare_glycans (slow, exact)
                                     for the seen/unseen classification
      EVAL_LIMIT=<int>               cap N_te after subset filtering (smoke tests)
      EVAL_OUTPUT=<path>             override PARETO_PATH for eval-only runs
    """
    print(f"PARETO_SWEEP_BEGIN", flush=True)
    X_te_list = to_list(X_te_raw); y_te_list = to_list(y_te_raw)
    y_te_str = [as_str(y) for y in y_te_list]
    N_te = len(X_te_list)
    print(f"X_test n={N_te} (before eval filters)", flush=True)

    # ---- Optional eval-only subset filtering ----
    _eval_subset = os.environ.get("EVAL_SUBSET", "all").strip().lower()
    if _eval_subset not in ("all", "seen", "unseen"):
        raise ValueError(f"EVAL_SUBSET must be all/seen/unseen, got {_eval_subset!r}")
    if _eval_subset in ("seen", "unseen"):
        # y_train (loaded at module scope) is authoritative for train structures.
        _use_graph = os.environ.get("EVAL_USE_GRAPH_EQUALITY", "0") == "1"
        train_strs = set(as_str(y) for y in y_train)
        # Fast path: string equality. compute a bool per test spectrum.
        is_seen = np.array([y in train_strs for y in y_te_str], dtype=bool)
        if _use_graph:
            from glycowork.motif.graph import compare_glycans
            train_list = sorted(train_strs)
            resolved = {}
            for i, y_te in enumerate(y_te_str):
                if is_seen[i] or y_te in resolved:
                    if y_te in resolved:
                        is_seen[i] = is_seen[i] or resolved[y_te]
                    continue
                found = False
                for y_tr in train_list:
                    try:
                        if compare_glycans(y_te, y_tr):
                            found = True; break
                    except Exception:
                        continue
                resolved[y_te] = found
                is_seen[i] = found
        keep = is_seen if _eval_subset == "seen" else ~is_seen
        n_kept = int(keep.sum())
        print(f"EVAL_SUBSET={_eval_subset} use_graph_equality={_use_graph}: "
              f"{n_kept}/{N_te} test spectra kept", flush=True)
        X_te_list = [x for x, k in zip(X_te_list, keep) if bool(k)]
        y_te_list = [y for y, k in zip(y_te_list, keep) if bool(k)]
        y_te_str  = [y for y, k in zip(y_te_str,  keep) if bool(k)]
        N_te = len(X_te_list)

    # ---- Optional trap-slice filter (for cross-instrument eval on CC2's
    # own metadata field). EVAL_TRAP is the integer code from TRAP_MAP:
    #   0=linear, 1=orbitrap, 2=amazon, 3=other/unknown
    # Filters test set where x[8] (trap_int in the 9-tuple aligned format)
    # matches the specified value. Set to empty/unset for no filter.
    _eval_trap = os.environ.get("EVAL_TRAP", "").strip()
    if _eval_trap:
        try:
            _trap_target = int(_eval_trap)
        except ValueError:
            raise ValueError(f"EVAL_TRAP must be an integer 0/1/2/3, got {_eval_trap!r}")
        keep_trap = np.array(
            [len(x) > 8 and int(x[8]) == _trap_target for x in X_te_list],
            dtype=bool,
        )
        n_kept_trap = int(keep_trap.sum())
        _trap_names = {0: "linear", 1: "orbitrap", 2: "amazon", 3: "other"}
        _trap_label = _trap_names.get(_trap_target, f"trap_{_trap_target}")
        print(f"EVAL_TRAP={_trap_target} ({_trap_label}): "
              f"{n_kept_trap}/{N_te} test spectra kept", flush=True)
        X_te_list = [x for x, k in zip(X_te_list, keep_trap) if bool(k)]
        y_te_list = [y for y, k in zip(y_te_list, keep_trap) if bool(k)]
        y_te_str  = [y for y, k in zip(y_te_str,  keep_trap) if bool(k)]
        N_te = len(X_te_list)

    _eval_limit = os.environ.get("EVAL_LIMIT")
    if _eval_limit is not None and _eval_limit.strip():
        cap = int(_eval_limit)
        if cap < N_te:
            X_te_list = X_te_list[:cap]
            y_te_list = y_te_list[:cap]
            y_te_str  = y_te_str[:cap]
            N_te = len(X_te_list)
            print(f"EVAL_LIMIT={cap}: capped to {N_te} spectra", flush=True)

    print(f"X_test n={N_te} (after eval filters)", flush=True)

    # Encode test features once
    MZ_te = np.zeros((N_te, MZ_LEN), dtype=np.float32)
    MZR_te = np.zeros((N_te, MZ_LEN), dtype=np.float32)
    PREC_te = np.zeros((N_te,), dtype=np.float32)
    for i in range(N_te):
        mz, mz_r, prec = get_feats(X_te_list[i])
        n = min(mz.shape[0], MZ_LEN); MZ_te[i, :n] = mz[:n]
        n = min(mz_r.shape[0], MZ_LEN); MZR_te[i, :n] = mz_r[:n]
        PREC_te[i] = prec
    MZ_te_t   = norm_spec(torch.tensor(MZ_te))
    MZR_te_t  = norm_spec(torch.tensor(MZR_te))
    PREC_te_t = (torch.tensor(PREC_te) - pm) / ps

    # Build the GT-encoded test labels and the inference template (BOS/EOS/PAD
    # preserved, all non-special positions replaced with MASK). The template
    # gives the model the structural skeleton (sequence length); it never sees
    # the GT tokens. Length structure is derived from y_test only and is
    # used solely to construct the BOS/EOS/PAD positions.
    Y_te_full = torch.tensor([encode(s) for s in y_te_str], dtype=torch.long)
    Tmpl_te = make_template(Y_te_full)

    # K_LIST overridable via env var (comma-separated); defaults to log-spaced
    # 1,2,4,8,16,32,64. Use to add intermediate values (e.g., "1,2,3,4,5,8,16,32,64")
    # for finer-grained low-K Pareto curves.
    _k_env = os.environ.get("K_LIST", "1,2,4,8,16,32,64")
    K_LIST = [int(k.strip()) for k in _k_env.split(",") if k.strip()]
    pareto = []

    # ---- Optional: one-time preprocessing benchmark (raw peak_d -> 9-tuple).
    # BENCH_E2E=1 or BENCH_PREPROCESS=1 turns this on. Loads raw peak_d dicts
    # from the HF-dataset mount (/data/test_peak_d.pkl by default) and times
    # the exact preprocess path used at ingestion (parse -> normalise ->
    # bin_intensities into 2048-bin (binned, remainder) vectors + trivial
    # metadata packing). Result is K-independent, copied to every row.
    preprocess_ms = None
    preprocess_ms_std = None
    preprocess_n = 0
    if os.environ.get("BENCH_E2E", "0") in ("1", "true", "yes") or \
       os.environ.get("BENCH_PREPROCESS", "0") in ("1", "true", "yes"):
        try:
            import statistics as _stats
            import ast as _ast
            try:
                import numpy_indexed as _npi
                _has_npi = True
            except ImportError:
                _has_npi = False
            _pkd_path = os.environ.get("BENCH_PEAK_D", "/data/test_peak_d.pkl")
            print(f"[bench-preprocess] loading raw peak_d from {_pkd_path}", flush=True)
            with open(_pkd_path, "rb") as _f:
                _peak_d_list = pickle.load(_f)
            print(f"[bench-preprocess] loaded {len(_peak_d_list):,} peak dicts", flush=True)
            _MIN_MZ, _MAX_MZ, _BIN_NUM = 39.714, 3000.0, 2048
            _FRAMES = np.linspace(_MIN_MZ, _MAX_MZ, _BIN_NUM)
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
            def _bin_intensities(pd_, frames=_FRAMES):
                nf = len(frames)
                b = np.zeros(nf); md = np.zeros(nf)
                mzs = np.array(list(pd_.keys()), dtype="float32")
                ints = np.array(list(pd_.values()))
                bi = np.digitize(mzs, frames, right=True)
                mzr = mzs - frames[bi - 1]
                if _has_npi:
                    maxi = _npi.group_by(bi - 1).max(ints)
                    mzr = mzr * np.isin(ints, maxi)
                    ub, sm = _npi.group_by(bi).sum(ints)
                    _, mmr = _npi.group_by(bi).max(mzr)
                    b[ub - 1] = sm
                    md[ub - 1] = mmr
                else:
                    for k, i_ in enumerate(bi):
                        if 0 < i_ <= nf:
                            b[i_ - 1] += ints[k]
                            if mzr[k] > md[i_ - 1]:
                                md[i_ - 1] = mzr[k]
                return b, md
            def _preprocess(pd_raw):
                sp = _safe_peak_parse(pd_raw)
                if sp is None or len(sp) == 0: return None
                sp = _normalise(sp)
                if sp is None: return None
                try:
                    b, r = _bin_intensities(sp)
                except Exception:
                    return None
                return (b.astype(np.float32), r.astype(np.float32),
                        800.0, 0, 0.3, 0, 0, 0, 1)
            _n_pp = min(int(os.environ.get("BENCH_PREPROCESS_N", "200")),
                        len(_peak_d_list))
            _warmup = 10
            _rng = np.random.default_rng(0)
            _idxs = _rng.choice(len(_peak_d_list),
                                size=min(_n_pp + _warmup, len(_peak_d_list)),
                                replace=False)
            for i in _idxs[:_warmup]:
                _ = _preprocess(_peak_d_list[i])
            _pp_times = []
            for i in _idxs[_warmup:_warmup + _n_pp]:
                _t0 = time.perf_counter()
                _ = _preprocess(_peak_d_list[i])
                _pp_times.append((time.perf_counter() - _t0) * 1000.0)
            preprocess_ms = float(_stats.mean(_pp_times))
            preprocess_ms_std = float(_stats.stdev(_pp_times)) if len(_pp_times) > 1 else 0.0
            preprocess_n = len(_pp_times)
            print(f"[bench-preprocess] DFM preprocess ms/spec = "
                  f"{preprocess_ms:.3f}±{preprocess_ms_std:.3f} (n={preprocess_n})",
                  flush=True)
        except Exception as _e:
            print(f"[bench-preprocess] SKIPPED (error: {_e})", flush=True)

    for K in K_LIST:
        print(f"PARETO K={K} sizing batch...", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            B_eval = find_max_batch(K, MZ_LEN, V, max_attempt=4096)
            # Cap batch at total dataset to avoid wasted slack
            B_eval = min(B_eval, N_te)
        else:
            B_eval = 8
        print(f"PARETO K={K} batch_size_eval={B_eval}", flush=True)

        # ---- Accuracy on full X_test (K=1 uses argmax; K>1 uses sampling) ----
        n_struct = 0
        sample_mode = (K > 1)
        # Reset auto-commit diagnostics for this K
        _AUTO_COMMIT_STATS["sum_first_all_committed_step"] = 0
        _AUTO_COMMIT_STATS["n_spectra"] = 0
        _AUTO_COMMIT_STATS["n_never_fully_committed"] = 0
        _AUTO_COMMIT_STATS["K_total"] = K
        _AUTO_COMMIT_STATS["sum_active_per_step"] = 0
        _AUTO_COMMIT_STATS["n_spectra_ee"] = 0
        # Optional prediction dump for failure-mode analysis. DUMP_PREDS=1
        # collects (idx_in_subset, y_true, y_pred, correct) per spectrum;
        # written to /output/preds_K{K}.jsonl (or DUMP_PREDS_PATH if set).
        _dump_preds = os.environ.get("DUMP_PREDS", "0") in ("1", "true", "yes")
        _dump_records = [] if _dump_preds else None
        t0 = time.time()
        for i in range(0, N_te, B_eval):
            mb  = MZ_te_t[i:i+B_eval].to(DEVICE)
            mrb = MZR_te_t[i:i+B_eval].to(DEVICE)
            pb  = PREC_te_t[i:i+B_eval].to(DEVICE)
            tmpl = Tmpl_te[i:i+B_eval].to(DEVICE)
            yb   = Y_te_full[i:i+B_eval]
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                enabled=(DEVICE == "cuda")):
                mem  = enc(mb, mrb, pb)
                pred = sample_dispatch(mem, tmpl, K, sample_mode)
            for b in range(pred.size(0)):
                pred_str = decode_ids(pred[b].cpu().tolist())
                true_str = decode_ids(yb[b].tolist())
                is_correct = struct_equal(pred_str, true_str)
                if is_correct:
                    n_struct += 1
                if _dump_preds:
                    _dump_records.append({
                        "idx": i + b,
                        "y_true": true_str,
                        "y_pred": pred_str,
                        "correct": bool(is_correct),
                    })
            if (i // B_eval) % 50 == 0:
                print(f"  K={K} progress={i+pred.size(0)}/{N_te} "
                      f"running_acc={n_struct/(i+pred.size(0)):.4f}", flush=True)
        acc = n_struct / N_te
        dt_full = time.time() - t0
        print(f"PARETO K={K} accuracy={acc:.4f} dt_full_sweep={dt_full:.1f}s", flush=True)

        if _dump_preds:
            _dump_default = f"/output/preds_K{K}.jsonl" if Path("/output").is_dir() \
                else str(OUT / f"preds_K{K}.jsonl")
            _dump_path = Path(os.environ.get("DUMP_PREDS_PATH", _dump_default))
            _dump_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_dump_path, "w") as _f:
                for _rec in _dump_records:
                    _f.write(json.dumps(_rec) + "\n")
            print(f"[dump-preds] wrote {len(_dump_records)} records to {_dump_path}",
                  flush=True)
            push_to_hub_if_configured(_dump_path)

        # ---- Wall-clock: 5-pass warmup, then time one batched call ----
        warm_b = min(B_eval, 32)
        mz_warm   = MZ_te_t[:warm_b].to(DEVICE)
        mzr_warm  = MZR_te_t[:warm_b].to(DEVICE)
        prec_warm = PREC_te_t[:warm_b].to(DEVICE)
        tmpl_warm = Tmpl_te[:warm_b].to(DEVICE)
        for _ in range(5):
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                enabled=(DEVICE == "cuda")):
                mem_w = enc(mz_warm, mzr_warm, prec_warm)
                _ = sample_dispatch(mem_w, tmpl_warm, K, sample_mode)
        if DEVICE == "cuda": torch.cuda.synchronize()

        bench_b = min(B_eval, N_te)
        mz_b   = MZ_te_t[:bench_b].to(DEVICE)
        mzr_b  = MZR_te_t[:bench_b].to(DEVICE)
        prec_b = PREC_te_t[:bench_b].to(DEVICE)
        tmpl_b = Tmpl_te[:bench_b].to(DEVICE)
        if DEVICE == "cuda": torch.cuda.synchronize()
        t_b0 = time.time()
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                            enabled=(DEVICE == "cuda")):
            mem_b = enc(mz_b, mzr_b, prec_b)
            _ = sample_dispatch(mem_b, tmpl_b, K, sample_mode)
        if DEVICE == "cuda": torch.cuda.synchronize()
        dt_batch = time.time() - t_b0
        ms_per_spec = 1000.0 * dt_batch / bench_b
        print(f"PARETO K={K} wallclock_ms_per_spectrum={ms_per_spec:.3f} "
              f"batch_size_eval={bench_b}", flush=True)

        # NFE per spectrum varies by sampler. Heun does 2 forward passes per
        # Euler step (predictor + corrector). All other samplers do 1 per step,
        # plus a final argmax commit pass for K>1 in case any masks remain.
        if SAMPLER == "heun":
            nfe = float(2 * K)
        else:
            nfe = float(K) + (1.0 if K > 1 else 0.0)

        row = {
            "sweep_value": K,
            "metric_name": "K",
            "accuracy": float(acc),
            "nfe_per_spectrum": nfe,
            "wallclock_ms_per_spectrum": float(ms_per_spec),
            "batch_size_eval": int(bench_b),
        }
        # Auto-commit effective-NFE diagnostic
        if SAMPLER == "soft_auto_commit" and _AUTO_COMMIT_STATS["n_spectra"] > 0:
            avg_eff_K = (_AUTO_COMMIT_STATS["sum_first_all_committed_step"]
                         / max(_AUTO_COMMIT_STATS["n_spectra"], 1))
            frac_never = (_AUTO_COMMIT_STATS["n_never_fully_committed"]
                          / max(_AUTO_COMMIT_STATS["n_spectra"], 1))
            row["effective_K_with_per_spec_early_exit"] = float(avg_eff_K)
            row["frac_spectra_never_fully_committed"] = float(frac_never)
            print(f"PARETO K={K} avg_effective_K (per-spec early exit) = "
                  f"{avg_eff_K:.2f} ; frac_never_committed = {frac_never:.4f}",
                  flush=True)
        # Early-exit measured effective NFE (sub-batch shrinking, the realistic case)
        if SAMPLER in ("crystallizing_flows", "soft_auto_commit_ee") and _AUTO_COMMIT_STATS.get("n_spectra_ee", 0) > 0:
            avg_eff_K_ee = (_AUTO_COMMIT_STATS["sum_active_per_step"]
                            / max(_AUTO_COMMIT_STATS["n_spectra_ee"], 1))
            row["effective_K_measured_ee"] = float(avg_eff_K_ee)
            print(f"PARETO K={K} measured effective NFE (sub-batch early exit) = "
                  f"{avg_eff_K_ee:.2f}", flush=True)

        # ---- Wall-clock: UNBATCHED (batch=1) — real-time streaming latency ----
        # BENCH_E2E=1 or BENCH_UNBATCHED=1 turns this on. Mirrors GlycoBART's
        # time_wallclock_unbatched: 5-pass warmup at batch=1, then n_unb timed
        # single-spectrum forwards. Reflects the cost when spectra arrive one
        # at a time (real-time deployment); batched is the offline-throughput
        # regime already reported above.
        if os.environ.get("BENCH_E2E", "0") in ("1", "true", "yes") or \
           os.environ.get("BENCH_UNBATCHED", "0") in ("1", "true", "yes"):
            import statistics as _stats
            _n_unb = min(int(os.environ.get("BENCH_UNBATCHED_N", "100")), N_te)
            _warmup_unb = 5
            with torch.no_grad():
                for i in range(min(_warmup_unb, _n_unb)):
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                        enabled=(DEVICE == "cuda")):
                        _mw = enc(MZ_te_t[i:i+1].to(DEVICE),
                                  MZR_te_t[i:i+1].to(DEVICE),
                                  PREC_te_t[i:i+1].to(DEVICE))
                        _ = sample_dispatch(_mw, Tmpl_te[i:i+1].to(DEVICE),
                                            K, sample_mode)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            _times_unb = []
            for i in range(_n_unb):
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                _t0 = time.time()
                with torch.no_grad():
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                        enabled=(DEVICE == "cuda")):
                        _m1 = enc(MZ_te_t[i:i+1].to(DEVICE),
                                  MZR_te_t[i:i+1].to(DEVICE),
                                  PREC_te_t[i:i+1].to(DEVICE))
                        _ = sample_dispatch(_m1, Tmpl_te[i:i+1].to(DEVICE),
                                            K, sample_mode)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                _times_unb.append((time.time() - _t0) * 1000.0)
            _ms_unb = float(_stats.mean(_times_unb))
            _ms_unb_std = float(_stats.stdev(_times_unb)) if len(_times_unb) > 1 else 0.0
            row["wallclock_ms_per_spectrum_unbatched"] = _ms_unb
            row["wallclock_ms_per_spectrum_unbatched_std"] = _ms_unb_std
            row["wallclock_n_unbatched"] = int(_n_unb)
            print(f"PARETO K={K} unbatched_ms_per_spectrum={_ms_unb:.2f}±"
                  f"{_ms_unb_std:.2f} (n={_n_unb})", flush=True)

        # Copy one-time preprocess timing into this row (K-independent).
        if preprocess_ms is not None:
            row["preprocess_ms_per_spectrum"] = float(preprocess_ms)
            row["preprocess_ms_per_spectrum_std"] = float(preprocess_ms_std)
            row["preprocess_n"] = int(preprocess_n)

        pareto.append(row)

    # Dump confidence-calibration logs if LOG_COMMIT_CONF was enabled.
    if os.environ.get("LOG_COMMIT_CONF", "0") in ("1", "true", "yes"):
        _conf_default = "/output/commit_confidences.json" if Path("/output").is_dir() \
            else str(OUT / "commit_confidences.json")
        _conf_path = Path(os.environ.get("LOG_COMMIT_CONF_PATH", _conf_default))
        _conf_path.parent.mkdir(parents=True, exist_ok=True)
        _conf_payload = {
            "K1_argmax_confidences": _CALIBRATION_LOG_K1,
            "multi_step_commit_events": _CALIBRATION_LOG_MULTI,
            "n_K1_events": len(_CALIBRATION_LOG_K1),
            "n_multi_commit_events": len(_CALIBRATION_LOG_MULTI),
        }
        with open(_conf_path, "w") as _f:
            json.dump(_conf_payload, _f)
        print(f"[calibration] wrote {len(_CALIBRATION_LOG_K1)} K=1 events + "
              f"{len(_CALIBRATION_LOG_MULTI)} commit events -> {_conf_path}",
              flush=True)
        push_to_hub_if_configured(_conf_path)

    return pareto

# ============================== ADAPTIVE t=0 ROUTING SWEEP ===================
# Per-spectrum K routing:
#
# 1. Compute c(x) = min over non-special positions of max-token probability
#    at t=eps (one routing forward pass per spectrum, all positions masked).
# 2. Map c(x) to K via a tau-parameterised cascade:
#       c >= tau           -> K = 1
#       c >= 0.9*tau       -> K = 2
#       c >= 0.8*tau       -> K = 4
#       c >= 0.7*tau       -> K = 8
#       c >= 0.6*tau       -> K = 16
#       otherwise          -> K = 32
# 3. Per K bucket, dispatch to a per-K sampler:
#       K=1                -> argmax (one-shot; equivalent to euler_sample(K=1, sample=False))
#       K in {2, 4, 8}     -> soft propagation
#       K in {16, 32}      -> greedy confidence
# 4. Sweep tau in {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}.
#
# NFE per spectrum = 1 (routing pass) + sampler NFE. For K=1 the routing
# pass IS the sampler (its argmax is the prediction), so NFE = 1.

@torch.no_grad()
def run_adaptive_t0_pareto_on_test(X_te_raw, y_te_raw):
    print(f"ADAPTIVE_T0_SWEEP_BEGIN", flush=True)
    X_te_list = to_list(X_te_raw); y_te_list = to_list(y_te_raw)
    y_te_str = [as_str(y) for y in y_te_list]
    N_te = len(X_te_list)
    print(f"X_test n={N_te}", flush=True)

    # Stack test features (mirrors run_pareto_on_test)
    MZ_te = np.zeros((N_te, MZ_LEN), dtype=np.float32)
    MZR_te = np.zeros((N_te, MZ_LEN), dtype=np.float32)
    PREC_te = np.zeros((N_te,), dtype=np.float32)
    for i in range(N_te):
        mz, mz_r, prec = get_feats(X_te_list[i])
        n = min(mz.shape[0], MZ_LEN); MZ_te[i, :n] = mz[:n]
        n = min(mz_r.shape[0], MZ_LEN); MZR_te[i, :n] = mz_r[:n]
        PREC_te[i] = prec
    MZ_te_t   = norm_spec(torch.tensor(MZ_te))
    MZR_te_t  = norm_spec(torch.tensor(MZR_te))
    PREC_te_t = (torch.tensor(PREC_te) - pm) / ps
    Y_te_full = torch.tensor([encode(s) for s in y_te_str], dtype=torch.long)
    Tmpl_te = make_template(Y_te_full)

    # ---------- Routing pass: compute c(x) for every test spectrum ----------
    # We also cache the K=1 argmax prediction since the routing pass IS the
    # K=1 sampler (no extra NFE for K=1 spectra). The cached argmax is used
    # only for spectra that ultimately route to K=1.
    print(f"ROUTING_PASS computing c(x) for all {N_te} spectra...", flush=True)
    eps = 1e-3
    B_route = 4096
    c_all = np.zeros(N_te, dtype=np.float32)
    pred_k1_all = torch.zeros((N_te, MAX_LEN), dtype=torch.long)
    t_route0 = time.time()
    for i in range(0, N_te, B_route):
        mb  = MZ_te_t[i:i+B_route].to(DEVICE)
        mrb = MZR_te_t[i:i+B_route].to(DEVICE)
        pb  = PREC_te_t[i:i+B_route].to(DEVICE)
        tmpl = Tmpl_te[i:i+B_route].to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                            enabled=(DEVICE == "cuda")):
            mem = enc(mb, mrb, pb)
            t1 = torch.full((tmpl.size(0),), 1.0 - eps, device=DEVICE)
            logits = dec(tmpl, t1, mem)
            probs = F.softmax(logits, dim=-1)
        # c(x) = min over non-special positions of max-token probability
        is_non_special = (tmpl == MASK_ID)                                 # [B, L]
        top1 = probs.max(-1).values                                        # [B, L]
        top1 = top1.masked_fill(~is_non_special, float("inf"))
        c_batch = top1.min(-1).values                                       # [B]
        c_all[i:i+tmpl.size(0)] = c_batch.float().cpu().numpy()
        # Cache K=1 argmax decode
        pred = probs.argmax(-1)
        pred = torch.where(is_non_special, pred, tmpl)
        pred_k1_all[i:i+tmpl.size(0)] = pred.cpu()
        if (i // B_route) % 5 == 0:
            print(f"  routing progress={i+tmpl.size(0)}/{N_te}", flush=True)
    print(f"ROUTING_PASS_DONE dt={time.time()-t_route0:.1f}s", flush=True)

    # Per-K best sampler dispatch
    def run_bucket_sampler(K_val, indices, B_eval):
        """Run the empirical-best sampler for this K on the given spectra."""
        N_b = len(indices)
        preds = torch.zeros((N_b, MAX_LEN), dtype=torch.long)
        for j in range(0, N_b, B_eval):
            sub = indices[j:j+B_eval]
            mb  = MZ_te_t[sub].to(DEVICE)
            mrb = MZR_te_t[sub].to(DEVICE)
            pb  = PREC_te_t[sub].to(DEVICE)
            tmpl = Tmpl_te[sub].to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                enabled=(DEVICE == "cuda")):
                mem = enc(mb, mrb, pb)
                if K_val == 1:
                    out = euler_sample(mem, tmpl, K=1, sample=False)
                elif K_val in (2, 4, 8):
                    out = euler_sample_soft(mem, tmpl, K=K_val)
                elif K_val in (16, 32):
                    out = euler_sample_greedy_confidence(mem, tmpl, K=K_val)
                else:
                    raise ValueError(f"Unsupported adaptive-bucket K={K_val}")
            preds[j:j+sub.shape[0]] = out.cpu()
        return preds

    K_VALUES = [1, 2, 4, 8, 16, 32]
    TAU_LIST = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    pareto = []

    for tau in TAU_LIST:
        print(f"ADAPTIVE_T0 tau={tau} bucketing spectra...", flush=True)
        # Cascade routing
        K_per = np.full(N_te, 32, dtype=np.int64)
        K_per[c_all >= 0.6 * tau] = 16
        K_per[c_all >= 0.7 * tau] = 8
        K_per[c_all >= 0.8 * tau] = 4
        K_per[c_all >= 0.9 * tau] = 2
        K_per[c_all >= tau]       = 1

        # Distribution
        budget = {int(K_val): int((K_per == K_val).sum()) for K_val in K_VALUES}
        budget_frac = {str(K_val): budget[K_val] / N_te for K_val in K_VALUES}
        print(f"ADAPTIVE_T0 tau={tau} budget_distribution={budget_frac}", flush=True)

        # Run each bucket and collect predictions
        all_preds = pred_k1_all.clone()  # default to K=1 (cached); overwritten for K>=2
        t_acc0 = time.time()
        for K_val in K_VALUES:
            indices_np = np.where(K_per == K_val)[0]
            if indices_np.size == 0:
                continue
            indices = torch.from_numpy(indices_np)
            if K_val == 1:
                # Already cached in pred_k1_all
                continue
            # Pick a batch size by best-known fit at each K (mirrors prior sweeps)
            B_eval = 4096
            preds_bucket = run_bucket_sampler(K_val, indices, B_eval)
            all_preds[indices] = preds_bucket
            print(f"  tau={tau} K={K_val} n={indices.numel()} done", flush=True)
        # Score the assembled predictions on full test set
        n_struct = 0
        for b in range(N_te):
            pred_str = decode_ids(all_preds[b].tolist())
            true_str = decode_ids(Y_te_full[b].tolist())
            if struct_equal(pred_str, true_str):
                n_struct += 1
        accuracy = n_struct / N_te
        dt_full = time.time() - t_acc0
        # NFE per spectrum: routing pass (1) + sampler NFE per K
        # K=1: routing pass IS the sampler -> NFE=1
        # K>=2: routing + K Euler steps + 1 final commit (rarely needed) ~= K + 2
        avg_nfe = 0.0
        for K_val in K_VALUES:
            n = budget[K_val]
            if K_val == 1:
                per = 1.0
            else:
                per = float(K_val) + 2.0      # 1 routing + K steps + 1 cleanup
            avg_nfe += n * per
        avg_nfe /= N_te

        print(f"ADAPTIVE_T0 tau={tau} accuracy={accuracy:.4f} avg_nfe={avg_nfe:.2f} "
              f"dt_full_sweep={dt_full:.1f}s", flush=True)

        # Wallclock: just report aggregated time per spectrum (not warm-up
        # benchmarked since the bucketing introduces structural variability)
        wallclock_ms = 1000.0 * dt_full / N_te

        pareto.append({
            "sweep_value": tau,
            "metric_name": "tau",
            "accuracy": float(accuracy),
            "avg_nfe_per_spectrum": float(avg_nfe),
            "wallclock_ms_per_spectrum": float(wallclock_ms),
            "budget_distribution": budget_frac,
        })

    return pareto

# ============================== ENTRY POINT ==================================
PARETO_ONLY = os.environ.get("PARETO_ONLY", "").lower() in ("1", "true", "yes")
PARETO_MODE = os.environ.get("PARETO_MODE", "k_sweep").lower()
if PARETO_MODE not in ("k_sweep", "adaptive_t0"):
    raise ValueError(f"Unknown PARETO_MODE={PARETO_MODE!r}; "
                     f"choose k_sweep or adaptive_t0")

def fetch_best_from_hub():
    """Pull best_model.pt from HF_OUTPUT_REPO into CKPT_PATH for PARETO_ONLY runs.
    Uses /tmp for the HF cache because /data is the read-only dataset mount."""
    from huggingface_hub import hf_hub_download
    print(f"PARETO_ONLY: fetching best_model.pt from {HF_OUTPUT_REPO}", flush=True)
    p = hf_hub_download(repo_id=HF_OUTPUT_REPO, filename="best_model.pt",
                        repo_type=HF_OUTPUT_REPO_TYPE, cache_dir="/tmp/hf-models")
    import shutil
    shutil.copy(p, CKPT_PATH)
    print(f"PARETO_ONLY: copied to {CKPT_PATH}", flush=True)

def main():
    if PARETO_ONLY:
        fetch_best_from_hub()
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        best_step  = int(ck.get("step", 0))
        best_epoch = float(ck.get("epoch", 0.0))
        best_struct = float(ck.get("top1_structural_accuracy",
                                   ck.get("val_acc", 0.0)))
        val_tok_acc, val_ppl = float("nan"), float("nan")
        print(f"PARETO_ONLY: loaded best step={best_step} epoch={best_epoch} "
              f"top1_structural_accuracy={best_struct:.4f}", flush=True)
    else:
        best_step, best_epoch, best_struct, val_tok_acc, val_ppl = run_training()

    # Pareto sweep on the test set (the only place X_test is touched)
    print("FETCH_TEST_BEGIN (touched only here, in final eval / Pareto)", flush=True)
    X_te_raw = hf_load("X_test")
    y_te_raw = hf_load("y_test")
    print(f"FETCH_TEST_DONE n={len(X_te_raw)}", flush=True)

    load_best_into_models()
    if PARETO_MODE == "adaptive_t0":
        pareto = run_adaptive_t0_pareto_on_test(X_te_raw, y_te_raw)
    else:
        pareto = run_pareto_on_test(X_te_raw, y_te_raw)

    final_payload = {
        "cell": "xp-dfm-50m",
        "checkpoint": str(CKPT_PATH),
        "pareto_mode": PARETO_MODE,
        "sampler": SAMPLER,
        "temperature": (TEMPERATURE if SAMPLER == "boltzmann" else None),
        "primary_metric": "top1_structural_accuracy",
        "best_top1_structural_accuracy_val": float(best_struct),
        "best_epoch": float(best_epoch),
        "best_step": int(best_step),
        "init_seed": INIT_SEED,
        "hardware": "H200 141GB BF16",
        "training_config": {
            "BATCH": BATCH, "BASE_LR": BASE_LR, "EPOCHS": EPOCHS,
            "WARMUP": WARMUP, "WEIGHT_DECAY": WD, "GRAD_CLIP": GRAD_CLIP,
            "LR_MAX": float(LR_MAX), "LR_FLOOR": float(LR_FLOOR),
            "VAL_SPLIT_SEED": VAL_SPLIT_SEED,
            "ARCH": {"d_model": D_MODEL, "n_heads": N_HEADS,
                     "n_enc": N_ENC, "n_dec": N_DEC,
                     "max_len": MAX_LEN, "vocab": V, "mz_len": MZ_LEN},
        },
        "pareto": pareto,
    }
    # EVAL_OUTPUT overrides the default PARETO_PATH when set by eval.py.
    _eval_output = os.environ.get("EVAL_OUTPUT")
    _write_path = Path(_eval_output) if _eval_output else PARETO_PATH
    _write_path.parent.mkdir(parents=True, exist_ok=True)
    # Tag payload with eval-only provenance if applicable.
    _eval_subset = os.environ.get("EVAL_SUBSET", "all").strip().lower()
    if _eval_subset != "all" or _eval_output:
        final_payload["eval_subset"] = _eval_subset
        final_payload["eval_use_graph_equality"] = (
            os.environ.get("EVAL_USE_GRAPH_EQUALITY", "0") == "1"
        )
        final_payload["eval_limit"] = os.environ.get("EVAL_LIMIT")
    with open(_write_path, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"PARETO_WRITTEN {_write_path}", flush=True)

    # Push end-of-run artifacts to HF (best_model.pt was already pushed on
    # every new best during training).
    # For eval-only runs (EVAL_OUTPUT set), push that path instead of the
    # default PARETO_PATH — otherwise the auto-push looks for the wrong file
    # and logs a spurious warning.
    push_to_hub_if_configured(_write_path)
    if PROGRESS_LOG.exists():
        push_to_hub_if_configured(PROGRESS_LOG)

    # Reported accuracy: K=1 row for k_sweep mode, best-tau row for adaptive.
    # If K=1 isn't in the partial sweep (e.g. K_LIST="3" only), fall back to
    # the best accuracy in the pareto list so METRIC_FINAL still prints.
    if PARETO_MODE == "adaptive_t0":
        k1_acc = max(r["accuracy"] for r in pareto)
    else:
        k1_acc = next((r["accuracy"] for r in pareto if r["sweep_value"] == 1),
                      max((r["accuracy"] for r in pareto), default=float("nan")))
    print(f"METRIC_FINAL "
          f"step={best_step} "
          f"best_top1_structural_accuracy={best_struct:.4f} "
          f"test_top1_structural_accuracy_k1={k1_acc:.4f} "
          f"best_epoch={best_epoch:.2f} "
          f"best_step={best_step} "
          f"final_epoch={EPOCHS} "
          f"val_token_acc={val_tok_acc:.4f} "
          f"val_ppl={val_ppl:.3f}", flush=True)

if __name__ == "__main__":
    main()
