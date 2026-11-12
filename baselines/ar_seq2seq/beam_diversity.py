# requirements: huggingface_hub glycowork numpy torch
"""Inspect the K beams per spectrum to check they are actually diverse.

Loads best_model.pt from the AR bucket, runs beam search at beam=4 on
the first ~30 aligned-test spectra, and logs:

  - the 4 decoded strings + cumulative log-prob per beam
  - how many of the 4 are byte-distinct
  - how many resolve via glycowork.compare_glycans to the same glycan
  - whether each beam matches the ground truth (structurally)
"""
import os, time, math, pickle, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16
HF_REPO = os.environ.get(
    "HF_DATASET_REPO",
    os.environ.get("HF_REPO", "rlacombe/candycrunch-v2-benchmark"),
)
CKPT_PATH = os.environ.get("CKPT_PATH", "/output/checkpoints/best_model.pt")
N_SAMPLES = 30   # number of test spectra to inspect

# ---------------- glycowork ---------------
from glycowork.motif.graph import compare_glycans
def struct_equal(a, b):
    if a == b: return True
    try: return bool(compare_glycans(a, b))
    except Exception: return False

# ---------------- data ---------------
def hf_pickle(name):
    p = hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="dataset")
    return pickle.loads(Path(p).read_bytes())

print("LOAD test split (first N)", flush=True)
X_te = hf_pickle("X_test.pkl")
y_te = hf_pickle("y_test.pkl")
N = min(N_SAMPLES, len(X_te))
X_te = X_te[:N]; y_te = y_te[:N]

def get_feats(x):
    mz = np.asarray(x[0], dtype=np.float32).flatten()
    mz_r = np.asarray(x[1], dtype=np.float32).flatten() if len(x) > 1 else np.zeros_like(mz)
    try: prec = float(x[2]) if len(x) > 2 else 0.0
    except (TypeError, ValueError): prec = 0.0
    return mz, mz_r, prec

MZ_LEN = get_feats(X_te[0])[0].shape[0]
def stack(source):
    n = len(source)
    MZ = np.zeros((n, MZ_LEN), dtype=np.float32)
    MZR = np.zeros((n, MZ_LEN), dtype=np.float32)
    PREC = np.zeros((n,), dtype=np.float32)
    for i, x in enumerate(source):
        mz, mz_r, prec = get_feats(x)
        MZ[i, :mz.shape[0]] = mz; MZR[i, :mz_r.shape[0]] = mz_r; PREC[i] = prec
    return MZ, MZR, PREC

MZ, MZR, PREC = stack(X_te)
def norm_spec(x):
    x = torch.log1p(x.clamp_min(0))
    return x / (x.max(-1, keepdim=True).values + 1e-6)

# ---------------- vocab + checkpoint ---------------
print(f"LOAD ckpt {CKPT_PATH}", flush=True)
state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
vocab = state["vocab"]; V = state["V"]; MAX_LEN = state["MAX_LEN"]
PAD_ID, BOS_ID, EOS_ID = vocab["<pad>"], vocab["<bos>"], vocab["<eos>"]
id2tok = {v: k for k, v in vocab.items()}
pm = state["spec_norm"]["prec_mean"]; ps = state["spec_norm"]["prec_std"]

MZ_t = norm_spec(torch.tensor(MZ))
MZR_t = norm_spec(torch.tensor(MZR))
PREC_t = (torch.tensor(PREC) - pm) / ps

def decode_ids(ids):
    out = []
    for i in ids:
        if i in (BOS_ID, PAD_ID): continue
        if i == EOS_ID: break
        out.append(id2tok.get(int(i), "?"))
    return "".join(out)

# ---------------- model (matched to train.py) ---------------
D_MODEL = 512; N_HEADS = 8; N_ENC = 4; N_DEC = 8; N_SPEC_TOKENS = 64

class SpecEncoder(nn.Module):
    def __init__(self, mz_len, d=D_MODEL, n_layers=N_ENC, n_heads=N_HEADS, n_tokens=N_SPEC_TOKENS):
        super().__init__()
        self.mz_len = mz_len; self.n_tokens = n_tokens
        self.chunk = mz_len // n_tokens + (1 if mz_len % n_tokens else 0)
        self.proj = nn.Linear(2 * self.chunk, d)
        self.pos = nn.Embedding(n_tokens + 1, d)
        self.prec_proj = nn.Linear(1, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                           dropout=0.1, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer("ids", torch.arange(n_tokens + 1).unsqueeze(0))

    def forward(self, mz, mz_r, prec):
        B = mz.shape[0]
        pad = self.chunk * self.n_tokens - self.mz_len
        mz_p = F.pad(mz, (0, pad)).view(B, self.n_tokens, self.chunk)
        mzr_p = F.pad(mz_r, (0, pad)).view(B, self.n_tokens, self.chunk)
        h = self.proj(torch.cat([mz_p, mzr_p], dim=-1))
        p = self.prec_proj(prec.view(B, 1, 1)).expand(B, 1, -1)
        h = torch.cat([h, p], dim=1) + self.pos(self.ids.expand(B, -1))
        return self.trunk(h)

class ARDecoder(nn.Module):
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
        self._cache = {}

    def causal_mask(self, L, device):
        key = (L, device)
        m = self._cache.get(key)
        if m is None:
            m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            self._cache[key] = m
        return m

    def forward(self, x, mem):
        B, L = x.shape
        h = self.emb(x) + self.pos(self.ids[:, :L])
        m = self.causal_mask(L, h.device)
        return self.head(self.trunk(h, mem, tgt_mask=m,
                                    tgt_key_padding_mask=(x == PAD_ID)))

enc = SpecEncoder(MZ_LEN).to(DEVICE)
dec = ARDecoder(V).to(DEVICE)
enc.load_state_dict(state["enc"])
dec.load_state_dict(state["dec"])
enc.eval(); dec.eval()
print(f"MODEL loaded V={V} MAX_LEN={MAX_LEN}", flush=True)

# ---------------- beam_search_all (returns ALL K beams) ---------------
@torch.no_grad()
def beam_search_all(mz, mzr, prec, beam, max_len=MAX_LEN):
    """Like beam_search but returns ALL K beams + their cumulative scores.
    Returns (tokens (B,K,max_len), scores (B,K))."""
    B = mz.shape[0]; K = beam
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        mem = enc(mz, mzr, prec)
    T_enc, d = mem.shape[1], mem.shape[2]
    mem_exp = mem.unsqueeze(1).expand(B, K, T_enc, d).reshape(B * K, T_enc, d)
    tokens = torch.full((B, K, max_len), PAD_ID, dtype=torch.long, device=DEVICE)
    tokens[:, :, 0] = BOS_ID
    scores = torch.zeros((B, K), device=DEVICE)
    if K > 1: scores[:, 1:] = float('-inf')
    finished = torch.zeros((B, K), dtype=torch.bool, device=DEVICE)

    for t in range(1, max_len):
        if bool(finished.all()): break
        prefix = tokens[:, :, :t].reshape(B * K, t)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits = dec(prefix, mem_exp)
        next_logits = logits[:, -1, :].float()
        log_probs = F.log_softmax(next_logits, dim=-1)
        finished_flat = finished.reshape(B * K)
        if finished_flat.any():
            log_probs[finished_flat, :] = float('-inf')
            log_probs[finished_flat, PAD_ID] = 0.0
        cand = scores.reshape(B * K, 1) + log_probs
        cand = cand.reshape(B, K * V)
        top_scores, top_idx = cand.topk(K, dim=1)
        beam_idx = top_idx // V
        token_idx = top_idx % V
        new_tokens = torch.gather(tokens, 1,
                                  beam_idx.unsqueeze(-1).expand(B, K, max_len)).clone()
        new_tokens[:, :, t] = token_idx
        tokens = new_tokens
        finished = torch.gather(finished, 1, beam_idx) | (token_idx == EOS_ID)
        scores = top_scores
    return tokens, scores


def as_str(y):
    if isinstance(y, str): return y
    if isinstance(y, (list, tuple)) and y and isinstance(y[0], str): return y[0]
    return str(y)

# ---------------- run + report ---------------
K = 4
print(f"\n=== BEAM-{K} DIVERSITY DIAGNOSTIC on {N} aligned-test spectra ===", flush=True)
mz = MZ_t.to(DEVICE); mzr = MZR_t.to(DEVICE); prec = PREC_t.to(DEVICE)
t0 = time.time()
tokens, scores = beam_search_all(mz, mzr, prec, beam=K)
print(f"BEAM_SEARCH done dt={time.time()-t0:.1f}s\n", flush=True)

# Aggregates
agg = {
    "n": N,
    "all_4_byte_distinct": 0,
    "n_distinct_strings_avg": 0.0,
    "all_4_match_same_glycan": 0,
    "best_beam_matches_truth": 0,
    "any_of_4_beams_matches_truth": 0,
}
distinct_count_total = 0

for b in range(N):
    truth = as_str(y_te[b])
    beams = []
    for k in range(K):
        s = decode_ids(tokens[b, k].tolist())
        beams.append((s, float(scores[b, k])))
    # Pretty print
    print(f"--- spectrum {b} ---")
    print(f"  truth:  {truth}")
    for k, (s, sc) in enumerate(beams):
        match_truth = struct_equal(s, truth)
        flag = "✓" if match_truth else " "
        print(f"  beam{k} {flag} score={sc:8.3f}  {s}")

    # Diversity
    strs = [s for s, _ in beams]
    distinct_strs = len(set(strs))
    distinct_count_total += distinct_strs
    if distinct_strs == K:
        agg["all_4_byte_distinct"] += 1
    # Do all 4 resolve to the same glycan graph?
    same_struct = all(struct_equal(strs[0], s) for s in strs[1:])
    if same_struct:
        agg["all_4_match_same_glycan"] += 1
    # Accuracy
    if struct_equal(strs[0], truth):
        agg["best_beam_matches_truth"] += 1
    if any(struct_equal(s, truth) for s in strs):
        agg["any_of_4_beams_matches_truth"] += 1

agg["n_distinct_strings_avg"] = distinct_count_total / N

print("\n=== AGGREGATES ===")
for k, v in agg.items():
    print(f"  {k}: {v}")
print("DONE", flush=True)
