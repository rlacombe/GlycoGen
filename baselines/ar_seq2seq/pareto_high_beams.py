# requirements: huggingface_hub glycowork numpy torch
"""Run beam=16 and beam=32 Pareto sweep on the AR checkpoint with safer
batch caps and the cuDNN MHA backend disabled.

Workarounds for a large-effective-batch (B * K) MHA-graph crash on
H200 + PyTorch 2.x cuDNN SDP kernel:

  1. Lower the OOM-search ceiling per beam: 1024 for beam=16, 512 for beam=32.
     Both keep the effective batch at ~16K.
  2. Disable cuDNN's SDP path so torch falls back to math/flash backends
     for MHA: torch.backends.cuda.enable_cudnn_sdp(False).

Merges the new beam=16/32 rows into the existing pareto.json (preserves
previously logged beam=1/2/4/8 rows).
"""
import os, time, math, pickle, re, json, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

torch.set_float32_matmul_precision("high")
# --- key cuDNN mitigation: disable cuDNN SDP path (it's the source of the
# mha_graph.execute crash at large effective batches on H200 + PyTorch 2.x).
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
    print("CUDNN_SDP disabled", flush=True)
except Exception as e:
    print(f"CUDNN_SDP toggle failed (non-fatal): {e}", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16
HF_REPO = os.environ.get(
    "HF_DATASET_REPO",
    os.environ.get("HF_REPO", "rlacombe/candycrunch-v2-benchmark"),
)
CKPT_PATH = os.environ.get("CKPT_PATH", "/output/checkpoints/best_model.pt")

# Per-beam max-batch ceiling (caps the OOM-search hi); keeps effective
# attention batch (B*K) at ~16K to stay clear of cuDNN limits.
BEAM_BATCH_CEIL = {16: 1024, 32: 512}

# Single-beam mode: each job sweeps one beam value, writes to a per-beam
# file (avoids parallel-write race on the shared pareto.json). The
# orchestrator merges per-beam files into pareto.json at the end.
BEAMS = [int(b) for b in os.environ.get("BEAMS", "16,32").split(",") if b]
PER_BEAM_OUT = Path("/output") / f"pareto_beams_{'_'.join(str(b) for b in BEAMS)}.json"
print(f"BEAMS={BEAMS} OUT={PER_BEAM_OUT}", flush=True)

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

print("LOAD aligned X_test", flush=True)
X_te = hf_pickle("X_test.pkl")
y_te = hf_pickle("y_test.pkl")

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

print("STACK test features...", flush=True); t0 = time.time()
MZ, MZR, PREC = stack(X_te)
print(f"STACK done dt={time.time()-t0:.1f}s", flush=True)

def as_str(y):
    if isinstance(y, str): return y
    if isinstance(y, (list, tuple)) and y and isinstance(y[0], str): return y[0]
    return str(y)
y_te_kept = [as_str(y) for y in y_te]

def norm_spec(x):
    x = torch.log1p(x.clamp_min(0))
    return x / (x.max(-1, keepdim=True).values + 1e-6)

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

# ---------------- model ---------------
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

# ---------------- beam_search ---------------
@torch.no_grad()
def beam_search(mz, mzr, prec, beam, max_len=MAX_LEN):
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
    best_idx = scores.argmax(dim=1)
    return tokens[torch.arange(B), best_idx]

# ---------------- OOM-aware batch finder + timing ---------------
def _try(beam, batch):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        sl = slice(0, batch)
        mb  = MZ_t[sl].to(DEVICE)
        mrb = MZR_t[sl].to(DEVICE)
        pb  = PREC_t[sl].to(DEVICE)
        _ = beam_search(mb, mrb, pb, beam=beam)
        torch.cuda.synchronize()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Catch RuntimeError too — that's how cuDNN's mha_graph.execute fails
        torch.cuda.empty_cache(); gc.collect()
        return False

def find_max_batch(beam, lo=8, hi=None):
    if hi is None: hi = BEAM_BATCH_CEIL.get(beam, 4096)
    print(f"  OOM-search beam={beam} hi_cap={hi} ...", flush=True)
    if not _try(beam, lo):
        print(f"  fail at lo={lo}", flush=True)
        return max(lo // 2, 1)
    while lo < hi and _try(beam, min(hi, lo * 2)):
        lo = min(hi, lo * 2)
    if lo >= hi:
        print(f"  capped at hi={hi}", flush=True)
        return hi
    a, b = lo, min(hi, lo * 2)
    while a + 1 < b:
        m = (a + b) // 2
        if _try(beam, m): a = m
        else: b = m
    print(f"  max-fit batch beam={beam} -> {a}", flush=True)
    return a

def measure_one_batch(beam, batch_size, n_warmup=5):
    sl = slice(0, batch_size)
    mb  = MZ_t[sl].to(DEVICE); mrb = MZR_t[sl].to(DEVICE); pb = PREC_t[sl].to(DEVICE)
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
    acc_bs = max(bs // 2, 1)
    wall_ms_total = measure_one_batch(beam, bs)
    ms_per_spec = wall_ms_total / bs
    print(f"  WALLCLOCK beam={beam} batch={bs} total_ms={wall_ms_total:.1f} per_spec={ms_per_spec:.3f}ms",
          flush=True)
    correct = 0; total = 0; len_sum = 0
    t0 = time.time()
    for i in range(0, len(y_te_kept), acc_bs):
        sl = slice(i, min(i + acc_bs, len(y_te_kept)))
        mb  = MZ_t[sl].to(DEVICE); mrb = MZR_t[sl].to(DEVICE); pb = PREC_t[sl].to(DEVICE)
        pred = beam_search(mb, mrb, pb, beam=beam)
        for b in range(pred.shape[0]):
            seq = pred[b].tolist()
            L = seq.index(EOS_ID) + 1 if EOS_ID in seq else MAX_LEN - 1
            len_sum += L
            truth = y_te_kept[i + b]
            if struct_equal(decode_ids(seq), truth):
                correct += 1
            total += 1
        if (i // acc_bs) % 25 == 0:
            print(f"  PROGRESS beam={beam} {total}/{len(y_te_kept)} acc_so_far={correct/max(total,1):.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    acc = correct / max(total, 1)
    mean_out_len = len_sum / max(total, 1)
    nfe = mean_out_len * beam
    print(f"  RESULT beam={beam} acc={acc:.4f} mean_out_len={mean_out_len:.2f} nfe={nfe:.2f} "
          f"dt={time.time()-t0:.1f}s", flush=True)
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

new_rows = []
for beam in BEAMS:
    try:
        new_rows.append(run_pareto_for_beam(beam))
    except Exception as e:
        print(f"  ERROR beam={beam}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache(); gc.collect()

# Write per-beam result file (orchestrator merges these into pareto.json).
out = {
    "beams": BEAMS,
    "rows": new_rows,
    "cudnn_sdp_disabled": True,
    "beam_batch_ceil": BEAM_BATCH_CEIL,
}
with open(PER_BEAM_OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"PER_BEAM_SAVED {PER_BEAM_OUT}", flush=True)
print("DONE", flush=True)
