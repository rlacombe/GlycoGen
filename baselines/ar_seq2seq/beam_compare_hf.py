# requirements: huggingface_hub glycowork numpy torch
"""A/B the local beam search against a transformers-style
BeamSearchScorer implementation on the same aligned-test slice.

The HF-style implementation:

  1. Each step picks top-2K candidates from the K*V expansion (not just K).
  2. Finished beams (EOS-emitting) are moved to a separate finished pool
     (capped at K per spectrum) and cannot be displaced by active beams.
  3. Active pool always carries K unfinished beams (or fewer near the end).
  4. Final selection: the K-best of finished∪active, ranked by
     length-penalty-normalized score `cum_log_prob / len^alpha` (alpha=1.0).

Outputs per-spectrum agreement counts (identical / structurally
equivalent / different).
"""
import os, time, math, pickle, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

torch.set_float32_matmul_precision("high")
try: torch.backends.cuda.enable_cudnn_sdp(False)
except Exception: pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16
HF_REPO = os.environ.get(
    "HF_DATASET_REPO",
    os.environ.get("HF_REPO", "rlacombe/candycrunch-v2-benchmark"),
)
CKPT_PATH = os.environ.get("CKPT_PATH", "/output/checkpoints/best_model.pt")
OUT_PATH = Path(os.environ.get("OUT_PATH", "/output/beam_compare_hf.json"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", 2000))
BEAM = int(os.environ.get("BEAM", 4))
LENGTH_PENALTY = float(os.environ.get("LENGTH_PENALTY", 1.0))

# ----- glycowork -----
from glycowork.motif.graph import compare_glycans
def struct_equal(a, b):
    if a == b: return True
    try: return bool(compare_glycans(a, b))
    except Exception: return False

# ----- data -----
def hf_pickle(name):
    p = hf_hub_download(repo_id=HF_REPO, filename=name, repo_type="dataset")
    return pickle.loads(Path(p).read_bytes())

print(f"LOAD aligned X_test (first {N_SAMPLES})", flush=True)
X_te = hf_pickle("X_test.pkl")[:N_SAMPLES]
y_te = hf_pickle("y_test.pkl")[:N_SAMPLES]

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

def as_str(y):
    if isinstance(y, str): return y
    if isinstance(y, (list, tuple)) and y and isinstance(y[0], str): return y[0]
    return str(y)
y_te_kept = [as_str(y) for y in y_te]

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

# ----- model (mirrors train.py) -----
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

# ============================================================================
# Variant A: our existing beam search (frozen-finished-in-active, top-K, raw)
# ============================================================================
@torch.no_grad()
def beam_search_ours(mz, mzr, prec, beam, max_len=MAX_LEN):
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
        log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
        ff = finished.reshape(B * K)
        if ff.any():
            log_probs[ff, :] = float('-inf')
            log_probs[ff, PAD_ID] = 0.0
        cand = (scores.reshape(B * K, 1) + log_probs).reshape(B, K * V)
        top_scores, top_idx = cand.topk(K, dim=1)
        beam_idx = top_idx // V; token_idx = top_idx % V
        new_tokens = torch.gather(tokens, 1,
                                  beam_idx.unsqueeze(-1).expand(B, K, max_len)).clone()
        new_tokens[:, :, t] = token_idx
        tokens = new_tokens
        finished = torch.gather(finished, 1, beam_idx) | (token_idx == EOS_ID)
        scores = top_scores
    best_idx = scores.argmax(dim=1)
    return tokens[torch.arange(B), best_idx]

# ============================================================================
# Variant B: HF-style beam search
#   - top 2K expansion per step (filters to K active + tracks K finished)
#   - separate finished pool capped at K
#   - length-penalized final selection (alpha=1.0 default)
# ============================================================================
@torch.no_grad()
def beam_search_hf_style(mz, mzr, prec, beam, max_len=MAX_LEN, alpha=LENGTH_PENALTY):
    B = mz.shape[0]; K = beam
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
        mem = enc(mz, mzr, prec)
    T_enc, d = mem.shape[1], mem.shape[2]
    mem_exp = mem.unsqueeze(1).expand(B, K, T_enc, d).reshape(B * K, T_enc, d)

    # Active pool: K beams of length-grow tokens, scores
    tokens = torch.full((B, K, max_len), PAD_ID, dtype=torch.long, device=DEVICE)
    tokens[:, :, 0] = BOS_ID
    scores = torch.full((B, K), float('-inf'), device=DEVICE)
    scores[:, 0] = 0.0

    # Finished pool (per spectrum): list of (length-norm score, raw score, length, tokens)
    finished_pool = [[] for _ in range(B)]

    def push_finished(b, raw_score, length, beam_tokens):
        ln_score = raw_score / (length ** alpha + 1e-9)
        finished_pool[b].append((ln_score, raw_score, length, beam_tokens))
        finished_pool[b].sort(key=lambda x: -x[0])
        if len(finished_pool[b]) > K:
            finished_pool[b] = finished_pool[b][:K]

    for t in range(1, max_len):
        # Stop when every spectrum has K finished beams (HF early_stopping=True)
        if all(len(finished_pool[b]) >= K for b in range(B)):
            break

        prefix = tokens[:, :, :t].reshape(B * K, t)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logits = dec(prefix, mem_exp)
        log_probs = F.log_softmax(logits[:, -1, :].float(), dim=-1)  # (B*K, V)

        # Mask out beams already at -inf (e.g. if pool shrunk because no candidates)
        cand = (scores.reshape(B * K, 1) + log_probs).reshape(B, K * V)

        # HF: pick top 2K candidates per spectrum
        topk_scores, topk_idx = cand.topk(2 * K, dim=1)  # (B, 2K)
        topk_beam = topk_idx // V                          # (B, 2K)
        topk_token = topk_idx % V                          # (B, 2K)

        new_scores = torch.full((B, K), float('-inf'), device=DEVICE)
        new_tokens_buf = tokens.clone()  # we'll overwrite per spectrum

        for b in range(B):
            slot = 0
            for c in range(2 * K):
                if slot >= K: break
                bi = int(topk_beam[b, c].item())
                ti = int(topk_token[b, c].item())
                sc = float(topk_scores[b, c].item())
                if sc == float('-inf'): break
                if ti == EOS_ID:
                    # Move to finished pool (length includes EOS step but excludes BOS)
                    parent = tokens[b, bi].clone()
                    parent[t] = ti
                    length = t  # number of generated tokens excluding BOS, including EOS
                    push_finished(b, sc, length, parent)
                else:
                    # Keep as active
                    new_tokens_buf[b, slot] = tokens[b, bi]
                    new_tokens_buf[b, slot, t] = ti
                    new_scores[b, slot] = sc
                    slot += 1
            # If we filled fewer than K active slots, the rest stay at -inf
            # so they won't be picked next step.

        tokens = new_tokens_buf
        scores = new_scores

    # Final selection: best length-normalized score from finished∪active per spectrum
    out = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=DEVICE)
    for b in range(B):
        candidates = list(finished_pool[b])
        # Add still-active beams (score them as if they ended here, no EOS appended)
        for k in range(K):
            if scores[b, k].item() > float('-inf'):
                # Find effective length (last non-PAD position + 1 minus BOS)
                row = tokens[b, k]
                # length = number of non-pad tokens minus BOS
                non_pad = (row != PAD_ID).sum().item()
                length = max(non_pad - 1, 1)
                raw = float(scores[b, k].item())
                ln = raw / (length ** alpha + 1e-9)
                candidates.append((ln, raw, length, row.clone()))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        out[b] = candidates[0][3]
    return out

# ----- run both, batched -----
mz_full = MZ_t.to(DEVICE)
mzr_full = MZR_t.to(DEVICE)
prec_full = PREC_t.to(DEVICE)
N = mz_full.shape[0]

def batched(beam_fn, batch_size, name):
    print(f"\n=== {name} beam={BEAM} N={N} bs={batch_size} ===", flush=True)
    all_pred = torch.full((N, MAX_LEN), PAD_ID, dtype=torch.long, device=DEVICE)
    t0 = time.time()
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        mb, mrb, pb = mz_full[sl], mzr_full[sl], prec_full[sl]
        all_pred[sl] = beam_fn(mb, mrb, pb, beam=BEAM)
        if (i // batch_size) % 5 == 0:
            print(f"  {sl.stop}/{N} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  done dt={time.time()-t0:.1f}s", flush=True)
    return all_pred

# Use a smaller batch for hf_style because it has python-level per-spectrum loop
pred_ours = batched(beam_search_ours, 512, "OURS")
pred_hf   = batched(beam_search_hf_style, 64, "HF_STYLE")

# ----- compare -----
str_ours = [decode_ids(pred_ours[b].tolist()) for b in range(N)]
str_hf   = [decode_ids(pred_hf[b].tolist())   for b in range(N)]

correct_ours = 0; correct_hf = 0
identical_strings = 0; struct_equiv_predictions = 0
ours_right_hf_wrong = 0; hf_right_ours_wrong = 0
both_right = 0; both_wrong = 0
for b in range(N):
    truth = y_te_kept[b]
    a_ok = struct_equal(str_ours[b], truth)
    h_ok = struct_equal(str_hf[b], truth)
    correct_ours += int(a_ok)
    correct_hf += int(h_ok)
    if str_ours[b] == str_hf[b]:
        identical_strings += 1
    elif struct_equal(str_ours[b], str_hf[b]):
        struct_equiv_predictions += 1
    if a_ok and not h_ok: ours_right_hf_wrong += 1
    if h_ok and not a_ok: hf_right_ours_wrong += 1
    if a_ok and h_ok: both_right += 1
    if (not a_ok) and (not h_ok): both_wrong += 1

agg = {
    "n": N,
    "beam": BEAM,
    "length_penalty_alpha": LENGTH_PENALTY,
    "accuracy_ours": correct_ours / N,
    "accuracy_hf_style": correct_hf / N,
    "delta_pp": (correct_hf - correct_ours) / N * 100.0,
    "identical_string_predictions": identical_strings,
    "struct_equivalent_predictions_when_strings_differ": struct_equiv_predictions,
    "ours_right_hf_wrong": ours_right_hf_wrong,
    "hf_right_ours_wrong": hf_right_ours_wrong,
    "both_right": both_right,
    "both_wrong": both_wrong,
}
print("\n=== AGGREGATES ===")
for k, v in agg.items():
    print(f"  {k}: {v}")

# Sample of disagreements
print("\n=== sample of disagreements ===")
shown = 0
for b in range(N):
    if shown >= 15: break
    if str_ours[b] != str_hf[b]:
        truth = y_te_kept[b]
        a_ok = struct_equal(str_ours[b], truth)
        h_ok = struct_equal(str_hf[b], truth)
        print(f"  spec {b}: truth={truth}")
        print(f"           ours{'✓' if a_ok else ' '} {str_ours[b]}")
        print(f"           hf  {'✓' if h_ok else ' '} {str_hf[b]}")
        shown += 1

OUT_PATH.write_text(json.dumps(agg, indent=2))
print(f"\nSAVED {OUT_PATH}", flush=True)
print("DONE", flush=True)
