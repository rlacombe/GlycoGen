# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy<2",
#   "huggingface_hub>=0.24",
#   "glycowork",
# ]
# ///
"""
GlycoGen-DFM eval-only CLI wrapper.

Loads the 50M checkpoint from HF and runs Crystallizing Flows inference
on the leak-free CandyCrunch v2 test set. No training.

Args:
  --subset {all, seen, unseen}   subset of the test set to evaluate on
                                  - 'seen'   : test spectra whose GT structure
                                                appears somewhere in y_train
                                  - 'unseen' : structures never in y_train
                                                (the ~0.6% novel-structure subset)
                                  - 'all'    : full test set
  --k SPEC                       single K ("1", "8"), comma list ("1,8,32"),
                                  or "pareto" for the default sweep
                                  {1, 2, 4, 8, 16, 32, 64}
  --sampler NAME                 sampler function; default 'crystallizing_flows'
                                  (Crystallizing Flows with early exit; legacy
                                  aliases 'soft_auto_commit_ee' and 'algorithm1'
                                  are preserved for compatibility)
  --use-graph-equality           for seen/unseen subset classification, use
                                  glycowork.compare_glycans (slow but exact)
                                  rather than pure string equality (fast)
  --limit N                      cap total test spectra before inference (smoke tests)
  --output PATH                  where to write results JSON

Implementation: wraps train.py's PARETO_ONLY code path. train.py reads
EVAL_SUBSET / EVAL_USE_GRAPH_EQUALITY env vars set here to filter its
test set before the Pareto sweep. The inference code path (sampler,
model architecture, HF checkpoint download) is train.py's; this file
adds only the subset filtering and CLI ergonomics.

Usage examples:
    # K=1 on unseen structures only (~570 spectra)
    python eval.py --subset unseen --k 1 --output unseen_k1.json

    # Pareto sweep on the unseen subset
    python eval.py --subset unseen --k pareto --output unseen_pareto.json

    # Full test set
    python eval.py --subset all --k pareto --output paper_reproduction.json

Test with:
    python -m pytest test_eval.py -v
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np


# =====================================================================
# Constants
# =====================================================================
SUBSET_CHOICES = ("all", "seen", "unseen")

# Default K sweep.
DEFAULT_PARETO_K = [1, 2, 4, 8, 16, 32, 64]

# Samplers implemented in train.py::sample_dispatch().
# First-class names: 'crystallizing_flows' (Crystallizing Flows with early exit)
# and 'ctmc_euler' (standard CTMC Euler unmasking). Legacy internal aliases
# 'soft_auto_commit_ee' and 'algorithm1' remain valid for backward compatibility.
VALID_SAMPLERS = (
    "crystallizing_flows",     # Crystallizing Flows (soft propagation + auto commit + early exit)
    "ctmc_euler",              # standard CTMC Euler unmasking baseline
    "algorithm1",              # backward-compatible alias for ctmc_euler
    "soft_auto_commit_ee",     # backward-compatible alias for crystallizing_flows
    "soft_auto_commit",        # CF without per-spectrum early exit
    "soft_propagation",        # soft propagation, no commits
    "greedy_confidence",       # MaskGIT-style top-N confidence commits
    "greedy_soft",             # greedy commit on soft state
    "velocity",                # velocity-explicit Euler
    "velocity_argmax_last",    # velocity + final-step argmax
    "all_argmax",              # argmax at every step, no sampling
    "boltzmann",               # temperature-scaled softmax sampling
    "heun",                    # 2nd-order predictor-corrector
)

SAMPLER_ALIASES = {
    "soft_auto_commit_ee": "crystallizing_flows",
    "algorithm1": "ctmc_euler",
}

DEFAULT_SAMPLER = "crystallizing_flows"

HF_DATASET_REPO = os.environ.get(
    "HF_DATASET_REPO", "rlacombe/candycrunch-v2-benchmark"
)
HF_MODEL_REPO = os.environ.get(
    "HF_OUTPUT_REPO", "rlacombe/glycogen-dfm-50m"
)

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


# =====================================================================
# Pure functions (unit-tested in test_eval.py)
# =====================================================================
def compute_seen_mask(
    y_test: Sequence[str],
    y_train: Sequence[str],
    use_graph_equality: bool = False,
) -> np.ndarray:
    """Boolean mask: True at index i iff y_test[i] shares its structure with
    at least one y_train entry.

    Fast path: string equality (`y in set(y_train)`).
    Slow path (`use_graph_equality=True`): for each y_test not string-matched
    in train, probe glycowork.compare_glycans against every unique y_train.
    """
    train_set = set(y_train)
    mask = np.array([y in train_set for y in y_test], dtype=bool)
    if not use_graph_equality:
        return mask

    try:
        from glycowork.motif.graph import compare_glycans
    except ImportError as e:
        raise RuntimeError(
            "use_graph_equality=True requires glycowork; install with "
            "`pip install glycowork`."
        ) from e

    train_list = sorted(train_set)
    # Only need to probe test items not already string-matched. Cache per
    # unique test string to avoid recomputation.
    resolved: dict[str, bool] = {}
    for i, y_te in enumerate(y_test):
        if mask[i]:
            continue
        if y_te in resolved:
            mask[i] = resolved[y_te]
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
        mask[i] = found
    return mask


def filter_by_mask(items: Sequence, mask: np.ndarray) -> list:
    """Return the subset of `items` where `mask` is True."""
    if len(items) != len(mask):
        raise ValueError(
            f"length mismatch: len(items)={len(items)} len(mask)={len(mask)}"
        )
    return [x for x, keep in zip(items, mask) if bool(keep)]


def parse_k_spec(spec: str) -> list[int]:
    """Parse the --k argument.

    Accepts:
      - a single int as string: "1", "8"
      - a comma-separated list: "1,8,32"
      - the literal "pareto" (any case): expands to DEFAULT_PARETO_K
    Rejects: empty string, non-integer tokens, non-positive integers.
    """
    if spec is None or not spec.strip():
        raise ValueError("--k cannot be empty")
    if spec.strip().lower() == "pareto":
        return list(DEFAULT_PARETO_K)
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError(f"empty token in --k spec: {spec!r}")
        try:
            k = int(tok)
        except ValueError as e:
            raise ValueError(f"invalid K in --k spec: {tok!r}") from e
        if k <= 0:
            raise ValueError(f"K must be a positive integer, got {k}")
        out.append(k)
    return out


def parse_subset_arg(value: str) -> str:
    """Normalize --subset argument."""
    if value is None:
        raise ValueError("--subset must be provided")
    normalized = value.strip().lower()
    if normalized not in SUBSET_CHOICES:
        raise ValueError(
            f"invalid --subset {value!r}; expected one of {SUBSET_CHOICES}"
        )
    return normalized


def validate_sampler(value: str) -> str:
    """Validate --sampler name against VALID_SAMPLERS."""
    if value not in VALID_SAMPLERS:
        raise ValueError(
            f"invalid --sampler {value!r}; expected one of {VALID_SAMPLERS}"
        )
    return value


def normalize_sampler(value: str) -> str:
    """Validate --sampler and map legacy aliases to primary public names."""
    validated = validate_sampler(value)
    return SAMPLER_ALIASES.get(validated, validated)


# =====================================================================
# CLI
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval.py",
        description="Eval-only wrapper for GlycoGen DFM (50M).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--subset",
        default="all",
        help="which test-set subset to evaluate: all / seen / unseen (structure)",
    )
    p.add_argument(
        "--k",
        default="1",
        help="K value(s): '1', '1,8,32', or 'pareto' for the default sweep",
    )
    p.add_argument(
        "--sampler",
        default=DEFAULT_SAMPLER,
        help="inference sampler; see VALID_SAMPLERS in eval.py",
    )
    p.add_argument(
        "--use-graph-equality",
        action="store_true",
        help=(
            "use glycowork.compare_glycans for seen/unseen classification "
            "(exact but slower); default is fast string equality"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap test spectra to this many before inference (smoke tests)",
    )
    p.add_argument(
        "--output",
        default="eval_output.json",
        help="path to write results JSON",
    )
    p.add_argument(
        "--hf-model-repo",
        default=HF_MODEL_REPO,
        help="HF model repo to pull best_model.pt from",
    )
    p.add_argument(
        "--hf-dataset-repo",
        default=HF_DATASET_REPO,
        help="HF dataset repo for X_test / y_test / y_train",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved config and exit without running train.py",
    )
    return p


def resolve_config(argv: list[str] | None = None) -> dict:
    """Parse and validate CLI args into a resolved config dict."""
    ns = build_parser().parse_args(argv)
    return {
        "subset": parse_subset_arg(ns.subset),
        "k_values": parse_k_spec(ns.k),
        "sampler": validate_sampler(ns.sampler),
        "use_graph_equality": bool(ns.use_graph_equality),
        "limit": ns.limit,
        "output": ns.output,
        "hf_model_repo": ns.hf_model_repo,
        "hf_dataset_repo": ns.hf_dataset_repo,
        "dry_run": ns.dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    cfg = resolve_config(argv)

    print("=== eval.py resolved config ===", flush=True)
    for k, v in cfg.items():
        print(f"  {k}: {v}", flush=True)
    print("=" * 32, flush=True)

    if cfg["dry_run"]:
        print("[dry-run] not launching train.py", flush=True)
        return 0

    # Build the env for train.py. train.py's PARETO_ONLY code path:
    #   - pulls best_model.pt from HF_OUTPUT_REPO
    #   - skips run_training()
    #   - runs run_pareto_on_test() on the full test set
    # We add EVAL_SUBSET / EVAL_USE_GRAPH_EQUALITY / EVAL_LIMIT / EVAL_OUTPUT
    # env vars that train.py will read (patched section in run_pareto_on_test).
    env = os.environ.copy()
    env["PARETO_ONLY"] = "1"
    env["SAMPLER"] = cfg["sampler"]
    env["K_LIST"] = ",".join(str(k) for k in cfg["k_values"])
    env["HF_OUTPUT_REPO"] = cfg["hf_model_repo"]
    env["HF_DATASET_REPO"] = cfg["hf_dataset_repo"]
    env["EVAL_SUBSET"] = cfg["subset"]
    env["EVAL_USE_GRAPH_EQUALITY"] = "1" if cfg["use_graph_equality"] else "0"
    if cfg["limit"] is not None:
        env["EVAL_LIMIT"] = str(cfg["limit"])
    env["EVAL_OUTPUT"] = cfg["output"]

    if not TRAIN_SCRIPT.exists():
        print(f"[error] train.py not found at {TRAIN_SCRIPT}", file=sys.stderr)
        return 2

    print(f"[launch] python {TRAIN_SCRIPT} (PARETO_ONLY=1, EVAL_SUBSET="
          f"{cfg['subset']}, K_LIST={env['K_LIST']}, SAMPLER={cfg['sampler']})",
          flush=True)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        env=env,
        check=False,
    )
    dt = time.time() - t0
    print(f"[done] train.py exit={result.returncode} dt={dt:.1f}s", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
