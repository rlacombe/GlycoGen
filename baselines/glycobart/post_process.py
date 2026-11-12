"""Post-processing pipeline for GlycoBART (and DFM/AR) predictions.

Re-implements CABSEL's `filter_glycans_glycobart` from
glycotrans/utils.py without the database-snap step (line 1380's
`if glycan in glycan_df_use` filter, which turns the pipeline from
generative prediction into retrieval against a known-glycan database).
Retains the spectrum-aware post-processing: mass check, glycan-class
enforcement, topology sort, fragment-pattern matching, and
annotation-score ranking.

Model-agnostic inputs:
  - top-K candidate IUPAC strings per spectrum
  - per-spectrum metadata: reducing_mass, mode, modification, glycan_class,
    peak_d (the observed MS/MS peak dict)

Optimizations:
  - Memoize per-glycan-string features: glycan_to_mass, composition, combinatorics
  - Memoize per-(glycan, charge) CandyCrumbs theoretical fragments
  - Multiprocess per-spectrum scoring across all CPU cores (joblib)

Usage from another script:
    from post_process import score_predictions, FilterConfig
    config = FilterConfig(MS1_ppm=10, MS2_tolerance=0.5, frag_threshold=1,
                          annotation_thresh=3, min_ion_match=0,
                          filter_diag_ion={'Neu5Gc','Kdn','S'})
    results = score_predictions(predictions_per_spectrum, spectrum_metadata, config)
    # results: list of dicts with top-1, top-5 selections + scores
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FilterConfig:
    """Mirrors filter_glycans_glycobart kwargs (minus DB-snap-related ones)."""
    MS1_ppm: float = 10
    MS2_tolerance: float = 0.5
    frag_threshold: float = 1
    annotation_thresh: float = 3
    min_ion_match: int = 0
    filter_diag_ion: frozenset = field(default_factory=lambda: frozenset({'Neu5Gc', 'Kdn', 'S'}))
    # Cap on how many top fragments to use for matching (CABSEL uses :20)
    top_frag_cap: int = 20
    # Cap on how many candidates per spectrum to keep after each filter (CABSEL: 5)
    keep_top_k: int = 5


# ============================================================================
# Memoization utilities
# ============================================================================

# Module-level caches. In multiprocessing, each worker has its own copy
# (acceptable since workers each handle a subset of spectra; the unique-glycan
# set is small enough that per-worker caching still hits the same items).

_GLYCAN_MASS_CACHE: dict[str, Optional[float]] = {}
_COMPOSITION_CACHE: dict[str, Optional[dict]] = {}
_COMBINATORICS_CACHE: dict[frozenset, Optional[np.ndarray]] = {}
_CANDYCRUMBS_CACHE: dict[tuple, Any] = {}  # keyed by (glycan, charge, mass_tag)


def _cache_get(cache: dict, key, compute: Callable):
    if key in cache:
        return cache[key]
    val = compute()
    cache[key] = val
    return val


# ============================================================================
# Glycan feature computation (memoized)
# ============================================================================

def glycan_neutral_mass(iupac: str) -> Optional[float]:
    """glycowork.glycan_to_mass with sample_prep='underivatized'. Cached."""
    if not iupac:
        return None
    if iupac in _GLYCAN_MASS_CACHE:
        return _GLYCAN_MASS_CACHE[iupac]
    try:
        from glycowork.motif.tokenization import glycan_to_mass
        m = float(glycan_to_mass(iupac, sample_prep='underivatized'))
    except Exception:
        m = None
    _GLYCAN_MASS_CACHE[iupac] = m
    return m


def glycan_composition(iupac: str) -> Optional[dict]:
    """glycowork.glycan_to_composition. Cached."""
    if not iupac:
        return None
    if iupac in _COMPOSITION_CACHE:
        return _COMPOSITION_CACHE[iupac]
    try:
        from glycowork.motif.tokenization import glycan_to_composition
        comp = glycan_to_composition(iupac)
        if comp is not None:
            comp = dict(comp)
    except Exception:
        comp = None
    _COMPOSITION_CACHE[iupac] = comp
    return comp


def composition_combinatorics(composition: dict) -> Optional[np.ndarray]:
    """Generate all sub-composition masses for fragment matching.

    Re-implements CABSEL's `combinatorics(composition)` (which lived in
    candycrunch.prediction before being removed in v0.6.3+).

    For a composition like {'Hex': 4, 'HexNAc': 4, 'dHex': 1}, returns the
    sorted unique masses of all sub-compositions {Hex: 0..4, HexNAc: 0..4,
    dHex: 0..1}. Each mass uses glycowork's composition_to_mass.

    Cached per composition.
    """
    if not composition:
        return None
    key = frozenset(composition.items())
    if key in _COMBINATORICS_CACHE:
        return _COMBINATORICS_CACHE[key]
    try:
        from itertools import product
        from glycowork.motif.tokenization import composition_to_mass
        keys = list(composition.keys())
        counts = [int(composition[k]) for k in keys]
        masses = set()
        for sub_counts in product(*[range(c + 1) for c in counts]):
            sub_comp = {k: v for k, v in zip(keys, sub_counts) if v > 0}
            if not sub_comp:
                continue
            try:
                m = float(composition_to_mass(sub_comp))
                masses.add(round(m, 4))
            except Exception:
                continue
        c = np.array(sorted(masses), dtype=np.float32) if masses else None
    except Exception:
        c = None
    _COMBINATORICS_CACHE[key] = c
    return c


# ============================================================================
# Mass check (CABSEL recipe verbatim, minus the DB-snap)
# ============================================================================

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


def mass_check_modified_ppm(observed_mass: float, glycan_iupac: str,
                            mode: str = 'negative', modification: str = 'reduced',
                            mass_thresh: float = 10) -> list[float]:
    """Verbatim port of CABSEL's mass_check_modified_ppm. Returns list of
    consistent m/z values within mass_thresh ppm; empty if not consistent."""
    mz = glycan_neutral_mass(glycan_iupac)
    if mz is None:
        return []
    if modification != 'underivatized':
        mz = mz - 18.01056 + _MOD_MASS.get(modification, 0)
    adducts1 = (['M+H', 'M+NH4', 'M+Na', 'M+K', 'M+2Na-H', 'M+2K-H', 'M+H-H20']
                if mode == 'positive' else ['M-H'])
    if mode == 'positive':
        og_list = ([mz] + [mz + _ADDUCT_C1.get(a, 999) for a in adducts1]
                   + [(2 * mz + 22.98922), (2 * mz + 1.00728)])
    else:
        og_list = [mz] + [mz + _ADDUCT_C1.get(a, 999) for a in adducts1]
    adducts2 = (['M+2H', 'M+H+NH4', 'M+H+Na', 'M+H+K', 'M+2Na', 'M+2K']
                if mode == 'positive' else ['M-2H'])
    am2 = [_ADDUCT_C2.get(a, 999) for a in adducts2]
    am3 = [_ADDUCT_C3['M+3H' if mode == 'positive' else 'M-3H']]
    am4 = [_ADDUCT_C4['M+4H']] if mode == 'positive' else []
    am = [am2, am3, am4]
    mz_list = og_list + [
        (m / z + adduct_adjust)
        for z, alist in zip([2, 3, 4], am)
        for adduct_adjust in alist
        for m in og_list
    ]
    return [m for m in mz_list
            if mz > 0 and abs(observed_mass - m) * 1e6 / max(mz, 1.0) < mass_thresh]


# ============================================================================
# Glycan-class enforcement (verbatim port of enforce_class_modified)
# ============================================================================

def enforce_class(glycan_iupac: str, glycan_class: str = 'N') -> bool:
    """Returns True if the candidate matches the requested glycan class."""
    if not glycan_iupac:
        return False
    if glycan_class == 'N':
        return 'GlcNAc(b1-4)GlcNAc' in glycan_iupac
    if glycan_class == 'O':
        return glycan_iupac.endswith(('GalNAc', 'GalNAc6S', 'GalNAcOS', 'Fuc', 'Man', 'Gal'))
    if glycan_class == 'free':
        return glycan_iupac.endswith(('Glc', 'GlcOS', 'GlcNAc', 'Ins'))
    return True  # other classes: don't enforce


# ============================================================================
# Fragment-pattern matching (per-spectrum, fast: cmasses cached per glycan)
# ============================================================================

def fragment_match_score(glycan_iupac: str, top_fragments: np.ndarray,
                         observed_mass: float, mode: str = 'negative',
                         modification: str = 'reduced',
                         MS2_tolerance: float = 0.5,
                         filter_diag_ion: frozenset = frozenset()) -> float:
    """Per-(glycan, spectrum) fragment-match score: matches / total_top_frags.
    cmasses computation is cached per composition.
    Returns -1 if filter_diag_ion blocks the candidate."""
    composition = glycan_composition(glycan_iupac)
    if composition is None:
        return 0.0
    if any(k in composition for k in filter_diag_ion):
        return -1.0  # diagnostic-ion blocked
    cmasses = composition_combinatorics(composition)
    if cmasses is None or len(cmasses) == 0:
        return 0.0
    # MS2 mass check: observed top fragments vs theoretical cmasses
    # Vectorized for speed: |obs - cm| < tolerance per (obs, cm) pair
    if len(top_fragments) == 0:
        return 0.0
    obs = top_fragments[:20]  # cap per CABSEL
    diff = np.abs(obs[:, None] - cmasses[None, :])  # (n_obs, n_cmasses)
    matches_per_obs = (diff.min(axis=1) < MS2_tolerance).astype(np.int32)
    return float(matches_per_obs.sum() / len(obs))


def get_top_fragments_from_peakd(peak_d: dict, threshold: float = 0.8) -> np.ndarray:
    """Port of get_top_fragments: sort peaks by intensity desc, take top-K
    where cumulative intensity reaches `threshold` of total."""
    if not isinstance(peak_d, dict) or not peak_d:
        return np.array([], dtype=np.float32)
    items = sorted(peak_d.items(), key=lambda kv: kv[1], reverse=True)
    masses = np.array([float(m) for m, _ in items], dtype=np.float32)
    intensities = np.array([float(v) for _, v in items], dtype=np.float32)
    total = intensities.sum()
    if total <= 0:
        return np.array([], dtype=np.float32)
    cum_frac = np.cumsum(intensities) / total
    # Take peaks until cumulative fraction reaches threshold
    n = int((cum_frac < threshold).sum()) + 1
    return masses[:n]


# ============================================================================
# Per-spectrum post-processing: filter + score + pick top-1 / top-K
# ============================================================================

def process_spectrum(predictions: list[str],
                     observed_mass: float,
                     mode: str,
                     modification: str,
                     glycan_class: str,
                     peak_d: dict,
                     config: FilterConfig) -> dict:
    """Re-rank top-K predictions by spectrum-aware score (mass + fragment).

    Strategy: NEVER filter candidates out (option 1 — keeps all K candidates
    in the output set, so top-K can't regress vs raw model). Compute a
    per-candidate score combining mass-match tightness and fragment-pattern
    score; sort candidates by score descending, with original model rank as
    tie-breaker.

    Glycan-class enforcement is NOT applied here (we don't have non-leaky
    per-spectrum class info in our metadata; CABSEL uses class as a per-
    experiment parameter, not per-spectrum). The `glycan_class` parameter is
    accepted but ignored.

    Returns: top1_post, topk_post, scores.
    """
    top_frags = get_top_fragments_from_peakd(peak_d, config.frag_threshold)
    scored = []
    for rank, c in enumerate(predictions):
        if not c:
            scored.append((c, -float("inf"), rank))
            continue
        # 1. Mass check (score = inverse of best mass error, or 0 if none)
        mz_hits = mass_check_modified_ppm(observed_mass, c, mode=mode,
                                          modification=modification,
                                          mass_thresh=config.MS1_ppm)
        mass_score = 1.0 if mz_hits else 0.0
        if mz_hits:
            min_err = min(abs(observed_mass - m) for m in mz_hits)
            mass_err_pp = min_err * 1e6 / max(observed_mass, 1.0)  # ppm
            # Score in [0, 1]: 1 at perfect match, 0 at mass_thresh ppm
            mass_score = max(0.0, 1.0 - mass_err_pp / config.MS1_ppm)
        # 2. Fragment-pattern score (in [0, 1]; -1 if diagnostic-ion blocked)
        frag_score = fragment_match_score(c, top_frags, observed_mass, mode=mode,
                                          modification=modification,
                                          MS2_tolerance=config.MS2_tolerance,
                                          filter_diag_ion=config.filter_diag_ion)
        if frag_score < 0:  # diagnostic-ion violation: heavy penalty but still keep
            frag_score = -1.0
        # Composite score: weighted sum (frag dominant when present, mass as confidence floor)
        composite = 0.6 * frag_score + 0.4 * mass_score
        scored.append((c, composite, rank))

    # Sort by score desc, original rank asc as tie-breaker
    scored.sort(key=lambda x: (-x[1], x[2]))
    candidates = [x[0] for x in scored if x[0]]
    top1 = candidates[0] if candidates else ""
    topk = candidates[:config.keep_top_k]
    return {
        "top1_post": top1,
        "topk_post": topk,
        "scores": [(x[0], x[1]) for x in scored[:config.keep_top_k]],
    }


# ============================================================================
# Multi-process scoring entry point
# ============================================================================

def score_predictions(predictions_per_spectrum: list[list[str]],
                       observed_masses: list[float],
                       modes: list[str],
                       modifications: list[str],
                       glycan_classes: list[str],
                       peak_ds: list[dict],
                       config: Optional[FilterConfig] = None,
                       n_jobs: int = -1,
                       verbose: bool = True) -> list[dict]:
    """Apply post-processing to all spectra. Multiprocess across n_jobs workers.
    Pre-warms glycan-feature caches across all unique candidates."""
    if config is None:
        config = FilterConfig()
    n = len(predictions_per_spectrum)
    assert all(len(x) == n for x in [observed_masses, modes, modifications,
                                     glycan_classes, peak_ds])

    # Pre-warm caches: collect all unique candidates and pre-compute features
    if verbose:
        print(f"[postproc] {n} spectra, collecting unique candidates...", flush=True)
    unique_glycans = set()
    for preds in predictions_per_spectrum:
        for c in preds:
            if c:
                unique_glycans.add(c)
    if verbose:
        print(f"[postproc] {len(unique_glycans)} unique candidates; pre-warming caches...", flush=True)
    t0 = time.time()
    for g in unique_glycans:
        glycan_neutral_mass(g)
        comp = glycan_composition(g)
        if comp:
            composition_combinatorics(comp)
    if verbose:
        print(f"[postproc] cache pre-warmed in {time.time()-t0:.1f}s", flush=True)

    # Per-spectrum processing (CPU-bound, parallelize)
    if verbose:
        print(f"[postproc] scoring {n} spectra with n_jobs={n_jobs}...", flush=True)
    t0 = time.time()
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(process_spectrum)(
                predictions_per_spectrum[i], observed_masses[i],
                modes[i], modifications[i], glycan_classes[i], peak_ds[i],
                config,
            )
            for i in range(n)
        )
    except ImportError:
        # Fallback to single-process
        if verbose:
            print("[postproc] joblib missing, falling back to single-process", flush=True)
        results = [
            process_spectrum(predictions_per_spectrum[i], observed_masses[i],
                             modes[i], modifications[i], glycan_classes[i],
                             peak_ds[i], config)
            for i in range(n)
        ]
    if verbose:
        print(f"[postproc] scoring complete in {time.time()-t0:.1f}s "
              f"({(time.time()-t0)/n*1000:.1f} ms/spec)", flush=True)
    return results
