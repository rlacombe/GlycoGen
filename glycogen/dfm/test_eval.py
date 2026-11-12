"""Unit tests for eval.py pure functions.

Run:
    python -m pytest test_eval.py -v

These tests exercise the pure helpers in eval.py (subset filtering, K spec
parsing, seen/unseen mask computation). They do NOT touch HF, download
checkpoints, or run inference — those are integration-level and require
GPU/network.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval import (
    compute_seen_mask,
    filter_by_mask,
    normalize_sampler,
    parse_k_spec,
    parse_subset_arg,
    resolve_config,
    validate_sampler,
    DEFAULT_SAMPLER,
    SAMPLER_ALIASES,
    SUBSET_CHOICES,
    DEFAULT_PARETO_K,
    VALID_SAMPLERS,
)


# ------------------------------------------------------------------
# compute_seen_mask — pure function, no HF, no glycowork if string mode
# ------------------------------------------------------------------

class TestComputeSeenMask:
    def test_all_seen_string(self):
        y_train = ["Glc", "Gal", "Man"]
        y_test = ["Glc", "Gal", "Man"]
        mask = compute_seen_mask(y_test, y_train, use_graph_equality=False)
        assert mask.dtype == bool
        assert mask.tolist() == [True, True, True]

    def test_all_unseen_string(self):
        y_train = ["Glc", "Gal"]
        y_test = ["Xyl", "Rha"]
        mask = compute_seen_mask(y_test, y_train, use_graph_equality=False)
        assert mask.tolist() == [False, False]

    def test_mixed_seen_unseen_string(self):
        y_train = ["A", "B", "C"]
        y_test = ["A", "X", "B", "Y", "A"]
        mask = compute_seen_mask(y_test, y_train, use_graph_equality=False)
        assert mask.tolist() == [True, False, True, False, True]

    def test_duplicates_in_train(self):
        y_train = ["A", "A", "B", "B", "C"]
        y_test = ["A", "D"]
        mask = compute_seen_mask(y_test, y_train, use_graph_equality=False)
        assert mask.tolist() == [True, False]

    def test_empty_test(self):
        mask = compute_seen_mask([], ["A", "B"], use_graph_equality=False)
        assert len(mask) == 0

    def test_empty_train(self):
        mask = compute_seen_mask(["A", "B"], [], use_graph_equality=False)
        assert mask.tolist() == [False, False]

    def test_graph_equality_string_match_shortcut(self):
        # When strings match, graph equality should also return True
        # without invoking glycowork (fast path).
        y_train = ["Neu5Ac(a2-6)Gal(b1-4)GlcNAc"]
        y_test = ["Neu5Ac(a2-6)Gal(b1-4)GlcNAc"]
        mask = compute_seen_mask(y_test, y_train, use_graph_equality=True)
        assert mask.tolist() == [True]


# ------------------------------------------------------------------
# filter_by_mask
# ------------------------------------------------------------------

class TestFilterByMask:
    def test_basic(self):
        items = ["a", "b", "c", "d"]
        mask = np.array([True, False, True, False], dtype=bool)
        assert filter_by_mask(items, mask) == ["a", "c"]

    def test_all_true(self):
        items = [1, 2, 3]
        mask = np.array([True, True, True], dtype=bool)
        assert filter_by_mask(items, mask) == [1, 2, 3]

    def test_all_false(self):
        items = [1, 2, 3]
        mask = np.array([False, False, False], dtype=bool)
        assert filter_by_mask(items, mask) == []

    def test_empty(self):
        assert filter_by_mask([], np.array([], dtype=bool)) == []

    def test_length_mismatch_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            filter_by_mask([1, 2, 3], np.array([True, False], dtype=bool))


# ------------------------------------------------------------------
# parse_k_spec
# ------------------------------------------------------------------

class TestParseKSpec:
    def test_single_k(self):
        assert parse_k_spec("1") == [1]
        assert parse_k_spec("8") == [8]

    def test_comma_list(self):
        assert parse_k_spec("1,8,32") == [1, 8, 32]
        assert parse_k_spec("1, 8, 32") == [1, 8, 32]  # tolerant of spaces

    def test_pareto_default(self):
        assert parse_k_spec("pareto") == DEFAULT_PARETO_K
        # sensible default: at least K=1 and a mid-range K
        assert 1 in DEFAULT_PARETO_K
        assert 8 in DEFAULT_PARETO_K or 16 in DEFAULT_PARETO_K

    def test_case_insensitive_pareto(self):
        assert parse_k_spec("PARETO") == DEFAULT_PARETO_K
        assert parse_k_spec("Pareto") == DEFAULT_PARETO_K

    def test_invalid_empty(self):
        with pytest.raises(ValueError):
            parse_k_spec("")

    def test_invalid_nonint(self):
        with pytest.raises(ValueError):
            parse_k_spec("foo")

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            parse_k_spec("-1")

    def test_invalid_zero(self):
        with pytest.raises(ValueError):
            parse_k_spec("0")


# ------------------------------------------------------------------
# parse_subset_arg
# ------------------------------------------------------------------

class TestParseSubsetArg:
    def test_valid_choices(self):
        for choice in SUBSET_CHOICES:
            assert parse_subset_arg(choice) == choice

    def test_case_insensitive(self):
        assert parse_subset_arg("ALL") == "all"
        assert parse_subset_arg("Unseen") == "unseen"

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_subset_arg("bogus")


# ------------------------------------------------------------------
# validate_sampler
# ------------------------------------------------------------------

class TestValidateSampler:
    def test_valid_samplers(self):
        assert validate_sampler("crystallizing_flows") == "crystallizing_flows"
        assert validate_sampler("ctmc_euler") == "ctmc_euler"
        assert validate_sampler("soft_auto_commit_ee") == "soft_auto_commit_ee"
        assert validate_sampler("algorithm1") == "algorithm1"

    def test_default_sampler_is_crystallizing_flows(self):
        assert DEFAULT_SAMPLER == "crystallizing_flows"
        cfg = resolve_config([])
        assert cfg["sampler"] == "crystallizing_flows"

    def test_legacy_aliases_accepted_and_mapped(self):
        assert SAMPLER_ALIASES["soft_auto_commit_ee"] == "crystallizing_flows"
        assert SAMPLER_ALIASES["algorithm1"] == "ctmc_euler"
        assert normalize_sampler("soft_auto_commit_ee") == "crystallizing_flows"
        assert normalize_sampler("crystallizing_flows") == "crystallizing_flows"
        assert normalize_sampler("algorithm1") == "ctmc_euler"
        assert normalize_sampler("ctmc_euler") == "ctmc_euler"
        assert resolve_config(["--sampler", "soft_auto_commit_ee"])["sampler"] == "soft_auto_commit_ee"
        assert resolve_config(["--sampler", "algorithm1"])["sampler"] == "algorithm1"
        assert resolve_config(["--sampler", "crystallizing_flows"])["sampler"] == "crystallizing_flows"
        assert resolve_config(["--sampler", "ctmc_euler"])["sampler"] == "ctmc_euler"

    def test_all_documented_samplers(self):
        for s in VALID_SAMPLERS:
            assert validate_sampler(s) == s

    def test_invalid_sampler_raises(self):
        with pytest.raises(ValueError):
            validate_sampler("not_a_sampler")
        with pytest.raises(ValueError):
            normalize_sampler("not_a_sampler")


# ------------------------------------------------------------------
# End-to-end mask+filter integration (still pure, no HF/GPU)
# ------------------------------------------------------------------

class TestMaskFilterIntegration:
    def test_unseen_subset_pipeline(self):
        y_train = ["A", "B", "C"]
        y_test = ["A", "X", "B", "Y", "A", "Z"]
        X_test = [f"spec_{i}" for i in range(len(y_test))]  # dummy features

        seen_mask = compute_seen_mask(y_test, y_train)
        unseen_mask = ~seen_mask

        X_unseen = filter_by_mask(X_test, unseen_mask)
        y_unseen = filter_by_mask(y_test, unseen_mask)

        assert X_unseen == ["spec_1", "spec_3", "spec_5"]
        assert y_unseen == ["X", "Y", "Z"]

    def test_seen_subset_pipeline(self):
        y_train = ["A", "B"]
        y_test = ["A", "X", "B", "Y"]
        X_test = [1, 2, 3, 4]

        seen_mask = compute_seen_mask(y_test, y_train)
        X_seen = filter_by_mask(X_test, seen_mask)
        y_seen = filter_by_mask(y_test, seen_mask)

        assert X_seen == [1, 3]
        assert y_seen == ["A", "B"]
