"""Tests for the fidelity metric's arithmetic, which is where it can lie.

Greedy divergence saturates; a KL that silently reports 0 for a broken build
would be worse than saturating, so identical / shifted / disagreeing cases are
pinned here.
"""

import math

import pytest

from fidelity_kl import chunk_tokens, compare


def _pos(target, dist):
    """dist: {token_id: probability} -> a position record in log space."""
    return {"target": target,
            "logprobs": {t: math.log(p) for t, p in dist.items()}}


def test_identical_distributions_score_zero_kl_and_full_agreement():
    ref = [_pos(1, {1: 0.7, 2: 0.2, 3: 0.1})]
    out = compare(ref, [dict(p) for p in ref])
    assert out["top1_agreement"] == 1.0
    assert out["kl_mean"] == pytest.approx(0.0, abs=1e-9)
    assert out["delta_logprob_mean"] == pytest.approx(0.0, abs=1e-9)


def test_a_shifted_distribution_scores_positive_kl_without_changing_top1():
    """The case greedy divergence cannot see: same argmax, worse distribution."""
    ref = [_pos(1, {1: 0.7, 2: 0.2, 3: 0.1})]
    cand = [_pos(1, {1: 0.5, 2: 0.3, 3: 0.2})]
    out = compare(ref, cand)
    assert out["top1_agreement"] == 1.0
    assert out["kl_mean"] > 0.01
    assert out["delta_logprob_mean"] < 0      # reference token got less likely


def test_disagreement_is_counted():
    ref = [_pos(1, {1: 0.7, 2: 0.3}), _pos(1, {1: 0.7, 2: 0.3})]
    cand = [_pos(1, {1: 0.7, 2: 0.3}), _pos(1, {2: 0.8, 1: 0.2})]
    assert compare(ref, cand)["top1_agreement"] == 0.5


def test_worse_builds_score_monotonically_worse():
    """The property greedy divergence lacks: it must rank, not just detect."""
    ref = [_pos(1, {1: 0.8, 2: 0.15, 3: 0.05})]
    near = compare(ref, [_pos(1, {1: 0.78, 2: 0.16, 3: 0.06})])["kl_mean"]
    far = compare(ref, [_pos(1, {1: 0.40, 2: 0.35, 3: 0.25})])["kl_mean"]
    assert 0 < near < far


def test_coverage_is_reported_so_a_lower_bound_is_not_read_as_a_kl():
    """Top-k truncation makes KL a lower bound; the run has to say so."""
    ref = [_pos(1, {1: 0.3, 2: 0.2})]          # only 0.5 of the mass covered
    out = compare(ref, [_pos(1, {1: 0.3, 2: 0.2})])
    assert out["coverage_mean"] == pytest.approx(0.5, abs=1e-6)


def test_mismatched_corpora_are_refused_not_averaged():
    with pytest.raises(SystemExit):
        compare([_pos(1, {1: 1.0})], [])


def test_chunking_drops_only_a_tail_too_short_to_score():
    # 488 is a perfectly good chunk; both runs see the same slicing.
    assert [len(c) for c in chunk_tokens(list(range(1000)), 512, None)] == [512, 488]
    # 8 leftover tokens is not, so it is dropped rather than compared.
    assert [len(c) for c in chunk_tokens(list(range(520)), 512, None)] == [512]
    assert len(chunk_tokens(list(range(4096)), 512, 3)) == 3
