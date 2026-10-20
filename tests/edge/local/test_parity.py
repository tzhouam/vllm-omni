# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference consistency: does the Omni stage path change the model's answer?

The comparison logic is unit-tested here; the run that actually loads two
engines is at the bottom and skips without a checkpoint. It is separated from
``test_e2e_local_text`` because it cannot share that module's engine fixture --
it needs the reference engine loaded and released *before* the Omni one, and
two copies of a 7.7 GB checkpoint do not fit the budget this milestone is about.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vllm_omni.edge.local.parity import compare_ids, summarize
from vllm_omni.edge.local.plan import plan_text_session

MAX_MODEL_LEN = 4096


# ------------------------------------------------------------------- unit
@pytest.mark.core_model
@pytest.mark.cpu
class TestCompare:
    def test_identical_sequences_are_exact(self):
        assert compare_ids([1, 2, 3], [1, 2, 3]) == (True, None, 3)

    def test_a_divergence_reports_where(self):
        assert compare_ids([1, 2, 3, 4], [1, 2, 9, 4]) == (False, 2, 2)

    def test_a_short_engine_sequence_is_not_exact(self):
        """A truncation is a divergence at the end, not agreement."""
        assert compare_ids([1, 2, 3], [1, 2]) == (False, 2, 2)

    def test_a_long_engine_sequence_is_not_exact(self):
        assert compare_ids([1, 2], [1, 2, 3]) == (False, 2, 2)

    def test_divergence_at_the_first_token(self):
        assert compare_ids([5, 6], [7, 6]) == (False, 0, 0)

    def test_empty_sequences_agree_vacuously(self):
        assert compare_ids([], []) == (True, None, 0)

    def test_summary_counts_and_reports_the_worst_prefix(self):
        from vllm_omni.edge.local.parity import PromptParity

        rows = [
            PromptParity("a", 4, 8, 8, True, None, 8),
            PromptParity("b", 4, 8, 8, False, 5, 5),
            PromptParity("c", 4, 8, 8, False, 1, 1),
        ]
        stats = summarize(rows)
        assert stats["exact"] == 1
        assert stats["prompts"] == 3
        assert stats["min_agreement_prefix"] == 1
        assert stats["mean_agreement_prefix"] == 3.0

    def test_an_all_exact_summary_has_no_divergence_stats(self):
        from vllm_omni.edge.local.parity import PromptParity

        stats = summarize([PromptParity("a", 4, 8, 8, True, None, 8)])
        assert stats["exact_fraction"] == 1.0
        assert stats["min_agreement_prefix"] is None


# -------------------------------------------------------------------- e2e
def _models_root() -> Path:
    return Path(__file__).resolve().parents[3].parent / "models"


def _platform() -> str:
    from vllm.platforms import current_platform

    return str(getattr(current_platform, "device_type", "") or "")


def _find_model() -> str:
    override = os.environ.get("VLLM_OMNI_LOCAL_TEST_MODEL")
    if override:
        return override
    candidates = ("Spark-X2.5-4B",) if _platform() == "cuda" else ("Spark-X2.5-1.7B-int8",)
    for name in candidates:
        if (_models_root() / name / "config.json").is_file():
            return str(_models_root() / name)
    pytest.skip(f"no Spark checkpoint for platform {_platform()!r}")


@pytest.mark.local_model
@pytest.mark.asyncio(loop_scope="module")
async def test_the_omni_path_decodes_what_plain_vllm_decodes():
    """The hard acceptance condition: same checkpoint, same limits, same
    greedy decode -- the only difference is the path through the Omni stage.

    Four prompts rather than the full twelve: this loads two engines and the
    point is to detect a systematic difference in the stage path, which four
    prompts of different shapes (short chat, tool-formatted, long, plain) show
    as well as twelve. The full set runs from the CLI's ``parity`` command.
    """
    from vllm_omni.edge.local.cli import _engine_token_ids
    from vllm_omni.edge.local.parity import reference_token_ids
    from vllm_omni.edge.local.prompts import acceptance_prompts

    model_dir = _find_model()
    plan = plan_text_session(model_dir, max_model_len=MAX_MODEL_LEN, max_num_seqs=1)
    if not plan.admitted:
        pytest.skip(f"this host cannot run {model_dir}")

    all_prompts = dict(acceptance_prompts())
    prompts = [(n, all_prompts[n]) for n in ("chat_short", "chat_tools", "chat_long", "plain_0")]
    max_tokens = 64

    reference = reference_token_ids(
        model_dir, prompts, engine_kwargs=plan.engine_kwargs, max_tokens=max_tokens
    )
    engine = await _engine_token_ids(plan, prompts, max_tokens)

    mismatches = []
    for name, _ in prompts:
        ref = reference[name]["token_ids"]
        eng = engine[name]["token_ids"]
        assert len(ref) == max_tokens, f"{name}: reference produced {len(ref)}"
        exact, first, prefix = compare_ids(ref, eng)
        if not exact:
            mismatches.append(f"{name}: diverged at {first}/{len(ref)} (prefix {prefix})")
    assert not mismatches, "the Omni stage path changed the decode:\n" + "\n".join(mismatches)
