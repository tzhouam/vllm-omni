# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Does the Omni path decode the same tokens as plain vLLM?

The validation plan makes reference consistency a *hard* initial acceptance
condition, and it is the one claim the rest of M0 cannot make for itself.
Everything else the local text mode reports -- the plan, the budget, the
cancellation -- is about the engine's own behaviour. This is about whether the
engine changed the model's answer.

The comparison is **token ids, not text**. Two different id sequences can
detokenize to the same string, and a text comparison would hide exactly the
kind of drift worth catching. Both sides run greedy (`temperature=0`) with
`ignore_eos` so the sequences are the same length and a divergence is a
divergence rather than an early stop.

Both sides also run on the same checkpoint, the same device and the same
request limits, so the only difference is the path: ``vllm.LLM`` straight into
vLLM, against ``AsyncOmni`` → Omni StageRuntime → the stage's EngineCore. A
mismatch here localises to the Omni stage path, which is precisely what M0
added and what nothing else tests.

**What a mismatch would and would not mean.** Greedy decoding is not bitwise
stable across batch shapes -- a different prefill chunking or batch composition
reorders reductions and can flip an argmax at a near-tie. So the first divergent
position is reported, not just a yes/no: a run that agrees for 120 of 128 tokens
and parts on one near-tie is a different finding from one that parts at token 3.

The two engines are loaded **sequentially in one process**, reference first,
with the reference released before the second starts. Two 7.7 GB models
resident at once would not fit the budget this milestone is about respecting.
"""

from __future__ import annotations

import contextlib
import gc
from dataclasses import asdict, dataclass, field
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PromptParity:
    """One prompt, both paths."""

    name: str
    prompt_tokens: int | None
    reference_tokens: int
    engine_tokens: int
    exact: bool
    first_divergence: int | None
    """Index of the first differing id, or ``None`` when they agree. Reported
    even when the lengths differ, so a truncation is distinguishable from a
    disagreement."""
    agreement_prefix: int
    """How many leading ids matched. ``== reference_tokens`` iff exact."""
    reference_head: list[int] = field(default_factory=list)
    engine_head: list[int] = field(default_factory=list)
    """The eight ids around the first divergence, for a look at what changed."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_ids(reference: list[int], engine: list[int]) -> tuple[bool, int | None, int]:
    """``(exact, first_divergence, agreement_prefix)``."""
    prefix = 0
    for a, b in zip(reference, engine):
        if a != b:
            return False, prefix, prefix
        prefix += 1
    if len(reference) != len(engine):
        return False, prefix, prefix
    return True, None, prefix


def summarize(rows: list[PromptParity]) -> dict[str, Any]:
    exact = sum(1 for r in rows if r.exact)
    divergences = [r.agreement_prefix for r in rows if not r.exact]
    return {
        "prompts": len(rows),
        "exact": exact,
        "exact_fraction": (exact / len(rows)) if rows else 0.0,
        "min_agreement_prefix": min(divergences) if divergences else None,
        "mean_agreement_prefix": (sum(divergences) / len(divergences)) if divergences else None,
        "note": (
            "Greedy decoding is not bitwise stable across batch shapes: a "
            "different prefill chunking reorders reductions and can flip an "
            "argmax at a near-tie. agreement_prefix is reported so a late "
            "single-token split is distinguishable from an early structural one."
        ),
    }


def reference_token_ids(
    model_dir: str,
    prompts: list[tuple[str, str]],
    *,
    engine_kwargs: dict[str, Any],
    max_tokens: int,
) -> dict[str, dict[str, Any]]:
    """Greedy decode through plain ``vllm.LLM``, then release the engine.

    ``engine_kwargs`` comes from the same :class:`~vllm_omni.edge.local.plan.
    ExecutionPlan` the Omni side will use, so the two differ in path and not in
    configuration.
    """
    import vllm_omni  # noqa: F401  - registers Spark's arch and config
    from vllm import LLM, SamplingParams

    from vllm_omni.edge.local.engine import apply_runtime_env

    # Applied here, not left to the caller: this function *starts an engine*,
    # and two of the three entries are load-time fatal on this host. A caller
    # that forgot would not get a subtly different comparison, it would get
    # "UVA is not available" -- which is how this line came to exist.
    apply_runtime_env()

    kwargs = dict(engine_kwargs)
    # ``LLM`` does not take Omni's spelling of these; everything else is shared.
    kwargs.pop("kv_cache_memory_bytes", None)
    llm = LLM(model=model_dir, trust_remote_code=False, disable_log_stats=True, **kwargs)
    try:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        outputs = llm.generate([text for _, text in prompts], params, use_tqdm=False)
        return {
            name: {
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
                "prompt_tokens": len(out.prompt_token_ids or ()),
            }
            for (name, _), out in zip(prompts, outputs)
        }
    finally:
        # Release before the Omni engine loads: the budget this milestone is
        # about does not have room for two copies of the weights.
        del llm
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
