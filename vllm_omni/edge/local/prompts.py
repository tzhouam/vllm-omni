# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fixed prompt set the M0 acceptance run uses, and why it is this one.

The roadmap asks for "同一组 12 个提示至少生成 128 token" -- one fixed set of 12
prompts, at least 128 tokens each. The set is not new: it is the same twelve
this repo already measures Spark's quantized builds against
(``analysis/experiments/spark_edge/quality_int4.py``), which is the point. A
new prompt set would make M0's numbers incomparable with the W4A16/W4A8/Q4_K_M
fidelity work that produced "2/12 and 1/12 greedy", and comparability is the
only reason to fix a set at all.

It is three chat prompts and nine plain completions:

* ``chat_short`` (28 tokens), ``chat_tools`` (137) and ``chat_long`` (2465)
  are stored **already rendered** through Spark's chat template, copied from
  the reference captures in ``analysis/experiments/spark_edge/ref*.json``. They
  are verbatim so that the tokens fed to the model here are byte-identical to
  the ones the parity and fidelity runs fed it -- a re-render through a newer
  template would silently change the comparison.
* the nine ``plain_*`` are raw completions. They exercise the path with no
  template at all, which is also the path a base-model deployment takes.

The spread of lengths is deliberate: ``chat_long`` at 2465 tokens crosses
Spark's 512-token sliding window several times, so a run over this set touches
window rollover rather than only short prefixes.

:func:`render_chat` is here as the adapter seam -- turning messages into a
prompt is a model-adapter job, not an engine job -- but the acceptance set does
not use it, for the reason above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
_CHAT_FILE = _HERE / "acceptance_chat_prompts.json"

# Verbatim from analysis/experiments/spark_edge/quality_int4.py::PLAIN.
PLAIN_PROMPTS: tuple[str, ...] = (
    "The capital of Australia is",
    "In 1969, the Apollo 11 mission",
    'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
    "The three laws of thermodynamics state that",
    "To make a proper espresso you need",
    "The difference between TCP and UDP is",
    "Photosynthesis converts",
    "A binary search tree is a data structure that",
    "The Treaty of Westphalia was signed in",
)


def acceptance_prompts() -> list[tuple[str, str]]:
    """The twelve ``(name, prompt)`` pairs, chat first then plain.

    Raises if the chat captures are missing rather than falling back to nine
    prompts: a run that quietly measured a different set would be worse than
    one that failed.
    """
    chat: dict[str, str] = json.loads(_CHAT_FILE.read_text(encoding="utf-8"))
    ordered = [(name, chat[name]) for name in ("chat_short", "chat_tools", "chat_long")]
    ordered += [(f"plain_{i}", text) for i, text in enumerate(PLAIN_PROMPTS)]
    if len(ordered) != 12:
        raise RuntimeError(f"the acceptance set must be 12 prompts, got {len(ordered)}")
    return ordered


def render_chat(model_dir: str, messages: list[dict[str, Any]], *, add_generation_prompt: bool = True) -> str:
    """Apply the checkpoint's own chat template. The adapter seam.

    Transformers 5 returns a list from ``apply_chat_template`` under some
    argument combinations and a string under others; the architecture record
    has a failed run from exactly that (``fp8_offload10``). Rendering to text
    explicitly, and asserting it, is cheaper than discovering it again.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"apply_chat_template returned {type(rendered).__name__}, not str; "
            "pass tokenize=False and render before tokenizing"
        )
    return rendered
