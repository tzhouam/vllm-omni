from __future__ import annotations

from typing import Any


def _get_stage_name(stage: Any) -> str:
    # OmniStage exposes `.model_stage`; fallback to stage_id.
    model_stage = getattr(stage, "model_stage", None)
    if model_stage:
        return str(model_stage)
    return f"stage_{getattr(stage, 'stage_id', 'unknown')}"


def _fmt(stage: Any, field: str, cur: Any, rec: str) -> str:
    return f"{_get_stage_name(stage)}: {field}={cur!r}. {rec}"


def check_sampling_params_list(
    sampling_params_list: list[dict],
    stage_list: list[Any] | None = None,
    *,
    model: str | None = None,
    config_path: str | None = None,
) -> list[str]:
    """
    Qwen3-Omni multi-stage sampling params checks.

    This function must NOT raise; it should return human-readable warning strings.
    The orchestrator will emit them via logger.warning_once.
    """
    print(
        "=======================================================================\n"
        "check_sampling_params_list\n"
        "=======================================================================\n",
        flush=True,
    )
    errors: list[str] = []
    stage_list = stage_list or []

    if stage_list and len(sampling_params_list) != len(stage_list):
        errors.append(
            f"len(sampling_params_list)={len(sampling_params_list)} does not match "
            f"len(stage_list)={len(stage_list)} (model={model!r}, config={config_path!r})"
        )

    # Generic per-stage range checks (best-effort).
    for i, sp in enumerate(sampling_params_list):
        stage = stage_list[i] if i < len(stage_list) else f"stage_{i}"
        if not isinstance(sp, dict):
            continue

        # temperature: >= 0
        if "temperature" in sp:
            try:
                t = float(sp["temperature"])
                if t < 0:
                    errors.append(_fmt(stage, "temperature", sp["temperature"], "Expected >= 0."))
            except Exception:
                errors.append(_fmt(stage, "temperature", sp["temperature"], "Expected a number >= 0."))

        # top_p: (0, 1]
        if "top_p" in sp and sp["top_p"] is not None:
            try:
                p = float(sp["top_p"])
                if p <= 0 or p > 1:
                    errors.append(_fmt(stage, "top_p", sp["top_p"], "Expected 0 < top_p <= 1."))
            except Exception:
                errors.append(_fmt(stage, "top_p", sp["top_p"], "Expected a number with 0 < top_p <= 1."))

        # top_k: >= -1
        if "top_k" in sp and sp["top_k"] is not None:
            try:
                k = int(sp["top_k"])
                if k < -1:
                    errors.append(_fmt(stage, "top_k", sp["top_k"], "Expected top_k >= -1."))
            except Exception:
                errors.append(_fmt(stage, "top_k", sp["top_k"], "Expected an int with top_k >= -1."))

        # repetition_penalty: > 0
        if "repetition_penalty" in sp and sp["repetition_penalty"] is not None:
            try:
                rp = float(sp["repetition_penalty"])
                if rp <= 0:
                    errors.append(_fmt(stage, "repetition_penalty", sp["repetition_penalty"], "Expected > 0."))
            except Exception:
                errors.append(_fmt(stage, "repetition_penalty", sp["repetition_penalty"], "Expected a number > 0."))

        # max_tokens: >= 1
        if "max_tokens" in sp and sp["max_tokens"] is not None:
            try:
                mt = int(sp["max_tokens"])
                if mt < 1:
                    errors.append(_fmt(stage, "max_tokens", sp["max_tokens"], "Expected max_tokens >= 1."))
            except Exception:
                errors.append(_fmt(stage, "max_tokens", sp["max_tokens"], "Expected an int with max_tokens >= 1."))

    # Qwen3-Omni stage-specific suggestions.
    for i, sp in enumerate(sampling_params_list):
        if i >= len(stage_list) or not isinstance(sp, dict):
            continue
        stage = stage_list[i]
        model_stage = getattr(stage, "model_stage", None)

        if model_stage == "talker":
            stop_ids = sp.get("stop_token_ids", None)
            if not stop_ids:
                errors.append(
                    _fmt(
                        stage,
                        "stop_token_ids",
                        stop_ids,
                        "Talker stage should set stop_token_ids (e.g., codec EOS) to stop at end of codec stream.",
                    )
                )

        if model_stage == "code2wav":
            temp = sp.get("temperature", None)
            # Code2Wav is a deterministic generation stage in our pipeline; recommend greedy.
            if temp is None:
                errors.append(_fmt(stage, "temperature", temp, "Recommend temperature=0.0 for code2wav stage."))
            else:
                try:
                    if float(temp) != 0.0:
                        errors.append(_fmt(stage, "temperature", temp, "Recommend temperature=0.0 for code2wav stage."))
                except Exception:
                    errors.append(_fmt(stage, "temperature", temp, "Recommend a numeric temperature=0.0 for code2wav."))

    return errors
