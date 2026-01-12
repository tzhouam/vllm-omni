from typing import Any

from vllm.sampling_params import SamplingParams


def _get_stage_name(stage: Any) -> str:
    # OmniStage exposes `.model_stage`; fallback to stage_id.
    model_stage = getattr(stage, "model_stage", None)
    if model_stage:
        return str(model_stage)
    return f"stage_{getattr(stage, 'stage_id', 'unknown')}"


def _fmt(stage: Any, field: str, cur: Any, rec: str) -> str:
    return f"{_get_stage_name(stage)}: {field}={cur!r}. {rec}"


def check_general_params(
    sampling_params_list: list[SamplingParams],
    stage_list: list[Any] | None = None,
    engine_args_list: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    General sampling params checks.

    This function must NOT raise; it should return human-readable warning strings.
    The orchestrator will emit them via logger.warning_once.
    """
    print("Checking general model params")
    errors: list[str] = []
    stage_list = stage_list or []
    engine_args_list = engine_args_list or []
    # Generic per-stage range checks (best-effort).
    for i, (sampling_params, engine_args, stage) in enumerate(zip(sampling_params_list, engine_args_list, stage_list)):
        if not isinstance(sampling_params, SamplingParams) or not isinstance(engine_args, dict):
            continue

        # temperature: >= 0
        if sampling_params.temperature is not None:
            try:
                if sampling_params.temperature < 0:
                    errors.append(_fmt(stage, "temperature", sampling_params.temperature, "Expected >= 0."))
            except Exception:
                errors.append(_fmt(stage, "temperature", sampling_params.temperature, "Expected a number >= 0."))

        # top_p: (0, 1]
        if sampling_params.top_p is not None:
            try:
                if sampling_params.top_p <= 0 or sampling_params.top_p > 1:
                    errors.append(_fmt(stage, "top_p", sampling_params.top_p, "Expected 0 < top_p <= 1."))
            except Exception:
                errors.append(_fmt(stage, "top_p", sampling_params.top_p, "Expected a number with 0 < top_p <= 1."))

        # top_k: >= -1
        if sampling_params.top_k is not None:
            try:
                if sampling_params.top_k < -1:
                    errors.append(_fmt(stage, "top_k", sampling_params.top_k, "Expected top_k >= -1."))
            except Exception:
                errors.append(_fmt(stage, "top_k", sampling_params.top_k, "Expected an int with top_k >= -1."))

        # repetition_penalty: > 0
        if sampling_params.repetition_penalty is not None:
            try:
                if sampling_params.repetition_penalty <= 0:
                    errors.append(
                        _fmt(stage, "repetition_penalty", sampling_params.repetition_penalty, "Expected > 0.")
                    )
            except Exception:
                errors.append(
                    _fmt(stage, "repetition_penalty", sampling_params.repetition_penalty, "Expected a number > 0.")
                )

        # max_tokens: >= 1
        if sampling_params.max_tokens is not None:
            try:
                if sampling_params.max_tokens < 1:
                    errors.append(_fmt(stage, "max_tokens", sampling_params.max_tokens, "Expected max_tokens >= 1."))
            except Exception:
                errors.append(
                    _fmt(stage, "max_tokens", sampling_params.max_tokens, "Expected a number with max_tokens >= 1.")
                )
    return errors
