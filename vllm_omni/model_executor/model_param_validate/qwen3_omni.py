from __future__ import annotations

from typing import Any

from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.model_param_validate.general_check import _fmt, check_general_params


def validate_model_params(
    sampling_params_list: list[SamplingParams],
    stage_list: list[Any] | None = None,
    engine_args_list: list[dict[str, Any]] | None = None,
    *,
    model: str | None = None,
    config_path: str | None = None,
) -> list[str]:
    """
    Qwen3-Omni multi-stage sampling params checks.

    This function must NOT raise; it should return human-readable warning strings.
    The orchestrator will emit them via logger.warning_once.
    """
    errors: list[str] = []
    stage_list = stage_list or []
    engine_args_list = engine_args_list or []

    if stage_list and len(sampling_params_list) != len(stage_list):
        errors.append(
            f"len(sampling_params_list)={len(sampling_params_list)} does not match "
            f"len(stage_list)={len(stage_list)} (model={model!r}, config={config_path!r})"
        )
    if engine_args_list and stage_list and len(engine_args_list) != len(stage_list):
        errors.append(
            f"len(engine_args_list)={len(engine_args_list)} does not match "
            f"len(stage_list)={len(stage_list)} (model={model!r}, config={config_path!r})"
        )

    errors.extend(check_general_params(sampling_params_list, stage_list, engine_args_list))

    # Cross-check engine args with stage identity / expected outputs.
    expected_output_type_by_stage = {
        "thinker": "latent",
        "talker": "latent",
        "code2wav": "audio",
    }
    for i, stage in enumerate(stage_list):
        if i >= len(engine_args_list):
            continue
        engine_args = engine_args_list[i]
        model_stage = getattr(stage, "model_stage", None)
        if model_stage in expected_output_type_by_stage:
            got = engine_args.get("engine_output_type", None)
            exp = expected_output_type_by_stage[model_stage]
            if got is not None and str(got) != exp:
                errors.append(
                    _fmt(stage, "engine_args.engine_output_type", got, f"Expected {exp!r} for {model_stage} stage.")
                )

    # Qwen3-Omni stage-specific suggestions.
    for i, (sampling_params, engine_args, stage) in enumerate(zip(sampling_params_list, engine_args_list, stage_list)):
        if not isinstance(sampling_params, SamplingParams) or not isinstance(engine_args, dict):
            continue
        model_stage = getattr(stage, "model_stage", None)

        if model_stage == "talker":
            stop_ids = sampling_params.stop_token_ids
            if not stop_ids:
                errors.append(
                    _fmt(
                        stage,
                        "stop_token_ids",
                        stop_ids,
                        "Talker stage should set stop_token_ids (e.g., codec EOS) to stop at end of codec stream.",
                    )
                )
            if sampling_params.temperature == 0 and i == 1:
                errors.append(
                    _fmt(
                        stage,
                        "temperature",
                        sampling_params.temperature,
                        "Expected temperature > 0 for talker stage to avoid infinite loop.",
                    )
                )
            # talker should have max_tokens larger than thinker's
            if sampling_params.max_tokens is not None:
                try:
                    if sampling_params.max_tokens < sampling_params_list[0].max_tokens:
                        errors.append(
                            _fmt(
                                stage,
                                "max_tokens",
                                sampling_params.max_tokens,
                                "Expected max_tokens >= thinker's max_tokens for talker stage.",
                            )
                        )
                except Exception:
                    errors.append(
                        _fmt(
                            stage,
                            "max_tokens",
                            sampling_params.max_tokens,
                            "Expected an int with max_tokens >= thinker's max_tokens for talker stage.",
                        )
                    )

        if model_stage == "code2wav":
            temp = sampling_params.temperature
            # Code2Wav is a deterministic generation stage in our pipeline; recommend greedy.
            if temp is None:
                errors.append(_fmt(stage, "temperature", temp, "Recommend temperature=0.0 for code2wav stage."))
            else:
                try:
                    if float(temp) != 0.0:
                        errors.append(_fmt(stage, "temperature", temp, "Recommend temperature=0.0 for code2wav stage."))
                except Exception:
                    errors.append(_fmt(stage, "temperature", temp, "Recommend a numeric temperature=0.0 for code2wav."))

            # code2wav should have max_tokens larger than talker's * 1280
            if "max_num_batched_tokens" in engine_args:
                try:
                    max_num_batched_tokens = int(engine_args["max_num_batched_tokens"])
                    if max_num_batched_tokens < 1280 * sampling_params_list[1].max_tokens:
                        errors.append(
                            _fmt(
                                stage,
                                "max_num_batched_tokens",
                                max_num_batched_tokens,
                                "Expected max_num_batched_tokens >= 1280 * talker's max_tokens for code2wav stage as ",
                                1280 * sampling_params_list[1].max_tokens,
                            )
                        )
                except Exception:
                    errors.append(
                        _fmt(
                            stage,
                            "max_num_batched_tokens",
                            max_num_batched_tokens,
                            (
                                "Expected an int with max_num_batched_tokens >= 1280 * talker's max_tokens "
                                "for code2wav stage as " + str(1280 * sampling_params_list[1].max_tokens)
                            ),
                        )
                    )

    return errors
