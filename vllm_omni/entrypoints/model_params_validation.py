"""
Sampling params validation for multi-stage Omni pipelines.

This module implements a config-linked validation hook (RFC #710 Design 1):
https://github.com/vllm-project/vllm-omni/issues/710

Key properties:
- Validator is configured per stage config YAML via a *top-level* key:
    model_params_validator: "pkg.module:func"
- Validation runs in the orchestrator (Omni / AsyncOmni) so it covers both:
  - offline Python APIs (Omni / AsyncOmni)
  - online OpenAI server (which uses AsyncOmni internally)
- Validation must not block requests: it only emits logger.warning_once.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)

ValidatorFn = Callable[..., list[str]]

# Default validator when config does not specify one.
_DEFAULT_VALIDATOR = "vllm_omni.model_executor.model_param_validate.general_check:check_general_params"


def _stringify_validator_errors(errors: list[str]) -> str:
    header = "=" * 10 + "Sampling params validation warnings"
    # Keep ordering stable for warning_once keying.
    body = "\n".join(f"- {e}" for e in errors)
    return f"{header}\n{body}\n" + "=" * 10


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of objects (SamplingParams / OmegaConf / dataclasses) to plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    # OmegaConf containers
    try:
        # Avoid importing DictConfig/ListConfig types directly.
        if obj.__class__.__module__.startswith("omegaconf"):
            as_container = OmegaConf.to_container(obj, resolve=True)  # type: ignore[arg-type]
            if isinstance(as_container, dict):
                return dict(as_container)
    except Exception:
        pass
    # pydantic v2
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return dict(model_dump())
        except Exception:
            pass
    # dataclass / regular object
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return dict(d)
    # fallback
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        return {"_repr": repr(obj)}


def _parse_validator_path(path: str) -> tuple[str, str] | None:
    # Accept "pkg.mod:func" (preferred) and "pkg.mod.func" (fallback).
    if not path or not isinstance(path, str):
        return None
    if ":" in path:
        mod, fn = path.split(":", 1)
        return mod.strip(), fn.strip()
    if "." in path:
        mod, fn = path.rsplit(".", 1)
        return mod.strip(), fn.strip()
    return None


@lru_cache(maxsize=32)
def _load_validator_from_config_path(config_path: str | None) -> ValidatorFn | None:
    """Load validator callable from YAML top-level key `model_params_validator`."""
    if not config_path:
        # No config: still use the default validator.
        return _load_validator_from_path(_DEFAULT_VALIDATOR, config_path="<none>")
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        logger.warning_once("Model params validator: failed to load config %r: %s", config_path, e)
        return _load_validator_from_path(_DEFAULT_VALIDATOR, config_path=str(config_path))

    validator_path = None
    try:
        validator_path = cfg.get("model_params_validator", None)
    except Exception:
        # OmegaConf objects can fail .get in some edge cases; ignore.
        validator_path = getattr(cfg, "model_params_validator", None)

    if not validator_path:
        # Fallback to default general validator.
        return _load_validator_from_path(_DEFAULT_VALIDATOR, config_path=str(config_path))

    parsed = _parse_validator_path(str(validator_path))
    if not parsed:
        logger.warning_once(
            "Model params validator: invalid path %r in %r (expected 'pkg.module:func')",
            validator_path,
            config_path,
        )
        return _load_validator_from_path(_DEFAULT_VALIDATOR, config_path=str(config_path))
    return _load_validator_from_path(str(validator_path), config_path=str(config_path))


def _load_validator_from_path(path: str, *, config_path: str) -> ValidatorFn | None:
    parsed = _parse_validator_path(str(path))
    if not parsed:
        logger.warning_once(
            "Model params validator: invalid path %r (expected 'pkg.module:func') [config=%r]",
            path,
            config_path,
        )
        return None
    module_path, fn_name = parsed
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
    except Exception as e:
        logger.warning_once(
            "Model params validator: failed to import %r from %r (config=%r): %s",
            fn_name,
            module_path,
            config_path,
            e,
        )
        return None

    if not callable(fn):
        logger.warning_once(
            "Model params validator: %r in %r (config=%r) is not callable",
            fn_name,
            module_path,
            config_path,
        )
        return None

    return fn  # type: ignore[return-value]


def validate_model_params(
    *,
    sampling_params_list: list[Any],
    stage_list: list[Any],
    model: str | None,
    config_path: str | None,
) -> None:
    """Run stage-level validator (if configured) and emit warning_once on violations.

    Validator is expected to check sampling params together with per-stage engine args.
    """
    validator = _load_validator_from_config_path(config_path)
    if validator is None:
        return

    # Normalize LLM-stage dicts into SamplingParams where possible.
    engine_args_list = [_to_plain_dict(getattr(s, "engine_args", None)) for s in stage_list]
    try:
        errors = validator(
            sampling_params_list,
            stage_list,
            engine_args_list,
            model=model,
            config_path=config_path,
        )
    except TypeError:
        # Backward-compat: try without engine_args_list, then with the oldest
        # signature check_sampling_params_list(sampling_params_list).
        try:
            errors = validator(
                sampling_params_list,
                stage_list,
                model=model,
                config_path=config_path,
            )
        except Exception as e:
            try:
                errors = validator(sampling_params_list)
            except Exception as e2:
                logger.warning_once(
                    "Sampling params validator raised (ignored). model=%r config=%r err=%s / %s",
                    model,
                    config_path,
                    e,
                    e2,
                )
                return
    except Exception as e:
        logger.warning_once(
            "Sampling params validator raised (ignored). model=%r config=%r err=%s",
            model,
            config_path,
            e,
        )
        return

    if not errors:
        return

    # One warning_once per process (by design).
    logger.warning_once(_stringify_validator_errors(list(errors)))
