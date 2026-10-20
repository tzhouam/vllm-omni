# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.config import get_config, get_hf_file_to_dict
from vllm.transformers_utils.repo_utils import file_or_path_exists

from vllm_omni.config.config_factory import with_trust_remote_code_override
from vllm_omni.config.config_factory import (
    StageConfigFactory,
    _materialize_object_storage_configs,
    _name_match_candidate,
)
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import _DEPLOY_DIR
from vllm_omni.diffusion.utils.hf_utils import (
    _looks_like_dreamzero,
    get_diffusion_model_index,
    resolve_native_diffusion_model_class,
)
from vllm_omni.entrypoints.stage_utils import _to_dict
from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.platforms import current_omni_platform

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

logger = init_logger(__name__)


def inject_omni_kv_config(stage: Any, omni_conn_cfg: dict[str, Any], omni_from: str, omni_to: str) -> None:
    """Inject connector configuration into stage engine arguments."""
    typed_connector_config = getattr(stage, "connector_config", None)
    if typed_connector_config is not None:
        omni_conf_dict = dict(typed_connector_config.omni_kv_config or {})
        omni_conf_dict.update(connector_config=omni_conn_cfg, omni_from_stage=omni_from, omni_to_stage=omni_to)
        typed_connector_config.omni_kv_config = omni_conf_dict
        return

    # Prepare omni_kv_config dict
    omni_conf_dict = {}
    try:
        # Access engine_args safely (might be OmegaConf or dict)
        existing_args = stage.engine_args
        if hasattr(existing_args, "get"):
            _oc = existing_args.get("omni_kv_config", None)
            if _oc:
                if hasattr(_oc, "items"):  # dict-like
                    omni_conf_dict = dict(_oc)
                else:  # object?
                    omni_conf_dict = _to_dict(_oc)
    except Exception:
        omni_conf_dict = {}

    # Inject connector info
    omni_conf_dict["connector_config"] = omni_conn_cfg
    omni_conf_dict["omni_from_stage"] = omni_from
    omni_conf_dict["omni_to_stage"] = omni_to

    # Write back to engine_args
    try:
        if hasattr(stage.engine_args, "__setitem__"):
            stage.engine_args["omni_kv_config"] = omni_conf_dict
        else:
            setattr(stage.engine_args, "omni_kv_config", omni_conf_dict)
    except Exception as e:
        # Fallback for OmegaConf or similar if direct set fails?
        logger.error(f"Failed to inject omni connector config into stage: {e}")


# [edge-infer] Kept across the v0.29.0rc1 merge: upstream removed
# ``resolve_model_config_path`` and its helpers from this module (its callers
# moved to ``vllm_omni.config.resolver``); the edge ``deploy_profile`` selector
# below still needs the model -> default deploy YAML lookup, so that chain is
# retained here. ``_convert_dataclasses_to_dict`` / ``load_and_resolve_stage_configs``
# now live in ``vllm_omni.config.resolver`` / ``vllm_omni.engine.async_omni_engine``.
_DIFFUSERS_CLASS_TO_CONFIG: dict[str, str] = {
    "GlmImagePipeline": "glm_image",
}


def _try_get_class_name_from_diffusers_config(model: str) -> str | None:
    """Try to get class name from diffusers model configuration files.

    Args:
        model: Model name or path

    Returns:
        Model type string if found, None otherwise
    """
    model_index = get_diffusion_model_index(model)
    if model_index and isinstance(model_index, dict) and "_class_name" in model_index:
        logger.debug(f"Found Diffusers model type '{model_index['_class_name']}'")
        return model_index["_class_name"]

    return None


def _try_resolve_omni_model_type(model: str) -> str | None:
    """Try to resolve model_type for omni models with empty config.json.

    Searches both the legacy ``stage_configs/*.yaml`` directory and the
    migrated ``deploy/*.yaml`` directory for a stem that substring-matches
    the model path basename (e.g. ``cosyvoice3`` in
    ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``). Only the basename is scanned
    so URI segments such as the bucket name cannot select an unrelated
    pipeline. The longest match wins so ``cosyvoice3`` beats ``cosyvoice``
    and ``bagel_single_stage`` beats ``bagel``.
    """
    model_lower = _name_match_candidate(model).lower().replace("-", "").replace("_", "")
    best_match: str | None = None
    best_len = 0
    for subdir in ("model_executor/stage_configs", "deploy"):
        config_dir = PROJECT_ROOT / "vllm_omni" / subdir
        if not config_dir.exists():
            continue
        for config_file in sorted(config_dir.glob("*.yaml")):
            candidate = config_file.stem.replace("-", "").replace("_", "")
            if candidate and candidate in model_lower and len(candidate) > best_len:
                best_match = config_file.stem
                best_len = len(candidate)
    return best_match


def _registry_default_deploy_path(model: str) -> str | None:
    """Return the registered pipeline's default deploy YAML path, if any.

    A checkpoint's HF ``model_type`` is not always the registered Omni pipeline
    key. MiniCPM-o 4.5, for example, reports ``minicpmo`` while its
    predicate-selected pipeline is ``minicpmo_4_5``. Stage construction already
    loads that pipeline's default deploy config, so connector discovery must
    resolve the same path.
    """
    pipeline_config = StageConfigFactory.get_pipeline_config(
        model=model,
        trust_remote_code=True,
    )
    if pipeline_config is None or pipeline_config.default_deploy_config_name is None:
        return None
    default_deploy_path = _DEPLOY_DIR / pipeline_config.default_deploy_config_name
    if default_deploy_path.is_file():
        return str(default_deploy_path)
    return None


def deploy_path_for_profile(model_type: str, profile: str) -> Path | None:
    """Return ``deploy/<profile>/<model_type>.yaml`` if it exists (pure path lookup).

    Looks first under the platform's default stage-config directory, then under
    the packaged ``vllm_omni/deploy`` directory. ``model_type`` is the deploy
    YAML stem (the same key ``resolve_model_config_path`` resolves to).
    """
    if not profile or not model_type:
        return None
    file_name = f"{model_type}.yaml"
    candidates = []
    try:
        candidates.append(PROJECT_ROOT / current_omni_platform.get_default_stage_config_path() / profile / file_name)
    except NotImplementedError:
        pass
    candidates.append(_DEPLOY_DIR / profile / file_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def available_deploy_profiles(model_type: str | None = None) -> list[str]:
    """List profile directories under ``vllm_omni/deploy`` (optionally only those defining ``model_type``)."""
    profiles = []
    if not _DEPLOY_DIR.exists():
        return profiles
    for entry in sorted(_DEPLOY_DIR.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if model_type is None or (entry / f"{model_type}.yaml").exists():
            profiles.append(entry.name)
    return profiles


def resolve_deploy_profile_path(model: str, profile: str) -> str:
    """Resolve a named deploy profile (e.g. ``edge``) for ``model`` to a YAML path.

    The model's default deploy YAML stem (from ``resolve_model_config_path``)
    is the key under ``deploy/<profile>/``. An explicit profile never silently
    falls back to the default deploy config.

    Raises:
        FileNotFoundError: if the profile does not define a YAML for the model.
    """
    default_path = resolve_model_config_path(model)
    model_type = Path(default_path).stem if default_path else None
    if model_type is None:
        raise FileNotFoundError(f"Cannot resolve a deploy profile for {model!r}: no default deploy config found.")
    path = deploy_path_for_profile(model_type, profile)
    if path is None:
        raise FileNotFoundError(
            f"Deploy profile {profile!r} has no config for model type {model_type!r}. "
            f"Profiles defining it: {available_deploy_profiles(model_type)}; "
            f"all profiles: {available_deploy_profiles()}."
        )
    logger.info("[deploy_profile] %s -> %s", profile, path)
    return str(path)


def apply_deploy_profile(model: str, args_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate ``deploy_profile`` into ``deploy_config`` in-place (used by CLI and engine)."""
    profile = args_dict.pop("deploy_profile", None)
    if not profile:
        # A deploy YAML carries the same `orchestrator:` block whichever flag
        # loaded it. Applying it only on the profile path made the file mean
        # different things depending on how it was passed, silently: pointing
        # --deploy-config at deploy/edge/qwen3_tts.yaml loaded its stages but
        # dropped `parallel_stage_init: true`, so an edge deployment started
        # sequentially -- 63.8 s instead of 30.5 s on Qwen3-TTS -- with nothing
        # in the log to say why.
        if args_dict.get("deploy_config"):
            _apply_profile_orchestrator_defaults(args_dict["deploy_config"], args_dict)
        return args_dict
    if args_dict.get("deploy_config"):
        raise ValueError("`deploy_profile` and `deploy_config` are mutually exclusive; pass only one.")
    if profile == "auto":
        from vllm_omni.edge.auto_profile import materialize_auto_deploy

        default_path = resolve_model_config_path(model)
        if default_path is None:
            raise FileNotFoundError(f"Cannot resolve a deploy config for {model!r} (needed by deploy_profile='auto').")
        edge_path = deploy_path_for_profile(Path(default_path).stem, "edge")
        base = str(edge_path) if edge_path is not None else default_path
        args_dict["deploy_config"], _ = materialize_auto_deploy(model, base)
        _apply_profile_orchestrator_defaults(base, args_dict)
        return args_dict
    args_dict["deploy_config"] = resolve_deploy_profile_path(model, profile)
    _apply_profile_orchestrator_defaults(args_dict["deploy_config"], args_dict)
    return args_dict


# Orchestrator-level settings a deploy profile may carry under a top-level
# ``orchestrator:`` block. They are engine/CLI arguments, not stage config, so
# ``load_deploy_config`` ignores the block and the profile selector applies it
# here (explicit user values win; only unset/False defaults are replaced).
PROFILE_ORCHESTRATOR_KEYS = frozenset({"parallel_stage_init", "init_timeout", "stage_init_timeout"})


def profile_orchestrator_defaults(deploy_path: str | Path) -> dict[str, Any]:
    """Return the ``orchestrator:`` block of a deploy YAML (only known keys)."""
    try:
        import yaml

        with open(deploy_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except (OSError, ValueError):
        return {}
    block = doc.get("orchestrator") if isinstance(doc, dict) else None
    if not isinstance(block, dict):
        return {}
    return {k: v for k, v in block.items() if k in PROFILE_ORCHESTRATOR_KEYS}


def _apply_profile_orchestrator_defaults(deploy_path: str | Path, args_dict: dict[str, Any]) -> None:
    for key, value in profile_orchestrator_defaults(deploy_path).items():
        if args_dict.get(key) in (None, False):
            args_dict[key] = value


def resolve_model_config_path(model: str) -> str | None:
    """Resolve the stage/deploy config file path from the model name.

    Resolves configuration path based on the model type and device type.
    Order:
    1. Device-specific stage config under ``stage_configs/{device_type}/``
    2. If HF ``model_type`` is not an ``OMNI_PIPELINES`` key, the registered
       pipeline's ``default_deploy_config_name`` (keeps connectors aligned with
       stage construction when HF type and pipeline key differ)
    3. ``deploy/{model_type}.yaml``
    4. Legacy ``stage_configs/{model_type}.yaml``
    5. Registered pipeline default deploy config (final fallthrough)

    Args:
        model: Model name or path (used to determine model_type)

    Returns:
        String path to the stage/deploy configuration file, or ``None`` if no
        matching config file exists.

    Raises:
        ValueError: If model_type cannot be determined
    """
    # Object-storage URIs are streamed by vLLM's Run:AI streamer only after
    # each stage builds its ModelConfig, so config resolution here reads the
    # local copy that vLLM-Omni materializes for such URIs. Name-based
    # fallbacks keep the original string (the materialized path is a hash).
    config_source = _materialize_object_storage_configs(model)
    # Try to get config from standard transformers format first
    try:
        hf_config = get_config(config_source, trust_remote_code=True)
        model_type = hf_config.model_type
    except (ValueError, Exception):
        # If standard transformers format fails, try diffusers format
        native_model_class = resolve_native_diffusion_model_class(config_source)
        if native_model_class is not None:
            model_type = native_model_class
        elif get_diffusion_model_index(config_source) is not None:
            model_type = _try_get_class_name_from_diffusers_config(config_source)
            if model_type is None:
                raise ValueError(
                    f"Could not determine model_type for diffusers model: {model}. "
                    "Please ensure its Diffusers pipeline index contains '_class_name'"
                )
        elif file_or_path_exists(config_source, "config.json", revision=None):
            # Try to read config.json manually for custom models like Bagel that fail get_config
            # but have a valid config.json with model_type
            try:
                config_dict = get_hf_file_to_dict("config.json", config_source, revision=None)
                if config_dict and "model_type" in config_dict:
                    model_type = config_dict["model_type"]
                else:
                    # For models with empty config.json (e.g. CosyVoice3),
                    # try matching against registered omni stage configs.
                    model_type = _try_resolve_omni_model_type(model)
                    if model_type is None:
                        raise ValueError(f"config.json found but missing 'model_type' for model: {model}")
            except Exception as e:
                raise ValueError(f"Failed to read config.json for model: {model}. Error: {e}") from e
        else:
            # No config.json at repo root (e.g. GLM-TTS stores configs in
            # subdirectories only).  Try matching against registered deploy
            # YAML filenames before giving up.
            model_type = _try_resolve_omni_model_type(model)
            if model_type is None:
                raise ValueError(
                    f"Could not determine model_type for model: {model}. "
                    "Model is not in standard transformers or Diffusers format. "
                    f"Please ensure the model has proper configuration files with 'model_type' field"
                )

    default_config_path = current_omni_platform.get_default_stage_config_path()
    if model_type == "vla" and _looks_like_dreamzero(config_source):
        model_type = "dreamzero"

    if model_type in _DIFFUSERS_CLASS_TO_CONFIG:
        normalized_model_type = _DIFFUSERS_CLASS_TO_CONFIG[model_type]
    else:
        normalized_model_type = model_type.replace("-", "_")
    model_type_str = f"{normalized_model_type}.yaml"
    complete_config_path = PROJECT_ROOT / default_config_path / model_type_str
    if os.path.exists(complete_config_path):
        return str(complete_config_path)

    # Prefer the registry default before deploy/<hf_model_type>.yaml when the
    # HF type is not itself a pipeline key. Otherwise a future
    # deploy/minicpmo.yaml would desync connectors from stages for MiniCPM-o 4.5.
    if normalized_model_type not in OMNI_PIPELINES:
        registry_path = _registry_default_deploy_path(model)
        if registry_path is not None:
            return registry_path

    deploy_config_path = _DEPLOY_DIR / model_type_str
    if os.path.exists(deploy_config_path):
        return str(deploy_config_path)

    stage_config_file = f"vllm_omni/model_executor/stage_configs/{normalized_model_type}.yaml"
    stage_config_path = PROJECT_ROOT / stage_config_file
    if os.path.exists(stage_config_path):
        return str(stage_config_path)

    return _registry_default_deploy_path(model)


def parse_stage_overrides(value: Any) -> dict[str, dict[str, Any]] | None:
    """Parse and validate the shape of per-stage JSON overrides."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--stage-overrides is not valid JSON: {exc}. Got: {value!r}") from exc
    else:
        parsed = value

    if not isinstance(parsed, dict):
        raise ValueError(
            "--stage-overrides must be a JSON object mapping stage_id -> overrides, "
            f"got {type(parsed).__name__}: {parsed!r}"
        )
    for stage_id, overrides in parsed.items():
        if not isinstance(stage_id, str) or not stage_id.isascii() or not stage_id.isdigit():
            raise ValueError(
                f"--stage-overrides keys must be non-negative integer stage ids (as strings), got {stage_id!r}"
            )
        if not isinstance(overrides, dict):
            raise ValueError(
                f"--stage-overrides[{stage_id!r}] must be an object, got {type(overrides).__name__}: {overrides!r}"
            )

    return parsed


@dataclass(frozen=True)
class StageConfigInputs:
    """Normalized inputs shared by all stage-config loading paths."""

    model: str
    kwargs: dict[str, Any]
    trust_remote_code: bool | None
    deploy_config_path: str | None
    stage_overrides: dict[str, dict[str, Any]] | None
    strategy_config_path: str | None


def prepare_stage_config_inputs(
    model: str,
    kwargs: Mapping[str, Any],
    *,
    trust_remote_code: bool | None,
    snapshot_model: bool = False,
) -> StageConfigInputs:
    """Normalize model/config arguments before resolving stage configs.

    The standard API worker, multi-API parent, and headless entrypoint must
    apply the same legacy-key filtering, trust-remote-code precedence, and
    config-file extraction before calling :func:`resolve_omni_config`.
    """
    resolved_model = model
    if snapshot_model:
        # Keep model materialization at the CLI boundary for the parent-owned
        # launch path. The API worker's OmniBase has already done this work.
        from vllm_omni.entrypoints.omni_base import omni_snapshot_download

        resolved_model = omni_snapshot_download(model)

    resolved_kwargs = dict(kwargs)
    for legacy_arg in ("stage_configs_path", "stage_configs"):
        if legacy_arg in resolved_kwargs:
            raise ValueError(f"`{legacy_arg}` is no longer supported; use `deploy_config` instead.")
    # Frontend process count is not a stage engine override.
    resolved_kwargs.pop("api_server_count", None)
    resolved_kwargs.pop("disable_log_stats", None)
    if resolved_kwargs.get("diffusion_streaming_output") and resolved_kwargs.get("streaming_output") is None:
        resolved_kwargs["streaming_output"] = True
    resolved_kwargs.pop("model_tag", None)

    resolved_kwargs = with_trust_remote_code_override(resolved_kwargs, trust_remote_code)
    deploy_config_path = resolved_kwargs.pop("deploy_config", None)
    strategy_config_path = resolved_kwargs.pop("strategy_config", None)
    stage_overrides = parse_stage_overrides(resolved_kwargs.pop("stage_overrides", None))
    return StageConfigInputs(
        model=resolved_model,
        kwargs=resolved_kwargs,
        trust_remote_code=trust_remote_code,
        deploy_config_path=deploy_config_path,
        stage_overrides=stage_overrides,
        strategy_config_path=strategy_config_path,
    )


def get_final_stage_id_for_e2e(
    output_modalities: list[str] | None, default_modalities: list[str], stage_list: list
) -> int:
    """Get the final stage id for e2e.

    Args:
        stage_list: List of stage configurations

    Returns:
        Final stage id for e2e
    """
    last_stage_id = len(stage_list) - 1
    if output_modalities is not None:
        prompt_modalities = []
        for modality in output_modalities:
            if modality not in default_modalities:
                logger.warning(f"Invalid output modality: {modality}, ignoring it")
                # TODO: if user specifies unsupported modalities, invalid it and raise an error
                continue
            prompt_modalities.append(modality)
        output_modalities = prompt_modalities
    else:
        output_modalities = default_modalities

    try:
        final_stage_id_for_e2e = last_stage_id
        for _sid in range(last_stage_id, -1, -1):
            if (
                getattr(stage_list[_sid], "final_output", False)
                and stage_list[_sid].final_output_type in output_modalities
            ):
                final_stage_id_for_e2e = _sid
                break
    except Exception as e:
        logger.debug(
            "[Orchestrator] Failed to determine final stage for E2E; \
                falling back to last: %s",
            e,
            exc_info=True,
        )
        final_stage_id_for_e2e = last_stage_id

    return final_stage_id_for_e2e


def filter_dataclass_kwargs(cls: Any, kwargs: dict) -> dict:
    """Filter kwargs to only include fields defined in the dataclass.

    Args:
        cls: Dataclass type
        kwargs: Keyword arguments to filter

    Returns:
        Filtered keyword arguments containing only valid dataclass fields
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary")

    def _filter_value(value: Any, annotation: Any) -> Any:
        """Recursively filter nested dict/list values based on dataclass annotations."""
        if annotation is None:
            return value

        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
                return filter_dataclass_kwargs(annotation, value)
            return value

        if origin in (list, tuple, set):
            args = get_args(annotation)
            inner = args[0] if args else None
            if isinstance(value, list | tuple | set):
                return type(value)(_filter_value(v, inner) for v in value)
            return value

        if origin is dict:
            args = get_args(annotation)
            val_type = args[1] if len(args) > 1 else None
            if isinstance(value, dict):
                return {k: _filter_value(v, val_type) for k, v in value.items()}
            return value

        if origin is types.UnionType or origin is getattr(types, "UnionType", None):
            for arg in get_args(annotation):
                if isinstance(arg, type) and is_dataclass(arg) and isinstance(value, dict):
                    return filter_dataclass_kwargs(arg, value)
                # Try container-style filtering for union members
                filtered = _filter_value(value, arg)
                if filtered is not value:
                    return filtered
            return value

        return value

    valid_fields = {f.name: f for f in fields(cls) if f.init}
    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k not in valid_fields:
            logger.warning(
                "Dropping unknown %s field %r (not declared on the dataclass)",
                cls.__name__,
                k,
            )
            continue
        field = valid_fields[k]
        filtered_kwargs[k] = _filter_value(v, field.type)

    return filtered_kwargs


# The following code detects if the process is running in a container and if
# PID host is available. If so, we can use process-scoped memory tracking;
# otherwise we need sequential init locks.


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def in_container() -> bool:
    # Common Docker signal
    if os.path.exists("/.dockerenv"):
        return True

    # cgroup markers (works for Docker/containerd/K8s/Podman in many setups)
    cg = _read_text("/proc/1/cgroup") or ""
    markers = ("docker", "containerd", "kubepods", "libpod", "podman")
    return any(m in cg for m in markers)


def has_pid_host() -> bool | None:
    """
    Returns:
      True  -> very likely running with --pid=host (host PID namespace)
      False -> very likely isolated PID namespace (default)
      None  -> cannot determine
    """
    # Strong signal: in host pid namespace, PID 2 is usually kthreadd
    comm2 = _read_text("/proc/2/comm")
    if comm2 is not None:
        comm2 = comm2.strip()
        if comm2 == "kthreadd":
            return True
        # If PID 2 exists and is NOT kthreadd, we're almost certainly not in host pid ns
        return False

    # Fallback: check for other low-numbered kernel threads (best-effort)
    for pid, name in [(3, "rcu_gp"), (4, "rcu_par_gp"), (10, "ksoftirqd/0")]:
        comm = _read_text(f"/proc/{pid}/comm")
        if comm is not None:
            if comm.strip() == name:
                return True
            else:
                return False

    return False


def detect_pid_host() -> bool:
    ic = in_container()
    if not ic:
        return True

    return has_pid_host() is True


### Helpers for handling delta messages
def coerce_param_message_types(params: list[OmniSamplingParams], is_streaming: bool):
    """Iterate over the sampling params and convert to the message types
    to DELTA messages, if streaming is enabled, or FINAL_ONLY if
    it's disabled, while respecting `.skip_clone` on the params.

    This is needed to avoid emitting redundant multimodal data.
    """
    # Coerce vLLM's default output kinds as needed to handle streaming
    # (i.e., DELTA output kind). Note that this is only applied to non
    # Diffusion sampling params.
    #
    # NOTE: Hidden states will still be passed between stages.
    for idx, sp in enumerate(params):
        # For OmniDiffusionParams don't set output kind
        if isinstance(sp, SamplingParams):
            params[idx] = maybe_coerce_to_message_type(sp, is_streaming)
    return params


def maybe_coerce_to_message_type(params: SamplingParams, is_streaming: bool):
    """If this is a CUMULATIVE message, coerce it to DELTA if streaming, otherwise FINAL_ONLY."""
    target_type = RequestOutputKind.DELTA if is_streaming else RequestOutputKind.FINAL_ONLY
    if params.output_kind == target_type:
        return params
    elif is_streaming and params.output_kind == RequestOutputKind.FINAL_ONLY:
        logger.debug("Coercing FINAL_ONLY output to DELTA for streaming")
    elif not is_streaming and params.output_kind == RequestOutputKind.DELTA:
        logger.debug("Coercing DELTA output to FINAL_ONLY for non-streaming")

    if not params.skip_clone:
        params = params.clone()
        params.skip_clone = True
    params.output_kind = target_type
    return params


class PureDiffusionLauncherAdapter:
    """vLLM launcher compatibility shim for pure-diffusion mode.

    The upstream launcher's shutdown path reads
    ``app.state.engine_client.vllm_config.shutdown_timeout``
    (vllm/entrypoints/launcher.py), but ``AsyncOmni.vllm_config`` returns
    ``None`` when the pipeline has no comprehension stage (pure diffusion),
    which crashes ``handle_shutdown`` with AttributeError and hangs server
    teardown (workers force-killed, spurious resource_tracker noise).

    This adapter only overrides the ``vllm_config`` property with a minimal
    fallback carrying ``shutdown_timeout`` and forwards every other attribute
    to the wrapped engine client, so the pure-diffusion detection
    (``get_vllm_config()`` still returns ``None``) is unaffected.
    """

    def __init__(self, engine_client: Any, shutdown_timeout: float) -> None:
        object.__setattr__(self, "_wrapped", engine_client)
        object.__setattr__(self, "_shutdown_timeout", float(shutdown_timeout))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def vllm_config(self) -> Any:
        return types.SimpleNamespace(shutdown_timeout=self._shutdown_timeout)
