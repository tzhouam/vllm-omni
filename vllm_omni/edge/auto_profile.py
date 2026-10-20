"""``deploy_profile="auto"``: materialize a hardware-adapted deploy YAML.

Chain: base deploy YAML (``edge/<model_type>.yaml`` if present, else the model
default) -> hardware-class overlay (``edge/hardware/<class>.yaml``) -> derived
overrides (:func:`vllm_omni.edge.adapt.derive_overrides`) -> cached calibration
(:mod:`vllm_omni.edge.calibrate`). The result is written to a temp file and
returned as a normal ``deploy_config`` path, so nothing downstream changes.
"""

from __future__ import annotations

import atexit
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger

from vllm_omni.config.stage_config import _DEPLOY_DIR, resolve_deploy_yaml
from vllm_omni.edge import calibrate
from vllm_omni.edge.adapt import derive_overrides, estimate_weights_bytes
from vllm_omni.edge.hardware_probe import HardwareProfile, describe, hardware_class, load_profile
from vllm_omni.edge.scheduling import KVGeometry

logger = init_logger(__name__)

HARDWARE_DIR = _DEPLOY_DIR / "edge" / "hardware"


def _merge_stage_list(base_stages: list[dict[str, Any]], overlay_stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {int(s.get("stage_id", i)): dict(s) for i, s in enumerate(base_stages)}
    for s in overlay_stages or []:
        sid = int(s.get("stage_id", 0))
        target = by_id.setdefault(sid, {"stage_id": sid})
        for k, v in s.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                target[k] = {**target[k], **v}
            else:
                target[k] = v
    return [by_id[k] for k in sorted(by_id)]


def _connector_extra(doc: dict[str, Any]) -> dict[str, Any] | None:
    connectors = doc.get("connectors")
    if not isinstance(connectors, dict) or not connectors:
        return None
    first = next(iter(connectors.values()))
    if not isinstance(first, dict):
        return None
    return first.setdefault("extra", {})


def apply_hardware_overlay(doc: dict[str, Any], hw_class: str) -> dict[str, Any]:
    path = HARDWARE_DIR / f"{hw_class}.yaml"
    if not path.exists():
        return doc
    with open(path, encoding="utf-8") as f:
        overlay = yaml.safe_load(f) or {}
    out = copy.deepcopy(doc)
    if overlay.get("stages"):
        out["stages"] = _merge_stage_list(out.get("stages", []), overlay["stages"])
    if isinstance(overlay.get("connectors"), dict):
        extra = _connector_extra(out)
        ov_extra = _connector_extra(overlay)
        if extra is not None and ov_extra:
            extra.update(ov_extra)
    for k, v in overlay.items():
        if k not in ("stages", "connectors", "platforms", "base_config"):
            out[k] = v
    return out


def apply_derived_overrides(doc: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(doc)
    stage_updates = []
    for sid, upd in overrides.get("stages", {}).items():
        entry: dict[str, Any] = {"stage_id": int(sid)}
        for k, v in upd.items():
            if k == "gpu_memory_utilization" and v is None:
                continue  # keep the YAML fraction unless an absolute budget replaces it
            entry[k] = v
        stage_updates.append(entry)
    out["stages"] = _merge_stage_list(out.get("stages", []), stage_updates)
    for k, v in overrides.get("top", {}).items():
        out[k] = v
    extra = _connector_extra(out)
    if extra is not None:
        extra.update(overrides.get("connector_extra", {}))
    return out


def apply_calibration(doc: dict[str, Any], cal: dict[str, Any] | None) -> dict[str, Any]:
    if not cal:
        return doc
    out = copy.deepcopy(doc)
    extra = _connector_extra(out)
    if extra is not None:
        extra.update(calibrate.connector_extra_from_calibration(cal))
    return out


def _model_dir(model: str) -> str | None:
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception:
        return None


def load_kv_geometry(model_dir: str | Path | None) -> KVGeometry | None:
    """Talker KV geometry from ``<model_dir>/config.json`` (None when unavailable)."""
    if not model_dir:
        return None
    cfg_path = Path(model_dir) / "config.json"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return KVGeometry.from_hf_config(json.load(f))
    except (OSError, ValueError):
        return None


def materialize_auto_deploy(
    model: str,
    base_yaml: str | Path,
    *,
    profile: HardwareProfile | None = None,
    out_dir: str | Path | None = None,
    use_calibration: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Build the adapted deploy YAML; returns ``(path, report)``."""
    prof = profile or load_profile()
    hw_class = hardware_class(prof)
    doc = resolve_deploy_yaml(Path(base_yaml))
    doc.pop("base_config", None)
    # ``platforms:`` overlays are applied later by the config factory and would
    # override the derived values (thread binding, budgets, eager switches);
    # the hardware-class overlay + derivation supersede them in auto mode.
    doc.pop("platforms", None)
    doc = apply_hardware_overlay(doc, hw_class)
    model_dir = _model_dir(model)
    weights = estimate_weights_bytes(model_dir)
    overrides = derive_overrides(
        prof,
        weights_bytes=weights,
        n_stages=len(doc.get("stages", [])) or 2,
        kv_geometry=load_kv_geometry(model_dir),
    )
    doc = apply_derived_overrides(doc, overrides)
    cal = None
    if use_calibration:
        cal = calibrate.load_calibration(calibrate.hardware_fingerprint(prof, model))
        doc = apply_calibration(doc, cal)
    target_dir = Path(out_dir) if out_dir else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="auto_", suffix=".yaml", dir=str(target_dir))
    else:
        fd, path = tempfile.mkstemp(prefix=f"omni_auto_{hw_class}_", suffix=".yaml")
        atexit.register(lambda p=path: Path(p).unlink(missing_ok=True))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"# auto-generated by vllm_omni.edge.auto_profile for {model}\n# {describe(prof)}\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    report = {
        "hardware_class": hw_class,
        "profile": prof.to_dict(),
        "base_yaml": str(base_yaml),
        "weights_bytes": weights,
        "overrides": overrides,
        "calibration": cal,
        "deploy_config": path,
    }
    logger.info("[auto_profile] %s -> %s (%s)", hw_class, path, "; ".join(overrides.get("notes", [])))
    return path, report
