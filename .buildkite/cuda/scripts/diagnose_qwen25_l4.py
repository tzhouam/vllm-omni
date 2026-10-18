# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Collect a controlled L4 backend/sampling comparison; this is not a CI gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict


class DiagnosticCase(TypedDict):
    backend: str
    case: str
    request_id: str
    text: str
    finish_reason: str
    usage: dict[str, object] | None
    request_overrides: dict[str, float]
    cuda_launch_blocking: str
    contains_expected_media_content: bool


class DiagnosticReport(TypedDict):
    purpose: str
    commit: str | None
    complete: bool
    cases: list[DiagnosticCase]


def prepare_configs(output: Path) -> list[tuple[str, Path]]:
    import yaml

    from tests.helpers.stage_config import get_deploy_config_path
    from vllm_omni.config.stage_config import load_deploy_config

    source = Path(get_deploy_config_path("ci/qwen2_5_omni.yaml"))
    configs = []
    for variant in ("control", "vision_sdpa", "cuda_sync"):
        config = yaml.safe_load(source.read_text())
        if variant == "vision_sdpa":
            thinker = next(stage for stage in config["stages"] if stage["stage_id"] == 0)
            thinker.setdefault("engine_extras", {})["mm_encoder_attn_backend"] = "TORCH_SDPA"
        path = output / f"{variant}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_deploy_config(path)
        configs.append((variant, path))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("qwen25-diagnostic"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_dir.resolve()))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = prepare_configs(output)
    if args.prepare_only:
        print(json.dumps({"configs": [str(path) for _, path in configs]}), flush=True)
        return

    from openai import OpenAI

    from examples.online_serving.openai_chat_completion_client_for_multimodal_generation import (
        get_mixed_modalities_query,
        get_system_prompt,
    )
    from tests.helpers.runtime import OmniServer

    model = "Qwen/Qwen2.5-Omni-7B"
    report: DiagnosticReport = {
        "purpose": "Diagnostic comparison, not a replacement for original quality assertions or full CI coverage.",
        "commit": os.environ.get("BUILDKITE_COMMIT"),
        "complete": False,
        "cases": [],
    }
    target = output / "results.json"
    target.write_text(json.dumps(report, indent=2))
    for variant, path in configs:
        with OmniServer(
            model,
            ["--deploy-config", str(path), "--stage-init-timeout", "600", "--init-timeout", "900"],
            port=None,
            env_dict={"CUDA_LAUNCH_BLOCKING": "1"} if variant == "cuda_sync" else None,
        ) as server:
            cases: list[tuple[str, str | None, dict[str, float]]] = [
                ("default", None, {}),
                ("penalty_one", None, {"repetition_penalty": 1.0}),
            ]
            if variant == "control":
                cases.append(
                    ("explicit_summary", "Analyze all the media content and provide a comprehensive summary.", {})
                )
            with OpenAI(
                api_key="EMPTY", base_url=f"http://127.0.0.1:{server.port}/v1", timeout=300, max_retries=0
            ) as client:
                for name, prompt, extra_body in cases:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[get_system_prompt(), get_mixed_modalities_query(custom_prompt=prompt)],
                        modalities=["text"],
                        extra_body=extra_body,
                    )
                    text = response.choices[0].message.content or ""
                    row: DiagnosticCase = {
                        "backend": variant,
                        "case": name,
                        "request_id": response.id,
                        "text": text,
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": response.usage.model_dump() if response.usage is not None else None,
                        "request_overrides": extra_body,
                        "cuda_launch_blocking": "1"
                        if variant == "cuda_sync"
                        else os.environ.get("CUDA_LAUNCH_BLOCKING", "unset"),
                        "contains_expected_media_content": all(
                            word in text.lower() for word in ("baby", "book", "lamb")
                        ),
                    }
                    report["cases"].append(row)
                    target.write_text(json.dumps(report, indent=2))
                    print(json.dumps(row), flush=True)
    report["complete"] = True
    target.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
