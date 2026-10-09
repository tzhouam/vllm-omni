# SPDX-License-Identifier: Apache-2.0
"""Pure metadata checks shared by live and packaged memory-fixture evaluators.

The expectation comes from committed seed readback before a timed request;
matching it to actual captured input is required, never inferred from an answer.
This module imports no benchmark, backend, platform, or model implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

MEMORY_PROVENANCE_SCHEMA = "omni-agent-memory-fixture-provenance-v1"
MEMORY_FIXTURE_SOURCE = "benchmark-fixture:memory"
MEMORY_FIXTURE_SCOPE = "isolated_seed_recall_within_one_open_controller"
MEMORY_SETUP_WORKSPACE_BYTES = 128 << 10
MEMORY_CONSUMER_PREFIXES = {
    "external.strata.text.v1": "strata-agent:",
    "external.llamacpp.text.v1": "llamacpp-agent:",
}


def text_identity(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "utf8_bytes": len(encoded),
        "chars": len(text),
    }


def memory_source_verified(case: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Fail closed unless committed seed and actual first input are linked."""
    if not isinstance(case, Mapping) or not isinstance(result, Mapping):
        return False
    try:
        setup = result["fixture_setup"]
        request = setup["profile_request"]
        expected = setup["memory_provenance"]
        input_contract = setup["input_contract"]
        evidence = result["placement_evidence"]
        records = evidence["model_step_identities"]
        first = evidence["first_model_prompt_identity"]
        captured = records[0]
        required_contract = {
            "submission_mode": "model_selected_tools_v1", "case_id": case["case_id"],
            "instruction_sha256": text_identity(case["prompt"])["sha256"],
            "instruction_utf8_bytes": text_identity(case["prompt"])["utf8_bytes"],
            "explicit_read_url": None,
        }
        contract_digest = hashlib.sha256(json.dumps(
            required_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        reference = case["reference"]
        actual = {key: captured[key] for key in ("sha256", "utf8_bytes", "chars")}
        return bool(
            case["task_class"] == "memory" and case["metadata"].get("kind") == "memory_recall"
            and case["metadata"].get("source") == MEMORY_FIXTURE_SOURCE
            and isinstance(reference, str) and reference.strip()
            and reference.casefold() not in case["prompt"].casefold()
            and result["complete_agent_trace"] and result["trace_scope"] == "agent_e2e"
            and not result["tool_decisions"]
            and setup.get("private_memory_reset") is True
            and isinstance(request, Mapping)
            and set(request) == {"run_id", "route_id", "case_id", "phase", "repetition"}
            and isinstance(request.get("run_id"), str) and bool(request["run_id"])
            and request.get("case_id") == case["case_id"]
            and request.get("phase") in {"warmup", "measured", "endurance"}
            and request["phase"] == setup.get("phase")
            and type(request.get("repetition")) is int and request["repetition"] >= 0
            and type(setup.get("repetition")) is int
            and request["repetition"] == setup["repetition"]
            and isinstance(expected, Mapping)
            and expected.get("schema") == MEMORY_PROVENANCE_SCHEMA
            and expected.get("scope") == MEMORY_FIXTURE_SCOPE
            and expected.get("case_id") == case["case_id"]
            and expected.get("route_id") == request.get("route_id")
            and expected.get("artifact_id") == result["artifact_id"]
            and expected.get("backend") == result["backend"]
            and expected.get("seed_source") == setup.get("seed_source") == MEMORY_FIXTURE_SOURCE
            and expected.get("seed_event_id") == setup.get("seed_event_id")
            and isinstance(expected.get("seed_event_id"), str)
            and 1 <= len(expected["seed_event_id"]) <= 128
            and expected.get("seed_kind") == "user_observation"
            and expected.get("seed_session_id") == f"fixture-{case['case_id']}"
            and expected.get("seed_request_id") == "memory-seed"
            and isinstance(expected.get("controller_session_id"), str)
            and bool(expected["controller_session_id"])
            and expected["controller_session_id"] != expected["seed_session_id"]
            and expected.get("task_sha256") == required_contract["instruction_sha256"]
            and expected.get("reference_sha256") == text_identity(reference)["sha256"]
            and expected.get("setup_workspace_bytes") == MEMORY_SETUP_WORKSPACE_BYTES
            and input_contract == {**required_contract, "contract_sha256": contract_digest}
            and isinstance(expected.get("consumer_identity_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", expected["consumer_identity_sha256"])
            and result["backend"] in MEMORY_CONSUMER_PREFIXES
            and result["artifact_id"] == MEMORY_CONSUMER_PREFIXES[result["backend"]]
            + expected["consumer_identity_sha256"]
            and type(evidence.get("model_prompt_step_count")) is int
            and evidence["model_prompt_step_count"] == 1 and len(records) == 1
            and type(captured.get("step")) is int and captured["step"] == 0
            and isinstance(captured.get("model_request_id"), str)
            and captured["model_request_id"].endswith("-step-0")
            and type(first.get("step")) is int
            and first == {"step": 0, **actual}
            and isinstance(actual["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", actual["sha256"])
            and all(type(actual[key]) is int and actual[key] > 0 for key in ("utf8_bytes", "chars"))
            and actual == expected.get("expected_first_model_input")
            and actual["utf8_bytes"] <= 16384
        )
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, UnicodeError):
        return False

