# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicitly versioned interpretation of raw Agent model responses.

No decoder constraint, response extraction, retry, tool execution or engine.
The caller must bind the returned consumer identity and admit its workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SOURCE_FILE = Path(__file__).resolve()
_SOURCE_SHA256 = hashlib.sha256(_SOURCE_FILE.read_bytes()).hexdigest()

SCHEMA = "omni-agent-output-contract-v1"
MODES = frozenset({"strict_raw_agent_json_v1", "strict_outer_json_fence_agent_v1"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _positive(value: Any, maximum: int, label: str) -> None:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"invalid {label}")


@dataclass(frozen=True)
class AgentOutputContract:
    mode: str
    max_response_bytes: int = 65536
    max_depth: int = 16
    max_nodes: int = 2048
    max_directive_bytes: int = 8192
    workspace_budget_bytes: int = 2 << 20
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.mode not in MODES:
            raise ValueError("unknown Agent output contract")
        for key, maximum in (
            ("max_response_bytes", 1 << 20),
            ("max_depth", 32),
            ("max_nodes", 4096),
            ("max_directive_bytes", 16384),
            ("workspace_budget_bytes", 32 << 20),
        ):
            _positive(getattr(self, key), maximum, key)
        # Declared conservative parser workspace; not an observed/process hard cap.
        if self.workspace_budget_bytes < self.minimum_workspace_bytes:
            raise ValueError("Agent output parser workspace is not admitted")

    @property
    def minimum_workspace_bytes(self) -> int:
        return 8 * self.max_response_bytes + 512 * self.max_nodes + 2 * self.max_directive_bytes + 65536

    @classmethod
    def from_dict(cls, value: Any) -> AgentOutputContract:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("output contract needs exact versioned fields")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity_sha256(self) -> str:
        return _hash(self.to_dict())

    def consumer_identity(self, base_artifact_id: str) -> dict[str, Any]:
        if not isinstance(base_artifact_id, str) or not base_artifact_id:
            raise ValueError("base engine artifact identity required")
        if hashlib.sha256(_SOURCE_FILE.read_bytes()).hexdigest() != _SOURCE_SHA256:
            raise ValueError("Agent output adapter source changed after import")
        value = {
            "schema": "omni-agent-output-consumer-v1",
            "base_artifact_id": base_artifact_id,
            "contract": self.to_dict(),
            "adapter_source_sha256": _SOURCE_SHA256,
        }
        return value | {"identity_sha256": _hash(value)}

    def admit(self, *, max_io_bytes: int, workspace_reserved_bytes: int) -> None:
        if (
            type(max_io_bytes) is not int
            or self.max_response_bytes > max_io_bytes
            or self.max_directive_bytes > max_io_bytes
            or type(workspace_reserved_bytes) is not int
            or workspace_reserved_bytes < self.workspace_budget_bytes
        ):
            raise ValueError("output contract exceeds existing I/O or explicit parser workspace admission")

    def directive(self, permitted_tools: frozenset[str]) -> dict[str, Any]:
        if (
            not isinstance(permitted_tools, frozenset)
            or len(permitted_tools) > 32
            or any(not isinstance(x, str) or not x or len(x) > 64 for x in permitted_tools)
        ):
            raise ValueError("invalid trusted tool subset")
        schema = {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["final"],
                    "additionalProperties": False,
                    "properties": {"final": {"type": "string"}},
                },
                {
                    "type": "object",
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                    "properties": {"tool": {"enum": sorted(permitted_tools)}, "args": {"type": "object"}},
                },
            ]
        }
        value = {
            "contract_schema": self.schema,
            "mode": self.mode,
            "instruction": "Return exactly one JSON object matching the envelope, without prose or fences. "
            "Tool arguments remain subject to the application's exact operation schema and permissions.",
            "envelope_schema": schema,
            "max_response_bytes": self.max_response_bytes,
        }
        if len(_canonical(value)) > self.max_directive_bytes:
            raise ValueError("bound output directive exceeds admitted bytes")
        return value


def _scan_depth(text: str, maximum: int) -> None:
    depth, quoted, escape = 0, False, False
    for char in text:
        if quoted:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > maximum:
                raise ValueError("Agent JSON depth exceeds its contract")
        elif char in "]}":
            depth -= 1
    # Syntax and balanced delimiters are checked by the full JSON parser.


def _strict_object(text: str, contract: AgentOutputContract) -> dict[str, Any]:
    _scan_depth(text, contract.max_depth)

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate Agent JSON key")
            value[key] = item
        return value

    def constant(_):
        raise ValueError("nonfinite Agent JSON constant")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("complete raw Agent response is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Agent response must be an object")
    pending, nodes = [value], 0
    while pending:
        item = pending.pop()
        nodes += 1
        if nodes > contract.max_nodes:
            raise ValueError("Agent JSON node count exceeds its contract")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite Agent JSON number")
    # Also rejects isolated escaped surrogate values before any tool proposal.
    _canonical(value)
    return value


class AgentOutputBuffer:
    """One byte buffer, no visible deltas; only a complete stop may be interpreted."""

    def __init__(
        self,
        contract: AgentOutputContract,
        *,
        request_id: str,
        worker_generation: str,
        stage_id: int,
        permitted_tools: frozenset[str],
        previous_epoch: int = 0,
    ) -> None:
        if (
            not isinstance(request_id, str)
            or not 0 < len(request_id) <= 256
            or not isinstance(worker_generation, str)
            or not 0 < len(worker_generation) <= 128
            or type(stage_id) is not int
            or stage_id < 0
        ):
            raise ValueError("exact model request/stage generation required")
        self.contract, self.request_id = contract, request_id
        self.generation, self.stage_id = worker_generation, stage_id
        if type(previous_epoch) is not int or previous_epoch < 0:
            raise ValueError("invalid previous model epoch")
        self.previous_epoch = previous_epoch
        self.permitted = permitted_tools
        self.directive = contract.directive(permitted_tools)
        self._raw = bytearray()
        self._terminal: dict | None = None
        self._finished = False

    def discard(self) -> None:
        """Drop consumer-owned payload before worker retirement can release its lease."""
        self._raw.clear()
        self.directive.clear()
        self._terminal = None
        self._finished = True

    def append(self, text: str) -> None:
        if self._terminal is not None or self._finished or not isinstance(text, str):
            raise ValueError("Agent output after terminal or invalid text")
        encoded = text.encode("utf-8")
        if len(self._raw) + len(encoded) > self.contract.max_response_bytes:
            raise ValueError("Agent model response exceeds its admitted bytes")
        self._raw.extend(encoded)

    def terminal(self, metrics: dict) -> None:
        event = metrics.get("stage_event") if isinstance(metrics, dict) else None
        if (
            self._terminal is not None
            or self._finished
            or not isinstance(metrics, dict)
            or metrics.get("finish_reason") != "stop"
            or not isinstance(event, dict)
            or event.get("terminal") is not True
            or event.get("kind") != "text"
            or event.get("request_id") != self.request_id
            or event.get("worker_generation") != self.generation
            or type(event.get("stage_id")) is not int
            or event.get("stage_id") != self.stage_id
            or type(event.get("epoch")) is not int
            or event["epoch"] <= self.previous_epoch
            or type(event.get("seq")) is not int
            or event["seq"] <= 0
            or (
                metrics.get("raw_model_output_sha256") is not None
                and metrics["raw_model_output_sha256"] != hashlib.sha256(self._raw).hexdigest()
            )
        ):
            raise ValueError("Agent output lacks an exact successful terminal")
        self._terminal = {
            key: event[key]
            for key in ("request_id", "worker_generation", "stage_id", "epoch", "seq", "kind", "terminal")
        }

    def finish(self) -> tuple[str, Any, dict]:
        if self._terminal is None or self._finished:
            raise ValueError("Agent output is incomplete or already interpreted")
        self._finished = True
        raw_hash = hashlib.sha256(self._raw).hexdigest()
        text = self._raw.decode("utf-8")
        self._raw.clear()
        normalized, body = "none", text
        if self.contract.mode == "strict_outer_json_fence_agent_v1" and text.strip(" \t\r\n").startswith("```"):
            lines = text.strip(" \t\r\n").replace("\r\n", "\n").split("\n")
            if (
                len(lines) < 3
                or lines[0] != "```json"
                or lines[-1] != "```"
                or any(line.strip().startswith("```") for line in lines[1:-1])
            ):
                raise ValueError("Agent response is not one exact outer json fence")
            body, normalized = "\n".join(lines[1:-1]), "single_outer_json_fence"
        value = _strict_object(body, self.contract)
        if set(value) == {"final"} and isinstance(value["final"], str):
            command, payload = "final", value["final"]
        elif (
            set(value) == {"tool", "args"}
            and isinstance(value["tool"], str)
            and value["tool"] in self.permitted
            and isinstance(value["args"], dict)
        ):
            command, payload = "tool", value
        else:
            raise ValueError("Agent output violates the exact command envelope")
        proof = {
            "schema": "omni-agent-output-interpretation-v1",
            "contract_sha256": self.contract.identity_sha256,
            "mode": self.contract.mode,
            "directive_sha256": _hash(self.directive),
            "raw_output_sha256": raw_hash,
            "canonical_output_sha256": _hash(value),
            "normalization": normalized,
            "stage_event": self._terminal,
            "constrained_decoding": False,
            "extraction_used": False,
            "retry_count": 0,
        }
        return command, payload, proof


async def collect_agent_command(chunks, *, buffer: AgentOutputBuffer, cancel):
    """Consume without exposing any proposal; close on consumer-side failure.

    The existing Omni generator retains stage ACK/cancel ownership. `cancel`
    is the exact owned model request cancellation callback, never a retry.
    """
    terminal_metrics = None
    try:
        async for chunk in chunks:
            if chunk.text:
                buffer.append(chunk.text)
            if chunk.terminal:
                terminal_metrics = dict(chunk.metrics)
                buffer.terminal(terminal_metrics)
        command, value, proof = buffer.finish()
        buffer.discard()
        return command, value, proof, terminal_metrics
    except BaseException as exc:
        needs_cancel = buffer._terminal is None
        buffer.discard()
        # Do not retain decoded input through parser exception frames while
        # cancelling a worker that may release the shared consumer lease.
        exc.__traceback__ = None
        exc.__cause__ = None
        exc.__context__ = None
        chunk = None
        if needs_cancel:
            await cancel(buffer.request_id)
        raise exc from None
    finally:
        closer = getattr(chunks, "aclose", None)
        if callable(closer):
            await closer()


def validate_output_contract_entry(entry: dict) -> AgentOutputContract | None:
    """Bind a configured consumer to the exact engine plan and imported source.

    Explicit modes initially use the existing Strata complete-model adapter.
    Plain-text entries do not change identity or behavior.
    """
    value = entry.get("model_output_contract")
    if value is None:
        if any(
            entry.get(key) is not None
            for key in ("base_artifact_id", "model_output_consumer_identity", "model_output_workspace_bytes")
        ):
            raise ValueError("consumer metadata without an explicit output contract")
        return None
    contract = AgentOutputContract.from_dict(value)
    if entry.get("backend") == "external.llamacpp.text.v1":
        from vllm_omni.edge.agent.llamacpp_route import validate_llamacpp_consumer_entry

        validate_llamacpp_consumer_entry(entry)
        return contract
    backend = entry.get("backend_config")
    if (
        not isinstance(backend, dict)
        or backend.get("name") not in {"external.strata.text.v1", "external.strata.multimodal.v1"}
        or entry.get("backend") != backend["name"]
    ):
        raise ValueError("explicit output contract requires a supported complete-model stage")
    base = "strata:" + _hash(backend)
    identity = contract.consumer_identity(base)
    if (
        entry.get("base_artifact_id") != base
        or entry.get("model_output_consumer_identity") != identity
        or entry.get("artifact_id") != "strata-agent:" + identity["identity_sha256"]
        or entry.get("model_output_workspace_bytes") != contract.workspace_budget_bytes
    ):
        raise ValueError("Agent consumer identity or explicit workspace differs")
    contract.admit(max_io_bytes=entry["max_io_bytes"], workspace_reserved_bytes=entry["model_output_workspace_bytes"])
    budget = backend["weight_tier_plan"]["budget"]
    if (
        type(budget.get("host_workspace_bytes")) is not int
        or budget["host_workspace_bytes"] < contract.workspace_budget_bytes
        or budget["host_transfer_bytes"] < entry["max_io_bytes"]
    ):
        raise ValueError("typed tier budget omits the Agent consumer workspace or I/O")
    return contract
