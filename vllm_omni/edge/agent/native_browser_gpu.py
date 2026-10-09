# SPDX-License-Identifier: Apache-2.0
"""Opt-in native browser GPU accounting under the existing companion lease.

Identity metadata is not model-operator placement, physical memory union, an
enforced cap or qualification. No work is scheduled and no allowance is chosen.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.engine.local_plan import CompanionResourceSpec
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable


def _sha256(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native GPU descriptor key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("non-finite native GPU descriptor constant: " + value)


@dataclass(frozen=True)
class NativeGpuAccountingDeclaration:
    path: str
    sha256: str
    cuda_ordinal: int
    gpu_pool: str
    gpu_identity_json: str

    @property
    def gpu_identity(self) -> dict[str, Any]:
        return json.loads(self.gpu_identity_json)


def resolve_native_gpu_accounting(path: Path, *, expected_sha256: str) -> NativeGpuAccountingDeclaration:
    """Resolve a pinned target identity; no budget or native calls here."""
    from vllm_omni.edge.windows_gpu_memory import normalize_pci_bus_id

    if not _sha256(expected_sha256):
        raise ValueError("an exact native GPU descriptor SHA256 is required")
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    if len(raw) > 64 << 10 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("native GPU descriptor bound or SHA256 differs")
    body = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    names = {"schema", "cuda_ordinal", "gpu_pool", "gpu_identity", "coverage",
             "all_descendants_covered", "hard_process_cap"}
    if (type(body) is not dict or set(body) != names
            or body["schema"] != "omni-browser-native-gpu-accounting-v1"
            or body["coverage"] != "exact_retained_handle_bound_set_only"
            or body["all_descendants_covered"] is not False or body["hard_process_cap"] is not False
            or type(body["cuda_ordinal"]) is not int or not 0 <= body["cuda_ordinal"] < 1 << 31
            or body["gpu_pool"] not in (
                {"vram", "vram:0"} if body["cuda_ordinal"] == 0 else {f"vram:{body['cuda_ordinal']}"})):
        raise ValueError("explicit bounded native GPU target and accounting scope are required")
    identity = body["gpu_identity"]
    if (type(identity) is not dict or set(identity) != {"uuid", "pci_bus_id", "name_sha256"}
            or type(identity["uuid"]) is not str or not re.fullmatch(
                r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", identity["uuid"])
            or not _sha256(identity["name_sha256"])):
        raise ValueError("native GPU target needs UUID/PCI/name identity")
    normalize_pci_bus_id(identity["pci_bus_id"])
    return NativeGpuAccountingDeclaration(str(path), expected_sha256, body["cuda_ordinal"],
        body["gpu_pool"], json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False))


class NativeBrowserGpuRegistryFactory:
    """App-owned provider until registry construction returns, with finite reports.

    The source callback belongs to the application, never a tool/model/page. It
    must not call the manager or owner guard: construction runs under the existing
    companion lock. The exact ledger/token is borrowed for identity checks only.
    No provider close is retried and an undrained failure blocks new inventory.
    """

    def __init__(self, declaration: NativeGpuAccountingDeclaration, spec: CompanionResourceSpec,
                 *, cdp_helper_capability: Any = None, review_metadata: Mapping[str, Any] | None = None) -> None:
        from vllm_omni.edge.windows_cdp_helper import CdpObservedHelperCapability

        if type(declaration) is not NativeGpuAccountingDeclaration or type(spec) is not CompanionResourceSpec:
            raise ValueError("resolved native GPU declaration and exact companion spec are required")
        gpu_claims = {key: value for key, value in spec.memory_demands.items()
                      if key == "vram" or key.startswith("vram:")}
        if gpu_claims != {declaration.gpu_pool: spec.memory_demands.get(declaration.gpu_pool)} or (
                type(gpu_claims.get(declaration.gpu_pool)) is not int or gpu_claims[declaration.gpu_pool] <= 0):
            raise ValueError("native GPU accounting needs one positive reviewed exact GPU pool allowance")
        if cdp_helper_capability is not None and type(cdp_helper_capability) is not CdpObservedHelperCapability:
            raise ValueError("only the resolved app-owned CDP helper capability is accepted")
        self.declaration = declaration
        self._claims = dict(spec.memory_demands)
        self._review_metadata = deepcopy(dict(review_metadata)) if review_metadata is not None else None
        self._helper = cdp_helper_capability
        self._source: Callable[[], Mapping[str, Any]] | None = None
        self._ledger: ResourceLedger | None = None
        self._token: Reservation | None = None
        self._generation: str | None = None
        self._lock = threading.RLock()
        self._attempt_count = 0
        self._records: list[dict[str, Any]] = []
        self._retained_failed_providers: list[Any] = []
        self._constructing_provider: Any = None
        self._retained_failed_registries: list[Any] = []
        self._constructing_registry: Any = None

    def bind_route_source(self, source: Callable[[], Mapping[str, Any]]) -> None:
        with self._lock:
            if not callable(source) or self._source is not None or self._token is not None or self._records:
                raise ValueError("native GPU source must be bound once while the application factory is cold")
            self._source = source

    def bind_resource_context(self, ledger: ResourceLedger, token: Reservation, *, joint_generation: str) -> None:
        """Synchronous exact-token storage only; no native work or manager calls."""
        with self._lock:
            self._check_no_failed_provider()
            if (self._source is None or not isinstance(ledger, ResourceLedger)
                    or not isinstance(token, Reservation) or not ledger.owns(token)
                    or dict(token.demands) != self._claims
                    or type(joint_generation) is not str or not 1 <= len(joint_generation) <= 256
                    or token.owner in ledger.snapshot()["quarantined"]
                    or (self._token is not None and (self._ledger is None
                        or not self._ledger.was_released(self._token) or self._generation == joint_generation))):
                raise ResourceUnavailable("native GPU factory requires a fresh exact companion lease")
            self._ledger, self._token, self._generation = ledger, token, joint_generation

    def _check_no_failed_provider(self) -> None:
        if (self._retained_failed_providers or self._constructing_provider is not None
                or self._retained_failed_registries or self._constructing_registry is not None):
            raise ResourceUnavailable("native GPU provider cleanup is terminally quarantined; no new inventory")

    def _source_snapshot(self, generation: str) -> dict[str, Any]:
        ledger, token = self._ledger, self._token
        if (ledger is None or token is None or generation != self._generation or self._source is None
                or not ledger.owns(token) or dict(token.demands) != self._claims):
            raise ResourceUnavailable("native GPU factory has no current exact companion token")
        status = ledger.snapshot()
        if token.owner in status["quarantined"]:
            raise ResourceUnavailable("native GPU companion token is quarantined")
        source = deepcopy(dict(self._source()))
        if (set(source) != {"gpu_identity", "gpu_index", "gpu_pool", "route_scope", "route_identity"}
                or source["gpu_identity"] != self.declaration.gpu_identity
                or type(source["gpu_index"]) is not int or source["gpu_index"] != self.declaration.cuda_ordinal
                or source["gpu_pool"] != self.declaration.gpu_pool
                or source["route_scope"] not in {"target_only_characterization", "current_loaded_route"}
                or (source["route_scope"] == "target_only_characterization" and source["route_identity"] is not None)
                or (source["route_scope"] == "current_loaded_route"
                    and not isinstance(source["route_identity"], dict))):
            raise ResourceUnavailable("current application GPU/route identity differs from the declared target")
        # Bound app-owned metadata as well as saved native capability arrays.
        encoded = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > 64 << 10:
            raise ValueError("native GPU route identity exceeds the metadata bound")
        if (self._ledger is not ledger or self._token is not token or self._generation != generation
                or not ledger.owns(token) or token.owner in ledger.snapshot()["quarantined"]
                or dict(token.demands) != self._claims):
            raise ResourceUnavailable("native GPU companion lease changed during source capture")
        return {"source": source, "ledger_capacities": dict(ledger.capacities),
                "companion_owner": token.owner, "companion_demands": dict(token.demands),
                "generation": generation}

    @staticmethod
    def _close_projection(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {"valid": False, "invalid_reason": "not_mapping"}
        names = ("drained", "adapter_handles_closed", "borrowed_process_handles_retained")
        if (value.get("schema") != "omni-windows-bound-process-gpu-memory-v1"
                or any(type(value.get(name)) is not bool for name in names)
                or type(value.get("process_handles_closed_by_provider")) is not int
                or value["process_handles_closed_by_provider"] != 0):
            return {"valid": False, "invalid_reason": "missing_or_invalid_terminal_fields"}
        return {"valid": True, **{name: value[name] for name in names}, "process_handles_closed_by_provider": 0}

    def _close_owned_failure(self, provider: Any, record: dict[str, Any]) -> None:
        self._retained_failed_providers.append(provider)
        record["provider_close_attempted"] = True
        try:
            value = provider.close()
        except BaseException as exc:
            record["provider_close_error_type"] = type(exc).__name__[:128]
            record["provider_close_verified"] = False
            return
        try:
            projection = self._close_projection(value)
            record["provider_close_receipt"] = projection
            verified = (projection.get("valid") is True and projection.get("drained") is True
                        and projection.get("adapter_handles_closed") is True
                        and projection.get("borrowed_process_handles_retained") is False)
        except BaseException as exc:
            record["provider_close_projection_error_type"] = type(exc).__name__[:128]
            verified = False
        record["provider_close_verified"] = verified
        if verified:
            if self._retained_failed_providers[-1] is not provider:
                raise RuntimeError("factory provider retention order changed")
            self._retained_failed_providers.pop()

    def _close_registry_failure(self, registry: Any, record: dict[str, Any], generation: str) -> None:
        """After constructor return, only the registry owns provider cleanup."""
        from vllm_omni.edge.agent.browser_resources import _ownership_coverage_verified

        self._retained_failed_registries.append(registry)
        record["registry_close_attempted"] = True
        try:
            receipt = registry.close()
            children = receipt.get("children") if isinstance(receipt, Mapping) else None
            gpu = self._close_projection(children.get("gpu_sampling_close")
                                         if isinstance(children, Mapping) else None)
            verified = (isinstance(receipt, Mapping)
                and receipt.get("schema") == "omni-windows-bound-process-memory-v1"
                and receipt.get("cohort_generation") == generation
                and receipt.get("observer_handles_closed") is True
                and receipt.get("all_descendants_retired") is False
                and isinstance(children, Mapping) and children.get("cohort_generation") == generation
                and children.get("bound_set_drain_verified") is True
                and children.get("known_bound_children_retired") is True
                and children.get("required_owned_bindings_verified") is True
                and children.get("all_descendants_retired") is False
                and _ownership_coverage_verified(receipt) and _ownership_coverage_verified(children)
                and gpu.get("valid") is True and gpu.get("drained") is True
                and gpu.get("adapter_handles_closed") is True
                and gpu.get("borrowed_process_handles_retained") is False)
            record["registry_close_receipt"] = {
                "receipt_is_mapping": isinstance(receipt, Mapping),
                "exact_generation_and_finite_ownership_drain_verified": verified,
                "gpu_sampling_close": gpu}
        except BaseException as exc:
            record["registry_close_error_type"] = type(exc).__name__[:128]
            verified = False
        record["registry_close_verified"] = verified
        if verified:
            if self._retained_failed_registries[-1] is not registry:
                raise RuntimeError("factory registry retention order changed")
            self._retained_failed_registries.pop()

    def __call__(self, generation: str) -> Any:
        with self._lock:
            self._check_no_failed_provider()  # Must precede source callback and any native inventory.
            if type(generation) is not str or not 1 <= len(generation) <= 256:
                raise ValueError("native GPU factory generation is required and bounded")
            if self._records and self._records[-1].get("generation") == generation:
                raise ResourceUnavailable("native GPU factory cannot reconstruct a generation")
            self._attempt_count += 1
            record: dict[str, Any] = {"attempt": self._attempt_count, "generation": generation,
                "phase": "descriptor_revalidation", "provider_created": False, "registry_returned": False,
                "provider_transferred_to_registry": False, "provider_close_attempted": False,
                "provider_close_verified": False}
            self._records.append(record)
            del self._records[:-16]
            provider = registry = None
            try:
                from vllm_omni.edge.agent.native_app import _dxgi_adapter_inventory, _resolve_native_gpu_pool_identity
                from vllm_omni.edge.windows_bound_gpu_memory import WindowsRetainedProcessGpuProvider
                from vllm_omni.edge.windows_process_memory import WindowsProcessMemoryRegistry

                if resolve_native_gpu_accounting(Path(self.declaration.path),
                        expected_sha256=self.declaration.sha256) != self.declaration:
                    raise ValueError("native GPU descriptor identity changed")
                record["phase"] = "application_context"
                before = self._source_snapshot(generation)
                record["source"] = before["source"]
                record["phase"] = "inventory"
                inventory = _dxgi_adapter_inventory(include_gpu_accounting=True)
                record["phase"] = "provider_constructor"
                provider = object.__new__(WindowsRetainedProcessGpuProvider)
                self._constructing_provider = provider
                record["provider_created"] = True
                WindowsRetainedProcessGpuProvider.__init__(provider, inventory)
                record["phase"] = "pool_identity_join"
                source = before["source"]
                record["join"] = _resolve_native_gpu_pool_identity(gpu_provider=provider,
                    gpu_identity=source["gpu_identity"], gpu_index=source["gpu_index"],
                    gpu_pool=source["gpu_pool"], ledger_capacities=before["ledger_capacities"])
                if before != self._source_snapshot(generation):
                    raise ResourceUnavailable("application GPU/route/ledger identity changed during native join")
                record["phase"] = "registry_constructor"
                registry = WindowsProcessMemoryRegistry(generation, gpu_provider=provider,
                    cdp_helper_capability=self._helper)
                # Constructor return is the ownership transfer boundary. Any
                # subsequent validation failure must close/retain this owner,
                # never bypass it by closing its provider directly.
                self._constructing_registry = registry
                self._constructing_provider = None
                record.update(registry_returned=True, provider_transferred_to_registry=True)
                if before != self._source_snapshot(generation):
                    raise ResourceUnavailable(
                        "application GPU/route/ledger identity changed during registry construction")
            except BaseException as primary:
                record["primary_error_type"] = type(primary).__name__[:128]
                record["status"] = "construction_failed"
                if registry is not None:
                    try:
                        self._close_registry_failure(registry, record, generation)
                    except BaseException as cleanup:
                        record["cleanup_bookkeeping_error_type"] = type(cleanup).__name__[:128]
                    if (record.get("registry_close_verified") is True
                            or any(row is registry for row in self._retained_failed_registries)):
                        self._constructing_registry = None
                elif provider is not None:
                    try:
                        self._close_owned_failure(provider, record)
                    except BaseException as cleanup:
                        record["cleanup_bookkeeping_error_type"] = type(cleanup).__name__[:128]
                if (provider is None or record.get("provider_close_verified") is True
                        or any(row is provider for row in self._retained_failed_providers)):
                    self._constructing_provider = None
                raise
            record.update(registry_returned=True, provider_transferred_to_registry=True,
                          status="transferred", phase="complete")
            self._constructing_provider = None
            self._constructing_registry = None
            return registry

    def snapshot(self) -> dict[str, Any]:
        """Bounded copied metadata only; no native query, guard or cleanup."""
        with self._lock:
            extra = (self._constructing_provider is not None
                     and not any(row is self._constructing_provider for row in self._retained_failed_providers))
            extra_registry = (self._constructing_registry is not None
                and not any(row is self._constructing_registry for row in self._retained_failed_registries))
            current = (self._records[-1] if self._records
                       and self._records[-1]["generation"] == self._generation else None)
            return deepcopy({"schema": "omni-browser-native-gpu-factory-report-v1", "enabled": True,
                "descriptor_path": self.declaration.path, "descriptor_sha256": self.declaration.sha256,
                "generation": self._generation, "current": current,
                "declared_incremental_memory_demands": self._claims, "review_metadata": self._review_metadata,
                "attempt_count": self._attempt_count, "retained_report_count": len(self._records),
                "report_limit": 16, "retained_failed_provider_count": len(self._retained_failed_providers) + int(extra),
                "constructing_provider_reference_retained": self._constructing_provider is not None,
                "retained_failed_registry_count": len(self._retained_failed_registries) + int(extra_registry),
                "constructing_registry_reference_retained": self._constructing_registry is not None,
                "failed_provider_close_retry_supported": False, "retained_failure_blocks_new_construction": True,
                "snapshot_performs_native_queries": False, "memory_allowance_selected": False,
                "model_operator_placement_verified": False, "hard_cap_verified": False, "qualification": False})
