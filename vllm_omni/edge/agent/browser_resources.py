# SPDX-License-Identifier: Apache-2.0
"""Optional application companion for cold model/browser joint admission.

This adapter owns an explicitly measured incremental allowance and one lazy
browser owner. It neither caps processes nor proves all descendant coverage.
The engine owns reservation release; Windows attribution remains in the app.
Nothing creates this adapter or chooses an allowance by default.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.engine.local_plan import CompanionResourceSpec
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable


@dataclass(frozen=True)
class ResolvedBrowserEnvelope:
    """Integrity-checked external review, not an inferred safety bound.

    Callers must review that the referenced cohort actually covers the intended
    browser workload and overhead. Hashes bind that decision to immutable bytes;
    they cannot turn sampled lower bounds into an enforceable memory cap.
    """

    resource_spec: CompanionResourceSpec
    measurement_references: tuple[tuple[str, str], ...]


def _sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _ownership_coverage_verified(receipt: Mapping[str, Any]) -> bool:
    coverage = receipt.get("coverage")
    if (not isinstance(coverage, Mapping)
            or coverage.get("scope") != "exact_retained_handle_bound_set_only"
            or coverage.get("all_descendants_covered") is not False
            or coverage.get("binding_registry_overflow") is not False
            or coverage.get("unadopted_handle_close_failures") != 0):
        return False
    count, details = coverage.get("unknown_count"), coverage.get("unknown_details")
    if type(count) is not int or count < 0 or not isinstance(details, list):
        return False
    # Dropping an already-taken counter sample is informational. Missing CDP
    # membership, failed adoption, unknown types, observer or handle failures
    # can conceal required owned work and must prevent release. If detail
    # truncation prevents reconciling every unknown count, remain quarantined.
    return (all(isinstance(row, Mapping) and row.get("reason") == "owner_checkpoint_queue_overflow"
                and type(row.get("count")) is int and row["count"] > 0 for row in details)
            and sum(row["count"] for row in details) == count)


def resolve_reviewed_browser_envelope(path: Path, *, expected_sha256: str) -> ResolvedBrowserEnvelope:
    """Resolve an explicit external allowance; there is no fallback/default.

    This candidate schema is intentionally opt-in and must be populated only
    after the real attribution cohort is reviewed. RAM and Windows commit are
    separate constraints; neither is added to the other as physical memory.
    Referenced evidence is local, finite, and hash-pinned. No model, browser,
    telemetry thread, or registry is created while resolving the declaration.
    """
    if not _sha256(expected_sha256):
        raise ValueError("an explicit lowercase envelope SHA256 is required")
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("browser envelope exceeds the metadata bound or hash differs")
    body = json.loads(raw)
    if (not isinstance(body, dict) or body.get("schema") != "omni-browser-resource-envelope-v1"
            or body.get("reviewed_for_joint_admission") is not True
            or body.get("coverage") != "exact_retained_handle_bound_set_only"
            or body.get("hard_process_cap") is not False
            or body.get("all_descendants_covered") is not False):
        raise ValueError("an explicit scoped browser allowance review is required")
    demands = body.get("memory_demands")
    if (not isinstance(demands, dict)
            or any(type(demands.get(pool)) is not int or demands[pool] <= 0
                   for pool in ("host_ram", "windows_commit"))):
        raise ValueError("reviewed host RAM and Windows commit allowances are required")
    rows = body.get("measurement_evidence")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("bounded measured evidence references are required")
    references: list[tuple[str, str]] = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("path"), str)
                or not row["path"] or len(row["path"]) > 4096 or not _sha256(row.get("sha256"))):
            raise ValueError("evidence needs an exact local path and SHA256")
        evidence = Path(row["path"])
        if not evidence.is_absolute():
            evidence = path.parent / evidence
        evidence = evidence.resolve(strict=True)
        digest = hashlib.sha256()
        with evidence.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("browser measurement evidence hash differs")
        references.append((str(evidence), row["sha256"]))
    spec = CompanionResourceSpec(
        purpose_id=body.get("purpose_id"), memory_demands=demands,
        envelope_sha256=expected_sha256, evidence_reference=str(path),
    )
    return ResolvedBrowserEnvelope(spec, tuple(references))


class WindowsBrowserResourceCompanion:
    """One cold-factory browser owner lent one exact engine resource token.

    ``bind_resource_lease`` only stores the lease. ``browser`` creates the
    browser/registry lazily after admission. Owner-worker observer callbacks
    never acquire the manager lock. ``close`` bypasses all admission guards,
    joins the browser worker and verifies only the retained finite bound set.

    A failed finite drain is terminal for this adapter generation. The current
    registry closes query handles after its bounded wait, so retrying a failed
    receipt cannot establish later retirement. Recovery requires explicit
    external reconciliation, not a fresh factory or a forgotten reservation.
    """

    def __init__(
        self,
        envelope: ResolvedBrowserEnvelope,
        *,
        browser_factory: Callable[..., Any] | None = None,
        registry_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not isinstance(envelope, ResolvedBrowserEnvelope):
            raise ValueError("an explicitly resolved browser allowance is required")
        if not isinstance(envelope.resource_spec, CompanionResourceSpec) or not envelope.measurement_references:
            raise ValueError("a frozen allowance and reviewed measurement references are required")
        self._resource_spec = envelope.resource_spec
        self._envelope = envelope
        self._browser_factory = browser_factory
        self._registry_factory = registry_factory
        self._lock = threading.RLock()
        self._ledger: ResourceLedger | None = None
        self._reservation: Reservation | None = None
        self._generation: str | None = None
        self._guard: Callable[[], None] | None = None
        self._registry: Any = None
        self._browser: Any = None
        self._factory_attempted = False
        self._factory_failed = False
        self._closing = False
        self._close_attempted = False
        self._close_verified = False
        self._quarantined = False
        self._release_evidence: dict[str, Any] | None = None
        self._last_close_evidence: dict[str, Any] | None = None
        self._tool_boundary: Any = None

    @property
    def resource_spec(self) -> CompanionResourceSpec:
        return self._resource_spec

    @property
    def release_evidence(self) -> Mapping[str, Any] | None:
        with self._lock:
            return deepcopy(self._release_evidence)

    @property
    def last_close_evidence(self) -> Mapping[str, Any] | None:
        """Last generation's diagnostic receipt survives a verified cold reset.

        Only release_evidence is authority for the current exact lease. This
        historical receipt is never accepted by the engine for a later lease.
        """
        with self._lock:
            return deepcopy(self._last_close_evidence)

    def is_cold(self) -> bool:
        with self._lock:
            # A bound-but-unstarted owner is cold for cleanup but cannot be
            # rebound: bind separately requires that no lease is attached.
            return (self._browser is None and self._registry is None
                    and not self._factory_attempted and not self._close_attempted
                    and not self._closing and not self._quarantined)

    def bind_resource_lease(
        self, ledger: ResourceLedger, reservation: Reservation,
        *, joint_generation: str, guard: Callable[[], None],
    ) -> None:
        with self._lock:
            if (not self.is_cold() or self._reservation is not None
                    or not isinstance(joint_generation, str) or not joint_generation
                    or not callable(guard) or not ledger.owns(reservation)
                    or dict(reservation.demands) != dict(self._resource_spec.memory_demands)
                    or reservation.owner in ledger.snapshot()["quarantined"]):
                raise ResourceUnavailable("browser requires its exact cold joint companion lease")
            # Synchronous storage only. No OS counters, browser constructor,
            # worker submission, manager callback, or process launch here.
            self._ledger, self._reservation = ledger, reservation
            self._generation, self._guard = joint_generation, guard

    def guard(self) -> None:
        with self._lock:
            ledger, token, callback = self._ledger, self._reservation, self._guard
            if (callback is None or ledger is None or token is None or self._closing
                    or self._close_attempted or self._quarantined):
                raise ResourceUnavailable("browser companion is unadmitted, closing, or quarantined")
        # Never call the manager while holding the adapter lock. Manager close
        # holds its lifecycle lock and then enters this adapter from its caller.
        callback()
        with self._lock:
            if (self._reservation is not token or self._guard is not callback
                    or self._closing or self._close_attempted or self._quarantined
                    or not ledger.owns(token) or token.owner in ledger.snapshot()["quarantined"]):
                raise ResourceUnavailable("browser companion lease became stale")

    def _guard_generation(self, generation: str, token: Reservation) -> None:
        with self._lock:
            if self._generation != generation or self._reservation is not token:
                raise ResourceUnavailable("browser factory owner belongs to a retired joint generation")
        self.guard()
        with self._lock:
            if self._generation != generation or self._reservation is not token:
                raise ResourceUnavailable("browser factory generation changed during its guard")

    def _process_checkpoint(self, registry: Any, generation: str,
                            action: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        # Registry callbacks run on the owner worker. They intentionally do
        # not call guard(), tools, Playwright, or any manager callback.
        with self._lock:
            if registry is None or registry is not self._registry or generation != self._generation:
                raise RuntimeError("browser observer is not bound to its exact registry generation")
        return registry.browser_checkpoint(action, payload)

    def browser(self) -> Any:
        self.guard()
        with self._lock:
            if self._closing or self._close_attempted or self._quarantined:
                raise ResourceUnavailable("browser owner cannot be constructed after close")
            if self._browser is None:
                self._factory_attempted = True
                try:
                    if self._registry_factory is None:
                        from vllm_omni.edge.windows_process_memory import WindowsProcessMemoryRegistry
                        self._registry = WindowsProcessMemoryRegistry(self._generation)
                    else:
                        self._registry = self._registry_factory(self._generation)
                    if self._browser_factory is None:
                        from .tools import ManagedEdgeBrowser
                        factory = ManagedEdgeBrowser
                    else:
                        factory = self._browser_factory
                    registry, generation, token = self._registry, self._generation, self._reservation
                    # Factory constructors must be non-launching and must
                    # not call guard or another manager callback. Only methods
                    # on the returned browser may submit owner-worker work.
                    self._browser = factory(
                        process_observer=lambda action, payload: self._process_checkpoint(
                            registry, generation, action, payload),
                        resource_guard=lambda: self._guard_generation(generation, token),
                    )
                except BaseException:
                    self._factory_failed = self._quarantined = True
                    raise
            browser = self._browser
        self.guard()
        return browser

    def make_tool_boundary(self, *, settings: Any = None, screen: Any = None,
                           boundary_factory: Callable[..., Any] | None = None) -> Any:
        """Cold optional wiring; the boundary never caches a factory browser.

        The app may attach this boundary before constructing its manager. No
        browser/registry is created until a later guarded browser operation.
        This does not grant or modify any existing tool approval permissions.
        """
        from .tools import WindowsToolBoundary
        with self._lock:
            if self._tool_boundary is not None:
                raise ValueError("one companion owns one tool boundary")
            factory = boundary_factory if boundary_factory is not None else WindowsToolBoundary
            self._tool_boundary = factory(
                settings=settings, screen=screen, browser_factory=self.browser,
                browser_resource_guard=self.guard, browser_close=self.close_owned_browser,
            )
            return self._tool_boundary

    def sample_process_memory(self) -> Mapping[str, Any] | None:
        """Optional caller telemetry; only retained counters, never CDP/guard.

        A caller must persist these scoped lower-bound samples and drain the
        bounded checkpoint queue often enough for its intended workload. No
        polling thread or automatic admission allowance is installed here.
        """
        with self._lock:
            registry = self._registry
        return registry.sample() if registry is not None else None

    def close_owned_browser(self) -> None:
        """Boundary close callback; manager retains release authority."""
        with self._lock:
            if self._reservation is None and self.is_cold():
                return  # A controller that never admitted a route owns no tool work.
        if self.close() is not True:
            raise ResourceUnavailable("owned browser finite bound-set drain is unverified; keep lease quarantined")

    def close(self) -> bool:
        # The manager may hold its RLock here. Do not invoke guard(), directly
        # or from the browser worker, while waiting for that worker to join.
        with self._lock:
            if self._close_attempted:
                return self._close_verified
            token, generation = self._reservation, self._generation
            if token is None or generation is None:
                raise ResourceUnavailable("an unbound browser companion has no release identity")
            self._close_attempted = self._closing = True
            browser, registry = self._browser, self._registry
            factory_failed = self._factory_failed
        failure: BaseException | None = None
        browser_receipt: Mapping[str, Any] | None = None
        registry_receipt: Mapping[str, Any] | None = None
        try:
            if browser is not None:
                try:
                    browser.close()
                except BaseException as exc:
                    failure = exc
                finally:
                    try:
                        browser_receipt = getattr(browser, "process_memory_close_receipt", None)
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                        else:
                            failure.add_note(
                                "secondary browser close receipt failure: " + type(exc).__name__ + ": " + str(exc))
            if registry is not None:
                try:
                    registry_receipt = registry.close()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(
                                "secondary browser registry close failure: " + type(exc).__name__ + ": " + str(exc))
            if browser is None and registry is None and not factory_failed:
                # The exact token was bound, but neither constructor ran.
                verified = True
                scope = "never_started_browser_owner_no_processes_or_worker"
            else:
                bound = browser_receipt.get("bound_set") if isinstance(browser_receipt, Mapping) else None
                children = registry_receipt.get("children") if isinstance(registry_receipt, Mapping) else None
                verified = (failure is None and not factory_failed and isinstance(bound, Mapping)
                    and isinstance(children, Mapping)
                    and browser_receipt.get("worker_joined") is True
                    and browser_receipt.get("observer_callback_error_count") == 0
                    and bound.get("cohort_generation") == generation
                    and children.get("cohort_generation") == generation
                    and bound.get("bound_set_drain_verified") is True
                    and children.get("bound_set_drain_verified") is True
                    and children.get("required_owned_bindings_verified") is True
                    and children.get("known_bound_children_retired") is True
                    and _ownership_coverage_verified(bound)
                    and _ownership_coverage_verified(children)
                    and registry_receipt.get("observer_handles_closed") is True
                    and bound.get("all_descendants_retired") is False
                    and children.get("all_descendants_retired") is False)
                scope = "joined_browser_owner_worker_and_exact_retained_handle_bound_set_only"
            with self._lock:
                self._close_verified = verified is True
                self._quarantined = not self._close_verified
                self._release_evidence = {
                    "schema": "omni-companion-release-v1", "joint_generation": generation,
                    "resource_owner": token.owner, "owned_work_drained": self._close_verified,
                    "required_ownership_verified": self._close_verified, "scope": scope,
                    "all_descendants_retired": False, "hard_process_cap": False,
                    "quarantine_required": self._quarantined,
                    "browser_close_receipt": deepcopy(browser_receipt),
                    "registry_close_receipt": deepcopy(registry_receipt),
                    "factory_failed": factory_failed,
                    "failure": (type(failure).__name__ + ": " + str(failure) if failure else None),
                }
                self._last_close_evidence = deepcopy(self._release_evidence)
        finally:
            with self._lock:
                self._closing = False
        if failure is not None:
            raise failure
        return self._close_verified

    def reset_after_verified_drain(self) -> None:
        with self._lock:
            if self._reservation is None and self.is_cold():
                # Engine may reconcile an exact never-bound token after bind
                # failed before attachment. There is no owner to reset, and
                # this no-op cannot erase attached or partially-started work.
                return
            if (not self._close_verified or self._quarantined or self._closing
                    or self._reservation is None or self._ledger is None
                    or not self._ledger.was_released(self._reservation)):
                raise ResourceUnavailable("browser reset requires verified drain and exact engine token release")
            # No constructor, observer, OS counters or manager callback. A
            # factory boundary retrieves the new owner on its next operation;
            # it cannot retain the previous shut-down browser executor.
            if self._tool_boundary is not None:
                self._tool_boundary.reset_browser_owner_after_verified_drain()
            self._ledger = self._reservation = self._generation = self._guard = None
            self._browser = self._registry = None
            self._factory_attempted = self._factory_failed = False
            self._closing = self._close_attempted = self._close_verified = False
            self._quarantined = False
            self._release_evidence = None
