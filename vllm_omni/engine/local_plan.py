# SPDX-License-Identifier: Apache-2.0
"""Complete local model-route lifecycle above the existing Omni StageRuntime.

The engine owns one shared resource ledger. StageRuntime adopts the exact
reservation token before loading; the Agent only holds a route lease.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable


class RouteMemorySpec(Protocol):
    route_id: str
    placement: str
    memory_demands: Mapping[str, int]


@dataclass(frozen=True)
class PlanAdmission:
    admitted: bool
    reason: str
    actual_placement: str | None = None


@dataclass(frozen=True)
class CompanionResourceSpec:
    """Explicit incremental allowance; evidence resolution belongs to the app.

    This records a reviewed workload envelope, not an enforced process cap.
    The already-resident application is outside the incremental allowance.
    """

    purpose_id: str
    memory_demands: Mapping[str, int]
    envelope_sha256: str
    evidence_reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.purpose_id, str) or not self.purpose_id or len(self.purpose_id) > 128:
            raise ValueError("a bounded companion purpose ID is required")
        if (not isinstance(self.memory_demands, Mapping) or not self.memory_demands
                or any(not isinstance(pool, str) or not pool or type(amount) is not int or amount < 0
                       for pool, amount in self.memory_demands.items())
                or not any(self.memory_demands.values())):
            raise ValueError("explicit nonnegative companion demands with a positive allowance are required")
        if (not isinstance(self.envelope_sha256, str) or len(self.envelope_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.envelope_sha256)):
            raise ValueError("a companion envelope SHA256 is required")
        if (not isinstance(self.evidence_reference, str) or not self.evidence_reference
                or len(self.evidence_reference) > 4096):
            raise ValueError("a bounded companion envelope evidence reference is required")
        object.__setattr__(self, "memory_demands", MappingProxyType(dict(self.memory_demands)))


class LocalResourceCompanion(Protocol):
    """Application adapter; bind must not start work or call another thread.

    Only the exact companion token is lent to this owner. ``close`` and its
    scoped evidence describe the adapter's owned cleanup contract, without an
    implicit claim about all OS descendants. ``reset_after_verified_drain``
    must prepare a cold owner without launching it. Bind/close/reset callbacks
    must not ask another thread to call back into the manager's lifecycle lock;
    cleanup on an owner worker must not invoke its admission guard.
    """

    resource_spec: CompanionResourceSpec
    release_evidence: Mapping[str, Any] | None

    def is_cold(self) -> bool: ...

    def bind_resource_lease(
        self, ledger: ResourceLedger, reservation: Reservation,
        *, joint_generation: str, guard: Callable[[], None],
    ) -> None: ...

    def close(self) -> bool: ...

    def reset_after_verified_drain(self) -> None: ...


class LocalPlanManager:
    """Serial, evidence-preserving admission for complete local model stages.

    ``free_bytes`` must return a fresh OS reading for each named physical pool.
    The initial capacities are fixed controller ceilings; they are never
    silently raised when another process releases memory.  A candidate route
    that needs the current resident to leave receives only a *provisional*
    router admission.  ``start`` closes the old worker, verifies its ledger
    drain, and repeats the live-memory check before reserving and loading.
    """

    def __init__(
        self,
        *,
        routes: Sequence[RouteMemorySpec],
        backends: Mapping[str, Any],
        capacities: Mapping[str, int],
        free_bytes: Callable[[], Mapping[str, int | None]],
        blocked_reasons: Mapping[str, str] | None = None,
        resource_ledger: ResourceLedger | None = None,
        companion: LocalResourceCompanion | None = None,
    ) -> None:
        self._routes = {route.route_id: route for route in routes}
        self._blocked_reasons = dict(blocked_reasons or {})
        if (
            len(self._routes) != len(routes)
            or set(backends) & set(self._blocked_reasons)
            or set(self._routes) != set(backends) | set(self._blocked_reasons)
        ):
            raise ValueError("each local route needs one backend or a capacity refusal")
        self._backends = dict(backends)
        if resource_ledger is not None and dict(resource_ledger.capacities) != dict(capacities):
            raise ValueError("shared ledger and route manager must agree on physical ceilings")
        self._ledger = resource_ledger if resource_ledger is not None else ResourceLedger(capacities)
        self._route_claims = {route.route_id: dict(route.memory_demands) for route in routes}
        self._route_placements = {route.route_id: route.placement for route in routes}
        for route in routes:
            if not route.memory_demands or any(
                type(amount) is not int or amount < 0 for amount in route.memory_demands.values()
            ):
                raise ValueError(f"{route.route_id}: nonnegative physical-pool demands are required")
            if route.route_id in self._backends and any(
                pool not in self._ledger.capacities or amount > self._ledger.capacities[pool]
                for pool, amount in route.memory_demands.items()
            ):
                raise ValueError(f"{route.route_id}: demand exceeds the fixed host ceiling")
        self._free_bytes = free_bytes
        self._lock = threading.RLock()
        self._owner: str | None = None
        self._reservation: Reservation | None = None
        self._resident_free_floor: dict[str, int] | None = None
        self._companion = companion
        self._companion_spec = companion.resource_spec if companion is not None else None
        if self._companion_spec is not None:
            if not isinstance(self._companion_spec, CompanionResourceSpec):
                raise ValueError("companion requires a frozen evidence-bound resource spec")
            if any(pool not in self._ledger.capacities
                   for pool in self._companion_spec.memory_demands):
                raise ValueError("companion uses an unknown physical pool")
        self._companion_reservation: Reservation | None = None
        self._joint_generation: str | None = None
        self._joint_route_id: str | None = None
        self._joint_startup_free: dict[str, int] | None = None
        self._joint_snapshot_window: tuple[int, int] | None = None
        self._joint_demands: dict[str, int] | None = None
        self._joint_needs_drain = False
        self._joint_loading = False
        self._companion_bound = False
        self._finalized = False

    def wrappers(self) -> dict[str, ManagedLocalBackend]:
        return {route_id: ManagedLocalBackend(self, route_id) for route_id in self._backends}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                "resident_route": self._owner,
                "resident_free_floor": dict(self._resident_free_floor or {}),
                "ledger": self._ledger.snapshot(),
            }
            if self._companion is not None:
                snapshot["joint_admission"] = {
                    "generation": self._joint_generation,
                    "route_id": self._joint_route_id,
                    "startup_free": dict(self._joint_startup_free or {}),
                    "snapshot_monotonic_ns": list(self._joint_snapshot_window or ()),
                    "combined_demands": dict(self._joint_demands or {}),
                    "protected_free_floor": dict(self._resident_free_floor or {}),
                    "needs_drain": self._joint_needs_drain,
                    "loading": self._joint_loading,
                    "model_owner": self._reservation.owner if self._reservation is not None else None,
                    "companion_owner": (self._companion_reservation.owner
                                        if self._companion_reservation is not None else None),
                    "envelope_sha256": self._companion_spec.envelope_sha256,
                    "evidence_reference": self._companion_spec.evidence_reference,
                    "scope": "declared_incremental_joint_allowance_not_process_hard_cap",
                }
            return snapshot

    def _live_free(self, demands: Mapping[str, int]) -> dict[str, int]:
        readings = self._free_bytes()
        available: dict[str, int] = {}
        for pool in demands:
            value = readings.get(pool)
            if type(value) is not int or value < 0:
                raise ResourceUnavailable(f"{pool} live free-memory measurement unavailable")
            available[pool] = value
        return available

    def _release_owner(self) -> None:
        owner, reservation = self._owner, self._reservation
        if owner is None:
            return
        if reservation is None:
            raise ResourceUnavailable("host memory reservation identity is missing")
        # A backend confirms worker/IO drain. Its StageRuntime may already
        # have released this exact shared token during the same shutdown.
        try:
            drained = self._backends[owner].close() is True
        except BaseException:
            self._ledger.release(reservation, drained=False)
            raise
        if not drained:
            self._ledger.release(reservation, drained=False)
            raise ResourceUnavailable(f"{owner}: worker drain unverified; host reservation quarantined")
        if not self._ledger.release(reservation, drained=True) and not self._ledger.was_released(reservation):
            raise ResourceUnavailable("host memory reservation release failed or was replaced")
        self._owner = None
        self._reservation = None
        if self._companion is None:
            self._resident_free_floor = None
        elif self._joint_generation is not None:
            self._joint_needs_drain = True

    def _companion_demands(self, route: RouteMemorySpec) -> dict[str, int]:
        if (self._companion is None or self._companion_spec is None
                or self._companion.resource_spec is not self._companion_spec):
            raise ResourceUnavailable("companion identity or memory declaration changed")
        if (dict(route.memory_demands) != self._route_claims[route.route_id]
                or route.placement != self._route_placements[route.route_id]):
            raise ResourceUnavailable("route identity or memory declaration changed")
        combined = dict(route.memory_demands)
        for pool, amount in self._companion_spec.memory_demands.items():
            combined[pool] = combined.get(pool, 0) + amount
        return combined

    def _check_joint_floor(self) -> None:
        if self._joint_demands is None or self._resident_free_floor is None:
            raise ResourceUnavailable("joint lifetime has no complete protected memory floor")
        if set(self._joint_demands) != set(self._resident_free_floor):
            raise ResourceUnavailable("joint lifetime memory floor is incomplete")
        live = self._live_free(self._joint_demands)
        for pool, minimum in self._resident_free_floor.items():
            if live[pool] < minimum:
                self._joint_needs_drain = True
                raise ResourceUnavailable(
                    f"{pool}: joint lifetime live free {live[pool]} bytes fell below "
                    f"declared reservation floor {minimum} bytes; drain and re-admit explicitly"
                )

    def _guard_companion(self, generation: str, token: Reservation) -> None:
        # Called by the application before submitting work to its owner worker;
        # do not call this across threads from bind while start holds this lock.
        with self._lock:
            if (generation != self._joint_generation or token is not self._companion_reservation
                    or not self._ledger.owns(token) or self._joint_needs_drain
                    or self._joint_loading
                    or token.owner in self._ledger.snapshot()["quarantined"]
                    or self._reservation is None or not self._ledger.owns(self._reservation)
                    or self._reservation.owner in self._ledger.snapshot()["quarantined"]
                    or not self._backends[self._joint_route_id].resident):
                raise ResourceUnavailable("companion joint lease is stale, quarantined, or needs full drain")
            self._companion_demands(self._routes[self._joint_route_id])
            self._check_joint_floor()

    def _release_companion(self) -> None:
        token = self._companion_reservation
        if token is None:
            return
        try:
            if not self._companion_bound and self._companion.is_cold() is True:
                # bind is a synchronous, non-launching operation. A failed
                # bind may release this never-started token only when the
                # adapter independently still confirms its cold state.
                if not self._ledger.release(token, drained=True) and not self._ledger.was_released(token):
                    raise ResourceUnavailable("unstarted companion exact resource token was replaced")
                self._companion_reservation = None
                return
            if self._companion.close() is not True:
                raise ResourceUnavailable("companion did not verify its owned cleanup contract")
            proof = self._companion.release_evidence
            if not (isinstance(proof, Mapping)
                    and proof.get("schema") == "omni-companion-release-v1"
                    and proof.get("joint_generation") == self._joint_generation
                    and proof.get("resource_owner") == token.owner
                    and proof.get("owned_work_drained") is True
                    and proof.get("required_ownership_verified") is True
                    and isinstance(proof.get("scope"), str) and proof["scope"]):
                raise ResourceUnavailable("companion scoped release evidence is missing or mismatched")
            if not self._ledger.release(token, drained=True) and not self._ledger.was_released(token):
                raise ResourceUnavailable("companion exact resource token was replaced")
            self._companion_reservation = None
        except BaseException:
            self._ledger.release(token, drained=False)
            self._joint_needs_drain = True
            raise

    def close_joint(self) -> None:
        """Drain both owned scopes; failed scopes stay charged/quarantined."""
        with self._lock:
            if self._companion is None:
                self._release_owner()
                return
            if self._joint_generation is None:
                if self._companion.is_cold() is not True:
                    raise ResourceUnavailable("unleased companion is not cold; explicit owned cleanup required")
                return
            self._joint_needs_drain = True
            failure: BaseException | None = None
            for closer in (self._release_owner, self._release_companion):
                try:
                    closer()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note("secondary joint drain failure: " + type(exc).__name__ + ": " + str(exc))
            if failure is not None:
                raise failure
            self._companion.reset_after_verified_drain()
            if self._companion.is_cold() is not True:
                raise ResourceUnavailable("companion reset did not produce a cold owner")
            self._joint_generation = self._joint_route_id = None
            self._joint_startup_free = self._joint_demands = None
            self._joint_snapshot_window = None
            self._resident_free_floor = None
            self._joint_needs_drain = False
            self._joint_loading = False
            self._companion_bound = False

    def finalize_close(self) -> dict[str, Any]:
        """Final controller check, after application/tool writers are drained."""
        with self._lock:
            self.close_joint()
            snapshot = self.snapshot()
            ledger = snapshot["ledger"]
            if ledger["owners"] or ledger["quarantined"] or any(ledger["reserved"].values()):
                raise ResourceUnavailable("final controller resource ledger is not empty")
            if any(backend.resident for backend in self._backends.values()):
                raise ResourceUnavailable("final controller still has a resident backend")
            self._finalized = True
            return snapshot

    def _check_resident_free(self, route: RouteMemorySpec) -> None:
        if self._reservation is None or not self._ledger.owns(self._reservation):
            raise ResourceUnavailable("resident route no longer owns its exact resource lease")
        if self._companion is not None:
            if (self._joint_route_id != route.route_id or self._joint_needs_drain
                    or self._reservation.owner in self._ledger.snapshot()["quarantined"]
                    or self._companion_reservation is None
                    or not self._ledger.owns(self._companion_reservation)
                    or self._companion_reservation.owner in self._ledger.snapshot()["quarantined"]
                    or self._companion_demands(route) != self._joint_demands):
                raise ResourceUnavailable("resident joint lifetime needs full drain and cold re-admission")
            self._check_joint_floor()
            return
        floor = self._resident_free_floor
        if floor is None or set(floor) != set(route.memory_demands):
            raise ResourceUnavailable("resident route has no complete memory floor")
        live = self._live_free(route.memory_demands)
        for pool, minimum in floor.items():
            if live[pool] < minimum:
                raise ResourceUnavailable(
                    f"{pool}: resident route live free {live[pool]} bytes fell below "
                    f"declared reservation floor {minimum} bytes; unload and re-admit explicitly"
                )

    def _quarantine_owner(self, route_id: str) -> None:
        with self._lock:
            if self._owner == route_id and self._reservation is not None:
                self._ledger.release(self._reservation, drained=False)
                if self._companion is not None:
                    self._joint_needs_drain = True

    def _admit_joint(self, route: RouteMemorySpec) -> PlanAdmission:
        combined = self._companion_demands(route)
        if self._owner == route.route_id and self._backends[route.route_id].resident:
            self._check_resident_free(route)
            plan = self._backends[route.route_id].execution_plan
            return PlanAdmission(
                True, "already resident under the immutable joint memory reservation",
                plan.get("observed_model_placement") if isinstance(plan, Mapping) else None,
            )
        ledger = self._ledger.snapshot()
        owned = [token for token in (self._reservation, self._companion_reservation)
                 if token is not None and self._ledger.owns(token)]
        if set(ledger["owners"]) - {token.owner for token in owned}:
            return PlanAdmission(False, "joint cold admission requires a reconciled controller ledger")
        if self._joint_generation is None and (
            self._companion.is_cold() is not True
            or any(backend.resident for backend in self._backends.values())
        ):
            return PlanAdmission(False, "model and companion must both drain before joint cold admission")
        live = self._live_free(combined)
        for pool, requested in combined.items():
            reclaim = sum(token.demands.get(pool, 0) for token in owned)
            possible = min(self._ledger.capacities[pool] - ledger["reserved"][pool] + reclaim,
                           live[pool] + reclaim)
            if requested > possible:
                return PlanAdmission(False,
                                     f"{pool}: joint need {requested} bytes; at most {possible} bytes after full drain")
        return PlanAdmission(
            True,
            "provisional joint transition; both owners must drain and one fresh snapshot will be checked"
            if self._joint_generation is not None
            else "joint live capacity preview passed; cold atomic reservation and load still required",
        )

    def admit(self, route: RouteMemorySpec) -> PlanAdmission:
        with self._lock:
            if self._finalized:
                return PlanAdmission(False, "local controller has been finalized")
            registered = self._routes.get(route.route_id)
            if (
                registered != route
                or dict(route.memory_demands) != self._route_claims[route.route_id]
                or route.placement != self._route_placements[route.route_id]
            ):
                return PlanAdmission(False, "route identity or memory declaration changed")
            if route.route_id in self._blocked_reasons:
                return PlanAdmission(False, self._blocked_reasons[route.route_id])
            try:
                if self._companion is not None:
                    return self._admit_joint(route)
                quarantined = set(self._ledger.snapshot()["quarantined"])
                if self._owner is not None and (self._owner in quarantined or not self._backends[self._owner].resident):
                    # A failed/cancelled backend may have been shut down by
                    # Omni.  Reconcile it only after close confirms drain.
                    self._release_owner()
                if self._owner == route.route_id:
                    self._check_resident_free(route)
                    plan = self._backends[route.route_id].execution_plan
                    return PlanAdmission(
                        True,
                        "already resident under the host memory reservation",
                        plan.get("observed_model_placement") if isinstance(plan, Mapping) else None,
                    )
                live = self._live_free(route.memory_demands)
                old_claim = self._reservation.demands if self._reservation is not None else {}
                ledger_reserved = self._ledger.snapshot()["reserved"]
                for pool, requested in route.memory_demands.items():
                    # Preview a turn-boundary switch without unloading during
                    # candidate ranking.  The old claim is only a possible
                    # reclaim; start() must remeasure after actual release.
                    possible = min(
                        self._ledger.capacities[pool] - ledger_reserved[pool] + old_claim.get(pool, 0),
                        live[pool] + old_claim.get(pool, 0),
                    )
                    if requested > possible:
                        return PlanAdmission(
                            False,
                            f"{pool}: need {requested} bytes; at most {possible} bytes after current route release",
                        )
                reason = (
                    "provisional switch; old worker must drain and live memory will be rechecked"
                    if self._owner is not None
                    else "live capacity gate passed; Omni stage load and placement still required"
                )
                return PlanAdmission(True, reason)
            except ResourceUnavailable as exc:
                return PlanAdmission(False, str(exc))

    def _start_joint(self, route: RouteMemorySpec, backend: Any) -> None:
        combined = self._companion_demands(route)
        if self._owner == route.route_id and backend.resident:
            self._check_resident_free(route)
            return
        if self._joint_generation is not None:
            self.close_joint()
            combined = self._companion_demands(route)
        if (self._owner is not None or self._reservation is not None
                or self._companion_reservation is not None
                or self._companion.is_cold() is not True
                or any(candidate.resident for candidate in self._backends.values())):
            raise ResourceUnavailable("model and companion must both be cold before joint reservation")
        snapshot = self._ledger.snapshot()
        if snapshot["owners"] or snapshot["quarantined"]:
            raise ResourceUnavailable("joint cold admission requires a reconciled controller ledger")
        # Exactly one fresh provider call supplies every prelaunch pool. Later
        # post-load/tool checks are separate observations of this same floor.
        sampling_started = time.monotonic_ns()
        live = self._live_free(combined)
        sampling_finished = time.monotonic_ns()
        for pool, requested in combined.items():
            if requested > live[pool]:
                raise ResourceUnavailable(f"{pool}: joint need {requested} bytes, live free {live[pool]} bytes")
        generation = uuid.uuid4().hex
        companion_owner = f"companion:{generation}:{self._companion_spec.purpose_id}"
        if companion_owner in self._routes:
            raise ResourceUnavailable("companion owner collides with a model route")
        tokens = self._ledger.reserve_many({route.route_id: route.memory_demands,
                                           companion_owner: self._companion_spec.memory_demands})
        self._owner, self._reservation = route.route_id, tokens[route.route_id]
        self._companion_reservation = tokens[companion_owner]
        self._joint_generation, self._joint_route_id = generation, route.route_id
        self._joint_startup_free, self._joint_demands = live, combined
        self._joint_snapshot_window = (sampling_started, sampling_finished)
        self._resident_free_floor = {pool: live[pool] - amount for pool, amount in combined.items()}
        self._joint_needs_drain = False
        self._joint_loading = True
        token = self._companion_reservation
        try:
            self._companion.bind_resource_lease(
                self._ledger, token, joint_generation=generation,
                guard=lambda: self._guard_companion(generation, token),
            )
            self._companion_bound = True
            if self._companion_demands(route) != combined:
                raise ResourceUnavailable("joint memory declarations changed during binding")
            binder = getattr(backend, "bind_resource_lease", None)
            if not callable(binder):
                raise ResourceUnavailable("joint model backend cannot adopt its exact model-only token")
            binder(self._ledger, self._reservation)
            backend.start()
            if self._companion_demands(route) != combined:
                raise ResourceUnavailable("joint memory declarations changed during loading")
            if not self._ledger.owns(self._reservation) or not self._ledger.owns(token):
                raise ResourceUnavailable("loaded joint lifetime lost an exact resource token")
            self._validate_loaded_route(route, backend)
            self._check_joint_floor()
            self._joint_loading = False
        except BaseException as primary:
            try:
                self.close_joint()
            except BaseException as cleanup:
                primary.add_note("joint cleanup failed: " + type(cleanup).__name__ + ": " + str(cleanup))
            raise

    @staticmethod
    def _validate_loaded_route(route: RouteMemorySpec, backend: Any) -> None:
        plan = backend.execution_plan
        if not backend.resident or not isinstance(plan, Mapping):
            raise RuntimeError("Omni route did not finish a healthy stage load")
        if plan.get("requested_device") != route.placement:
            raise RuntimeError("Omni route loaded on a different device")
        observed = plan.get("observed_model_placement")
        if getattr(route, "backend", None) in {
            "external.strata.text.v1", "external.strata.multimodal.v1"
        }:
            if route.backend == "external.strata.text.v1":
                from vllm_omni.engine.backends.strata import validate_strata_load_plan

                validate_strata_load_plan(dict(plan), route.placement)
            else:
                from vllm_omni.engine.backends.strata_multimodal import validate_strata_multimodal_load_plan

                validate_strata_multimodal_load_plan(dict(plan), route.placement)
        elif observed != route.placement and not (
            route.placement.startswith("Vulkan_Host+") and observed is None
            and plan.get("placement_evidence_level") == "override_selection_only"
        ):
            raise RuntimeError("Omni route has no verified model placement")
        if dict(plan.get("reserved_bytes", {})) != dict(route.memory_demands):
            raise RuntimeError("Omni stage and host memory claims differ")

    def start(self, route_id: str) -> None:
        with self._lock:
            if self._finalized:
                raise ResourceUnavailable("local controller has been finalized")
            if route_id in self._blocked_reasons:
                raise ResourceUnavailable(self._blocked_reasons[route_id])
            route = self._routes[route_id]
            backend = self._backends[route_id]
            if (
                dict(route.memory_demands) != self._route_claims[route_id]
                or route.placement != self._route_placements[route_id]
            ):
                raise ResourceUnavailable("route identity or memory declaration changed")
            if self._companion is not None:
                self._start_joint(route, backend)
                return
            if self._owner == route_id and backend.resident:
                self._check_resident_free(route)
                return
            if self._owner is not None:
                self._release_owner()
            live = self._live_free(route.memory_demands)
            for pool, requested in route.memory_demands.items():
                if requested > live[pool]:
                    raise ResourceUnavailable(
                        f"{pool}: need {requested} bytes, live free {live[pool]} bytes after route release"
                    )
            reservation = self._ledger.reserve(route_id, route.memory_demands)
            self._owner, self._reservation = route_id, reservation
            try:
                binder = getattr(backend, "bind_resource_lease", None)
                if callable(binder):
                    binder(self._ledger, reservation)
                backend.start()
                if not self._ledger.owns(reservation):
                    raise ResourceUnavailable("loaded route lost its exact resource lease")
                plan = backend.execution_plan
                if not backend.resident or not isinstance(plan, Mapping):
                    raise RuntimeError("Omni route did not finish a healthy stage load")
                if plan.get("requested_device") != route.placement:
                    raise RuntimeError("Omni route loaded on a different device")
                observed = plan.get("observed_model_placement")
                if getattr(route, "backend", None) in {
                    "external.strata.text.v1", "external.strata.multimodal.v1"
                }:
                    # Explicit experimental Strata execution verifies the
                    # loaded GPU/CPU-pool configuration. It does not convert
                    # that capability into observed per-request computation
                    # or whole-Agent route qualification.
                    if route.backend == "external.strata.text.v1":
                        from vllm_omni.engine.backends.strata import validate_strata_load_plan

                        validate_strata_load_plan(dict(plan), route.placement)
                    else:
                        from vllm_omni.engine.backends.strata_multimodal import (
                            validate_strata_multimodal_load_plan,
                        )

                        validate_strata_multimodal_load_plan(dict(plan), route.placement)
                elif observed != route.placement and not (
                    route.placement.startswith("Vulkan_Host+")
                    and observed is None
                    and plan.get("placement_evidence_level") == "override_selection_only"
                ):
                    raise RuntimeError("Omni route has no verified model placement")
                if dict(plan.get("reserved_bytes", {})) != dict(route.memory_demands):
                    raise RuntimeError("Omni stage and host memory claims differ")
                # Available memory can shrink after load when the worker lazily
                # allocates KV/workspace. The full declared claim covers that
                # growth. Keep the pre-load free amount outside the claim as a
                # floor; external use cannot silently consume *that* margin.
                floor = {pool: max(0, live[pool] - amount) for pool, amount in route.memory_demands.items()}
                post_load = self._live_free(route.memory_demands)
                if any(post_load[pool] < minimum for pool, minimum in floor.items()):
                    raise ResourceUnavailable("loaded route exceeded its declared physical-pool claim")
                self._resident_free_floor = floor
            except BaseException:
                # A failed load may have spawned a worker.  Never release the
                # host claim merely because the Python start call raised.
                self._release_owner()
                raise

    def close(self, route_id: str) -> bool:
        with self._lock:
            if route_id != self._owner:
                return True
            self._release_owner()
            return True


class ManagedLocalBackend:
    """Keep the complete-model client contract while sharing engine admission."""

    def __init__(self, coordinator: LocalPlanManager, route_id: str) -> None:
        self._coordinator = coordinator
        self._route_id = route_id
        self.release_evidence: Mapping[str, Any] | None = None

    @property
    def execution_plan(self) -> Mapping[str, Any] | None:
        return self._coordinator._backends[self._route_id].execution_plan

    @property
    def model_output_contract_identity(self) -> Mapping[str, Any] | None:
        return getattr(self._coordinator._backends[self._route_id], "model_output_contract_identity", None)

    def last_model_output(self) -> Mapping[str, Any] | None:
        getter = getattr(self._coordinator._backends[self._route_id], "last_model_output", None)
        return getter() if callable(getter) else None

    def start(self) -> None:
        self._coordinator.start(self._route_id)

    async def generate(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        image_data_url: str | None = None,
    ) -> AsyncIterator[Any]:
        backend = self._coordinator._backends[self._route_id]
        if self._coordinator._companion is not None:
            with self._coordinator._lock:
                if self._coordinator._owner != self._route_id or not backend.resident:
                    raise ResourceUnavailable("model request has no healthy admitted joint lifetime")
                self._coordinator._check_resident_free(self._coordinator._routes[self._route_id])
        chunks = backend.generate(prompt, request_id=request_id, max_tokens=max_tokens,
                                  image_data_url=image_data_url)
        try:
            async for chunk in chunks:
                yield chunk
        finally:
            closer = getattr(chunks, "aclose", None)
            if callable(closer):
                await closer()

    async def cancel(self, request_id: str) -> None:
        await self._coordinator._backends[self._route_id].cancel(request_id)

    def request_state_released(self, request_id: str) -> bool:
        backend = self._coordinator._backends[self._route_id]
        self.release_evidence = None
        reporter = getattr(backend, "request_state_released", None)
        try:
            if callable(reporter) and reporter(request_id) is True:
                # The stage has drained; reconcile its exact physical-pool
                # lease before announcing release. Other shared owners remain.
                proof = getattr(backend, "release_evidence", None)
                if not (
                    isinstance(proof, Mapping)
                    and proof.get("request_id") == request_id
                    and proof.get("worker_exit_confirmed") is True
                    and proof.get("stage_ledger_empty") is True
                ):
                    self._coordinator._quarantine_owner(self._route_id)
                    return False
                token = self._coordinator._reservation
                if proof.get("schema") == "omni-resource-release-v2" and (
                    token is None
                    or proof.get("resource_owner") != token.owner
                    or proof.get("resource_claim_released") is not True
                ):
                    self._coordinator._quarantine_owner(self._route_id)
                    return False
                if self._coordinator.close(self._route_id):
                    snapshot = self._coordinator.snapshot()
                    ledger = snapshot["ledger"]
                    owner = proof.get("resource_owner", self._route_id)
                    if (
                        snapshot["resident_route"] is None
                        and owner not in ledger["owners"]
                        and owner not in ledger["quarantined"]
                        and token is not None
                        and self._coordinator._ledger.was_released(token)
                    ):
                        self.release_evidence = {
                            **proof,
                            "schema": "omni-resource-release-v2",
                            "resource_claim_released": True,
                            "resource_owner": token.owner,
                            "exact_resource_token_released": True,
                            "release_scope": "exact_model_lease",
                            "joint_generation": self._coordinator._joint_generation,
                            "tool_lease_retained": (
                                self._coordinator._companion_reservation is not None
                                and self._coordinator._ledger.owns(self._coordinator._companion_reservation)
                            ),
                            "host_claim_released": True,
                            "host_ledger_empty": not ledger["owners"] and not ledger["quarantined"],
                        }
                        return True
                self._coordinator._quarantine_owner(self._route_id)
                return False
        except Exception:
            self._coordinator._quarantine_owner(self._route_id)
            raise
        self._coordinator._quarantine_owner(self._route_id)
        return False

    def close(self) -> bool:
        return self._coordinator.close(self._route_id)
