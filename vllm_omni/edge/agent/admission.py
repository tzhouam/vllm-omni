# SPDX-License-Identifier: Apache-2.0
"""One physical-host memory claim across resident Omni Agent routes.

Each StageRuntime retains its own stage ledger.  That ledger cannot see a
second, separately constructed runtime, so the Windows Agent also owns one
outer reservation.  Only one complete model route is resident at a time:
route changes happen at turn boundaries, after the old worker has drained.
Live free-memory readings are used for new loads and resident reuse. A resident
model is not charged twice, but reuse fails closed if available memory drops
below the post-load baseline (which may mean external use or lazy allocations).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, AsyncIterator

from vllm_omni.edge.agent.router import Admission, Route
from vllm_omni.engine.resource_ledger import Reservation, ResourceLedger, ResourceUnavailable


class HostMemoryCoordinator:
    """Serial, evidence-preserving admission for complete local model stages.

    ``free_bytes`` must return a fresh OS reading for each named physical pool.
    The initial capacities are fixed controller ceilings; they are never
    silently raised when another process releases memory.  A candidate route
    that needs the current resident to leave receives only a *provisional*
    router admission.  ``start`` closes the old worker, verifies its ledger
    drain, and repeats the live-memory check before reserving and loading.
    """

    def __init__(
        self, *, routes: Sequence[Route], backends: Mapping[str, Any],
        capacities: Mapping[str, int], free_bytes: Callable[[], Mapping[str, int | None]],
        blocked_reasons: Mapping[str, str] | None = None,
    ) -> None:
        self._routes = {route.route_id: route for route in routes}
        self._blocked_reasons = dict(blocked_reasons or {})
        if (
            len(self._routes) != len(routes)
            or set(backends) & set(self._blocked_reasons)
            or set(self._routes) != set(backends) | set(self._blocked_reasons)
        ):
            raise ValueError("each Agent route needs one backend or a capacity refusal")
        self._backends = dict(backends)
        self._ledger = ResourceLedger(capacities)
        for route in routes:
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

    def wrappers(self) -> dict[str, ManagedRouteBackend]:
        return {route_id: ManagedRouteBackend(self, route_id)
                for route_id in self._backends}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"resident_route": self._owner,
                    "resident_free_floor": dict(self._resident_free_floor or {}),
                    "ledger": self._ledger.snapshot()}

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
        # OmniLlamaBackend.close() returns True only after its stage ledger and
        # worker have both drained.  A False/unknown result remains quarantined
        # in the outer ledger; no other model may load over those bytes.
        if self._backends[owner].close() is not True:
            self._ledger.release(reservation, drained=False)
            raise ResourceUnavailable(
                f"{owner}: worker drain unverified; host reservation quarantined"
            )
        if not self._ledger.release(reservation, drained=True):
            raise ResourceUnavailable("host memory reservation release failed")
        self._owner = None
        self._reservation = None
        self._resident_free_floor = None

    def _check_resident_free(self, route: Route) -> None:
        floor = self._resident_free_floor
        if floor is None or set(floor) != set(route.memory_demands):
            raise ResourceUnavailable("resident route has no complete post-load memory baseline")
        live = self._live_free(route.memory_demands)
        for pool, minimum in floor.items():
            if live[pool] < minimum:
                raise ResourceUnavailable(
                    f"{pool}: resident route live free {live[pool]} bytes fell below "
                    f"post-load baseline {minimum} bytes; unload and re-admit explicitly"
                )

    def _quarantine_owner(self, route_id: str) -> None:
        with self._lock:
            if self._owner == route_id and self._reservation is not None:
                self._ledger.release(self._reservation, drained=False)

    def admit(self, route: Route) -> Admission:
        with self._lock:
            registered = self._routes.get(route.route_id)
            if registered != route:
                return Admission(False, "route identity or memory declaration changed")
            if route.route_id in self._blocked_reasons:
                return Admission(False, self._blocked_reasons[route.route_id])
            try:
                quarantined = set(self._ledger.snapshot()["quarantined"])
                if self._owner is not None and (
                    self._owner in quarantined or not self._backends[self._owner].resident
                ):
                    # A failed/cancelled backend may have been shut down by
                    # Omni.  Reconcile it only after close confirms drain.
                    self._release_owner()
                if self._owner == route.route_id:
                    self._check_resident_free(route)
                    plan = self._backends[route.route_id].execution_plan
                    return Admission(
                        True, "already resident under the host memory reservation",
                        plan.get("observed_model_placement") if isinstance(plan, Mapping)
                        else None,
                    )
                live = self._live_free(route.memory_demands)
                old_claim = (self._reservation.demands if self._reservation is not None else {})
                for pool, requested in route.memory_demands.items():
                    # Preview a turn-boundary switch without unloading during
                    # candidate ranking.  The old claim is only a possible
                    # reclaim; start() must remeasure after actual release.
                    possible = min(
                        self._ledger.capacities[pool],
                        live[pool] + old_claim.get(pool, 0),
                    )
                    if requested > possible:
                        return Admission(
                            False, f"{pool}: need {requested} bytes; at most {possible} "
                            "bytes after current route release",
                        )
                reason = (
                    "provisional switch; old worker must drain and live memory will be rechecked"
                    if self._owner is not None else
                    "live capacity gate passed; Omni stage load and placement still required"
                )
                return Admission(True, reason)
            except ResourceUnavailable as exc:
                return Admission(False, str(exc))

    def start(self, route_id: str) -> None:
        with self._lock:
            if route_id in self._blocked_reasons:
                raise ResourceUnavailable(self._blocked_reasons[route_id])
            route = self._routes[route_id]
            backend = self._backends[route_id]
            if self._owner == route_id and backend.resident:
                self._check_resident_free(route)
                return
            if self._owner is not None:
                self._release_owner()
            live = self._live_free(route.memory_demands)
            for pool, requested in route.memory_demands.items():
                if requested > live[pool]:
                    raise ResourceUnavailable(
                        f"{pool}: need {requested} bytes, live free {live[pool]} "
                        "bytes after route release"
                    )
            reservation = self._ledger.reserve(route_id, route.memory_demands)
            self._owner, self._reservation = route_id, reservation
            try:
                backend.start()
                plan = backend.execution_plan
                if not backend.resident or not isinstance(plan, Mapping):
                    raise RuntimeError("Omni route did not finish a healthy stage load")
                if plan.get("requested_device") != route.placement:
                    raise RuntimeError("Omni route loaded on a different device")
                observed = plan.get("observed_model_placement")
                if observed != route.placement and not (
                    route.placement.startswith("Vulkan_Host+") and
                    observed is None and
                    plan.get("placement_evidence_level") == "override_selection_only"
                ):
                    raise RuntimeError("Omni route has no verified model placement")
                if dict(plan.get("reserved_bytes", {})) != dict(route.memory_demands):
                    raise RuntimeError("Omni stage and host memory claims differ")
                self._resident_free_floor = self._live_free(route.memory_demands)
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


class ManagedRouteBackend:
    """Keep the Agent's existing ModelBackend contract while sharing admission."""

    def __init__(self, coordinator: HostMemoryCoordinator, route_id: str) -> None:
        self._coordinator = coordinator
        self._route_id = route_id
        self.release_evidence: Mapping[str, Any] | None = None

    @property
    def execution_plan(self) -> Mapping[str, Any] | None:
        return self._coordinator._backends[self._route_id].execution_plan

    def start(self) -> None:
        self._coordinator.start(self._route_id)

    async def generate(
        self, prompt: str, *, request_id: str, max_tokens: int,
        image_data_url: str | None = None,
    ) -> AsyncIterator[Any]:
        backend = self._coordinator._backends[self._route_id]
        async for chunk in backend.generate(
            prompt, request_id=request_id, max_tokens=max_tokens,
            image_data_url=image_data_url,
        ):
            yield chunk

    async def cancel(self, request_id: str) -> None:
        await self._coordinator._backends[self._route_id].cancel(request_id)

    def request_state_released(self, request_id: str) -> bool:
        backend = self._coordinator._backends[self._route_id]
        self.release_evidence = None
        reporter = getattr(backend,
                           "request_state_released", None)
        try:
            if callable(reporter) and reporter(request_id) is True:
                # The stage has drained; release its host-wide physical-pool
                # claim before the controller announces state_released.
                proof = getattr(backend, "release_evidence", None)
                if not (isinstance(proof, Mapping) and
                        proof.get("request_id") == request_id and
                        proof.get("worker_exit_confirmed") is True and
                        proof.get("stage_ledger_empty") is True):
                    self._coordinator._quarantine_owner(self._route_id)
                    return False
                if self._coordinator.close(self._route_id):
                    snapshot = self._coordinator.snapshot()
                    ledger = snapshot["ledger"]
                    if (snapshot["resident_route"] is None and
                        not ledger["owners"] and not ledger["quarantined"]):
                        self.release_evidence = {
                            **proof, "host_claim_released": True,
                            "host_ledger_empty": True,
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
