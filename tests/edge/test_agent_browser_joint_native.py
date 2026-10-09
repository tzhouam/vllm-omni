# SPDX-License-Identifier: Apache-2.0
"""Explicitly opted-in real owned-browser companion/manager contract tests.

The model stage and available-memory provider are synthetic accounting only.
The browser, Windows retained-handle registry, HTTP fixture and cleanup are
real. Fixture claim amounts do NOT represent a measured browser RAM envelope.
There is no inference, memory-cap, all-descendant or NN qualification here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.edge_agent.paired_suite import FixtureSite, ReadOnlyFixtureTools
from vllm_omni.edge import windows_process_memory as wpm
from vllm_omni.edge.agent import browser_resources as br
from vllm_omni.edge.agent import tools
from vllm_omni.engine import local_plan as lp
from vllm_omni.engine.resource_ledger import ResourceLedger, ResourceUnavailable

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("OMNI_RUN_NATIVE_BROWSER_JOINT_TESTS") != "1",
    reason="requires native Windows Edge and explicit OMNI_RUN_NATIVE_BROWSER_JOINT_TESTS=1",
)


class _ModelFreeStage:
    """No StageClient, model, worker or allocation; tests only manager lifetime."""

    def __init__(self, route: Any) -> None:
        self.route = route
        self.resident = False
        self.execution_plan = None
        self.loads = self.closes = 0
        self.ledger = self.token = None

    def bind_resource_lease(self, ledger: ResourceLedger, token: Any) -> None:
        assert ledger.owns(token) and dict(token.demands) == dict(self.route.memory_demands)
        self.ledger, self.token = ledger, token

    def start(self) -> None:
        assert self.ledger.owns(self.token) and len(self.ledger.snapshot()["owners"]) == 2
        self.loads += 1
        self.resident = True
        self.execution_plan = {"requested_device": "cpu", "observed_model_placement": "cpu",
            "reserved_bytes": dict(self.route.memory_demands),
            "test_scope": "synthetic_model_free_lifetime_no_neural_execution"}

    def close(self) -> bool:
        self.closes += 1
        self.resident = False
        self.execution_plan = None
        # The real manager releases the exact model token. This stub does not
        # release a tool token or fabricate worker/process retirement evidence.
        return True


class _ContractRun:
    def __init__(self, tmp_path: Path, fixture: FixtureSite, *, registry_limit: int = 128) -> None:
        self.path = tmp_path
        self.fixture = fixture
        self.path.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.guards: list[dict[str, Any]] = []
        self.http_requests: list[str] = []
        self.browser_owner_threads: set[int] = set()
        self.browsers: list[tools.ManagedEdgeBrowser] = []
        self.registries: list[wpm.WindowsProcessMemoryRegistry] = []
        self.close_proofs: list[Any] = []
        self.samples: list[Any] = []
        self.snapshots: list[Any] = []
        self.factory_count = 0
        self.free = {"host_ram": 100, "windows_commit": 100}
        self.route = SimpleNamespace(route_id="native-contract-model-free", placement="cpu",
            memory_demands={"host_ram": 50, "windows_commit": 40})
        self.stage = _ModelFreeStage(self.route)
        self.ledger = ResourceLedger(self.free)
        declaration = br.ResolvedBrowserEnvelope(
            lp.CompanionResourceSpec("native-browser-contract-fixture", {"host_ram": 20, "windows_commit": 30},
                "a" * 64, "unit-fixture-not-a-reviewed-measured-RAM-envelope"),
            (("synthetic-accounting-fixture-not-measurement", "b" * 64),),
        )

        def registry_factory(generation: str) -> wpm.WindowsProcessMemoryRegistry:
            self.journal("construct_real_registry", generation=generation, max_bindings=registry_limit)
            value = wpm.WindowsProcessMemoryRegistry(generation, max_bindings=registry_limit)
            self.registries.append(value)
            return value

        def browser_factory(**kwargs: Any) -> tools.ManagedEdgeBrowser:
            self.factory_count += 1
            assert self.guards and self.stage.resident
            assert len(self.ledger.snapshot()["owners"]) == 2
            assert self.guards[-1]["generation"] == self.companion._generation
            assert self.guards[-1]["owner"] == self.companion._reservation.owner
            assert self.guards[-1]["thread_id"] == threading.get_ident()
            observer = kwargs["process_observer"]

            def observed(action: str, payload: Any) -> Any:
                # This wrapper logs only metadata and forwards the exact real
                # callback result. It never supplies a fabricated receipt.
                if action != "children_closed" and threading.get_ident() == browser._owner_thread:
                    self.browser_owner_threads.add(threading.get_ident())
                self.journal("real_observer_callback", action=action)
                return observer(action, payload)

            kwargs["process_observer"] = observed
            browser = tools.ManagedEdgeBrowser(
                profile_dir=tmp_path / f"owned-context-{self.factory_count}", headless=True, **kwargs)
            self.browsers.append(browser)
            self.journal("construct_real_browser_owner", count=self.factory_count)
            return browser

        self.companion = br.WindowsBrowserResourceCompanion(
            declaration, browser_factory=browser_factory, registry_factory=registry_factory)
        self.manager = lp.LocalPlanManager(routes=[self.route], backends={self.route.route_id: self.stage},
            capacities=self.ledger.capacities, resource_ledger=self.ledger,
            free_bytes=lambda: dict(self.free), companion=self.companion)
        original_guard = self.manager._guard_companion

        def guarded(generation: str, token: Any) -> None:
            caller = threading.get_ident()
            # Fail before acquiring the manager lock if a regression invokes
            # this callback on the browser owner, avoiding that lock cycle.
            if caller in self.browser_owner_threads or any(
                caller == thread.ident for browser in self.browsers for thread in browser._worker._threads
            ):
                raise AssertionError("browser owner worker called manager admission guard")
            row = {"thread_id": caller, "generation": generation, "owner": token.owner,
                   "factory_count": self.factory_count, "http_request_count": len(self.http_requests)}
            self.guards.append(row)
            original_guard(generation, token)

        self.manager._guard_companion = guarded
        original_close = self.companion.close

        def recorded_close() -> bool:
            try:
                return original_close()
            finally:
                # Record the real current proof before manager reset clears it.
                self.close_proofs.append(self.companion.release_evidence)

        self.companion.close = recorded_close
        self.boundary = self.companion.make_tool_boundary(
            boundary_factory=lambda **kwargs: ReadOnlyFixtureTools(fixture.origin, **kwargs))
        handler = fixture._server.RequestHandlerClass
        original = handler.do_GET

        def get(request: Any) -> None:
            self.http_requests.append(request.path)
            return original(request)

        handler.do_GET = get

    def journal(self, phase: str, **details: Any) -> None:
        row = {"phase": phase, "sequence": len(self.events) + 1,
               "monotonic_ns": time.monotonic_ns(), "thread_id": threading.get_ident(), **details}
        self.events.append(row)
        # Close each writer immediately so the parent can inspect the last
        # completed checkpoint if its controlled process deadline expires.
        with (self.path / "native-joint-contract-journal.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sample(self) -> Any:
        value = self.companion.sample_process_memory()
        self.samples.append(value)
        return value

    def snapshot(self) -> Any:
        value = self.manager.snapshot()
        self.snapshots.append(value)
        return value

    def write_receipt(self, *, failure: BaseException | None, injected_fault: str | None = None) -> None:
        modules = {"tools": tools, "registry": wpm, "local_plan": lp, "companion": br}
        source = {name: {"path": str(Path(module.__file__).resolve()),
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name, module in modules.items()}
        receipt = {"schema": "omni-native-joint-browser-contract-v1",
            "scope": "real_owned_browser_registry_contract_with_synthetic_model_and_ledger_units",
            "measured_browser_envelope": False, "model_execution": False,
            "ledger_unit_scope": "synthetic_units_not_a_physical_RAM_or_commit_allowance",
            "browser_gpu_memory_measured": False,
            "memory_hard_cap": False, "all_descendants_covered": False,
            "inference_or_performance_qualification": False,
            "injected_fault": injected_fault, "hardware_os": platform.platform(),
            "python": sys.version, "playwright": importlib.metadata.version("playwright"),
            "loaded_source": source, "model_load_count_synthetic": self.stage.loads,
            "events": self.events, "guards": self.guards, "http_requests": self.http_requests,
            "samples": self.samples, "snapshots": self.snapshots,
            "real_companion_close_proofs": self.close_proofs,
            "final_ledger": self.ledger.snapshot(),
            "browser_workers_joined": [all(not thread.is_alive() for thread in browser._worker._threads)
                                        for browser in self.browsers],
            "failure": ({"type": type(failure).__name__, "message": str(failure),
                         "notes": list(getattr(failure, "__notes__", []))} if failure else None)}
        (self.path / "native-joint-contract-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _successful_real_proof(proof: Any) -> None:
    assert proof["owned_work_drained"] is True and proof["required_ownership_verified"] is True
    assert proof["all_descendants_retired"] is False and proof["hard_process_cap"] is False
    assert proof["quarantine_required"] is False
    browser = proof["browser_close_receipt"]
    assert browser["worker_joined"] is True and browser["observer_callback_error_count"] == 0
    children = proof["registry_close_receipt"]["children"]
    assert children["cohort_generation"] == proof["joint_generation"]
    assert children["bound_set_drain_verified"] is True
    assert children["required_owned_bindings_verified"] is True
    assert children["known_bound_children_retired"] is True
    assert children["all_descendants_retired"] is False
    assert children["coverage"]["scope"] == "exact_retained_handle_bound_set_only"
    assert children["coverage"]["all_descendants_covered"] is False
    assert br._ownership_coverage_verified(children) is True
    assert br._ownership_coverage_verified(browser["bound_set"]) is True
    assert all(row["signaled"] is True and row["observer_handle_closed"] is True
               for row in children["processes"])


def _receipt_preserving_primary(run: _ContractRun, failure: BaseException | None,
                               *, injected_fault: str | None = None) -> BaseException | None:
    try:
        run.write_receipt(failure=failure, injected_fault=injected_fault)
    except BaseException as exc:
        if failure is None:
            return exc
        failure.add_note("secondary native contract receipt failure: " + type(exc).__name__ + ": " + str(exc))
    return failure


def test_native_joint_guards_floor_release_close_and_fresh_browser(tmp_path: Path) -> None:
    failure: BaseException | None = None
    with FixtureSite() as fixture:
        run = _ContractRun(tmp_path, fixture)
        try:
            first = fixture.origin + "/text/en"
            second = fixture.origin + "/text/zh"
            run.boundary.register_explicit_url("first", first)
            run.journal("cold_before_admission")
            with pytest.raises(ResourceUnavailable):
                run.boundary.execute(tools.ToolAction("browser_open", {"url": first}, "first"))
            assert run.factory_count == 0 and run.registries == [] and run.http_requests == []
            run.manager.start(run.route.route_id)
            state = run.snapshot()
            assert state["joint_admission"]["protected_free_floor"] == {"host_ram": 30, "windows_commit": 30}
            assert state["joint_admission"]["startup_free"] == {"host_ram": 100, "windows_commit": 100}
            assert state["joint_admission"]["combined_demands"] == {"host_ram": 70, "windows_commit": 70}
            old_model, old_tool = run.stage.token, run.companion._reservation
            assert dict(old_model.demands) == {"host_ram": 50, "windows_commit": 40}
            assert dict(old_tool.demands) == {"host_ram": 20, "windows_commit": 30}
            guards_before = len(run.guards)
            run.boundary.execute(tools.ToolAction("browser_open", {"url": first}, "first"))
            assert run.guards[guards_before]["factory_count"] == 0
            assert run.guards[guards_before]["http_request_count"] == 0
            assert run.guards[-1]["factory_count"] == 1  # Caller postcondition after the real operation.
            result = run.boundary.execute(tools.ToolAction("browser_read", {}, "first"))
            assert "CEDAR-4827" in result.data["text"] and result.source == first
            old_browser, old_registry = run.browsers[0], run.registries[0]
            sample = run.sample()
            assert sample["coverage"]["node_spawn_identity_verified"] is True
            assert sample["coverage"]["browser_root_identity_verified"] is True
            assert sample["coverage"]["all_descendants_covered"] is False
            assert {"playwright_node", "edge_browser"} <= {row["role"] for row in sample["processes"]}
            run.journal("real_browser_read_complete")
            requests_before, guards_before = len(run.http_requests), len(run.guards)
            run.free["host_ram"] = 29  # Synthetic fixture floor refusal, not measured system RAM.
            run.boundary.register_explicit_url("refused", second)
            with pytest.raises(ResourceUnavailable):
                run.boundary.execute(tools.ToolAction("browser_open", {"url": second}, "refused"))
            assert len(run.guards) > guards_before and len(run.http_requests) == requests_before
            assert run.factory_count == 1 and len(run.registries) == 1
            refusal = run.snapshot()
            assert refusal["joint_admission"]["needs_drain"] is True
            assert (refusal["joint_admission"]["protected_free_floor"]
                    == state["joint_admission"]["protected_free_floor"])
            run.free["host_ram"] = 100
            with pytest.raises(ResourceUnavailable):
                run.manager.start(run.route.route_id)  # A fresh free sample cannot relax the latched floor refusal.
            run.sample()
            guards_before = len(run.guards)
            run.journal("manager_lock_close_started")
            with run.manager._lock:
                run.manager.close_joint()
            run.journal("manager_lock_close_finished")
            assert len(run.guards) == guards_before  # Cleanup never invokes manager admission.
            assert run.ledger.was_released(old_model) and run.ledger.was_released(old_tool)
            assert run.companion.is_cold() and run.companion._browser is None
            assert run.ledger.snapshot()["owners"] == run.ledger.snapshot()["quarantined"] == []
            _successful_real_proof(run.close_proofs[-1])
            assert all(not thread.is_alive() for thread in old_browser._worker._threads)
            run.manager.start(run.route.route_id)
            assert run.companion._reservation is not old_tool and run.stage.token is not old_model
            assert run.companion._generation != state["joint_admission"]["generation"]
            assert run.companion.is_cold() and run.factory_count == 1
            with pytest.raises(ResourceUnavailable, match="retired joint generation"):
                old_browser._resource_guard()
            with pytest.raises(RuntimeError, match="exact registry generation"):
                old_browser._process_observer("cdp_begin", {})
            run.boundary.register_explicit_url("fresh", first)
            run.boundary.execute(tools.ToolAction("browser_open", {"url": first}, "fresh"))
            assert run.browsers[1] is not old_browser and run.registries[1] is not old_registry
            assert run.browsers[1]._worker is not old_browser._worker
            run.sample()
            current_model, retained_tool = run.stage.token, run.companion._reservation
            assert run.manager.close(run.route.route_id) is True
            assert run.ledger.was_released(current_model) and run.ledger.owns(retained_tool)
            assert run.ledger.snapshot()["reserved"] == {"host_ram": 20, "windows_commit": 30}
            with pytest.raises(ResourceUnavailable):
                run.boundary.execute(tools.ToolAction("browser_read", {}, "fresh"))
            run.sample()
            # Drain the retained real tool, reset, then take fresh model/tool leases.
            run.manager.start(run.route.route_id)
            assert run.ledger.was_released(retained_tool) and run.factory_count == 2
            _successful_real_proof(run.close_proofs[-1])
            run.snapshot()
        except BaseException as exc:
            failure = exc
        finally:
            try:
                run.journal("final_resource_close_started")
                final = run.manager.finalize_close()
                run.snapshots.append(final)
                run.journal("final_resource_close_finished")
                assert final["ledger"]["owners"] == final["ledger"]["quarantined"] == []
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note("secondary native joint close failure: " + type(exc).__name__ + ": " + str(exc))
            failure = _receipt_preserving_primary(run, failure)
    if failure is not None:
        raise failure


def test_native_joint_registry_bound_limit_keeps_real_cleanup_unknown_and_tool_quarantined(tmp_path: Path) -> None:
    """Deliberately refuse bindings; success here means truthful quarantine.

    This does not classify a normal device/browser route as unsupported, nor
    claim global ledger emptiness. Actual browser cleanup still runs; the
    exact synthetic tool token remains charged because ownership was unknown.
    """
    failure: BaseException | None = None
    with FixtureSite() as fixture:
        run = _ContractRun(tmp_path, fixture, registry_limit=1)
        try:
            url = fixture.origin + "/text/en"
            run.boundary.register_explicit_url("limited", url)
            run.manager.start(run.route.route_id)
            model, tool = run.stage.token, run.companion._reservation
            run.boundary.execute(tools.ToolAction("browser_open", {"url": url}, "limited"))
            sampled = run.sample()
            assert sampled["coverage"]["binding_registry_overflow"] is True
            run.journal("intentional_bound_limit_close_started")
            with pytest.raises(RuntimeError, match="bound-set drain|owned cleanup contract"):
                with run.manager._lock:
                    run.manager.close_joint()
            proof = run.companion.release_evidence
            assert proof["owned_work_drained"] is False and proof["quarantine_required"] is True
            assert proof["all_descendants_retired"] is False
            children = proof["registry_close_receipt"]["children"]
            assert children["coverage"]["binding_registry_overflow"] is True
            assert children["required_owned_bindings_verified"] is False
            assert children["bound_set_drain_verified"] is False
            assert run.ledger.was_released(model) and run.ledger.owns(tool)
            assert run.ledger.snapshot()["quarantined"] == [tool.owner]
            assert all(not thread.is_alive() for thread in run.browsers[0]._worker._threads)
            with pytest.raises(ResourceUnavailable):
                run.companion.reset_after_verified_drain()
            assert run.companion.close() is False  # A cached failed finite wait cannot manufacture recovery.
            run.snapshot()
        except BaseException as exc:
            failure = exc
        finally:
            # Always attempt real manager cleanup. In this deliberate fault
            # case the exact tool token MUST stay quarantined; do not force
            # drained=True, synthesize a receipt or forget the failed owner.
            try:
                with pytest.raises(ResourceUnavailable, match="owned cleanup contract"):
                    run.manager.close_joint()
                run.journal("intentional_quarantine_preserved")
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note("secondary intentional quarantine close: " + type(exc).__name__)
            failure = _receipt_preserving_primary(
                run, failure, injected_fault="real_registry_max_bindings_1_intentional_contract_fault")
    if failure is not None:
        raise failure
