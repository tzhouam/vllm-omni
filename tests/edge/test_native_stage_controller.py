# SPDX-License-Identifier: Apache-2.0
"""Compile the C++ mobile control boundary against the shared v2 fixture."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "packages/omni-stage-contracts/conformance/v2.json"


def _assign(kind, values):
    statements = []
    for name, value in values.items():
        if name in {"state", "inputs", "buffers", "operation"} or value is None:
            continue
        literal = "true" if value is True else "false" if value is False else json.dumps(value)
        statements.append(f"value.{name} = {literal};")
    return f"omni::{kind} value;\n" + "\n".join(statements) + "\nreturn value;"


def test_native_controller_uses_shared_fixture_and_enforces_state_credit_release(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++17 compiler unavailable; Android device qualification is separate")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request, events, state = fixture["request"], fixture["events"], fixture["request"]["state"]
    rejected = []
    for override in fixture["reject_event_overrides"]:
        event = {**events[0], **override}
        rejected.append("{ " + _assign("StageEvent", event).replace("return value;", "") +
                        "rejects([&] { controller.try_emit(value); }); }")
    source = r'''
#include "omni_stage_controller.h"
#include <cassert>
#include <iostream>

omni::StageRequest fixture_request() { REQUEST }
omni::StateHandle fixture_state() { STATE }
omni::StageEvent fixture_first() { FIRST }
omni::StageEvent fixture_last() { LAST }

template<typename F> void rejects(F operation) {
    bool refused = false;
    try { operation(); } catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
}

int main() {
    const auto request = fixture_request();
    const auto state = fixture_state();
    const std::map<std::uint64_t, omni::BackendIdentity> backends{
        {request.stage_id, {state.backend, state.backend_instance_id, state.worker_generation}}
    };
    omni::StageController controller(backends, MAX_EVENTS, MAX_BYTES);
    auto seed = request; seed.request_id = "seed"; seed.state.reset();
    controller.begin(seed);
    auto seeded = fixture_first(); seeded.request_id = "seed";
    seeded.state = state; seeded.terminal = true;
    assert(controller.try_emit(seeded) == omni::EmitResult::accepted);
    controller.finish(seed);
    auto continuation = request; continuation.state = state;
    rejects([&] { controller.begin(continuation); });
    assert(controller.receive(seed));
    assert(controller.outstanding_bytes() == seeded.payload_nbytes);
    assert(controller.acknowledge(seed, seeded.release_token));
    assert(!controller.acknowledge(seed, seeded.release_token));
    controller.begin(continuation);
    REJECTED
    assert(controller.try_emit(fixture_first()) == omni::EmitResult::accepted);
    assert(!controller.acknowledge(continuation, fixture_first().release_token));
    assert(controller.try_emit(fixture_last()) == omni::EmitResult::accepted);
    controller.finish(continuation);
    rejects([&] { controller.request_release(request.stage_id, state); });
    assert(controller.receive(continuation));
    assert(!controller.acknowledge(seed, fixture_first().release_token));
    assert(controller.acknowledge(continuation, fixture_first().release_token));
    assert(controller.receive(continuation));
    assert(controller.acknowledge(continuation, fixture_last().release_token));
    const auto release = controller.request_release(request.stage_id, state);
    assert(release == controller.request_release(request.stage_id, state));
    auto next = request; next.request_id = "next"; next.state.reset();
    rejects([&] { controller.begin(next); });
    rejects([&] { controller.confirm_release(request.stage_id, state, "wrong-release"); });
    controller.confirm_release(request.stage_id, state, release);
    rejects([&] { auto stale = continuation; stale.request_id = "stale-state"; controller.begin(stale); });

    omni::StageController pressured(backends, MAX_EVENTS, MAX_BYTES);
    pressured.begin(request);
    auto first = fixture_first(); auto second = fixture_last(); second.terminal = false;
    assert(pressured.try_emit(first) == omni::EmitResult::accepted);
    assert(pressured.try_emit(second) == omni::EmitResult::accepted);
    auto third = second; third.seq = 3; third.release_token = "third"; third.payload_nbytes = 1;
    assert(pressured.try_emit(third) == omni::EmitResult::backpressure);
    assert(pressured.receive(request));
    assert(pressured.acknowledge(request, first.release_token));
    assert(pressured.try_emit(third) == omni::EmitResult::accepted);
    assert(pressured.receive(request));
    pressured.cancel(request);
    assert(pressured.outstanding_bytes() == second.payload_nbytes);
    assert(pressured.try_emit(third) == omni::EmitResult::retired);
    rejects([&] { pressured.begin(next); });
    pressured.finish(request);  // Real adapter must ACK quiescence first.
    rejects([&] { pressured.begin(next); });
    assert(pressured.acknowledge(request, second.release_token));
    pressured.begin(next);
    rejects([&] { pressured.finish(next); });  // Missing terminal is a failure.

    omni::StageController identity(backends, MAX_EVENTS, MAX_BYTES);
    auto wrong_worker = request; wrong_worker.worker_generation = "other-worker";
    rejects([&] { identity.begin(wrong_worker); });
    identity.begin(request);
    auto wrong_state = first; wrong_state.state = state;
    wrong_state.state->artifact_id = "another-artifact";
    rejects([&] { identity.try_emit(wrong_state); });
    std::cout << "native v2 conformance passed\n";
}
'''
    replacements = {
        "REQUEST": _assign("StageRequest", request), "STATE": _assign("StateHandle", state),
        "FIRST": _assign("StageEvent", events[0]), "LAST": _assign("StageEvent", events[1]),
        "MAX_EVENTS": str(fixture["stream_limits"]["max_events"]),
        "MAX_BYTES": str(fixture["stream_limits"]["max_bytes"]), "REJECTED": "\n".join(rejected),
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    unit = tmp_path / "conformance.cpp"
    unit.write_text(source, encoding="utf-8")
    binary = tmp_path / "conformance"
    compiled = subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
        "-I", str(ROOT / "packages/omni-stage-controller/include"), str(unit), "-o", str(binary),
    ], text=True, capture_output=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    tested = subprocess.run([str(binary)], text=True, capture_output=True, timeout=10)
    assert tested.returncode == 0, tested.stdout + tested.stderr
    assert "native v2 conformance passed" in tested.stdout
