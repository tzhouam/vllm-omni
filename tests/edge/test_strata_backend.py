# SPDX-License-Identifier: Apache-2.0
"""Supervisor tests use a pinned loopback fixture, never claim model evidence."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.engine.backends.strata import (
    BACKEND_NAME,
    PINNED_STRATA_REVISION,
    StrataTextStageClient,
    _check_live_memory,
    _safe_telemetry,
    _sanitize_diagnostic,
    _stream_request,
    _verify_python_environment,
)
from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest, WeightTierBudget, WeightTierPlan

SERVER = r"""
import argparse, json, signal, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('engine','config','host','port'): p.add_argument('--'+key)
a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
completed=[]
args=cfg['args']; ctx=int(args[args.index('--max-context')+1])
Path(a.config).with_name('observed.json').write_text(json.dumps(cfg))
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def reply(self,obj):
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
        self.wfile.write(json.dumps(obj).encode())
    def do_GET(self):
        if self.path=='/health':
            self.reply({'status':'ok','loaded':True,'service':'strata','images':False})
        elif self.path=='/props':
            self.reply({'total_slots':1,'model_alias':cfg['model_name'],
                        'default_generation_settings':{'n_ctx':ctx}})
        elif self.path=='/metrics':
            self.reply({'totals':{'requests':len(completed)}, 'requests':completed[::-1]})
    def do_POST(self):
        if self.headers.get('Authorization')!='Bearer '+cfg['api_key']:
            self.send_error(401); return
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Path(a.config).with_name('request.json').write_text(json.dumps(body))
        text=body['messages'][0]['content']
        if text=='refuse': self.send_error(400,'context capacity'); return
        self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
        def emit(delta,finish=None):
            self.wfile.write(b'data: '+json.dumps({'choices':[{'delta':delta,'finish_reason':finish}]}).encode()+b'\n\n')
            self.wfile.flush()
        try:
            if text=='many':
                for _ in range(200): emit({'content':'x'})
            elif text=='slow':
                for _ in range(100): emit({'content':'x'}); time.sleep(.1)
            else:
                emit({'content':'A'}); emit({'content':'B'})
            emit({},'stop')
            completed.append({'hit_rate':0.5,'file_mb':2.5,'ram_blobs':3,'prompt':'private prompt'})
            self.wfile.write(b'data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':12,'completion_tokens':2}}).encode()+b'\n\n')
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError): pass
ThreadingHTTPServer((a.host,int(a.port)),Handler).serve_forever()
"""


def _manifest(root: Path, *, revision: str = "source-revision"):
    records = []
    for file in sorted(root.rglob("*")):
        if file.is_file():
            records.append(
                ArtifactFile(
                    file.relative_to(root).as_posix(),
                    file.stat().st_size,
                    hashlib.sha256(file.read_bytes()).hexdigest(),
                    role="auxiliary",
                )
            )
    return ArtifactManifest("fixture", revision, "MIT", tuple(records))


class Ledger:
    def __init__(self):
        self.releases = []

    def release(self, reservation, *, drained):
        self.releases.append(drained)


@pytest.fixture
def stage_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.engine.backends.strata._probe_memory",
        lambda _: {
            "host_ram_available_bytes": 1 << 30,
            "gpu_total_bytes": 2 << 20,
            "gpu_free_bytes": 2 << 20,
            "windows_commit_available_bytes": 1 << 30,
            "wsl_ram_available_bytes": 1 << 30,
            "windows_host_available_bytes": 1 << 30,
            "gpu_source": "test fixture only",
            "windows_host_verification": "fixture",
        },
    )
    runtime = tmp_path / "runtime"
    source = tmp_path / "source"
    pack = tmp_path / "pack"
    (runtime / "serve").mkdir(parents=True)
    (runtime / "engine").mkdir()
    (runtime / "serve" / "server.py").write_text(SERVER)
    (runtime / "engine" / "strata").write_text("fixture native executable")
    source.mkdir()
    (source / "weights.gguf").write_bytes(b"weights")
    (source / "ple.gguf").write_bytes(b"ple")
    (pack / "tokenizer").mkdir(parents=True)
    for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
        (pack / "tokenizer" / name).write_text("fixture")
    (pack / "index.txt").write_text("fixture packed tensor index")
    runtime_manifest = _manifest(runtime, revision=PINNED_STRATA_REVISION)
    artifact_manifest, prepared_manifest = _manifest(source), _manifest(pack)
    return {
        "name": BACKEND_NAME,
        "runtime_revision": PINNED_STRATA_REVISION,
        "runtime_root": str(runtime),
        "runtime_manifest": runtime_manifest.to_dict(),
        "python_bin": sys.executable,
        "python_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "engine_file": "engine/strata",
        "artifact_root": str(source),
        "artifact_manifest": artifact_manifest.to_dict(),
        "prepared_model_dir": str(pack),
        "prepared_pack_manifest": prepared_manifest.to_dict(),
        "conversion_manifest": {
            "complete": True,
            "source_manifest_sha256": artifact_manifest.manifest_sha256,
            "prepared_manifest_sha256": prepared_manifest.manifest_sha256,
            "tool_revision": PINNED_STRATA_REVISION,
            "conversions": [],
        },
        "native_file": "weights.gguf",
        "ple_file": "ple.gguf",
        "expert_ram_budget_bytes": 1 << 20,
        "host_overhead_bytes": 1 << 20,
        "gpu_budget_bytes": 1 << 20,
        "gpu_total_bytes": 2 << 20,
        "max_io_bytes": 65536,
        "start_timeout_s": 5,
        "request_timeout_s": 5,
    }


def _stage(config):
    ledger = Ledger()
    reservation = SimpleNamespace(demands={"host_ram": 4 << 20, "vram:0": 2 << 20})
    stage = StrataTextStageClient(SimpleNamespace(stage_id=0), config, ledger, reservation)
    return stage, ledger


def test_supervisor_builds_tool_free_bounded_configuration(stage_config):
    stage, ledger = _stage(stage_config)
    try:
        generated = json.loads((Path(stage._temporary.name) / "observed.json").read_text())
        assert generated["mcp_servers"] == {}
        assert "before_load" not in generated and "vision" not in generated
        assert generated["parallel"] == 1 and generated["fit_max_tokens"] is False
        args = generated["args"]
        assert args[args.index("--spec") + 1] == "2"
        assert args[args.index("--suffix-draft") + 1] == "0"
        assert args[args.index("--lookup-chain") + 1] == "0"
        assert args[args.index("--ple-io") + 1] == "direct"
        assert stage.execution_plan["mtp_enabled"] is False
        assert stage.execution_plan["native_verify_window"] == 2
        assert args[args.index("--prompt-cache") + 1] == "0"
        assert args[args.index("--conversation-cache-mib") + 1] == "0"
        assert stage.execution_plan["observed_model_placement"] is None
        assert stage.execution_plan["actual_io_mode"] is None
    finally:
        stage.shutdown()
    assert ledger.releases[-1] is True


def test_agent_forwarded_vram_pool_is_only_a_gpu_zero_alias(stage_config):
    config = stage_config | {"gpu_pool": "vram"}
    stage = StrataTextStageClient(
        SimpleNamespace(stage_id=0),
        config,
        Ledger(),
        SimpleNamespace(demands={"host_ram": 4 << 20, "vram": 2 << 20}),
    )
    try:
        assert stage.execution_plan["gpu_pool"] == "vram"
        assert stage.execution_plan["requested_device"] == "cpu+cuda:0"
        assert stage.execution_plan["observed_model_placement"] is None
    finally:
        stage.shutdown()
    with pytest.raises(ValueError, match="physical GPU index"):
        StrataTextStageClient(
            SimpleNamespace(stage_id=0),
            config | {"gpu_index": 1},
            Ledger(),
            SimpleNamespace(demands={"host_ram": 4 << 20, "vram": 2 << 20}),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"mcp_servers": {"shell": {"command": "bad"}}},
        {"before_load": ["bad"]},
        {"args": ["--batch", "2"]},
        {"runtime_revision": "0" * 40},
        {"expert_ram_budget_bytes": 10 << 20},
        {"conversion_manifest": {"complete": False}},
        {"start_timeout_s": float("nan")},
        {"ple_prefetch": "false"},
        {"spec_tokens": 4},
        {"spec_tokens": 1},
        {"ple_io": "ram"},
        {"gpu_index": 1},
    ],
)
def test_admission_rejects_unbound_or_unsafe_configuration(stage_config, change):
    ledger = Ledger()
    with pytest.raises((ValueError, ResourceUnavailable)):
        StrataTextStageClient(
            SimpleNamespace(stage_id=0),
            stage_config | change,
            ledger,
            SimpleNamespace(demands={"host_ram": 4 << 20, "vram:0": 2 << 20}),
        )
    assert ledger.releases == [True]


def test_modified_secondary_artifact_is_refused(stage_config):
    (Path(stage_config["artifact_root"]) / "ple.gguf").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        _stage(stage_config)


def test_mtp_draft_and_gpu_cache_limit_are_part_of_identity(stage_config):
    pack = Path(stage_config["prepared_model_dir"])
    (pack / "draft").mkdir()
    (pack / "draft" / "weights.bin").write_bytes(b"pinned draft")
    prepared = _manifest(pack)
    config = stage_config | {
        "prepared_pack_manifest": prepared.to_dict(),
        "mtp_directory": "draft",
        "spec_tokens": 4,
        "vram_reserve_mib": 0,
    }
    config["conversion_manifest"] = stage_config["conversion_manifest"] | {
        "prepared_manifest_sha256": prepared.manifest_sha256
    }
    stage, _ = _stage(config)
    try:
        generated = json.loads((Path(stage._temporary.name) / "observed.json").read_text())
        args = generated["args"]
        assert args[args.index("--spec") + 1] == "4"
        assert args[args.index("--mtp") + 1] == str(pack / "draft")
        assert args[args.index("--vram-reserve-mib") + 1] == "1"
        assert stage.execution_plan["mtp_enabled"] is True
    finally:
        stage.shutdown()


def test_typed_tier_plan_uses_exact_stage_lease_and_keeps_observation_unknown(stage_config):
    source = ArtifactManifest.from_dict(stage_config["artifact_manifest"])
    prepared = ArtifactManifest.from_dict(stage_config["prepared_pack_manifest"])
    budget = WeightTierBudget(
        cpu_expert_cache_bytes=1 << 20,
        host_transfer_bytes=65536,
        host_workspace_bytes=1 << 19,
        host_headroom_bytes=1 << 19,
        gpu_weights_bytes=1 << 19,
        gpu_expert_cache_bytes=1 << 19,
        ssd_artifact_bytes=source.total_size_bytes + prepared.total_size_bytes,
    )
    plan = WeightTierPlan(
        "fixture",
        source.manifest_sha256,
        BACKEND_NAME,
        PINNED_STRATA_REVISION,
        budget,
        ssd_experts=True,
        lookup_tables_on_demand=True,
    )
    config = stage_config | {"weight_tier_plan": plan.to_dict(), "route_id": "fixture"}
    stage = StrataTextStageClient(
        SimpleNamespace(stage_id=0), config, Ledger(), SimpleNamespace(demands=budget.resource_demands())
    )
    try:
        assert stage.execution_plan["weight_tier_plan"] == plan.to_dict()
        assert stage.execution_plan["placement_report"]["compute"] == ()
        assert stage.execution_plan["weight_tier_evidence"] == "declared_admission_budget_not_measured_allocation"
    finally:
        stage.shutdown()
    with pytest.raises(ValueError, match="exact stage resource lease"):
        _stage(config)


@pytest.mark.asyncio
async def test_ordered_stream_ack_isolation_and_explicit_usage(stage_config):
    stage, _ = _stage(stage_config)
    try:
        await stage.add_request_async("one", {"text": "hello", "stream_agent": True})
        with pytest.raises(ResourceUnavailable):
            await stage.add_request_async("two", {"text": "hello"})
        seen = []
        while (delta := await stage.receive_agent_delta("one")) is not None:
            seen.append(delta[0])
        await stage._task
        out = stage.get_graph_output_nowait()
        assert seen == ["A", "B"]
        assert out.outputs[0].text == "AB"
        assert out.metrics["usage"]["completion_tokens"] == 2
        events = out.metrics["stage_events"]
        terminal = out._custom_output["stage_event"]
        assert [event["seq"] for event in events] == [1, 2]
        assert terminal["seq"] == 3 and terminal["terminal"]
        assert "prompt" not in out.metrics["runtime_telemetry"]
        assert out.metrics["runtime_telemetry"]["physical_ssd_read_bytes"] is None
        stage.acknowledge("one", terminal["epoch"] - 1, terminal["worker_generation"])
        assert stage._active == "one"
        stage.acknowledge("one", terminal["epoch"], terminal["worker_generation"])
        await stage.add_request_async("two", {"text": "hello"})
        await stage._task
        assert stage.get_graph_output_nowait()._custom_output["stage_event"]["epoch"] == 2
    finally:
        stage.shutdown()


@pytest.mark.asyncio
async def test_backpressure_cancel_retires_and_suppresses_late_output(stage_config):
    stage, ledger = _stage(stage_config)
    await stage.add_request_async("blocked", {"text": "many", "stream_agent": True})
    for _ in range(100):
        if stage._agent_stream.qsize() == 64:
            break
        await asyncio.sleep(0.01)
    assert stage._agent_stream.qsize() == 64
    assert not stage._task.done()
    stage.acknowledge("blocked", stage._epoch, stage._generation)
    assert stage._ack_pending is None
    await stage.abort_requests_async(["blocked"])
    assert stage._proc.poll() is not None
    assert stage.get_graph_output_nowait() is None
    assert ledger.releases[-1] is True
    with pytest.raises(Exception):
        await stage.add_request_async("next", {"text": "hello"})
    stage.shutdown()


@pytest.mark.asyncio
async def test_context_refusal_retires_worker_without_output_success(stage_config):
    stage, ledger = _stage(stage_config)
    try:
        await stage.add_request_async("too-long", {"text": "refuse"})
        await stage._task
        output = stage.get_graph_output_nowait()
        assert output._custom_output["stage_event"]["kind"] == "error"
        assert ledger.releases[-1] is True
        assert stage._proc.poll() is not None
    finally:
        stage.shutdown()


def _sse(*rows, done=True):
    data = b"".join(b"data: " + json.dumps(row).encode() + b"\n\n" for row in rows)
    return io.BytesIO(data + (b"data: [DONE]\n\n" if done else b""))


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, *args, **kwargs):
        return self.response


@pytest.mark.parametrize(
    "rows,done,match",
    [
        ([{"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}], False, "before terminal"),
        ([{"choices": [{"delta": {"tool_calls": [{"id": "bad"}]}}]}], True, "tool calls"),
        (
            [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, {"choices": [{"delta": {"content": "late"}}]}],
            True,
            "after its finish",
        ),
    ],
)
def test_sse_refuses_incomplete_tools_and_late_content(monkeypatch, rows, done, match):
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener(_sse(*rows, done=done)))
    with pytest.raises(RuntimeError, match=match):
        _stream_request(
            "http://127.0.0.1/",
            {},
            token="fixture",
            timeout=1,
            limit=1024,
            cancelled=threading.Event(),
            on_delta=lambda *a: None,
        )


def test_physical_io_and_compute_cannot_be_inferred_from_logical_counter():
    telemetry = _safe_telemetry(
        {"file_mb": 42, "actual_io_mode": "direct", "observed_compute_units": ["CPU"], "hit_rate": float("nan")}
    )
    assert telemetry["file_mb"] == 42
    assert telemetry["hit_rate"] is None
    assert telemetry["actual_io_mode"] is None
    assert telemetry["physical_ssd_read_bytes"] is None
    assert telemetry["observed_compute_units"] is None


def test_diagnostics_never_retain_prompt_paths_or_infer_steady_io():
    assert _sanitize_diagnostic("prompt: private secret") is None
    assert _sanitize_diagnostic("strata generate: expert arena read unbuffered (private/path)") == (
        "strata arena_load_io_mode: unbuffered"
    )
    assert _sanitize_diagnostic("[strata] failed to open private/path") == "strata diagnostic: backend_error"
    assert _sanitize_diagnostic(
        "FileExpertSource: --resident-budget-gib 40.00 exceeds available physical RAM secret"
    ) == ("strata diagnostic: resident_budget_clamped")
    assert _sanitize_diagnostic("strata generate: the file tier reads through the file cache (changed) (private)") == (
        "strata file_tier_io_policy: buffered"
    )


@pytest.mark.parametrize(
    "observation",
    [
        {"host_ram_available_bytes": 1},
        {"gpu_free_bytes": 1},
        {"gpu_total_bytes": 3 << 20},
        {"gpu_total_bytes": None, "gpu_free_bytes": None},
    ],
)
def test_fresh_admission_refuses_before_process_start(stage_config, monkeypatch, observation):
    snapshot = {
        "host_ram_available_bytes": 1 << 30,
        "gpu_total_bytes": 2 << 20,
        "gpu_free_bytes": 2 << 20,
        "windows_commit_available_bytes": 1 << 30,
    } | observation
    monkeypatch.setattr("vllm_omni.engine.backends.strata._probe_memory", lambda _: snapshot)

    def unexpected_start(*args, **kwargs):
        raise AssertionError("fresh admission must precede process launch")

    monkeypatch.setattr("vllm_omni.engine.backends.strata.subprocess.Popen", unexpected_start)
    ledger = Ledger()
    with pytest.raises(ResourceUnavailable, match="fresh admission refused") as exc:
        StrataTextStageClient(
            SimpleNamespace(stage_id=0),
            stage_config,
            ledger,
            SimpleNamespace(demands={"host_ram": 4 << 20, "vram:0": 2 << 20}),
        )
    assert json.loads(str(exc.value).split("; snapshot=", 1)[1]) == snapshot
    assert ledger.releases == [True]


def test_wsl_quota_and_commit_are_distinct_constraints():
    snapshot = {
        "host_ram_available_bytes": 100,
        "gpu_total_bytes": 200,
        "gpu_free_bytes": 100,
        "wsl_ram_available_bytes": 20,
        "windows_commit_available_bytes": None,
        "windows_host_available_bytes": 15,
        "windows_host_verification": "measured",
    }
    with pytest.raises(ResourceUnavailable, match="wsl_ram: required=30"):
        _check_live_memory(snapshot, {"host_ram": 30, "vram:0": 50, "wsl_ram": 30}, "vram:0", 200)
    with pytest.raises(ResourceUnavailable, match="windows_commit: required=1"):
        _check_live_memory(snapshot, {"host_ram": 10, "vram:0": 50, "windows_commit": 1}, "vram:0", 200)
    with pytest.raises(ResourceUnavailable, match="Windows physical RAM is unverified"):
        _check_live_memory(
            snapshot | {"is_wsl": True, "windows_host_available_bytes": None},
            {"host_ram": 10, "vram:0": 50},
            "vram:0",
            200,
        )


def test_python_dependency_identity_is_rechecked():
    import importlib.metadata

    names = ("numpy", "jinja2", "regex", "PyYAML", "psutil", "Pillow", "gguf")
    dependencies = {}
    for name in names:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    identity = {
        "sys_version": sys.version,
        "dependencies": dependencies,
        "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
    }
    assert _verify_python_environment(Path(sys.executable), identity)["dependencies"] == dependencies
    with pytest.raises(ValueError, match="environment changed"):
        _verify_python_environment(Path(sys.executable), identity | {"sys_version": "different"})


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_retirement_finds_reparented_term_ignoring_child(tmp_path):
    import psutil

    marker = tmp_path / "child.pid"
    child_code = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(60)"
    )
    leader_code = (
        "import subprocess,sys,time;from pathlib import Path;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"p=Path({str(marker)!r});\nwhile not p.exists():time.sleep(.01)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    leader.wait(timeout=5)
    child = psutil.Process(int(marker.read_text()))
    stage = object.__new__(StrataTextStageClient)
    stage._retirement_lock = threading.Lock()
    stage._proc, stage._known_children = leader, {}
    stage._diagnostics = stage._temporary = None
    try:
        assert child.is_running()
        assert stage._terminate() is True
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()
