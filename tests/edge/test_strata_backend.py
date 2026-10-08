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
    _BOOTSTRAP,
    BACKEND_NAME,
    PINNED_STRATA_REVISION,
    StrataTextStageClient,
    _check_live_memory,
    _native_cache_control,
    _native_compute_observation,
    _safe_telemetry,
    _sanitize_diagnostic,
    _stream_request,
    _verify_cache_bounds,
    _verify_load_configuration,
    _verify_python_environment,
    validate_strata_load_plan,
)
from vllm_omni.engine.resource_ledger import ResourceUnavailable
from vllm_omni.engine.weight_tiers import ArtifactFile, ArtifactManifest, WeightTierBudget, WeightTierPlan

SERVER = r"""
import argparse, json, os, signal, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('engine','config','host','port'): p.add_argument('--'+key)
a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
completed=[]
args=cfg['args']; ctx=int(args[args.index('--max-context')+1])
workers=int(args[args.index('--pool-workers')+1])
cache_slots=int(args[args.index('--expert-cache')+1])
layout=(Path(args[args.index('--pack')+1])/'native_experts.txt').read_text()
blob=max(int(line.split()[4]) for line in layout.splitlines() if line and not line.startswith('#'))
cache_mib=(cache_slots*blob)>>20
arena_mib=int(float(args[args.index('--resident-budget-gib')+1])*2**30)>>20
if cfg['model_name']=='fixture-cache-too-large': cache_mib+=2
if cfg['model_name']=='fixture-cache-wrong-slots': cache_slots+=1
if cfg['model_name']=='fixture-ram-too-large': arena_mib+=1
info={'engine':'0.1.40.3','context':ctx,'kv':args[args.index('--kv')+1],
      'spec':int(args[args.index('--spec')+1]), 'lookup':0,
      'conversation_cache_mib':0,'conversation_cache_slots':0,
      'pool_workers':workers,'expert_slots':cache_slots,'expert_cache_mib':cache_mib,'arena_mib':arena_mib}
print('strata supervisor: native process started',flush=True)
if os.name=='nt': print('strata supervisor: native contained in Windows kill-on-close job',flush=True)
print('strata generate: GPU 0: Fixture GPU, compute capability 12.0',flush=True)
print(f'strata generate: CPU pool tasks/phase: {3*(workers+1)} (automatic), participating threads: {workers+1}',flush=True)
print(f'strata generate: {workers} expert-pool workers + the host thread',flush=True)
print('strata generate: native pack: /private/fixture experts (largest blob 1.00 MB), token embedding Q8_0 in mapped host memory (1 MiB, 0.1 s)',flush=True)
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
            self.reply({'engine':info,'totals':{'requests':len(completed)}, 'requests':completed[::-1]})
    def do_POST(self):
        if self.headers.get('Authorization')!='Bearer '+cfg['api_key']:
            self.send_error(401); return
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Path(a.config).with_name('request.json').write_text(json.dumps(body))
        text=body['messages'][0]['content']
        if text=='refuse': self.send_error(400,'context capacity'); return
        if text=='restart': print('strata supervisor: native process started',flush=True)
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
            print('strata supervisor: native done '+json.dumps({'generated':2,'prompt_tokens':12,
                  'hits':5,'lookups':10,'offloaded':0,'prompt_read':12}),flush=True)
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
            "gpu_name_sha256": hashlib.sha256(b"Fixture GPU").hexdigest(),
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
    (pack / "native_experts.txt").write_text(
        "# strata native experts v4: fixture (n_expert 8)\n"
        "0 12 7 0 131072 0 0 0 weights.gguf\n"
        "1 12 7 1048576 131072 0 0 0 weights.gguf\n"
    )
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
        "gpu_expert_cache_bytes": 1 << 20,
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


@pytest.mark.parametrize("tampered_conversion", [False, True])
def test_registered_stage_config_roundtrip_preserves_strict_conversion_proof(stage_config, tampered_conversion):
    """Exercise the production config/factory path, not only a raw dictionary."""
    from omegaconf import DictConfig, ListConfig

    from vllm_omni.config.stage_config import (
        DeployConfig,
        PipelineConfig,
        StageDeployConfig,
        StageExecutionType,
        StagePipelineConfig,
        merge_pipeline_deploy,
    )
    from vllm_omni.engine.stage_runtime import StageRuntime

    config = json.loads(json.dumps(stage_config))
    if tampered_conversion:
        config["conversion_manifest"]["source_manifest_sha256"] = "0" * 64
    pipeline = PipelineConfig(
        model_type="strata_protocol_fixture",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="agent",
                execution_type=StageExecutionType.GRAPH,
                final_output=True,
                final_output_type="text",
            ),
        ),
    )
    deploy = DeployConfig(
        async_chunk=False,
        stages=[
            StageDeployConfig(
                stage_id=0,
                backend=config,
                resource_budget={
                    "capacities": {"host_ram": 1 << 30, "vram:0": 2 << 20},
                    "demands": {"host_ram": 4 << 20, "vram:0": 2 << 20},
                },
            )
        ],
    )
    configs = [stage.to_omegaconf() for stage in merge_pipeline_deploy(pipeline, deploy)]
    conversion = configs[0].engine_args.backend.conversion_manifest
    assert isinstance(conversion, DictConfig)
    assert isinstance(conversion.conversions, ListConfig)
    runtime = StageRuntime(configs, "strata-protocol-fixture", "", stage_init_timeout=5, async_chunk=False)
    try:
        if tampered_conversion:
            with pytest.raises(ValueError, match="source-bound conversion manifest"):
                runtime.initialize()
        else:
            runtime.initialize()
            stage = runtime.stage_pools[0].stage_client
            assert isinstance(stage, StrataTextStageClient)
            validate_strata_load_plan(stage.execution_plan, "cpu+cuda:0")
            generated = json.loads((Path(stage._temporary.name) / "observed.json").read_text())
            args = generated["args"]
            assert args[args.index("--expert-cache") + 1] == "8"
            assert stage.execution_plan["gpu_expert_cache_control"]["requested_slots"] == 8
            assert (
                stage.execution_plan["artifact_manifest_sha256"]
                == ArtifactManifest.from_dict(config["artifact_manifest"]).manifest_sha256
            )
    finally:
        runtime.shutdown()
    assert runtime.resource_ledger.snapshot()["owners"] == []
    assert runtime.resource_ledger.snapshot()["quarantined"] == []


def test_supervisor_builds_tool_free_bounded_configuration(stage_config):
    stage, ledger = _stage(stage_config)
    try:
        generated = json.loads((Path(stage._temporary.name) / "observed.json").read_text())
        assert generated["mcp_servers"] == {}
        assert "before_load" not in generated and "vision" not in generated
        assert generated["parallel"] == 1 and generated["fit_max_tokens"] is False
        args = generated["args"]
        assert args[args.index("--expert-cache") + 1] == "8"
        assert args[args.index("--spec") + 1] == "2"
        assert args[args.index("--suffix-draft") + 1] == "0"
        assert args[args.index("--lookup-chain") + 1] == "0"
        assert args[args.index("--ple-io") + 1] == "direct"
        assert stage.execution_plan["mtp_enabled"] is False
        assert stage.execution_plan["native_verify_window"] == 2
        assert args[args.index("--prompt-cache") + 1] == "0"
        assert args[args.index("--conversation-cache-mib") + 1] == "0"
        assert stage.execution_plan["observed_model_placement"] is None
        assert stage.execution_plan["verified_execution_configuration"] == "cpu+cuda:0"
        assert stage.execution_plan["execution_configuration_evidence"]["scope"] == (
            "loaded_backend_execution_configuration_not_per_request_compute"
        )
        assert stage.execution_plan["observed_compute_units"] is None
        validate_strata_load_plan(stage.execution_plan, "cpu+cuda:0")
        assert stage.execution_plan["actual_io_mode"] is None
        control = stage.execution_plan["gpu_expert_cache_control"]
        assert control["allocation_upper_bytes"] == 1 << 20
        assert (
            stage.execution_plan["expert_cache_component_bounds"]["gpu_expert_cache"]["allocation_upper_bytes"]
            == 1 << 20
        )
        assert stage.execution_plan["gpu_aggregate_hard_cap_verified"] is False
        assert stage.execution_plan["three_tier_memory_qualified"] is False
        assert len(stage.execution_plan["route_controls_sha256"]) == 64
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


@pytest.mark.parametrize("budget", [0, 65536, 2 << 20])
def test_gpu_cache_budget_refuses_zero_too_small_or_aggregate_overflow(stage_config, budget):
    with pytest.raises((ValueError, ResourceUnavailable)):
        _stage(stage_config | {"gpu_expert_cache_bytes": budget})


@pytest.mark.parametrize("alias", ["fixture-cache-too-large", "fixture-cache-wrong-slots", "fixture-ram-too-large"])
def test_native_info_component_mismatch_refuses_and_drains(stage_config, alias):
    ledger = Ledger()
    with pytest.raises(ResourceUnavailable, match="expert cache|explicit uniform cache"):
        StrataTextStageClient(
            SimpleNamespace(stage_id=0),
            stage_config | {"model_alias": alias},
            ledger,
            SimpleNamespace(demands={"host_ram": 4 << 20, "vram:0": 2 << 20}),
        )
    assert ledger.releases[-1] is True


def test_cache_control_identity_changes_with_bound_and_requires_verified_layout(stage_config):
    first, _ = _stage(stage_config)
    try:
        first_identity = first.execution_plan["route_controls_sha256"]
    finally:
        first.shutdown()
    second, _ = _stage(stage_config | {"gpu_expert_cache_bytes": 1 << 19})
    try:
        assert second.execution_plan["route_controls_sha256"] != first_identity
        assert second.execution_plan["gpu_expert_cache_control"]["requested_slots"] == 4
    finally:
        second.shutdown()
    pack = Path(stage_config["prepared_model_dir"])
    with pytest.raises(ValueError, match="hash-verified"):
        _native_cache_control(pack, set(), 1 << 20)


@pytest.mark.parametrize(
    "layout",
    [
        "# strata native experts v5:\n0 12 7 0 131072\n",
        "# strata native experts v4:\n",
        "# strata native experts v4:\n1 12 7 0 131072\n",
        "# strata native experts v4:\n0 12 7 0 0\n",
    ],
)
def test_cache_control_refuses_unknown_empty_or_malformed_layout(tmp_path, layout):
    path = tmp_path / "native_experts.txt"
    path.write_text(layout)
    with pytest.raises(ValueError):
        _native_cache_control(tmp_path, {path}, 1 << 20)


def test_cache_control_accounts_for_native_profile_alignment(tmp_path):
    path = tmp_path / "native_experts.txt"
    path.write_text("# strata native experts v4:\n0 12 7 0 257\n")
    control = _native_cache_control(tmp_path, {path}, 1024)
    assert control["requested_slots"] == 2
    assert control["aligned_max_blob_bytes"] == 512
    assert control["allocation_upper_bytes"] == 1024


@pytest.fixture
def heterogeneous_profile_control(tmp_path):
    # Contract-only layout fixture: one-MiB and three-MiB layer blobs. Native
    # generate.cpp:4364-4386 can fill more small slots than the uniform N.
    path = tmp_path / "native_experts.txt"
    path.write_text("# strata native experts v4: fixture (n_expert 8)\n0 12 7 0 1048576\n1 13 8 8388608 3145728\n")
    return _native_cache_control(tmp_path, set(_manifest(tmp_path).verify(tmp_path)), 6 << 20)


def test_profile_sized_cache_accepts_expanded_slots_at_heterogeneous_byte_cap(heterogeneous_profile_control):
    control = heterogeneous_profile_control
    assert control["requested_slots"] == 2
    assert control["allocation_upper_bytes"] == 6 << 20
    # Three small experts plus one large expert exactly exhaust min(N*max_blob,
    # free_room) when free_room >= six MiB. A fifth small expert cannot fit.
    info = {"expert_slots": 4, "expert_cache_mib": 6, "arena_mib": 1}
    report = _verify_cache_bounds(info, control, 1 << 20, profiled=True)
    cache = report["gpu_expert_cache"]
    assert cache["observed_slots"] > control["requested_slots"]
    assert cache["allocation_lower_bytes"] == cache["allocation_upper_bytes"] == 6 << 20
    assert report["aggregate_gpu_hard_cap_verified"] is False
    assert report["total_process_peak_verified"] is False
    with pytest.raises(ResourceUnavailable, match="explicit uniform cache"):
        _verify_cache_bounds(info, control, 1 << 20, profiled=False)


def test_profile_sized_cache_accepts_free_room_limit_and_refuses_cap_exhaustion(heterogeneous_profile_control):
    control = heterogeneous_profile_control
    # With 3.5 MiB free_room the same ordered profile stops before its large
    # expert. INFO is floor-MiB precision; the adapter does not observe free_room.
    report = _verify_cache_bounds(
        {"expert_slots": 3, "expert_cache_mib": 3, "arena_mib": 1}, control, 1 << 20, profiled=True
    )
    assert report["gpu_expert_cache"]["observed_slots"] == 3
    assert report["gpu_expert_cache"]["allocation_lower_bytes"] == 3 << 20
    assert report["gpu_expert_cache"]["allocation_upper_bytes"] == (4 << 20) - 1
    # Reporting the fifth small expert's seventh MiB would violate the native
    # control byte cap even though expanded slot counts themselves are valid.
    with pytest.raises(ResourceUnavailable, match="component bound"):
        _verify_cache_bounds(
            {"expert_slots": 5, "expert_cache_mib": 7, "arena_mib": 1}, control, 1 << 20, profiled=True
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
    config = stage_config | {
        "weight_tier_plan": plan.to_dict(),
        "route_id": "fixture",
        "gpu_expert_cache_bytes": budget.gpu_expert_cache_bytes,
    }
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
        native_compute = out.metrics["runtime_telemetry"]["native_compute"]
        assert native_compute["cpu_expert_entries"] == 5
        assert native_compute["gpu_expert_entries"] == 5
        assert out.metrics["placement_report"]["compute"][0]["component"] == "routed_decode_experts"
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
    stage, ledger = _stage(stage_config | {"request_timeout_s": 15})
    await stage.add_request_async("blocked", {"text": "many", "stream_agent": True})
    deadline = asyncio.get_running_loop().time() + 8
    while stage._agent_stream.qsize() < 64 and not stage._task.done() and asyncio.get_running_loop().time() < deadline:
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


@pytest.mark.asyncio
async def test_hidden_native_restart_cannot_reuse_loaded_configuration(stage_config):
    stage, ledger = _stage(stage_config)
    try:
        await stage.add_request_async("restarted", {"text": "restart"})
        await stage._task
        output = stage.get_graph_output_nowait()
        assert output._custom_output["stage_event"]["kind"] == "error"
        assert stage._diagnostics.native_starts == 2
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


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_native_load_evidence_accepts_platform_line_terminators(ending):
    lines = [
        "strata generate: GPU 0: Fixture GPU, compute capability 12.0",
        "strata generate: CPU pool tasks/phase: 27 (automatic), participating threads: 9",
        "strata generate: 8 expert-pool workers + the host thread",
        "strata generate: native pack: C:/private/pack experts (largest blob 1.00 MB), "
        "token embedding Q8_0 in mapped host memory (1 MiB, 0.1 s)",
    ]
    for line in lines:
        expected = _sanitize_diagnostic(line)
        assert expected is not None
        assert _sanitize_diagnostic(line + ending) == expected


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


def test_agent_load_proof_rejects_gpu_mismatch_missing_pool_or_wrong_runtime(stage_config):
    stage, _ = _stage(stage_config)
    try:
        plan = stage.execution_plan
        for field, value in (("gpu_name_sha256", "0" * 64),):
            changed = json.loads(json.dumps(plan))
            changed["fresh_memory_admission"][field] = value
            with pytest.raises(RuntimeError, match="loaded backend execution configuration"):
                validate_strata_load_plan(changed, "cpu+cuda:0")
        changed = json.loads(json.dumps(plan))
        changed["execution_configuration_evidence"]["cpu_expert_pool"].pop("workers")
        with pytest.raises(RuntimeError):
            validate_strata_load_plan(changed, "cpu+cuda:0")
        changed = json.loads(json.dumps(plan))
        changed["execution_configuration_evidence"]["engine_info"]["engine"] = "different"
        with pytest.raises(RuntimeError):
            validate_strata_load_plan(changed, "cpu+cuda:0")
        with pytest.raises(RuntimeError):
            validate_strata_load_plan(plan, "cpu+cuda:1")
    finally:
        stage.shutdown()


def test_ram_storage_or_initialized_cpu_pool_does_not_prove_request_cpu_compute():
    proof = _verify_load_configuration(
        {}, {}, {"gpu_name_sha256": "0" * 64}, gpu=0, context=4096, kv="fp16", verify_window=2
    )
    assert proof["status"] == "unverified"
    assert _native_compute_observation(None, verified_native_pack=True, gpu=0)["units"] is None
    counters = {"lookups": 10, "hits": 8, "offloaded": 7}
    assert _native_compute_observation(counters, verified_native_pack=False, gpu=0)["units"] is None
    observed = _native_compute_observation(counters, verified_native_pack=True, gpu=0)
    assert observed["cpu_expert_entries"] == 2
    assert observed["gpu_expert_entries"] == 15
    assert observed["scope"] == "routed_decode_experts_only"
    assert observed["units"] == ["cpu", "cuda:0"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX stdout-observer fixture; native Windows uses pinned job module")
def test_bootstrap_observes_native_counters_without_changing_native_stdout(tmp_path):
    shim, server = tmp_path / "bootstrap.py", tmp_path / "server.py"
    shim.write_text(_BOOTSTRAP)
    rows = ["T 123456", "DONE 2 12 1.0 2.0 stop 0 0 0 8 10 1 2 3.0 12 7"]
    child = "\n".join(f"print({row!r})" for row in rows)
    server.write_text(
        "import subprocess,sys,json\n"
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True)\n"
        f"assert [x.strip() for x in p.stdout]=={rows!r}\n"
        "p.wait(timeout=5);print('upstream stdout unchanged')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", str(shim), sys.executable, str(server)],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "upstream stdout unchanged"
    assert "T 123456" not in result.stderr
    line = next(line for line in result.stderr.splitlines() if "native done " in line)
    counters = json.loads(line.split("native done ", 1)[1])
    assert counters["hits"] == 8 and counters["lookups"] == 10 and counters["offloaded"] == 7
    assert _sanitize_diagnostic(line) is not None
