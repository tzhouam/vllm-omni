# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``python -m vllm_omni.edge.local`` -- plan, run, and the M0 acceptance run.

``accept`` is the one that matters: it is the roadmap's M0 acceptance criteria
executed as a single command, writing one JSON file that carries the versions,
the plan, the twelve requests, the cancellation check and the memory record.
What it does *not* do is decide whether the numbers are good. It records the
conditions with them so somebody can.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from vllm_omni.edge.hardware_probe import load_profile
from vllm_omni.edge.local.capabilities import describe, enumerate_devices, parse_mask
from vllm_omni.edge.local.engine import LocalTextEngine, apply_runtime_env
from vllm_omni.edge.local.manifest import runtime_versions
from vllm_omni.edge.local.plan import ExecutionPlan, plan_text_session
from vllm_omni.edge.local.parity import (
    PromptParity,
    compare_ids,
    reference_token_ids,
    summarize,
)
from vllm_omni.edge.local.prompts import acceptance_prompts


def _write(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


def cmd_devices(args: argparse.Namespace) -> int:
    profile = load_profile()
    devices = enumerate_devices(profile, mask=parse_mask(args.mask))
    print(describe(devices))
    _write(args.json, {"devices": [d.to_dict() for d in devices], "runtime": runtime_versions().to_dict()})
    return 0


def _plan_from_args(args: argparse.Namespace) -> ExecutionPlan:
    return plan_text_session(
        args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=not args.cuda_graphs,
        mask=parse_mask(args.mask),
        digest_weights=getattr(args, "digest_weights", False),
    )


def cmd_plan(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    print(plan.summary())
    _write(args.json, plan.to_dict())
    # A refusal is a successful plan run. It gets exit code 2 so a script can
    # tell "refused with reasons" from "the planner itself broke" (1).
    return 0 if plan.admitted else 2


def cmd_run(args: argparse.Namespace) -> int:
    plan = _plan_from_args(args)
    if not plan.admitted:
        print(plan.summary(), file=sys.stderr)
        return 2
    print(plan.summary())

    async def go() -> dict[str, Any]:
        async with LocalTextEngine(plan) as engine:
            session = engine.open_session()
            rid, stream = await engine.submit(
                session, args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
            )
            async for event in stream:
                if event.kind == "token":
                    print(event.payload["text"], end="", flush=True)
                elif event.kind == "error":
                    print(f"\n[error] {event.error}", file=sys.stderr)
            print()
            return {
                "plan": plan.to_dict(),
                "placement": engine.report_placement(),
                "usage": engine.report_usage(),
            }

    payload = asyncio.run(go())
    _write(args.json, payload)
    return 0


async def _acceptance(plan: ExecutionPlan, args: argparse.Namespace) -> dict[str, Any]:
    prompts = acceptance_prompts()
    results: list[dict[str, Any]] = []
    async with LocalTextEngine(plan) as engine:
        print(f"loaded in {engine.load_seconds:.2f} s")

        for name, text in prompts:
            session = engine.open_session()
            rid, stream = await engine.submit(
                session, text,
                max_tokens=args.max_tokens,
                temperature=0.0,
                # The acceptance criterion is "at least 128 tokens", and a
                # short greedy answer that hits EOS would not meet it. The
                # length is forced so the criterion means the same thing for
                # every prompt; this is a measurement setting, not a sampling
                # recommendation.
                ignore_eos=True,
            )
            async for _ in stream:
                pass
            record = engine.records[rid]
            engine.close_session(session.session_id)
            results.append({"prompt": name, **record.to_dict(), "stream": stream.stats()})
            print(
                f"  {name:<12} in={record.prompt_tokens} out={record.output_tokens} "
                f"ttft={record.ttft_s:.3f}s decode={record.decode_tok_per_s:.2f} tok/s"
                if record.ttft_s and record.decode_tok_per_s
                else f"  {name:<12} out={record.output_tokens} (timing unavailable)"
            )

        # Cancellation: interrupt a long request part-way and prove that
        # nothing from the retired epoch reaches the consumer afterwards.
        cancel_session = engine.open_session()
        rid, stream = await engine.submit(
            cancel_session, prompts[0][1], max_tokens=4096, ignore_eos=True
        )
        seen = 0
        async for event in stream:
            if event.kind == "token":
                seen += 1
            if seen >= 8:
                break
        cancel_report = await engine.cancel(rid)
        cancel_report["events_before_cancel"] = seen
        leaked = []
        while True:
            try:
                event = await asyncio.wait_for(stream.get(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                break
            if event is None:
                break
            leaked.append(event.to_dict())
        cancel_report["events_after_cancel"] = leaked
        cancel_report["clean"] = not leaked
        print(
            f"  cancel: aborted={cancel_report['aborted_in_backend']} "
            f"after {seen} events, leaked={len(leaked)}"
        )

        payload = {
            "started_unix": time.time(),
            "argv": sys.argv,
            "acceptance": {
                "prompts": len(prompts),
                "min_tokens_required": args.max_tokens,
                "all_reached_min": all(r["output_tokens"] >= args.max_tokens for r in results),
                "cancel_clean": cancel_report["clean"],
            },
            "plan": plan.to_dict(),
            "placement": engine.report_placement(),
            "usage": engine.report_usage(),
            "requests": results,
            "cancel": cancel_report,
        }
    return payload


def cmd_accept(args: argparse.Namespace) -> int:
    """The M0 acceptance run: both startups, twelve prompts, cancel, memory."""
    profile = load_profile()

    # 1. The capability-masked startup. Planned first and on purpose: the
    #    criterion is that the no-NVIDIA class produces a compatible plan *or*
    #    an explicit refusal, and a refusal here is a pass, not a failure.
    masked_plan = plan_text_session(
        args.model,
        profile=profile,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=not args.cuda_graphs,
        mask=frozenset({"gpu_discrete"}),
    )
    print("=" * 72)
    print("capability-masked startup (--mask gpu_discrete): the no-NVIDIA class")
    print("=" * 72)
    print(masked_plan.summary())

    # 2. The device-visible startup, which is the one that runs.
    plan = _plan_from_args(args)
    print()
    print("=" * 72)
    print("device-visible startup")
    print("=" * 72)
    print(plan.summary())
    print()

    payload: dict[str, Any] = {
        "masked_plan": masked_plan.to_dict(),
        "masked_outcome": "admitted" if masked_plan.admitted else "refused_explicitly",
    }
    if not plan.admitted:
        payload["plan"] = plan.to_dict()
        payload["outcome"] = "refused_explicitly"
        _write(args.json, payload)
        return 2

    payload.update(asyncio.run(_acceptance(plan, args)))
    payload["outcome"] = "completed"
    ok = payload["acceptance"]["all_reached_min"] and payload["acceptance"]["cancel_clean"]
    print(f"\nacceptance: {'PASS' if ok else 'INCOMPLETE'} "
          f"({payload['acceptance']})")
    _write(args.json, payload)
    return 0 if ok else 3


async def _engine_token_ids(plan: ExecutionPlan, prompts, max_tokens: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    async with LocalTextEngine(plan) as engine:
        for name, text in prompts:
            session = engine.open_session()
            request_id, stream = await engine.submit(
                session, text, max_tokens=max_tokens, temperature=0.0, ignore_eos=True
            )
            async for _ in stream:
                pass
            record = engine.records[request_id]
            engine.close_session(session.session_id)
            out[name] = {
                "token_ids": list(record.output_token_ids),
                "text": record.text,
                "prompt_tokens": record.prompt_tokens,
            }
    return out


def cmd_parity(args: argparse.Namespace) -> int:
    """Greedy token-id parity: the Omni stage path against plain vLLM.

    Exit 0 when every prompt matches, 3 when any diverges (the run succeeded;
    the models disagreed), 2 when the plan refused.
    """
    # Both sides get the same environment, applied before either engine starts.
    # Not cosmetic: the reference is a plain vllm.LLM in this same process and
    # loads *first*, so anything the engine would have set at its own start is
    # set too late for it. Two of the three entries here are load-time fatal on
    # this host (see RUNTIME_ENV), and a comparison between a process that has
    # them and one that does not measures the configuration, not the path.
    env = apply_runtime_env()

    plan = _plan_from_args(args)
    if not plan.admitted:
        print(plan.summary(), file=sys.stderr)
        return 2
    print(plan.summary())
    prompts = acceptance_prompts()

    print(f"\nreference: plain vllm.LLM, {len(prompts)} prompts x {args.max_tokens} greedy tokens")
    reference = reference_token_ids(
        plan.manifest.model_dir, prompts,
        engine_kwargs=plan.engine_kwargs, max_tokens=args.max_tokens,
    )
    print("engine: AsyncOmni -> Omni StageRuntime -> EngineCore")
    engine = asyncio.run(_engine_token_ids(plan, prompts, args.max_tokens))

    rows: list[PromptParity] = []
    for name, _ in prompts:
        ref = reference[name]["token_ids"]
        eng = engine[name]["token_ids"]
        exact, first, prefix = compare_ids(ref, eng)
        lo = max(0, (first or 0) - 2)
        rows.append(PromptParity(
            name=name,
            prompt_tokens=reference[name]["prompt_tokens"],
            reference_tokens=len(ref),
            engine_tokens=len(eng),
            exact=exact,
            first_divergence=first,
            agreement_prefix=prefix,
            reference_head=[] if exact else ref[lo:lo + 8],
            engine_head=[] if exact else eng[lo:lo + 8],
        ))
        mark = "exact" if exact else f"diverges at {first} (of {len(ref)})"
        print(f"  {name:<12} ref={len(ref)} omni={len(eng)}  {mark}")

    stats = summarize(rows)
    print(f"\nparity: {stats['exact']}/{stats['prompts']} exact")
    if stats["min_agreement_prefix"] is not None:
        print(f"  shortest agreement prefix: {stats['min_agreement_prefix']} tokens")
    _write(args.json, {
        "summary": stats,
        "env": env,
        "max_tokens": args.max_tokens,
        "plan": plan.to_dict(),
        "prompts": [r.to_dict() for r in rows],
        "reference": reference,
        "engine": engine,
    })
    return 0 if stats["exact"] == stats["prompts"] else 3


def _synthetic_inputs(graph: str, spec: tuple[dict[str, Any], ...], fmt: str) -> dict[str, Any]:
    """Example inputs for the profiled placement run.

    Placement can only be read off a graph that has actually executed, so the
    probe needs something to feed it. Shapes come from the graph itself; a
    dynamic dimension is an error rather than a guess, because guessing one
    wrong produces a session that fails for a reason unrelated to placement.
    """
    import numpy as np

    if not spec and not fmt.startswith("onnx:"):
        raise SystemExit(
            f"a {fmt} artifact declares no input shapes, so they cannot be "
            "synthesized. Pass --inputs with an .npz holding the real tensors; "
            "placement is verified from an actual run, and a run needs real inputs."
        )
    if not spec:
        import onnx

        model = onnx.load(graph, load_external_data=False)
        spec = tuple(
            {
                "name": i.name,
                "shape": [d.dim_value if d.HasField("dim_value") else None
                          for d in i.type.tensor_type.shape.dim],
                "elem_type": i.type.tensor_type.elem_type,
            }
            for i in model.graph.input
        )

    dtypes = {1: np.float32, 6: np.int32, 7: np.int64, 10: np.float16, 11: np.float64}
    inputs: dict[str, Any] = {}
    for item in spec:
        shape = list(item.get("shape") or [])
        if any(d is None or (isinstance(d, str)) or d == 0 for d in shape):
            raise SystemExit(
                f"input {item['name']!r} has a dynamic shape {shape}; pass --inputs "
                "with an .npz holding real tensors instead of asking for synthetic ones"
            )
        dtype = dtypes.get(int(item.get("elem_type", 1)), np.float32)
        if np.issubdtype(dtype, np.integer):
            inputs[item["name"]] = np.zeros(shape, dtype=dtype)
        else:
            inputs[item["name"]] = np.random.RandomState(0).randn(*shape).astype(dtype)
    return inputs


def cmd_external(args: argparse.Namespace) -> int:
    """Plan an exported graph onto the iGPU or the NPU, and prove the placement."""
    if args.worker_peak_rss_hint_bytes is not None and not args.device:
        print("--worker-peak-rss-hint-bytes requires --device: the peak is device-specific")
        return 2
    import numpy as np

    from vllm_omni.edge.local.external.stage import (
        ExternalStage,
        PlacementRefused,
        plan_external_stage,
    )
    from vllm_omni.edge.local.manifest import build_graph_artifact

    profile = load_profile()
    devices = enumerate_devices(profile, mask=parse_mask(args.mask))
    artifact = build_graph_artifact(
        args.graph,
        fmt=args.format,
        opset=args.opset,
        source_model=args.source_model or "(unrecorded)",
        component=args.component,
        exporter=args.exporter or "(unrecorded)",
    )
    plan = plan_external_stage(
        artifact, devices,
        prefer=args.prefer, require=args.device,
        min_fraction_on_target=args.min_placement,
        worker_peak_rss_hint_bytes=args.worker_peak_rss_hint_bytes,
    )
    print(plan.summary())
    record: dict[str, Any] = {
        "argv": sys.argv, "runtime": runtime_versions().to_dict(),
        "devices": [d.to_dict() for d in devices], "plan": plan.to_dict(),
    }
    if not plan.admitted:
        _write(args.json, record)
        return 2

    if args.inputs:
        loaded = np.load(args.inputs)
        example = {k: loaded[k] for k in loaded.files}
    else:
        example = _synthetic_inputs(artifact.path, artifact.input_spec, artifact.fmt)

    stage = ExternalStage(plan)
    try:
        report = stage.open(example, profile_dir=args.profile_dir)
        print()
        print(plan.summary())
        timings = []
        for _ in range(args.runs):
            _, timing = stage.run(example)
            timings.append(timing)
        if timings:
            worker = sorted(t.worker_s for t in timings)
            trip = sorted(t.round_trip_s for t in timings)
            transport = sorted(t.transport_s for t in timings)
            mid = len(timings) // 2
            print(
                f"\n  {args.runs} runs: device {worker[mid] * 1e3:.3f} ms median, "
                f"round trip {trip[mid] * 1e3:.3f} ms, "
                f"transport {transport[mid] * 1e3:.3f} ms"
            )
            print(
                "  transport is the cost of leaving the process; a stage only earns "
                "its place here when the gain beats it"
            )
            record["timings"] = [t.to_dict() for t in timings]
        record["stats"] = stage.stats()
        record["report"] = report.to_dict()
        record["plan"] = plan.to_dict()
    except PlacementRefused as exc:
        print(f"\nREFUSED [{exc.refusal.code}] {exc.refusal.message}")
        print(f"  remedy: {exc.refusal.remedy}")
        record["plan"] = plan.to_dict()
        _write(args.json, record)
        return 2
    finally:
        stage.close()

    _write(args.json, record)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vllm_omni.edge.local",
        description="M0 local text mode: plan, run and audit a single-stage text session.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_plan_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, help="checkpoint directory")
        sp.add_argument("--max-model-len", type=int, default=4096)
        sp.add_argument("--max-num-seqs", type=int, default=1)
        sp.add_argument("--max-num-batched-tokens", type=int, default=None)
        sp.add_argument("--cuda-graphs", action="store_true",
                        help="enable graph capture (M0 runs eager; this also adds a 2 GiB reserve)")
        sp.add_argument("--mask", default=None,
                        help="comma-separated device kinds to hide, e.g. gpu_discrete")
        sp.add_argument("--json", default=None, help="write the record here")

    sp = sub.add_parser("devices", help="what this process can actually run on")
    sp.add_argument("--mask", default=None)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_devices)

    sp = sub.add_parser("plan", help="plan a session; exit 2 means an explicit refusal")
    add_plan_args(sp)
    sp.add_argument("--digest-weights", action="store_true",
                    help="sha256 the weight files (slow; use when publishing a number)")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="stream one prompt")
    add_plan_args(sp)
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--max-tokens", type=int, default=128)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser(
        "parity",
        help="greedy token-id parity of the Omni path against plain vllm.LLM; exit 3 on divergence",
    )
    add_plan_args(sp)
    sp.add_argument("--max-tokens", type=int, default=128)
    sp.set_defaults(func=cmd_parity)

    sp = sub.add_parser(
        "external",
        help="place an exported graph on the iGPU or the NPU; exit 2 is an explicit refusal",
    )
    sp.add_argument("--graph", required=True, help="the exported .onnx")
    sp.add_argument("--format", default="onnx:a16w8",
                    choices=["onnx:a16w8", "onnx:fp16", "onnx:fp32", "pt:fp16", "pt:fp32"])
    sp.add_argument("--opset", type=int, default=21)
    sp.add_argument("--component", default="graph", help="which part of the model this is")
    sp.add_argument("--source-model", default=None)
    sp.add_argument("--exporter", default=None)
    sp.add_argument("--device", default=None,
                    help="require this device id; without it a refusal falls through to the other")
    sp.add_argument("--prefer", default=None, help="try this device id first, then fall through")
    sp.add_argument("--min-placement", type=float, default=0.5,
                    help="fraction of the graph the target EP must take for the placement to count")
    sp.add_argument("--worker-peak-rss-hint-bytes", type=int, default=None,
                    help="measured load peak for --device on this EP/driver; reserves it before launch")
    sp.add_argument("--inputs", default=None, help=".npz of real inputs; otherwise synthesized")
    sp.add_argument("--runs", type=int, default=10)
    sp.add_argument("--profile-dir", default=None, help="where ORT writes its placement profile")
    sp.add_argument("--mask", default=None)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_external)

    sp = sub.add_parser("accept", help="the M0 acceptance run")
    add_plan_args(sp)
    sp.add_argument("--max-tokens", type=int, default=128,
                    help="tokens per prompt; the criterion is at least 128")
    sp.set_defaults(func=cmd_accept)

    return p


def _use_selector_event_loop_on_windows() -> None:
    """zmq.asyncio needs a selector loop; Windows defaults to Proactor.

    [edge-infer W1] Python 3.8+ on Windows defaults to
    ``WindowsProactorEventLoopPolicy``, which does not implement the
    ``add_reader`` family. ``zmq.asyncio`` -- which the Omni orchestrator and
    every stage client use -- requires it, and the failure is a hard
    ``RuntimeError`` at the first async socket, not a degradation.

    This is a no-op everywhere else, and it is set here rather than at import
    so that importing the CLI does not reach into a host application's loop
    policy. A synchronous ``vllm.LLM`` run never hits this, which is why it
    surfaces only once the async engine is involved.
    """
    if sys.platform != "win32":
        return
    policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy is None:  # pragma: no cover - non-Windows
        return
    if not isinstance(asyncio.get_event_loop_policy(), policy):
        asyncio.set_event_loop_policy(policy())


def main(argv: list[str] | None = None) -> int:
    _use_selector_event_loop_on_windows()
    args = build_parser().parse_args(argv)
    return int(args.func(args))
