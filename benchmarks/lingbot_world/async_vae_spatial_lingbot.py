# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cases D/E: sequential/overlapping DiT with four-way spatial Wan VAE."""

import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch
from async_vae_spatial_worker import worker
from common import _CAMERA_ACTION_SCHEMA, _load_events, parse_args


async def main():
    args = parse_args()
    epochs = args.epochs
    os.environ["CAMPAIGN_PORT_BASE"] = str(args.port)
    mode = args.case
    allocated = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    degree = args.ulysses_degree
    assert degree in (2, 4) and args.tensor_parallel_size == 1
    assert mode in ("D", "E") and len(allocated) == 4
    decoder_device = ",".join(allocated)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(allocated[:degree])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    events = _load_events(Path(args.events))
    if len(events) != 10:
        raise ValueError("Exactly 10 supported ticks per epoch required")
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.lingbot_world.actions import LingBotCameraControlReducer
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import (
        ARDiffusionSessionEvent,
        ARDiffusionSessionManager,
        ARDiffusionWorkerLifecycle,
    )
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    profile_enabled = os.environ.get("LINGBOT_BENCH_PROFILE") == "1"
    engine_kwargs = {}
    if os.environ.get("CAMPAIGN_PORT_BASE"):
        port = int(os.environ["CAMPAIGN_PORT_BASE"])
        engine_kwargs.update(
            master_port=port, scheduler_port=port + 2, omni_master_address="127.0.0.1", omni_master_port=port + 3
        )
    if profile_enabled:
        engine_kwargs["profiler_config"] = dict(
            profiler="torch",
            torch_profiler_dir=str(output_dir / "traces"),
            torch_profiler_record_shapes=True,
            torch_profiler_with_stack=False,
            torch_profiler_with_memory=False,
            torch_profiler_use_gzip=True,
        )

    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    decoder = ctx.Process(target=worker, args=(child, decoder_device, args.model))
    decoder.start()

    def receive():
        if not conn.poll(1800):
            raise TimeoutError("Decoder response timeout")
        result = conn.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    engine = None
    try:
        hardware = await asyncio.to_thread(receive)
        started = time.perf_counter()
        engine = AsyncOmni(
            model=args.model,
            engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
            enforce_eager=args.enforce_eager,
            init_timeout=1800,
            stage_init_timeout=1800,
            parallel_config=DiffusionParallelConfig(
                tensor_parallel_size=args.tensor_parallel_size, ulysses_degree=args.ulysses_degree
            ),
            max_num_seqs=1,
            model_config=dict(
                ar_diffusion_height=args.height,
                ar_diffusion_width=args.width,
                ar_diffusion_kv_config=dict(gpu_memory_fraction=args.gpu_memory_fraction, warmup_cudagraph=True),
            ),
            **engine_kwargs,
        )
        init_seconds = time.perf_counter() - started
        sampling = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=9,
            num_inference_steps=4,
            max_sequence_length=512,
            seed=args.seed,
            output_type="latent",
            extra_args={"flow_shift": 5.0},
        )
        consumer = ARDiffusionOmniTickConsumer(
            engine,
            prompt_provider=lambda tick: {
                "prompt": tick.prompt,
                "multi_modal_data": {"image": str(Path(args.image).resolve())},
            },
            sampling_params_list=[sampling],
            diffusion_stage_id=0,
        )
        manager = ARDiffusionSessionManager(
            tick_consumer=consumer,
            lifecycle=ARDiffusionWorkerLifecycle(engine, stage_ids=[0], timeout=1800),
            max_pending_events=32,
            control_reducer_factory=LingBotCameraControlReducer,
        )
        rows = []
        sessions = []
        decodes = []
        validation = None
        for epoch in range(epochs):
            session_id = f"{args.session_id}-epoch{epoch:03d}"
            t = time.perf_counter()
            session = await manager.create_session(session_id)
            session_info = dict(epoch=epoch, create_seconds=time.perf_counter() - t)
            conn.send(dict(op="reset"))
            await asyncio.to_thread(receive)
            previous = None
            epoch_start = time.perf_counter()
            epoch_decodes = []
            epoch_latents = []

            def submit(latent, index):
                sent = time.perf_counter()
                conn.send(
                    dict(
                        op="decode",
                        latent=latent.numpy(),
                        index=index,
                        retain=epoch == 1,
                        save=epoch == 0 and index < 4,
                        save_path=str(output_dir / f"epoch_{epoch:03d}_pixels_{index:02d}.pt"),
                        profile_path=str(output_dir / "vae_trace.json")
                        if profile_enabled and epoch == 1 and index == 5
                        else None,
                    )
                )
                return sent

            async def collect_decode(sent):
                result = await asyncio.to_thread(receive)
                result.update(epoch=epoch, sent=sent, received=time.perf_counter())
                assert result["finite"] and result["frames"] == (9 if result["chunk_index"] == 0 else 12)
                epoch_decodes.append(result)
                decodes.append(result)
                with (output_dir / "decode.jsonl").open("a") as f:
                    f.write(json.dumps(result) + "\n")

            try:
                for i, event in enumerate(events):
                    controls = (
                        ()
                        if event["frames"] is None
                        else (
                            ARDiffusionControlInput(
                                track="camera",
                                schema=_CAMERA_ACTION_SCHEMA,
                                data={"mode": "script", "frames": event["frames"]},
                            ),
                        )
                    )
                    await session.accept_event(
                        ARDiffusionSessionEvent(
                            event_id=event["event_id"],
                            prompt=event["prompt"]
                            if event["prompt"] is not None
                            else (args.prompt if i == 0 else None),
                            controls=controls,
                        )
                    )
                    profiled = profile_enabled and epoch == 1 and i == 6
                    if profiled:
                        await engine.start_profile(profile_prefix=args.label, stages=[0])
                    pending = None
                    if mode == "E" and previous is not None:
                        sent = submit(previous, i - 1)
                        pending = asyncio.create_task(collect_decode(sent))
                    t = time.perf_counter()
                    output = await session.next_chunk()
                    elapsed = time.perf_counter() - t
                    if profiled:
                        await engine.stop_profile(stages=[0])
                    latent = output.images[0].detach().float().cpu()
                    row = dict(
                        epoch=epoch,
                        chunk_index=i,
                        latency_seconds=elapsed,
                        dit_start=t,
                        dit_end=t + elapsed,
                        latent_ready=time.perf_counter(),
                        latent_sha256=hashlib.sha256(latent.numpy().tobytes()).hexdigest(),
                        finite=bool(torch.isfinite(latent).all()),
                        shape=list(latent.shape),
                        steady=i >= 5,
                        profiled=profiled,
                    )
                    rows.append(row)
                    with (output_dir / "blocks.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    if not row["finite"]:
                        raise RuntimeError("Non-finite LingBot latents")
                    if pending is not None:
                        await pending
                    if mode == "D":
                        await collect_decode(submit(latent, i))
                    previous = latent
                    if epoch == 1:
                        epoch_latents.append(latent)
                    torch.save(latent, output_dir / f"epoch_{epoch:03d}_latent_{i:02d}.pt")
                if mode == "E":
                    await collect_decode(submit(previous, len(events) - 1))
                session_info.update(
                    start=epoch_start,
                    end=time.perf_counter(),
                    frames=sum(d["frames"] for d in epoch_decodes),
                    ttfc=epoch_decodes[0]["ready"] - epoch_start,
                )
                if epoch == 1:
                    for index, z in enumerate(epoch_latents):
                        torch.save(z, output_dir / f"steady_latent_{index:02d}.pt")
                    conn.send(dict(op="save_retained", folder=str(output_dir)))
                    await asyncio.to_thread(receive)
                if epoch == 0:
                    conn.send(dict(op="validate", output=str(output_dir)))
                    validation = await asyncio.to_thread(receive)
                    (output_dir / "decode_validation.json").write_text(json.dumps(validation, indent=2))
                    assert validation["passed"], validation
            finally:
                t = time.perf_counter()
                await manager.close_session(session_id)
                session_info["close_seconds"] = time.perf_counter() - t
                sessions.append(session_info)
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            if decoder.is_alive():
                try:
                    conn.send(dict(op="stop"))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                decoder.join(timeout=30)
            if decoder.is_alive():
                decoder.terminate()
                decoder.join(timeout=30)
            conn.close()
            child.close()
    (output_dir / "summary.json").write_text(
        json.dumps(
            dict(
                case=mode,
                usp=degree,
                gpu_count=4,
                vae_parallel_size=4,
                vae_parallel_mode="spatial_shard_width",
                hardware=hardware,
                decode_validation=validation,
                decodes=decodes,
                height=args.height,
                width=args.width,
                epochs=epochs,
                ticks_per_epoch=10,
                steady_start=5,
                sampling_scope=(
                    "four-rank width-sharded VAE in separate processes; paired scheduling mock; "
                    "CPU-mediated latent transfer; CPU-ready uint8 frames; "
                    "epoch0 warmup excluded; no media encoding or network"
                ),
                engine_init_seconds=init_seconds,
                sessions=sessions,
                chunks=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
