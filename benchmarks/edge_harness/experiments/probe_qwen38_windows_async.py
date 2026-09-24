"""Exercise public AsyncOmni text and image requests with the pinned FP8 model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--abort-check", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    report = {"model": args.model, "entrypoint": "AsyncOmni.generate", "started": time.time()}
    engine = None
    try:
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import SamplingParams

        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.windows.aio import install_selector_policy

        install_selector_policy()
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        image = Image.new("RGB", (96, 96), (255, 0, 0))
        sampling = SamplingParams(temperature=0, max_tokens=16)
        config = {
            "model": args.model,
            "language_model_only": False,
            "max_model_len": 512,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 512,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.80,
            "cpu_offload_gb": 12,
            "linear_backend": "triton",
            "dtype": "bfloat16",
            "stage_init_timeout": 180,
            "init_timeout": 240,
        }
        report["config"] = config
        start = time.perf_counter()
        engine = AsyncOmni(**config)
        report["startup_s"] = time.perf_counter() - start
        cases = (
            ("text", "Reply with the single word ready.", None, "ready"),
            ("image", "What is the dominant color in the image? Answer in one word.", image, "Red"),
        )
        report["cases"] = []
        for kind, question, visual, expected in cases:
            content = question if visual is None else [
                {"type": "image", "image": visual}, {"type": "text", "text": question}
            ]
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            request = {"prompt": prompt}
            if visual is not None:
                request["multi_modal_data"] = {"image": visual}
            outputs = []
            start = time.perf_counter()
            async for output in engine.generate(request, sampling, request_id=f"public-qwen38-{kind}"):
                outputs.append({"request_id": output.request_id, "error": output.error,
                                "text": output.outputs[0].text if output.outputs else None,
                                "finished": output.finished, "stage_id": output.stage_id})
            row = {"kind": kind, "wall_s": time.perf_counter() - start,
                   "outputs": outputs, "expected": expected}
            report["cases"].append(row)
            if not outputs or any(item["error"] for item in outputs) or outputs[-1]["text"].strip() != expected:
                raise RuntimeError(f"public {kind} request did not complete with {expected!r}")
        if args.abort_check:
            cancel_id = "public-qwen38-cancel"
            long_prompt = processor.apply_chat_template(
                [{"role": "user", "content": "Write the integers from 1 through 200, one per line, with no explanation."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            emitted = []

            async def collect_long() -> None:
                async for output in engine.generate(
                    {"prompt": long_prompt}, SamplingParams(temperature=0, max_tokens=256), request_id=cancel_id
                ):
                    emitted.append({"at_monotonic": time.perf_counter(),
                                    "text": output.outputs[0].text if output.outputs else None,
                                    "finished": output.finished,
                                    "finish_reason": output.outputs[0].finish_reason if output.outputs else None})

            task = asyncio.create_task(collect_long())
            deadline = time.monotonic() + 20
            while not any(state.external_request_id == cancel_id for state in engine.request_states.values()):
                if task.done():
                    raise RuntimeError("long request completed before cancellation check")
                if time.monotonic() >= deadline:
                    raise TimeoutError("long request was not admitted")
                await asyncio.sleep(0.02)
            await asyncio.sleep(2)
            if task.done():
                raise RuntimeError("long request completed during cancellation wait")
            before_abort = len(emitted)
            abort_started = time.perf_counter()
            await engine.abort(cancel_id)
            abort_completed = time.perf_counter()
            error = None
            try:
                await asyncio.wait_for(task, timeout=30)
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(0.3)
            report["abort_check"] = {
                "request_id": cancel_id, "submitted": True, "task_active_before_abort": True,
                "emitted_before_abort": before_abort, "emissions": emitted,
                "abort_started_monotonic": abort_started,
                "abort_completed_monotonic": abort_completed,
                "task_error": error, "remaining_request_states": len(engine.request_states),
            }
            after_abort = [row for row in emitted if row["at_monotonic"] > abort_completed]
            if len(after_abort) != 1 or after_abort[0]["finish_reason"] != "abort" or not after_abort[0]["finished"]:
                raise RuntimeError("abort did not produce exactly one terminal abort marker")
            fresh_outputs = []
            async for output in engine.generate(
                {"prompt": processor.apply_chat_template(
                    [{"role": "user", "content": "Reply with the single word ready."}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )}, sampling, request_id="public-qwen38-after-abort"
            ):
                fresh_outputs.append(output.outputs[0].text if output.outputs else None)
            report["abort_check"]["fresh_outputs"] = fresh_outputs
            if not fresh_outputs or fresh_outputs[-1] != "ready":
                raise RuntimeError("fresh request after abort failed")
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.shutdown()
        report["ended"] = time.time()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
