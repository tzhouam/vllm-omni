# SPDX-License-Identifier: Apache-2.0
"""Real-weight TTS slow consumer, abort/recovery and owned vocoder-process failure."""

import argparse
import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

from profile_local_text import save


async def run(args):
    import psutil
    import torch
    from transformers import AutoTokenizer

    from vllm_omni import AsyncOmni
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tts"))
    from stream_latency_bench import _cat_audio

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "argv": sys.argv,
        "start_unix": time.time(),
        "scope": (
            "Short real audio delivery, slow consumer, abort fence/recovery, owned vocoder termination. "
            "Not audio quality or 64 MiB buffer saturation."
        ),
    }
    omni, owned = None, []
    save(args.out / "report.json", report)
    try:
        cfg = Qwen3TTSConfig.from_pretrained(args.model)
        tok = AutoTokenizer.from_pretrained(args.model)

        def prompt(text):
            info = {"task_type": ["CustomVoice"], "text": [text], "language": ["English"], "speaker": ["Vivian"]}
            length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=info,
                task_type="CustomVoice",
                tokenize_prompt=lambda x: tok(x, padding=False)["input_ids"],
                codec_language_id=cfg.talker_config.codec_language_id,
                spk_is_dialect=cfg.talker_config.spk_is_dialect,
            )
            return {"prompt_token_ids": [0] * length, "additional_information": info}

        short = prompt("The garden is quiet. We can hear the birds singing.")
        long = prompt("This longer sentence keeps the speech model generating while we test interruption. " * 12)
        omni = AsyncOmni(model=args.model, deploy_profile="edge", stage_init_timeout=180)
        params = omni.resolve_sampling_params_list(None, allow_delta_coercion=True)

        def event(output, rid):
            assert output.request_id == rid, "Unexpected request ID in audio stream"
            mm = output.outputs[0].multimodal_output if output.outputs else None
            audio = _cat_audio(mm.get("audio") if mm else None)
            return {
                "request_id": output.request_id,
                "finished": output.finished,
                "samples": int(audio.numel()) if audio is not None else 0,
                "finite": bool(torch.isfinite(audio).all()) if audio is not None else True,
            }

        async def first_audio(stream, rid):
            async for output in stream:
                row = event(output, rid)
                if row["samples"]:
                    assert row["finite"], "Nonfinite audio before fault probe"
                    return row
            raise RuntimeError("No audio before fault probe")

        async def collect(stream, rid, pause=False):
            rows = []
            async for output in stream:
                rows.append(event(output, rid))
                if pause and len(rows) == 1:
                    await asyncio.sleep(3)
            return rows

        for rid, slow in [("baseline", False), ("slow-consumer", True)]:
            rows = await collect(omni.generate(short, request_id=rid, sampling_params_list=params), rid, slow)
            report[rid] = rows
            assert any(r["finished"] for r in rows) and any(r["samples"] for r in rows)
            assert all(r["finite"] for r in rows)
        rid = "abort-probe"
        stream = omni.generate(long, request_id=rid, sampling_params_list=params)
        first = await asyncio.wait_for(first_audio(stream, rid), 60)
        start = time.perf_counter()
        await omni.abort(rid)
        report["abort"] = {"first": first, "ack_s": time.perf_counter() - start}
        try:
            late = await asyncio.wait_for(collect(stream, rid), 10)
            report["abort"]["late_events"] = late
            report["abort"]["late_audio_samples"] = sum(r["samples"] for r in late)
        except asyncio.CancelledError:
            report["abort"]["terminal"] = "CancelledError"
        except Exception as error:
            report["abort"]["terminal_error"] = repr(error)
        rid = "after-abort"
        rows = await collect(omni.generate(short, request_id=rid, sampling_params_list=params), rid)
        report[rid] = rows
        assert any(r["finished"] for r in rows) and any(r["samples"] for r in rows)
        save(args.out / "report.json", report)
        descendants = {p.pid: p for p in psutil.Process().children(recursive=True)}
        core_handles = [
            p for client in omni.engine.stage_clients for p in client.resources.engine_manager.processes if p.is_alive()
        ]
        assert core_handles and all(p.pid in descendants for p in core_handles)
        manager_tree = {}
        for handle in core_handles:
            core = descendants[handle.pid]
            manager_tree[core.pid] = core
            manager_tree.update({p.pid: p for p in core.children(recursive=True)})
        owned = [(p.pid, p.create_time()) for p in manager_tree.values()]
        target = omni.engine.stage_clients[-1].resources.engine_manager.processes[0]
        rid = "vocoder-crash"
        stream = omni.generate(long, request_id=rid, sampling_params_list=params)
        report["crash_first"] = await asyncio.wait_for(first_audio(stream, rid), 60)
        report["crash"] = {"pid": target.pid, "name": target.name, "created": descendants[target.pid].create_time()}
        start = time.perf_counter()
        target.terminate()
        try:
            report["crash"]["events"] = await asyncio.wait_for(collect(stream, rid), 30)
            report["crash"]["explicit_error_reported"] = False
        except asyncio.TimeoutError:
            report["crash"]["timeout"] = True
        except (Exception, asyncio.CancelledError) as error:
            report["crash"].update(explicit_error_reported=True, error=repr(error))
        report["crash"]["observation_s"] = time.perf_counter() - start
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if omni is not None:
            try:
                omni.shutdown()
            except (Exception, asyncio.CancelledError) as error:
                report.update(status="failed", cleanup_error=repr(error))
                report.setdefault("error", f"Engine shutdown failed after probe: {error!r}")
        await asyncio.sleep(2)
        leftovers = []
        for pid, created in owned:
            try:
                p = psutil.Process(pid)
                if p.create_time() == created and p.is_running():
                    leftovers.append({"pid": pid, "created": created, "status": p.status()})
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        report["owned_processes_requiring_harness_cleanup"] = leftovers
        report["end_unix"] = time.time()
        save(args.out / "report.json", report)
    return report["status"] == "completed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm_omni.windows.aio import install_selector_policy

    install_selector_policy()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
