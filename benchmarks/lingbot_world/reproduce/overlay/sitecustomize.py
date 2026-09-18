# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-chunk wall-clock instrumentation for the LingBot-World served path.

Installed by putting this directory first on PYTHONPATH. It patches nothing at
import time; a ``builtins.__import__`` wrapper patches each target the first
time its module is loaded, so it survives the meta_path finders other overlays
install at index 0.

Every patched span appends one JSONL record to
``$LINGBOT_TRACE_DIR/spans_<pid>.jsonl`` with the span name, its wall duration,
and the wall clock at entry and exit, so spans from the four worker processes
and the API-server process can be laid on one timeline afterwards and the
question "is the muxing overlapped with the next chunk's compute?" is answered
by looking at the timeline rather than by reasoning about the code.
"""

from __future__ import annotations

import builtins
import functools
import json
import os
import sys
import threading
import time

_TRACE_DIR = os.environ.get("LINGBOT_TRACE_DIR")
if _TRACE_DIR:
    os.makedirs(_TRACE_DIR, exist_ok=True)
    _PATH = os.path.join(_TRACE_DIR, f"spans_{os.getpid()}.jsonl")
    _LOCK = threading.Lock()
    _FH = open(_PATH, "a", buffering=1)

    def _emit(name: str, t0: float, t1: float, **extra) -> None:
        rec = {"pid": os.getpid(), "name": name, "t0": t0, "t1": t1, "ms": (t1 - t0) * 1000.0}
        if extra:
            rec.update(extra)
        with _LOCK:
            _FH.write(json.dumps(rec) + "\n")

    def _wrap(obj, attr: str, name: str) -> None:
        original = getattr(obj, attr, None)
        if original is None or getattr(original, "_lingbot_traced", False):
            return

        @functools.wraps(original)
        def traced(*args, **kwargs):
            t0 = time.time()
            try:
                return original(*args, **kwargs)
            finally:
                _emit(name, t0, time.time())

        traced._lingbot_traced = True
        setattr(obj, attr, traced)

    def _patch_pipeline(module) -> bool:
        cls = getattr(module, "LingBotWorldCausalDMDPipeline", None)
        if cls is None:
            return False
        # Candidates first: a candidate replaces a method outright, so wrapping
        # has to happen afterwards or the span would time the discarded one.
        if _candidates is not None:
            try:
                _candidates.apply(module)
            except Exception as exc:
                print(f"[lingbot-cand] failed: {exc}", flush=True)
        for attr, name in (
            ("prepare_encode", "pipe.prepare_encode"),
            ("denoise_step", "pipe.denoise_step"),
            ("post_decode", "pipe.post_decode"),
            ("_decode_chunk_to_pixels", "pipe.decode_to_pixels"),
            ("_streaming_decode_chunk", "pipe.vae_streaming_decode"),
            ("_prepare_condition_chunk", "pipe.vae_condition_encode"),
            ("_generate_block", "pipe.dit_generate_block"),
            ("_prepare_next_chunk", "pipe.prepare_next_chunk"),
        ):
            _wrap(cls, attr, name)
        return True

    def _patch_post_process(module) -> bool:
        # The PIL conversion lives in a closure returned by the factory, so the
        # factory is wrapped and its product re-wrapped on the way out.
        factory = getattr(module, "get_lingbot_world_post_process_func", None)
        if factory is None:
            return False
        if getattr(factory, "_lingbot_traced", False):
            return True

        @functools.wraps(factory)
        def traced_factory(*args, **kwargs):
            fn = factory(*args, **kwargs)

            @functools.wraps(fn)
            def traced_fn(*a, **k):
                t0 = time.time()
                try:
                    return fn(*a, **k)
                finally:
                    _emit("pipe.postprocess_video", t0, time.time())

            return traced_fn

        traced_factory._lingbot_traced = True
        module.get_lingbot_world_post_process_func = traced_factory
        return True

    def _patch_video_api(module) -> bool:
        cls = getattr(module, "FragmentedMP4VideoEncoder", None)
        if cls is None:
            return False
        _wrap(cls, "encode", "api.mux_encode")
        _wrap(module, "_coerce_video_to_uint8_frames", "api.coerce_frames")
        return True

    def _patch_serving(module) -> bool:
        cls = None
        for candidate in vars(module).values():
            if isinstance(candidate, type) and hasattr(candidate, "handle_session"):
                cls = candidate
                break
        if cls is None:
            return False
        _wrap(cls, "handle_session", "api.session")
        return True

    def _patch_platform(module) -> bool:
        plat = getattr(module, "current_omni_platform", None)
        if plat is None:
            return False
        _wrap(plat, "synchronize", "platform.synchronize")
        return True

    def _patch_runner(module) -> bool:
        cls = getattr(module, "ARDiffusionModelRunner", None)
        if cls is None:
            return False
        _wrap(cls, "execute_stepwise", "runner.execute_stepwise")
        return True

    def _patch_model_runner(module) -> bool:
        # The pipeline profiler (enable_diffusion_pipeline_profiler) attaches
        # its per-stage durations to each chunk's DiffusionOutput here; mirror
        # them into the span file so the profiled arm's split is on disk.
        original = getattr(module, "attach_stage_durations", None)
        if original is None:
            return False
        if getattr(original, "_lingbot_traced", False):
            return _patch_model_runner_class(module)

        @functools.wraps(original)
        def traced(state, output):
            durations = getattr(state, "stage_durations", None) or {}
            now = time.time()
            for stage, seconds in durations.items():
                _emit("profiler." + stage, now - float(seconds), now, chunk=getattr(state, "chunk_index", None))
            return original(state, output)

        traced._lingbot_traced = True
        module.attach_stage_durations = traced
        return _patch_model_runner_class(module)

    def _patch_model_runner_class(module) -> bool:
        # The module is in sys.modules while it is still executing, so the
        # first patch pass can run before the class exists; report "not yet"
        # so the hook retries on a later import.
        # Residual host work between post_decode and the runner's return.
        cls = getattr(module, "DiffusionModelRunner", None)
        if cls is None:
            return False
        if cls is not None and os.environ.get("LINGBOT_SYNC_DEBUG") and not getattr(cls, "_lingbot_sync_debug", False):
            # After the model is loaded in a worker, make every device
            # synchronisation warn, and print where it came from (the frames
            # inside vllm_omni only), capped so the log stays readable.
            cls._lingbot_sync_debug = True
            original_load = cls.load_model

            @functools.wraps(original_load)
            def load_model_then_debug(self, *args, **kwargs):
                out = original_load(self, *args, **kwargs)
                import traceback
                import warnings

                import torch

                budget = [int(os.environ.get("LINGBOT_SYNC_DEBUG_MAX", "400"))]
                previous = warnings.showwarning

                def show(message, category, filename, lineno, file=None, line=None):
                    text = str(message)
                    if "synchroniz" in text.lower() and budget[0] > 0:
                        budget[0] -= 1
                        frames = [
                            f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
                            for f in traceback.extract_stack()[:-1]
                            if "vllm_omni" in f.filename
                        ]
                        sys.stderr.write(f"[sync-debug pid={os.getpid()}] {text} <- {' > '.join(frames[-6:])}\n")
                        return
                    previous(message, category, filename, lineno, file, line)

                warnings.showwarning = show
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
                sys.stderr.write(f"[sync-debug pid={os.getpid()}] armed\n")
                return out

            cls.load_model = load_model_then_debug
        if cls is not None:
            _wrap(cls, "_prepare_output_for_transport", "runner.prepare_output_for_transport")
            _wrap(cls, "_attach_stepwise_metadata", "runner.attach_stepwise_metadata")
            _wrap(cls, "_update_states_after", "runner.update_states_after")
            _wrap(cls, "_update_states", "runner.update_states")
            _wrap(cls, "_prepare_batch_inputs", "runner.prepare_batch_inputs")
        return True

    _TARGETS = {
        "vllm_omni.diffusion.worker.diffusion_model_runner": (_patch_model_runner,),
        "vllm_omni.diffusion.models.lingbot_world.pipeline": (_patch_pipeline, _patch_post_process),
        "vllm_omni.entrypoints.openai.video_api_utils": (_patch_video_api,),
        "vllm_omni.entrypoints.openai.serving_video_output_stream": (_patch_serving,),
        "vllm_omni.experimental.ar_diffusion.runner": (_patch_runner, _patch_platform),
    }

    try:
        import lingbot_candidates as _candidates
    except Exception as exc:
        _candidates = None
        print(f"[lingbot-cand] module unavailable: {exc}", flush=True)

    _real_import = builtins.__import__
    _done: set[str] = set()
    _patching = [False]

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _real_import(name, globals, locals, fromlist, level)
        if _TARGETS and not _patching[0]:
            _patching[0] = True
            try:
                _run_patchers()
            finally:
                _patching[0] = False
        return module

    def _run_patchers() -> None:
        if True:
            for target, patchers in list(_TARGETS.items()):
                if target in _done:
                    continue
                loaded = sys.modules.get(target)
                if loaded is None:
                    continue
                ok = True
                for patcher in patchers:
                    try:
                        ok = patcher(loaded) and ok
                    except Exception as exc:  # never break the server for a trace
                        print(f"[lingbot-trace] failed to patch {target}: {exc}", flush=True)
                        ok = False
                if ok:
                    _done.add(target)
                    print(f"[lingbot-trace] patched {target}", flush=True)

    builtins.__import__ = _import
    print(f"[lingbot-trace] installed, writing {_PATH}", flush=True)
