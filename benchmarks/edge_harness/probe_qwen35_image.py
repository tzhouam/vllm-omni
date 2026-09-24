"""One real-checkpoint Omni image-to-text probe with a synthetic red image."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
    record = {"model": args.model, "scope": "one synthetic-image Omni image-to-text request", "start": time.time()}
    engine = None
    try:
        from PIL import Image
        from transformers import AutoProcessor
        from vllm import SamplingParams

        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.windows.aio import install_selector_policy

        install_selector_policy()
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        image = Image.new("RGB", (96, 96), (255, 0, 0))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is the dominant color in the image? Answer in one word."},
        ]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        config = {
            "model": args.model,
            "language_model_only": False,
            "max_model_len": 512,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 512,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.85,
            "cpu_offload_gb": 2,
            "dtype": "bfloat16",
            "stage_init_timeout": 180,
        }
        record["config"] = config
        engine = Omni(**config)
        outputs = engine.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            SamplingParams(temperature=0, max_tokens=16),
        )
        record["outputs"] = [out.outputs[0].text for out in outputs if out.outputs]
        if not record["outputs"]:
            raise RuntimeError("No image-conditioned output")
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.close()
        record["end"] = time.time()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
