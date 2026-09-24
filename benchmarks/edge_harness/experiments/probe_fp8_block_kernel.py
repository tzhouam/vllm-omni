"""Check the installed Windows vLLM block-FP8 kernel before loading 27B weights."""

import json
import sys
import traceback
from pathlib import Path

import torch


def main() -> None:
    out = Path(sys.argv[1])
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    }
    try:
        import vllm.model_executor.kernels.linear.scaled_mm.triton  # noqa: F401

        a = torch.ones((4, 128), device="cuda", dtype=torch.float8_e4m3fn)
        b = torch.ones((128, 128), device="cuda", dtype=torch.float8_e4m3fn)
        sa = torch.ones((4, 1), device="cuda", dtype=torch.float32)
        sb = torch.ones((1, 1), device="cuda", dtype=torch.float32)
        y = torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
            a, b, sa, sb, [128, 128], torch.bfloat16
        )
        torch.cuda.synchronize()
        result.update(status="pass", shape=list(y.shape), value=float(y[0, 0]))
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
