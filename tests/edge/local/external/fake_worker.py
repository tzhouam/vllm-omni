# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A stage worker that speaks the protocol and never touches a device.

Selected with ``VLLM_OMNI_EXTERNAL_WORKER_ORT_CPU``, so the client, the
framing, the placement gate and every refusal path are exercised on a machine
with no AMD hardware, no onnxruntime and no Windows. What it replies is read
from the JSON file named by ``VLLM_OMNI_FAKE_WORKER_JSON``:

``load``      body merged into the load reply -- set ``node_counts`` and
              ``fraction_on_target`` to stage a declined graph.
``run``       ``{"scale": float}``; outputs are the inputs times it.
``die_on``    op name after which the worker exits without replying, for the
              "worker vanished mid-request" path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_PROTOCOL = _HERE.parents[4] / "vllm_omni" / "edge" / "local" / "external" / "protocol.py"

spec = importlib.util.spec_from_file_location("_fake_protocol", _PROTOCOL)
proto = importlib.util.module_from_spec(spec)
sys.modules["_fake_protocol"] = proto
spec.loader.exec_module(proto)


def main() -> int:
    host, port, token = sys.argv[2], int(sys.argv[4]), sys.argv[6]
    config = {}
    config_path = os.environ.get("VLLM_OMNI_FAKE_WORKER_JSON")
    if config_path and Path(config_path).is_file():
        config = json.loads(Path(config_path).read_text())

    import socket

    sock = socket.create_connection((host, port), timeout=30)
    proto.send_message(
        sock,
        proto.OP_HELLO,
        {
            "token": config.get("token", token),
            "executable": sys.executable,
            "python": sys.version,
            "platform": sys.platform,
            "pid": os.getpid(),
            "onnxruntime": "fake",
            "numpy": np.__version__,
            "available_providers": ["CPUExecutionProvider"],
            "rss_bytes": 1024,
        },
    )

    while True:
        op, body, tensors = proto.recv_message(sock)
        if config.get("die_on") == op:
            sock.close()
            return 0
        if op == proto.OP_LOAD:
            reply = {
                "ep": body.get("ep", "cpu"),
                "session_providers": ["CPUExecutionProvider"],
                "available_providers": ["CPUExecutionProvider"],
                "node_counts": {"CPUExecutionProvider": 4},
                "total_nodes": 4,
                "target_nodes": 4,
                "fraction_on_target": 1.0,
                "session_create_s": 0.01,
                "warmup_s": 0.001,
                "rss_bytes": 111 * 2**20,
                "inputs": [{"name": n, "type": "tensor(float)", "shape": list(t.shape)} for n, t in tensors.items()],
                "outputs": [{"name": "y", "type": "tensor(float)", "shape": []}],
            }
            reply.update(config.get("load", {}))
            proto.send_message(sock, proto.OP_OK, reply)
        elif op == proto.OP_RUN:
            import time

            time.sleep(float(config.get("run", {}).get("sleep_s", 0)))
            scale = float(config.get("run", {}).get("scale", 2.0))
            proto.send_message(
                sock,
                proto.OP_OK,
                {"run_s": 0.001},
                {f"{k}_out": (v * scale) for k, v in tensors.items()},
            )
        elif op == proto.OP_STATS:
            stats = {
                "runs": 1,
                "total_run_s": 0.001,
                "rss_bytes": 111 * 2**20,
                "peak_rss_bytes": 122 * 2**20,
                "report": {},
            }
            stats.update(config.get("stats", {}))
            proto.send_message(
                sock,
                proto.OP_OK,
                stats,
            )
        elif op == proto.OP_CLOSE:
            proto.send_message(sock, proto.OP_OK, {})
            return 0
        else:
            proto.send_message(sock, proto.OP_ERR, {"message": f"unknown op {op}", "code": "ValueError"})


if __name__ == "__main__":
    raise SystemExit(main())
