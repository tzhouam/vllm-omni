"""Opt-in MiniCPM-o KV handoff hook for a bounded whole-request experiment."""

import os

if os.environ.get("VLLM_OMNI_MINICPMO_KV_GRAPH"):
    from minicpmo_kv_patch import install

    install()
