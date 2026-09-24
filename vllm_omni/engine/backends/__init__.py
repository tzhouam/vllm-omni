# SPDX-License-Identifier: Apache-2.0
"""Factories for non-vLLM stages owned by the existing StageRuntime."""


def create_graph_client(metadata, config, ledger, reservation):
    backend = config.get("backend", {})
    name = backend.get("name")
    if name == "external.graph.v1":
        from .graph import GraphStageClient

        return GraphStageClient(metadata, backend, ledger, reservation)
    if name == "external.llamacpp.text.v1":
        from .llamacpp import LlamaCppTextStageClient

        return LlamaCppTextStageClient(metadata, backend, ledger, reservation)
    if name == "external.llamacpp.multimodal.v1":
        from .llamacpp import LlamaCppMultimodalStageClient

        return LlamaCppMultimodalStageClient(metadata, backend, ledger, reservation)
    if name == "external.crisp.tts.v1":
        from .crisp_tts import CrispTTSStageClient

        return CrispTTSStageClient(metadata, backend, ledger, reservation)
    if name == "external.qwen_tts.cpu.v1":
        from .qwen_tts_cpu import QwenTTSCPUStageClient

        return QwenTTSCPUStageClient(metadata, backend, ledger, reservation)
    if name == "external.minicpmo.gguf.v1":
        from .minicpmo_cpp import MiniCPMOCppStageClient

        return MiniCPMOCppStageClient(metadata, backend, ledger, reservation)
    if name == "external.internvla.policy.v1":
        from .internvla import InternVLAStageClient

        return InternVLAStageClient(metadata, backend, ledger, reservation)
    raise ValueError(f"unsupported complete-request stage backend: {name!r}")
