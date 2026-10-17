# Installation

!!! important
    vLLM-Omni is released against the matching upstream vLLM major/minor version. This source checkout targets the vLLM 0.32.0 release-branch wheel at `5df63a7203cdb9328d824a27b682e77c7b9928be`, tested with CUDA 13.0 and PyTorch 2.13.0. The vLLM 0.32.0 tag and commit-index CUDA wheel are available; the GitHub release page, PyPI wheel and matching production CUDA/ROCm/XPU images are not yet available. The published vLLM-Omni 0.30.0 wheel instructions use vLLM 0.30.0; choose the installation path that matches your Omni revision.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)
