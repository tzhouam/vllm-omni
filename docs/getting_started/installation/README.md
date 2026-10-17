# Installation

!!! important
    vLLM-Omni is released against the matching upstream vLLM major/minor version. This source checkout targets the vLLM 0.32.0rc3 release-branch wheel at `82eb574d959031c670ecbc9930ed7b756a119bfe`, tested with CUDA 13.0 and PyTorch 2.13.0. A final vLLM 0.32.0 release and matching production CUDA/ROCm/XPU images are not yet available. The published vLLM-Omni 0.30.0 wheel instructions use vLLM 0.30.0; choose the installation path that matches your Omni revision.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)
