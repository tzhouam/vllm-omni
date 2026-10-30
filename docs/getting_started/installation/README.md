# Installation

!!! important
    This source checkout targets vLLM 0.33.0rc1 at `cbafc4690fda85d74b8aad8f93f3d11aa0366b8b`, with CUDA 13.0 and PyTorch 2.14.0. Full CI and ROCm/XPU runtime validation remain pending. CUDA Dockerfiles upgrade the published 0.32.0 bootstrap image using the frozen candidate wheel; ROCm builds the candidate from source. The published Omni 0.30.0 wheel instructions below retain their matching vLLM 0.30.0 dependency.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)
