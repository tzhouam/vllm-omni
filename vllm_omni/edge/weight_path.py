# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which weight format and which kernel, per class of device.

A laptop, an edge gateway and a phone do not want the same build of the same
model, and the reason is not preference -- it is that the fast kernel on each
one needs a different weight layout, and the layouts are mutually exclusive:

* an x86 core with AMX tiles has a 4-bit GEMM (`_C::int4_scaled_mm_cpu`) that
  wants GPTQ-packed weights and quantizes activations to int8 -- int8
  activations are what llama.cpp's Q4_K_M does too, though not at the same bit
  budget (see below) -- and it is the fastest thing measured here;
* an x86 core without AMX falls back to PyTorch's tinygemm path, which wants
  its own packing and keeps activations in bf16 -- slower, but more faithful.
  This module used to call that "the only option", which was false: vLLM 0.28
  also has `ops.cpu_gemm_wna16`, selected whenever `layer.use_w4a8` is off
  (`vllm/model_executor/kernels/linear/mixed_precision/cpu.py:202`). It has
  now been measured, on the same GPTQ checkpoint with only
  `VLLM_CPU_INT4_W4A8` differing, 3 interleaved passes on 24 pinned cores:

      int4_scaled_mm_cpu (W4A8, AMX)   74.4 tok/s   spread 1.1%
      tinygemm W4A16 (same cores)      71.6 tok/s
      cpu_gemm_wna16                   32.3 tok/s   spread 1.5%

  So the fallback is real but **2.2x slower than tinygemm**, and the routing
  here was right for a reason nobody had checked. One caveat on reading that
  number outside x86: `_get_isa_hint` returns `"amx"` whenever AMX is present
  and activations are bf16, *regardless* of the W4A8 flag, so this measured
  `cpu_gemm_wna16` on its AMX path -- its best case on this host. The `"vec"`
  path a genuinely AMX-less machine takes is untested and presumably no
  faster, which only strengthens the choice;
* a phone NPU does not run vLLM at all. The deliverable there is an exported
  graph, and its quantization is decided by the export toolchain, not by us.

So the choice is a property of the device, and picking it at deploy time from
a probe is the difference between "4-bit is 22% faster" and "4-bit is broken
here". Measured on a Xeon 8480C, 24 threads, Spark-X2.5-1.7B. The two 4-bit
rows are from per-op profiles taken back to back on the same cores, because
this host's throughput swings ~30% and the gap between them is ~8%:

    path                 GEMM/step   step     tok/s
    w4a8 AMX g128          5.91 ms   11.55    86.6    fastest; least faithful
    w4a16 tinygemm g128    6.85 ms   12.51    80.0    more faithful
    llama.cpp Q4_K_M            --   11.11    90.0    the bar
    bf16 (no 4-bit kernel)      --      --    41.0

W4A8 wins twice over: `int4_scaled_mm_cpu` is 0.94 ms cheaper than tinygemm
per step, and it is reached through vLLM's own kernel dispatch, so it also
avoids the 0.48 ms/step custom-op boundary the tinygemm path needs to route
prefill and decode to different kernels.

One trap worth naming: a GPTQ checkpoint must quantize the output head
(`lm_head`), or that one GEMM runs in bf16 through `onednn_mm` and costs
2.97 ms/step by itself -- more than the whole 4-bit saving. Spark ties its
embedding to its head, so the head has to be untied and emitted separately;
`quantize_gptq_int4.py` does that by default.

Fidelity is the reason both 4-bit rows stay: against each build's own bf16
output over 12 greedy prompts, llama.cpp's Q4_K_M reproduced 4/12 exactly,
w4a16 2/12 and w4a8 1/12.

Two caveats on that comparison, both of which were stated wrongly here before.
It is **not** at a matched bit budget: w4a16 and w4a8 store 4 bits plus two
bf16 per 128-weight group = **4.25 bits/weight**, while plain Q4_K is 4.5 bits
over groups of 32 and Q4_K_M promotes some tensors further. And the symmetric
grid is **our export's choice, not the kernel's limit**: vLLM's W4A8 path uses
the checkpoint's zero points whenever `config.zero_points` is set and
synthesises the constant 8 only otherwise (same file, `:98`). Our exporter
passes `"sym": true`. An asymmetric export is a config change, and it is the
cheapest untried fidelity experiment.

That difference has a task-level consequence, which is why ``priority`` is not
a matter of taste. Driving a phone through the Android agent harness
(``analysis/experiments/spark_edge/android_agent``), every build navigates a
multi-screen task correctly -- and only some of them notice they have arrived.
W4A8 turned Wi-Fi on and then tapped the switch off again, on both runs;
W4A16 called ``task_complete`` citing the switch reading ``checked=true``, and
on a second task it picked the right app where bf16 picked the wrong one.
**Navigation survives quantization; recognising completion does not.** For
agentic work choose ``priority="fidelity"``; the 8% of decode it costs buys
the difference between finishing a task and undoing it. (Two tasks, one run
per cell -- enough to show the failure reproduces, not to rank success rates.)
"""

from dataclasses import dataclass, field
from typing import Literal

from vllm_omni.edge.hardware_probe import (
    HW_CLASS_ARM64_CPU,
    HW_CLASS_CUDA_DISCRETE,
    HW_CLASS_JETSON,
    HW_CLASS_NPU_PHONE,
    HardwareProfile,
    hardware_class,
)

Priority = Literal["speed", "fidelity"]


@dataclass(frozen=True)
class WeightPath:
    """How to build and run the weights on one class of device."""

    name: str
    quantization: str | None
    """vLLM ``--quantization`` name, or None to keep the model dtype."""
    group_size: int | None
    bits_per_weight: float | None
    """Stored bits per weight, group overhead included: a 4-bit code plus a
    bf16 scale and a bf16 zero per group is ``4 + 32 / group_size``, i.e.
    **4.25** at g128. (This field read 4.16 until 2026-09-13, which matched no
    arithmetic and made the llama.cpp fidelity comparison look budget-matched
    when it is not.)"""
    fused_cpu_norms: bool
    """vLLM's fused CPU RMSNorm. False everywhere on measurement: it is
    3.6-4.5x faster in isolation and *slower* in situ, because inductor already
    fuses the native form with the residual add. ``VLLM_OMNI_CPU_FUSED_NORMS``
    defaults to "0" for the same reason."""
    runs_in_vllm: bool
    """False means vLLM cannot execute here and an exported artifact is the
    deliverable -- the phone case."""
    verified: bool
    """True only where this combination was actually measured on hardware."""
    rationale: str
    build_with: str | None = None
    """Script that produces a checkpoint in this format."""
    notes: list[str] = field(default_factory=list)

    def as_engine_kwargs(self) -> dict[str, object]:
        """The subset that goes straight into ``LLM(...)`` / deploy overrides."""
        if not self.runs_in_vllm:
            return {}
        kwargs: dict[str, object] = {}
        if self.quantization:
            kwargs["quantization"] = self.quantization
        return kwargs


_AMX_FLAGS = ("amx_tile", "amx_bf16")


def has_amx(profile: HardwareProfile) -> bool:
    """AMX tiles, which is what gates vLLM's CPU 4-bit GEMM."""
    return profile.arch == "x86_64" and any(f in profile.cpu_flags for f in _AMX_FLAGS)


def select_weight_path(
    profile: HardwareProfile, priority: Priority = "speed"
) -> WeightPath:
    """Pick the weight format for this device.

    ``priority='fidelity'`` keeps bf16 activations wherever a choice exists,
    which costs ~7% of decode on an AMX machine and buys back the int8-activation
    error.
    """
    cls = hardware_class(profile)

    if cls == HW_CLASS_NPU_PHONE:
        return WeightPath(
            name="export-qnn-w8a16",
            quantization=None,
            group_size=None,
            bits_per_weight=8.0,
            fused_cpu_norms=False,
            runs_in_vllm=False,
            verified=True,
            rationale=(
                "vLLM has no NPU backend; the artifact is an exported graph. "
                "Calibrated w8a16 is the only scheme measured correct on device "
                "(36.9 dB); int8 activations break it (5.1 dB) and 4-bit weights "
                "fail the converter on the calibrated path."
            ),
            build_with="vllm_omni/edge/spark_export.py",
            notes=[
                "Calibration must use real activations: uncalibrated w4a16 "
                "scores -1.6 dB on device, i.e. uncorrelated with the reference.",
                "4-bit saves memory, not time, on this NPU: 0.917 vs 0.912 ms "
                "per layer against 8-bit.",
            ],
        )

    if cls in (HW_CLASS_CUDA_DISCRETE, HW_CLASS_JETSON):
        return WeightPath(
            name="bf16" if profile.supports_bf16 else "fp16",
            quantization=None,
            group_size=None,
            bits_per_weight=16.0,
            fused_cpu_norms=False,
            runs_in_vllm=True,
            verified=cls == HW_CLASS_CUDA_DISCRETE,
            rationale=(
                "The CPU 4-bit kernels are x86 CPU kernels. On a GPU the weights "
                "stay in the model dtype and the GPU quantization paths "
                "(marlin/AWQ) are a separate question this selector does not make."
            ),
        )

    if cls == HW_CLASS_ARM64_CPU:
        return WeightPath(
            name="w4a16-tinygemm",
            quantization="cpu_int4",
            group_size=128,
            bits_per_weight=4.25,
            fused_cpu_norms=False,
            runs_in_vllm=True,
            verified=False,
            rationale=(
                "No AMX, so no int4_scaled_mm_cpu. PyTorch's 4-bit CPU GEMM has "
                "an ARM path, but it dispatches to a different kernel "
                "(KleidiAI-style packing) than the AVX-512 one measured here."
            ),
            build_with="analysis/experiments/spark_edge/quantize_int4.py",
            notes=["Unverified on this host: x86 only. Measure before trusting."],
        )

    # x86 CPU: the one case with a real choice to make.
    if has_amx(profile) and priority == "speed":
        return WeightPath(
            name="w4a8-amx",
            quantization="gptq",
            group_size=128,
            bits_per_weight=4.25,
            fused_cpu_norms=False,
            runs_in_vllm=True,
            verified=True,
            rationale=(
                "AMX tiles present, so vLLM's own int4_scaled_mm_cpu applies: "
                "84.5 tok/s decode against 78.7 for tinygemm and 41.0 for bf16, "
                "and 3.0x llama.cpp on prefill."
            ),
            build_with="analysis/experiments/spark_edge/quantize_gptq_int4.py",
            notes=[
                "Least faithful of the three: 1/12 greedy prompts reproduced "
                "exactly against its own bf16, vs 2/12 for w4a16.",
                "Activations are quantized to int8 by the kernel; that is what "
                "llama.cpp's Q4_K_M also does.",
            ],
        )

    return WeightPath(
        name="w4a16-tinygemm",
        quantization="cpu_int4",
        group_size=128,
        bits_per_weight=4.25,
        fused_cpu_norms=False,
        runs_in_vllm=True,
        verified=True,
        rationale=(
            "No AMX tiles, or fidelity was asked for: keep activations in bf16 "
            "and take the tinygemm 4-bit GEMM. 78.7 tok/s decode, and the more "
            "faithful of the two 4-bit paths."
        ),
        build_with="analysis/experiments/spark_edge/quantize_int4.py",
    )
