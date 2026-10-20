# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Give vLLM's CPU backend the fused RMSNorm kernels it already ships.

vLLM's CPU build registers `_C::rms_norm` and `_C::fused_add_rms_norm`, and
`vllm/kernels/vllm_c.py` already wraps both as IR-op implementations -- but it
gates them on ``GPGPU_DEVICE = is_cuda_alike() or is_xpu()``, so on CPU the IR
layer marks them unsupported and every norm falls through to the PyTorch
implementation.  Nothing then fuses it: inductor emits an elementwise loop and
spreads a 4 KB reduction across every core, which at one token is almost
entirely thread barrier.

Measured on a Xeon 8480C at Spark-X2.5's decode shape (1 token, hidden 2048),
24 threads:

    rms_norm             native 20.99 us   fused  5.76 us   3.6x
    fused_add_rms_norm   native 41.14 us   fused  9.24 us   4.5x

Two norms per layer over 28 layers is ~1.8 ms of an ~12 ms decode step.

The same file's `gelu_and_mul` is deliberately *not* re-registered here: the
fused CPU kernel measured 64.11 us against the native path's 23.36 us at
Spark's GEGLU shape ([1, 13312] -> [1, 6656]), i.e. 2.7x slower, so the native
form is the right choice there and the asymmetry is the point -- "use the
fused kernel" is not a rule, it is a per-op measurement.

Registering under our own provider name rather than editing `vllm_c.py` keeps
this working across vLLM wheel upgrades; it is the mechanism vLLM documents for
out-of-tree kernels (see `vllm/kernels/oink_ops.py`). The equivalent upstream
change is two lines: add `is_cpu()` to `GPGPU_DEVICE` and give `CpuPlatform` a
`get_default_ir_op_priority`.
"""

import os

import torch
from torch import Tensor

from vllm import ir
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

PROVIDER = "vllm_c_cpu"
_FUSED_NORM_OPS = ("rms_norm", "fused_add_rms_norm")


def _cpu_kernels_available() -> bool:
    if not current_platform.is_cpu():
        return False
    try:
        current_platform.import_kernels()
        return all(hasattr(torch.ops._C, name) for name in _FUSED_NORM_OPS)
    except Exception as exc:  # pragma: no cover - build without the extension
        logger.debug("CPU fused norm kernels unavailable: %s", exc)
        return False


CPU_FUSED_NORMS = _cpu_kernels_available()


def _supports(x: Tensor, weight, epsilon, variance_size=None) -> bool:
    # The kernel normalizes over the last dim of a contiguous 2-D buffer and
    # has no variance_size override; anything else stays on the native path.
    return (
        variance_size is None
        and (weight is None or weight.dtype == x.dtype)
        and x.is_contiguous()
    )


def _supports_add(x: Tensor, x_residual: Tensor, weight, epsilon, variance_size=None):
    return _supports(x, weight, epsilon, variance_size) and x_residual.is_contiguous()


@ir.ops.rms_norm.register_impl(
    PROVIDER, supports_args=_supports, supported=CPU_FUSED_NORMS
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    shape = x.shape
    x2d = x if x.dim() == 2 else x.reshape(-1, shape[-1])
    out = torch.empty_like(x2d)
    torch.ops._C.rms_norm(out, x2d, weight, epsilon)
    return out if x.dim() == 2 else out.reshape(shape)


@ir.ops.fused_add_rms_norm.register_impl(
    PROVIDER, supports_args=_supports_add, supported=CPU_FUSED_NORMS, inplace=True
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert variance_size is None
    if x.dim() == 2:
        torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)
        return x, x_residual
    shape = x.shape
    x2d, r2d = x.reshape(-1, shape[-1]), x_residual.reshape(-1, shape[-1])
    torch.ops._C.fused_add_rms_norm(x2d, r2d, weight, epsilon)
    return x2d.reshape(shape), r2d.reshape(shape)


def prefer_cpu_fused_norms() -> None:
    """Rank these kernels ahead of the native path -- opt-in only.

    The implementations are always registered (so they can be selected through
    ``kernel_config.ir_op_priority`` like any other provider); this function
    only changes the *default* ranking, and it does nothing unless
    ``VLLM_OMNI_CPU_FUSED_NORMS=1``. The platform hook fills in ops the user
    left unset, so an explicit priority still wins either way.
    """
    if not CPU_FUSED_NORMS:
        return
    # Off by default, and the default is the measurement: in isolation these
    # kernels are 3.6x and 4.5x faster than the native path, but in the
    # compiled model they measured 72.7 tok/s against 74.0, because inductor
    # was already fusing the native norm into the residual add and an opaque
    # call breaks that fusion. Set VLLM_OMNI_CPU_FUSED_NORMS=1 to try them on
    # a machine where that trade may land differently.
    if os.environ.get("VLLM_OMNI_CPU_FUSED_NORMS", "0") != "1":
        return
    from vllm.platforms.cpu import CpuPlatform

    if getattr(CpuPlatform, "_omni_fused_norm_priority", False):
        return

    base = CpuPlatform.get_default_ir_op_priority.__func__

    def get_default_ir_op_priority(cls, vllm_config):
        priority = base(cls, vllm_config)
        for op_name in _FUSED_NORM_OPS:
            current = list(getattr(priority, op_name))
            if PROVIDER not in current:
                setattr(priority, op_name, [PROVIDER] + current)
        return priority

    CpuPlatform.get_default_ir_op_priority = classmethod(get_default_ir_op_priority)
    CpuPlatform._omni_fused_norm_priority = True
    logger.info(
        "CPU fused RMSNorm kernels enabled (provider %s) ahead of the native path.",
        PROVIDER,
    )
