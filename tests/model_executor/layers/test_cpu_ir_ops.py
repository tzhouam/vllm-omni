# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fused-RMSNorm IR registration."""

import pytest
import torch

from vllm import ir
from vllm_omni.model_executor.layers import cpu_ir_ops

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

pytest.importorskip("vllm")

_SKIP = not cpu_ir_ops.CPU_FUSED_NORMS
_REASON = "vLLM CPU build without _C rms_norm kernels"


def _snr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(20 * torch.log10(a.norm() / (a - b).norm().clamp(min=1e-9)))


def test_registered_under_its_own_provider():
    """Registering a new provider, rather than editing vllm_c.py, is what keeps
    this working across wheel upgrades."""
    assert cpu_ir_ops.PROVIDER in ir.ops.rms_norm.impls
    assert cpu_ir_ops.PROVIDER in ir.ops.fused_add_rms_norm.impls


@pytest.mark.skipif(_SKIP, reason=_REASON)
@pytest.mark.parametrize("tokens", [1, 5, 64])
def test_rms_norm_matches_the_native_impl(tokens):
    torch.manual_seed(0)
    x = torch.randn(tokens, 2048, dtype=torch.bfloat16)
    w = torch.randn(2048, dtype=torch.bfloat16)
    native = ir.ops.rms_norm.impls["native"].impl_fn(x, w, 1e-6)
    ours = ir.ops.rms_norm.impls[cpu_ir_ops.PROVIDER].impl_fn(x, w, 1e-6)
    assert ours.shape == native.shape and ours.dtype == native.dtype
    # Both accumulate in their own order at bf16 output precision, so ~48 dB is
    # the ceiling; a wrong kernel or layout lands far below.
    assert _snr(native, ours) > 40.0, _snr(native, ours)


@pytest.mark.skipif(_SKIP, reason=_REASON)
def test_fused_add_rms_norm_writes_both_outputs():
    torch.manual_seed(0)
    x = torch.randn(4, 2048, dtype=torch.bfloat16)
    res = (x * 0.5).contiguous()
    w = torch.randn(2048, dtype=torch.bfloat16)
    n_out, n_res = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
        x.clone(), res.clone(), w, 1e-6
    )
    o_out, o_res = ir.ops.fused_add_rms_norm.impls[cpu_ir_ops.PROVIDER].impl_fn(
        x.clone(), res.clone(), w, 1e-6
    )
    assert _snr(n_out, o_out) > 40.0, _snr(n_out, o_out)
    # The residual is a plain sum, so it should agree far more tightly.
    assert _snr(n_res, o_res) > 100.0, _snr(n_res, o_res)


@pytest.mark.skipif(_SKIP, reason=_REASON)
def test_three_dimensional_input_round_trips_its_shape():
    x = torch.randn(2, 3, 2048, dtype=torch.bfloat16)
    w = torch.randn(2048, dtype=torch.bfloat16)
    out = ir.ops.rms_norm.impls[cpu_ir_ops.PROVIDER].impl_fn(x, w, 1e-6)
    assert out.shape == x.shape


def test_declines_arguments_the_kernel_cannot_take():
    """variance_size and a non-contiguous input must fall back to native."""
    impl = ir.ops.rms_norm.impls[cpu_ir_ops.PROVIDER]
    x = torch.randn(4, 2048, dtype=torch.bfloat16)
    w = torch.randn(2048, dtype=torch.bfloat16)
    assert impl.supports_args(x, w, 1e-6)
    assert not impl.supports_args(x, w, 1e-6, 1024)
    assert not impl.supports_args(x.t(), w, 1e-6)
    # Mismatched weight dtype is the other case the kernel rejects.
    assert not impl.supports_args(x, w.float(), 1e-6)


def test_priority_patch_is_opt_in_and_idempotent(monkeypatch):
    """Off by default, because that is what the measurement says.

    In isolation these kernels are 3.6x and 4.5x faster than the native path;
    in the compiled model they measured 72.7 tok/s against 74.0, because
    inductor already fuses the native norm into the residual add. So the
    registration always happens -- the provider stays selectable through
    `kernel_config.ir_op_priority` -- but the default ranking does not change
    unless it is asked for.
    """
    if not cpu_ir_ops.CPU_FUSED_NORMS:
        pytest.skip(_REASON)
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.delenv("VLLM_OMNI_CPU_FUSED_NORMS", raising=False)
    cpu_ir_ops.prefer_cpu_fused_norms()
    assert CpuPlatform.get_default_ir_op_priority(None).rms_norm == ["native"]

    monkeypatch.setenv("VLLM_OMNI_CPU_FUSED_NORMS", "1")
    cpu_ir_ops.prefer_cpu_fused_norms()
    cpu_ir_ops.prefer_cpu_fused_norms()  # second call must not double-wrap
    priority = CpuPlatform.get_default_ir_op_priority(None)
    assert priority.rms_norm[0] == cpu_ir_ops.PROVIDER
    assert priority.rms_norm.count(cpu_ir_ops.PROVIDER) == 1
    assert "native" in priority.rms_norm
