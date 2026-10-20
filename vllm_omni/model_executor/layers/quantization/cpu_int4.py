# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W4A16 group-wise weights for vLLM's CPU backend.

Single-stream decode on a CPU is a bandwidth problem: the whole weight matrix
is read to produce one token, so the bits per weight, not the FLOPs, set the
token rate.  Four bits is how llama.cpp's Q4_K_M builds win single-stream
decode against an engine that beats them everywhere else.

vLLM 0.28 does reach four bits on CPU, through the GPTQ/AWQ kernel selector
(`int4_scaled_mm_cpu`, AMX W4A8 -- see `gptq_cpu_export`).  That path pins the
zero point at 8, so its grid is symmetric and its activations are int8.  This
one is the other trade: an asymmetric min/max grid and bf16 activations, which
costs some speed and buys back fidelity.

Weights are quantized to 4 bits on groups of ``group_size`` columns with a
bf16 scale and offset per group; at the default group of 64 that is 4.5 bits
per weight, the same budget Q4_K_M spends.  The GEMM is PyTorch's ``_weight_int4pack_mm_for_cpu`` tinygemm
kernel, which has AVX-512 paths and, unlike the fp32 entry point, keeps
bf16 activations in bf16 end to end.

Checkpoints carry, per quantized linear:

    <name>.weight_packed   uint8  [N, K // 2]  two 4-bit codes per byte,
                                               byte j = w[2j] | w[2j+1] << 4
    <name>.weight_scale    bf16   [N, K // group]
    <name>.weight_zero     bf16   [N, K // group]

and dequantize as ``w = (q - 8) * scale + zero``.  The row-major nibble layout
is the portable one; the tinygemm layout the kernel wants is built in
``process_weights_after_loading``.  That layout was long treated as opaque
here, which is what made the second copy below look unavoidable.  Measured on
this host -- pack an index pattern, read back where each nibble landed -- it
is ``[N/64][K][32 bytes]``: K is the middle axis, so a K range is contiguous
inside each n-block, and within every 32-byte run byte ``d`` holds output
channel ``d`` in its low nibble and ``d + 32`` in its high one, the same
permutation for every ``k`` and every block.  See ``TINYGEMM_N_BLOCK``.

The tinygemm kernel dequantizes once per row of activations, so its cost is
linear in the batch and it loses to AMX bf16 above about eight rows -- fine
for decode, ruinous for prefill.  Large batches therefore take a second path:
the weight is unpacked to bf16 once per call and handed to the ordinary
oneDNN/AMX GEMM, which costs one extra pass over the weights and wins back
prefill.

That path used to read a row-major duplicate, so a layer held two packed
layouts -- 8 bits per weight resident against 4.25 stored.  It can instead
dequantize straight from the tinygemm layout (``_dequantize_from_int4pack``),
bit-exact against the old path, and then the duplicate is never allocated: on
Spark-X2.5-1.7B, **615 MB of resident memory and ~915 MB of peak** (3 arms x 2
interleaved passes, RssAnon over the process tree).

It is not free, because the duplicate was not only a copy.
``to_split_halves`` de-interleaves the nibbles *once at load*, so each prefill
call reads two contiguous planes; reading the kernel layout redoes that
de-interleave every call.  pp512 drops **2190 -> 1344 tok/s**, i.e. **1.63x**,
pooled over 7 interleaved passes per arm -- and the ratio is only good to about
1.4-1.8x, because both dequantizing arms swing 10-18% on a shared host while
the tinygemm-only arm reproduces to 0.3%.  Dequantizing to ``[K, N]`` and using
``matmul`` to dodge the transpose was tried and is worse still (20.3 ms vs 14.4
per ``gate_up`` at 8 threads).  Recovering it wants a C++ de-interleave, not an
ATen expression chain.

So this is a dial, not a free win:

* versus ``prefill_dequant: false`` it is strictly better -- same memory to
  within the noise, **2.97x** the prefill (1344 vs 453), and it keeps
  ``dequant_threshold`` in force so multi-row batches never reach the tinygemm
  kernel, avoiding the SIGILL hazard below.  That setting is dominated, and
  this is the reliable comparison: the tinygemm-only arm repeats to 0.3%.
* versus keeping both layouts it trades 615 MB for ~1.6x prefill.  Default on,
  because this path exists for devices picked for their memory ceiling; set
  ``direct_dequant: false`` in the checkpoint's ``quantization_config`` to
  restore the old behaviour.

``supports_direct_dequant`` gates it: N has to tile the 64-channel block, which
every Spark linear does (2048, 3072, 13312, 131072).  An N like 80 packs as one
64 block plus a 16 tail that the reshape would mis-read, so those layers keep
the duplicate rather than being quietly wrong.
``VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT=0`` restores the old path, which is how the
two were compared under identical code.

``dequant_threshold`` defaults to 2, i.e. only a single-row step takes the
tinygemm path.  That is not just a speed choice -- and it is now also the only
safe setting, since ``prefill_dequant: false`` (which routes every batch into
the tinygemm kernel) no longer saves memory but still carries the hazard
below.  On a Sapphire Rapids host,
running the tinygemm kernel on a multi-row batch and then vLLM's
``cpu_attention_with_kv_cache`` in the same forward kills the process with
SIGILL inside the attention kernel -- the signature of an AMX instruction
issued with no valid tile configuration loaded.  The same prompt is fine in
bf16, and fine at any batch size once every multi-row GEMM goes through the
dequantize path, so the two kernels disagree about who owns the tile state.
Single-row steps never trip it, which is why the crossover sits at 2 rather
than at the ~8 rows where the dequantize path starts winning on speed.
Batched decode therefore pays the dequantize path; single-stream decode, the
case 4-bit weights exist for, does not.
"""

from typing import Any

import os

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op

# The tinygemm packer refuses anything else.
PACK_N_ALIGN = 16
INNER_K_TILES = 1


def pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """[N, K] codes in [0, 15] -> [N, K // 2] uint8, low nibble first."""
    q = q.to(torch.uint8)
    lo, hi = q[:, 0::2], q[:, 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """[N, K // 2] uint8 -> [N, K] int32 codes in [0, 15]."""
    n, half = packed.shape
    out = torch.empty(n, half * 2, dtype=torch.int32)
    out[:, 0::2] = (packed & 0x0F).to(torch.int32)
    out[:, 1::2] = (packed >> 4).to(torch.int32)
    return out


def quantize_groupwise(w: torch.Tensor, group: int):
    """Min/max 4-bit quantization of [N, K] -> (packed, scale, zero).

    ``scale = (max - min) / 15`` and ``zero = min + 8 * scale`` make the
    kernel's ``(q - 8) * scale + zero`` reproduce the group's extremes
    exactly, which is the same asymmetric grid llama.cpp's K-quants use.
    """
    n, k = w.shape
    if k % group:
        raise ValueError(f"input dim {k} is not a multiple of group size {group}")
    wf = w.float().reshape(n, k // group, group)
    wmin, wmax = wf.amin(dim=-1), wf.amax(dim=-1)
    scale = (wmax - wmin).clamp(min=1e-9) / 15.0
    zero = wmin + 8.0 * scale
    q = ((wf - wmin.unsqueeze(-1)) / scale.unsqueeze(-1)).round_().clamp_(0, 15)
    return pack_nibbles(q.reshape(n, k)), scale, zero


REPACK_ROWS = 2048
"""Output channels repacked at a time, to bound the load-time peak.

2048 because it is both the fastest and nearly the leanest. Repacking the
131072-row tied embedding, 24 threads:

    rows        chunks    time      int32 transient
    whole            1   208 ms         1024 MiB
    512            256   116 ms            4 MiB
    2048            64    79 ms           16 MiB
    8192            16    89 ms           64 MiB
    32768            4   129 ms          256 MiB

Chunking is *faster* than packing the whole matrix -- the working set stays in
cache -- so this costs nothing at load; the earlier suspicion that it cost
30 s of init came from a pass the harness had flagged as contended.

``unpack_nibbles`` expands 4-bit codes to int32 -- 4 bytes per weight, eight
times the packed form -- and the kernel's packer wants int32. Doing that for a
whole matrix makes the transient the largest thing in the process: 104 MiB for
``gate_up``, and **1024 MiB for the 131072-row tied embedding**, against 24 MiB
of packed weights for an entire layer. Since the packed layout is
``[N/64][K][32 bytes]``, n-blocks are independent, so the expansion can cover
one chunk at a time: verified byte-identical to packing the whole matrix at
13312x2048, 2048x6656, 3072x2048 and 131072x2048.

A multiple of ``TINYGEMM_N_BLOCK``; the loop rounds down to one.
``VLLM_OMNI_CPU_INT4_REPACK_ROWS`` overrides it, and 0 means "the whole matrix
at once", which restores the old behaviour for an A/B. Unlike the prefill-path
switch this only affects load time and produces identical tensors, so it does
not perturb the compiled graph or its cache key.
"""


def _repack_rows() -> int:
    raw = os.environ.get("VLLM_OMNI_CPU_INT4_REPACK_ROWS")
    return REPACK_ROWS if raw is None else int(raw)


def _pack_blocked(packed: torch.Tensor) -> torch.Tensor:
    """Kernel layout, without ever expanding the whole matrix to int32."""
    n = packed.shape[0]
    rows = _repack_rows()
    if rows <= 0:                       # explicit opt-out: old behaviour
        return torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            unpack_nibbles(packed), INNER_K_TILES
        )
    step = max(rows - rows % TINYGEMM_N_BLOCK, TINYGEMM_N_BLOCK)
    if n <= step:
        return torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            unpack_nibbles(packed), INNER_K_TILES
        )
    out = []
    for i in range(0, n, step):
        codes = unpack_nibbles(packed[i : i + step].contiguous())
        out.append(
            torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, INNER_K_TILES)
        )
        del codes
    return torch.cat(out, dim=0)


def _to_tinygemm(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    """Row-major nibbles + [N, G] scales -> kernel layout + [G, N, 2] scales."""
    inner = _pack_blocked(packed)
    sz = torch.stack([scale.float(), zero.float()], dim=-1)  # [N, G, 2]
    sz = sz.transpose(0, 1).contiguous().to(torch.bfloat16)  # [G, N, 2]
    return inner, sz


def to_split_halves(packed: torch.Tensor) -> torch.Tensor:
    """Interleaved nibbles -> byte j holding w[j] and w[j + K/2].

    The on-disk layout interleaves so that a tensor-parallel split along K
    stays a plain slice.  Once the shard is local, splitting the halves makes
    both nibble planes contiguous runs, and the dequantize loop drops its
    lane shuffle: measured 3.1 ms per layer instead of 4.5.
    """
    n = packed.shape[0]
    rows = _repack_rows()
    step = n if rows <= 0 else max(rows - rows % TINYGEMM_N_BLOCK, TINYGEMM_N_BLOCK)
    out = []
    for i in range(0, n, step):                 # bounded transient, as above
        codes = unpack_nibbles(packed[i : i + step].contiguous())
        k = codes.shape[1]
        lo, hi = codes[:, : k // 2], codes[:, k // 2 :]
        out.append((lo.to(torch.uint8) | (hi.to(torch.uint8) << 4)).contiguous())
        del codes
    return torch.cat(out, dim=0) if len(out) > 1 else out[0]


TINYGEMM_N_BLOCK = 64
"""Output channels per block in the AVX-512 tinygemm layout.

Measured on this host rather than assumed: packing an index pattern and
reading it back shows the buffer is ``[N/64][K][32 bytes]`` -- K is the middle
axis, so a K range is contiguous inside each n-block -- and that within each
32-byte run, byte ``d`` holds output channel ``d`` in its low nibble and
channel ``d + 32`` in its high nibble, the same permutation for every ``k`` and
every n-block. Below 64 the block is N itself (16, 32 and 48 were measured);
between, e.g. N=80, there is a 64 block plus a 16 tail this does not handle.
"""


def direct_dequant_enabled(config_value: bool = True) -> bool:
    """Whether to dequantize from the tinygemm layout instead of a duplicate.

    The supported way to choose is the ``direct_dequant`` field of the
    checkpoint's ``quantization_config``, because that lives in the model
    config and so lands in vLLM's compile-cache key
    (``ModelConfig.compute_hash`` hashes its fields against an ignore list).
    The two paths build *different graphs* -- one passes ``weight_halves`` into
    the custom op and the other passes ``None`` -- so they must not share a
    cached artifact.

    ``VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT`` overrides it for experiments, and
    **does not** participate in that key. Flipping it against a warm compile
    cache loads an artifact built for the other path and dies with
    ``KeyError: 'weight_halves'`` inside the AOT-loaded graph -- observed, not
    theorised. Give each arm its own ``VLLM_CACHE_ROOT`` when A/B-ing, which
    is what the benchmark specs now do.
    """
    override = os.environ.get("VLLM_OMNI_CPU_INT4_DIRECT_DEQUANT")
    if override is not None:
        return override != "0"
    return config_value


def supports_direct_dequant(n: int) -> bool:
    """Whether ``n`` output channels tile the block cleanly.

    Every Spark linear does (2048, 3072, 13312, 131072). An N like 80 packs as
    one 64 block plus a 16 tail, which the reshape below would mis-read, so
    those layers keep the second copy rather than being silently wrong.
    """
    if n < TINYGEMM_N_BLOCK:
        return n % 16 == 0
    return n % TINYGEMM_N_BLOCK == 0


def _dequantize_from_int4pack(
    int4pack: torch.Tensor, sz: torch.Tensor, group: int
) -> torch.Tensor:
    """bf16 weight straight from the kernel's own layout.

    This is what lets the prefill path exist without a second weight copy: the
    layout was previously treated as opaque, so a row-major duplicate was kept
    beside it purely to have something to dequantize from.
    """
    n = int4pack.shape[0]
    k = int4pack.shape[1] * 2
    b = TINYGEMM_N_BLOCK if n >= TINYGEMM_N_BLOCK else n
    blocks = int4pack.reshape(n // b, k, b // 2)
    lo = (blocks & 0x0F).to(torch.bfloat16)
    hi = (blocks >> 4).to(torch.bfloat16)
    # index 2 is the output channel within the block: d from the low nibble,
    # d + 32 from the high one.
    codes = torch.cat([lo, hi], dim=2).permute(0, 2, 1).reshape(n, k)
    scale = sz[..., 0].transpose(0, 1)          # [G, N] -> [N, G]
    zero = sz[..., 1].transpose(0, 1)
    w = (codes.reshape(n, k // group, group) - 8.0) * scale.unsqueeze(
        -1
    ) + zero.unsqueeze(-1)
    return w.reshape(n, k)


def _dequantize_halves(
    halves: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group: int,
) -> torch.Tensor:
    n, half = halves.shape
    g2 = half // group
    lo = (halves & 0x0F).to(torch.bfloat16).reshape(n, g2, group)
    hi = (halves >> 4).to(torch.bfloat16).reshape(n, g2, group)
    a = (lo - 8.0) * scale[:, :g2].unsqueeze(-1) + zero[:, :g2].unsqueeze(-1)
    b = (hi - 8.0) * scale[:, g2:].unsqueeze(-1) + zero[:, g2:].unsqueeze(-1)
    return torch.cat([a.reshape(n, half), b.reshape(n, half)], dim=1)


_compiled_dequant = None
_compiled_dequant_direct = None


def dequantize_direct(int4pack, sz, group):
    """bf16 weight from the tinygemm layout; compiled, like the halves path."""
    global _compiled_dequant_direct
    if _compiled_dequant_direct is None:
        _compiled_dequant_direct = torch.compile(
            _dequantize_from_int4pack, dynamic=False
        )
    return _compiled_dequant_direct(int4pack, sz, group)


def dequantize(halves, scale, zero, group):
    """bf16 weight from the row-major halves; compiled, so it is one pass."""
    global _compiled_dequant
    if _compiled_dequant is None:
        # Compiled outside any vLLM graph: this runs inside an opaque custom
        # op, so dynamo never traces it and never guards on the batch size.
        _compiled_dequant = torch.compile(_dequantize_halves, dynamic=False)
    return _compiled_dequant(halves, scale, zero, group)


def _cpu_int4_linear(
    x: torch.Tensor,
    int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    halves: torch.Tensor | None,
    scale: torch.Tensor | None,
    zero: torch.Tensor | None,
    group: int,
    threshold: int,
    direct_dequant: bool = False,
) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if x.shape[0] >= threshold:
        if direct_dequant:
            weight = dequantize_direct(int4pack, scales_and_zeros, group)
            return torch.nn.functional.linear(x, weight)
        if halves is not None:
            weight = dequantize(halves, scale, zero, group)
            return torch.nn.functional.linear(x, weight)
    return torch.ops.aten._weight_int4pack_mm_for_cpu(
        x, int4pack, group, scales_and_zeros
    )


def _cpu_int4_linear_fake(
    x: torch.Tensor,
    int4pack: torch.Tensor,
    scales_and_zeros: torch.Tensor,
    halves: torch.Tensor | None,
    scale: torch.Tensor | None,
    zero: torch.Tensor | None,
    group: int,
    threshold: int,
    direct_dequant: bool = False,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], int4pack.shape[0]), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="cpu_int4_linear",
    op_func=_cpu_int4_linear,
    mutates_args=[],
    fake_impl=_cpu_int4_linear_fake,
    dispatch_key="CPU",
)


@register_quantization_config("cpu_int4")
class CPUInt4Config(QuantizationConfig):
    """Group-wise 4-bit weight-only quantization for the CPU backend."""

    def __init__(
        self,
        group_size: int = 64,
        ignore: list[str] | None = None,
        quantize_embedding: bool = True,
        prefill_dequant: bool = True,
        dequant_threshold: int = 2,
        use_custom_op: bool = True,
        direct_dequant: bool = True,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.ignore = list(ignore or [])
        self.quantize_embedding = quantize_embedding
        self.prefill_dequant = prefill_dequant
        self.dequant_threshold = dequant_threshold
        self.use_custom_op = use_custom_op
        # In the checkpoint config, so it reaches the compile-cache key: the
        # two prefill paths produce different graphs and must not share an
        # AOT artifact.
        self.direct_dequant = direct_dequant

    def __repr__(self) -> str:
        return (
            f"CPUInt4Config(group_size={self.group_size}, "
            f"ignore={self.ignore}, quantize_embedding={self.quantize_embedding}, "
            f"prefill_dequant={self.prefill_dequant}, "
            f"dequant_threshold={self.dequant_threshold}, "
            f"use_custom_op={self.use_custom_op}, "
            f"direct_dequant={self.direct_dequant})"
        )

    def get_name(self) -> str:
        return "cpu_int4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # The fp32 entry point of the tinygemm kernel has no vectorized path
        # and runs ~30x slower, so bf16 is the only activation type offered.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CPUInt4Config":
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 64),
            ignore=cls.get_from_keys_or(config, ["ignore"], []),
            quantize_embedding=cls.get_from_keys_or(
                config, ["quantize_embedding"], True
            ),
            prefill_dequant=cls.get_from_keys_or(config, ["prefill_dequant"], True),
            dequant_threshold=cls.get_from_keys_or(config, ["dequant_threshold"], 2),
            use_custom_op=cls.get_from_keys_or(config, ["use_custom_op"], True),
            direct_dequant=cls.get_from_keys_or(config, ["direct_dequant"], True),
        )

    def _is_ignored(self, prefix: str) -> bool:
        return any(prefix == p or prefix.endswith("." + p) for p in self.ignore)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            if not self.quantize_embedding or self._is_ignored(prefix):
                return None
            return CPUInt4EmbeddingMethod(self)
        if isinstance(layer, LinearBase):
            if self._is_ignored(prefix):
                return UnquantizedLinearMethod()
            return CPUInt4LinearMethod(self)
        return None


def _create_int4_weights(
    method: "CPUInt4LinearMethod",
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    params_dtype: torch.dtype,
    extra_weight_attrs: dict,
) -> None:
    group = method.quant_config.group_size
    out_size = sum(output_partition_sizes)
    if input_size_per_partition % group:
        raise ValueError(
            f"input dim {input_size_per_partition} is not a multiple of "
            f"group size {group}"
        )
    if out_size % PACK_N_ALIGN:
        raise ValueError(
            f"output dim {out_size} is not a multiple of {PACK_N_ALIGN}, which "
            "the tinygemm packer requires"
        )
    weight_loader = extra_weight_attrs.pop("weight_loader")

    packed = PackedvLLMParameter(
        data=torch.empty(out_size, input_size_per_partition // 2, dtype=torch.uint8),
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=2,
        weight_loader=weight_loader,
    )
    layer.register_parameter("weight_packed", packed)
    for name in ("weight_scale", "weight_zero"):
        p = GroupQuantScaleParameter(
            data=torch.empty(
                out_size, input_size_per_partition // group, dtype=torch.bfloat16
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter(name, p)
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = out_size
    set_weight_attrs(packed, extra_weight_attrs)


def _finalize_int4(
    layer: torch.nn.Module,
    keep_row_major: bool,
    prefill_dequant: bool,
    direct_dequant: bool = True,
) -> None:
    """Repack into the layouts the two GEMM paths want; once per layer."""
    if getattr(layer, "_cpu_int4_ready", False):
        return
    packed = layer.weight_packed.data
    scale = layer.weight_scale.data
    zero = layer.weight_zero.data
    inner, sz = _to_tinygemm(packed, scale, zero)
    layer.weight_int4pack = torch.nn.Parameter(inner, requires_grad=False)
    layer.weight_scales_and_zeros = torch.nn.Parameter(sz, requires_grad=False)
    layer.weight_halves = None
    layer.weight_group_scale = None
    layer.weight_group_zero = None
    layer.cpu_int4_direct_dequant = False
    if prefill_dequant:
        if direct_dequant_enabled(direct_dequant) and supports_direct_dequant(
            inner.shape[0]
        ):
            # The prefill path reads the kernel's own layout, so there is no
            # second copy and no duplicated scale/zero. This is the whole
            # 740 MB.
            layer.cpu_int4_direct_dequant = True
        else:
            # N does not tile the 64-channel block (e.g. 80 = 64 + 16), so
            # keep the row-major duplicate rather than mis-read the layout.
            layer.weight_halves = torch.nn.Parameter(
                to_split_halves(packed), requires_grad=False
            )
            layer.weight_group_scale = torch.nn.Parameter(
                scale.clone(), requires_grad=False
            )
            layer.weight_group_zero = torch.nn.Parameter(
                zero.clone(), requires_grad=False
            )
    if keep_row_major:
        # Only the vocab table needs this: the tinygemm layout interleaves
        # rows inside blocks of 16, so a single-token embedding lookup cannot
        # address it. One row is a few hundred bytes, so keeping the portable
        # copy costs no bandwidth at decode.
        layer.weight_packed = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        layer.weight_zero = torch.nn.Parameter(zero, requires_grad=False)
    else:
        del layer.weight_packed
        del layer.weight_scale
        del layer.weight_zero
    layer._cpu_int4_ready = True


def _int4_mm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    group: int,
    threshold: int,
    custom_op: bool = True,
) -> torch.Tensor:
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if x2d.dtype != torch.bfloat16:
        # Static at trace time: the config advertises bf16 activations only,
        # and the fp32 entry point of the tinygemm kernel has no vectorized
        # path (~30x slower), so never fall into it by accident.
        x2d = x2d.to(torch.bfloat16)
    if custom_op:
        # Opaque to dynamo, so the batch-size branch never becomes a guard on
        # vLLM's single traced graph. Costs ~5 us a call in dispatch.
        out = torch.ops.vllm.cpu_int4_linear(
            x2d,
            layer.weight_int4pack,
            layer.weight_scales_and_zeros,
            layer.weight_halves,
            layer.weight_group_scale,
            layer.weight_group_zero,
            group,
            threshold,
            getattr(layer, "cpu_int4_direct_dequant", False),
        )
    elif getattr(layer, "cpu_int4_direct_dequant", False) and x2d.shape[0] >= threshold:
        out = torch.nn.functional.linear(
            x2d,
            _dequantize_from_int4pack(
                layer.weight_int4pack, layer.weight_scales_and_zeros, group
            ),
        )
    elif layer.weight_halves is not None and x2d.shape[0] >= threshold:
        out = torch.nn.functional.linear(
            x2d,
            _dequantize_halves(
                layer.weight_halves,
                layer.weight_group_scale,
                layer.weight_group_zero,
                group,
            ),
        )
    else:
        out = torch.ops.aten._weight_int4pack_mm_for_cpu(
            x2d, layer.weight_int4pack, group, layer.weight_scales_and_zeros
        )
    return out.reshape(*orig_shape[:-1], out.shape[-1])


class CPUInt4LinearMethod(QuantizeMethodBase):
    """Group-wise 4-bit linear."""

    def __init__(self, quant_config: CPUInt4Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _create_int4_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _finalize_int4(
            layer,
            keep_row_major=False,
            prefill_dequant=self.quant_config.prefill_dequant,
            direct_dequant=self.quant_config.direct_dequant,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = _int4_mm(
            layer,
            x,
            self.quant_config.group_size,
            self.quant_config.dequant_threshold,
            self.quant_config.use_custom_op,
        )
        if bias is not None:
            out = out + bias
        return out


class CPUInt4EmbeddingMethod(CPUInt4LinearMethod):
    """The same weights, used as a vocab table and as the output head.

    With tied embeddings these are one tensor, so quantizing it pays twice:
    the lm_head GEMM is the single largest read in a decode step, and the
    table itself is 16% of the model's bytes.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _create_int4_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            extra_weight_attrs,
        )
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # No dequantize path for the vocab table: the head only ever sees the
        # tokens being sampled (one row per sequence), and unpacking a
        # 131072 x 2048 matrix to bf16 would cost more than the GEMM saves.
        _finalize_int4(layer, keep_row_major=True, prefill_dequant=False)

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Share every quantized tensor, not just ``weight``."""
        for name in (
            "weight_packed",
            "weight_scale",
            "weight_zero",
            "weight_int4pack",
            "weight_scales_and_zeros",
            "weight_halves",
            "weight_group_scale",
            "weight_group_zero",
        ):
            src = getattr(embed_tokens, name, None)
            if src is not None:
                setattr(layer, name, src)
        layer.cpu_int4_direct_dequant = getattr(
            embed_tokens, "cpu_int4_direct_dequant", False
        )
        layer._cpu_int4_tied_to = embed_tokens
        return layer

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        group = self.quant_config.group_size
        rows = layer.weight_packed.data.index_select(0, input_.reshape(-1))
        codes = unpack_nibbles(rows).float()
        n, k = codes.shape
        scale = layer.weight_scale.data.index_select(0, input_.reshape(-1)).float()
        zero = layer.weight_zero.data.index_select(0, input_.reshape(-1)).float()
        deq = (codes.reshape(n, k // group, group) - 8.0) * scale.unsqueeze(
            -1
        ) + zero.unsqueeze(-1)
        out = deq.reshape(n, k).to(torch.bfloat16)
        return out.reshape(*input_.shape, k)
