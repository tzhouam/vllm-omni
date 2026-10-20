"""WP4/mobile: the decode-step graph exported for phone accelerators (QNN / TFLite / ORT).

Checks the maths of the grouped-query attention layout against a straightforward reference, the KV-cache contract,
and that the exported ONNX is free of the three constructs the Qualcomm toolchains reject.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from vllm_omni.edge import qnn_export as qx

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CFG = qx.DecodeStepConfig(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    intermediate_size=48,
    vocab_size=16,
    rope_theta=10000.0,
)
L = 6


def _module(seed: int = 0) -> qx.DecodeStep:
    torch.manual_seed(seed)
    module = qx.DecodeStep(CFG).eval()
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.15)
    return module


def test_config_rejects_head_counts_that_do_not_group():
    with pytest.raises(ValueError):
        qx.DecodeStepConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=6, num_key_value_heads=4, head_dim=8, intermediate_size=48, vocab_size=16)


def test_names_and_example_input_shapes_line_up():
    names, outs = qx.input_names(CFG), qx.output_names(CFG)
    args = qx.example_inputs(CFG, L)
    assert len(names) == len(args) == 4 + 2 * CFG.num_hidden_layers
    assert names[:4] == ["x", "cos", "sin", "mask"] and names[4:6] == ["k_cache_0", "v_cache_0"]
    assert outs[0] == "logits" and outs[1:3] == ["k_new_0", "v_new_0"]
    x, cos, sin, mask, *caches = args
    assert x.shape == (1, 1, CFG.hidden_size) and cos.shape == sin.shape == (1, 1, 1, CFG.head_dim)
    assert mask.shape == (1, 1, 1, L + 1)
    assert all(c.shape == (1, CFG.num_key_value_heads, L, CFG.head_dim) for c in caches)


def test_output_contract():
    module = _module()
    out = module(*qx.example_inputs(CFG, L))
    assert len(out) == 1 + 2 * CFG.num_hidden_layers
    assert out[0].shape == (1, CFG.vocab_size)
    assert all(t.shape == (1, CFG.num_key_value_heads, 1, CFG.head_dim) for t in out[1:])


def test_wrong_number_of_caches_is_rejected():
    module = _module()
    x, cos, sin, mask, *caches = qx.example_inputs(CFG, L)
    with pytest.raises(ValueError):
        module(x, cos, sin, mask, *caches[:-1])


def test_grouped_query_layout_matches_expanded_reference():
    """The exported layout reshapes q; the reference expands k/v. They must agree."""
    module = _module(1)
    x, cos, sin, mask, *caches = qx.example_inputs(CFG, L)
    layer = module.layers[0]
    got, k_new, v_new = layer(x, cos, sin, caches[0], caches[1], mask)

    d, kv, g = CFG.head_dim, CFG.num_key_value_heads, CFG.group_size
    h = qx._rms(x, layer.input_layernorm, CFG.rms_norm_eps)
    q = qx._rope(qx._rms(layer.q_proj(h).view(1, CFG.num_attention_heads, 1, d), layer.q_norm, CFG.rms_norm_eps), cos, sin)
    k = qx._rope(qx._rms(layer.k_proj(h).view(1, kv, 1, d), layer.k_norm, CFG.rms_norm_eps), cos, sin)
    v = layer.v_proj(h).view(1, kv, 1, d)
    keys = torch.cat([caches[0], k], 2).repeat_interleave(g, dim=1)
    vals = torch.cat([caches[1], v], 2).repeat_interleave(g, dim=1)
    scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(d) + mask
    attn = torch.matmul(torch.softmax(scores, -1), vals).transpose(1, 2).reshape(1, 1, CFG.num_attention_heads * d)
    expected = x + layer.o_proj(attn)
    hh = qx._rms(expected, layer.post_attention_layernorm, CFG.rms_norm_eps)
    expected = expected + layer.down_proj(torch.nn.functional.silu(layer.gate_proj(hh)) * layer.up_proj(hh))

    assert torch.allclose(got, expected, atol=1e-5)
    assert torch.allclose(k_new, k) and torch.allclose(v_new, v)


def test_masked_cache_slots_do_not_change_the_result():
    """Slots the mask disables may hold anything: that is what lets the host keep one fixed-size cache."""
    module = _module(2)
    args = list(qx.example_inputs(CFG, L, position=2))
    first = module(*args)[0]
    for i in range(4, len(args)):
        cache = args[i].clone()
        cache[:, :, 2:L] = 7.0  # masked region
        args[i] = cache
    assert torch.allclose(first, module(*args)[0], atol=1e-5)


def test_llama_style_config_skips_qk_norm():
    """MiniCPM-o's TTS head is a Llama decoder: no per-head q/k RMSNorm, so those params must not be applied."""
    cfg = qx.DecodeStepConfig(**{**{f.name: getattr(CFG, f.name) for f in CFG.__dataclass_fields__.values()}, "qk_norm": False})
    module = qx.DecodeStep(cfg).eval()
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.15)
    args = qx.example_inputs(cfg, L)
    before = module(*args)[0].clone()
    with torch.no_grad():  # q_norm/k_norm must be dead weights when qk_norm is off
        module.layers[0].q_norm.mul_(3.0)
        module.layers[0].k_norm.mul_(0.1)
    assert torch.equal(before, module(*args)[0])


def test_minicpmo_config_maps_the_tts_head():
    cfg = qx.config_from_minicpmo({"tts_config": {"num_hidden_layers": 20, "hidden_size": 768, "num_attention_heads": 12,
                                                  "num_key_value_heads": 12, "intermediate_size": 3072, "num_audio_tokens": 6562}})
    assert cfg.qk_norm is False and cfg.head_dim == 64 and cfg.vocab_size == 6562 and cfg.group_size == 1


def test_placement_table_prefers_npu_for_the_step_and_gpu_for_the_vocoder():
    p = qx.PLACEMENT_MS_PER_FRAME
    assert p["talker_step"]["npu"] < p["talker_step"]["gpu"]
    assert p["code2wav_chunk"]["gpu"] < p["code2wav_chunk"]["npu"]


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None, reason="onnx not installed")
def test_export_onnx_is_clean_for_the_qualcomm_toolchains(tmp_path):
    import onnx

    module = _module(3)
    manifest = qx.export_onnx(module, L, tmp_path / "step.onnx")
    assert manifest["cache_len"] == L and manifest["config"]["hidden_size"] == CFG.hidden_size
    model = onnx.load(str(tmp_path / "step.onnx"))
    onnx.checker.check_model(model, full_check=True)
    assert [i.name for i in model.graph.input] == qx.input_names(CFG)
    assert [o.name for o in model.graph.output] == qx.output_names(CFG)
    assert not [n for n in model.graph.node if n.op_type == "IsNaN"]
    assert not [n for n in model.graph.node for a in n.attribute if n.op_type == "Reshape" and a.name == "allowzero" and a.i]
    inits = {i.name: i for i in model.graph.initializer}
    assert not [n for n in model.graph.node if n.op_type in ("Unsqueeze", "Squeeze") and len(n.input) > 1 and n.input[1] in inits and not inits[n.input[1]].dims]


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("onnx") is None, reason="onnx/onnxruntime not installed")
def test_exported_graph_reproduces_eager(tmp_path):
    import onnxruntime as ort

    module = _module(4)
    args = qx.example_inputs(CFG, L)
    with torch.inference_mode():
        expected = module(*args)
    qx.export_onnx(module, L, tmp_path / "step.onnx")
    sess = ort.InferenceSession(str(tmp_path / "step.onnx"), providers=["CPUExecutionProvider"])
    feed = {name: arg.numpy() for name, arg in zip(qx.input_names(CFG), args)}
    got = sess.run(None, feed)
    for name, want, have in zip(qx.output_names(CFG), expected, got):
        assert torch.allclose(want, torch.from_numpy(have), atol=1e-4), name
