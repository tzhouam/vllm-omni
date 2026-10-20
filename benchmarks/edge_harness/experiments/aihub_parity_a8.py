"""On-device fidelity of the quantized sliding layer.

Speed is only half the answer: this repo has already found automatic int8
numerically broken on Qwen3-TTS and MiniCPM-o. This runs the real layer on a
real S25 at each quantisation and compares against the fp32 reference.
"""
import json
import numpy as np
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
OPT = {
    "fp16": "--target_runtime qnn_context_binary --quantize_full_type float16",
    "w4a16": "--target_runtime qnn_context_binary --quantize_full_type w4a16 --quantize_io",
    "w4a8": "--target_runtime qnn_context_binary --quantize_full_type w4a8 --quantize_io",
}

ref = np.load(f"{HERE}/ring3_ref_inputs.npz")
# The graph's live inputs; ONNX prunes the full-attention rotary/mask tensors
# a sliding layer never reads.
names = ["x", "cos_sw", "sin_sw", "mask_sw", "k_cache_0", "v_cache_0"]
inputs = {n: [ref[n].astype(np.float32)] for n in names}
ref_out = ref["hidden_out"]


def snr(a, b):
    a, b = a.ravel(), b.ravel()
    return float(20 * np.log10(np.linalg.norm(a) / max(np.linalg.norm(a - b), 1e-12)))


mdl = hub.upload_model(f"{HERE}/onnx/ring3_sliding.onnx", name="spark25_ring3_parity")
res, pend = {}, {}
for q, o in OPT.items():
    cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                name=f"spark parity {q}", options=o)
    pend[q] = cj
    print("compile", q, cj.job_id, flush=True)
infj = {}
for q, cj in pend.items():
    st = cj.wait()
    if st.success:
        infj[q] = hub.submit_inference_job(model=cj.get_target_model(),
                                           device=hub.Device(DEV), inputs=inputs,
                                           name=f"spark parity {q}")
    else:
        print("compile", q, "FAILED", (st.message or "")[:160], flush=True)
for q, ij in infj.items():
    st = ij.wait()
    if st.success:
        out = ij.download_output_data()
        key = [k for k in out if "hidden" in k or "output" in k.lower()] or list(out)
        got = np.asarray(out[key[0]][0])
        res[q] = round(snr(ref_out, got), 2)
        print(f"  {q}: {res[q]} dB", flush=True)
    else:
        res[q] = None
        print(f"  {q} inference FAILED:", (st.message or "")[:160], flush=True)
    json.dump(res, open(f"{HERE}/aihub_parity_a8.json", "w"), indent=1)
print("done", flush=True)
