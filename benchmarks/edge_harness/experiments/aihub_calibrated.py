"""Calibrated quantisation of the sliding layer, judged on real activations.

Uncalibrated w4a16/w4a8 scored -1.6 / -1.28 dB against the reference -- output
uncorrelated with the truth. This feeds the quantiser real activations
captured from the reference model on real text, then measures the same way.
"""
import json
import numpy as np
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
NAMES = ["x", "cos_sw", "sin_sw", "mask_sw", "k_cache_0", "v_cache_0"]

d = np.load(f"{HERE}/ring3_real_acts.npz")
n_samples = d["x"].shape[0]
calib = {n: [d[n][i] for i in range(n_samples)] for n in NAMES}
probe = {n: [d[n][0]] for n in NAMES}
ref_out = d["hidden_out"][0]

SCHEMES = {
    "w4a16": (hub.QuantizeDtype.INT4, hub.QuantizeDtype.INT16),
    "w8a16": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT16),
    "w4a8": (hub.QuantizeDtype.INT4, hub.QuantizeDtype.INT8),
    "w8a8": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT8),
}


def snr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(20 * np.log10(np.linalg.norm(a) / max(np.linalg.norm(a - b), 1e-12)))


src = hub.upload_model(f"{HERE}/onnx/ring3_sliding.onnx", name="spark25_calib_src")
qjobs = {}
for tag, (w, a) in SCHEMES.items():
    qj = hub.submit_quantize_job(model=src, calibration_data=calib,
                                 weights_dtype=w, activations_dtype=a,
                                 name=f"spark calib {tag}")
    qjobs[tag] = qj
    print("quantize", tag, qj.job_id, flush=True)

res = {}
cjobs = {}
for tag, qj in qjobs.items():
    st = qj.wait()
    print("quantize", tag, st.code, (st.message or "")[:150], flush=True)
    if st.success:
        cjobs[tag] = hub.submit_compile_job(
            model=qj.get_target_model(), device=hub.Device(DEV),
            name=f"spark calib {tag}",
            options="--target_runtime qnn_context_binary --quantize_io")

ijobs, pjobs = {}, {}
for tag, cj in cjobs.items():
    st = cj.wait()
    print("compile", tag, st.code, (st.message or "")[:150], flush=True)
    if st.success:
        tgt = cj.get_target_model()
        ijobs[tag] = hub.submit_inference_job(model=tgt, device=hub.Device(DEV),
                                              inputs=probe, name=f"spark calib {tag}")
        pjobs[tag] = hub.submit_profile_job(model=tgt, device=hub.Device(DEV),
                                            name=f"spark calib {tag}",
                                            options="--compute_unit npu")

for tag in list(ijobs):
    st = ijobs[tag].wait()
    entry = res.setdefault(tag, {})
    if st.success:
        out = ijobs[tag].download_output_data()
        got = np.asarray(list(out.values())[0][0])
        entry["snr_db"] = round(snr(ref_out, got), 2)
    else:
        entry["snr_db"] = None
        print(tag, "inference FAILED", (st.message or "")[:150], flush=True)
    st = pjobs[tag].wait()
    if st.success:
        p = pjobs[tag].download_profile()
        entry["ms"] = p["execution_summary"]["estimated_inference_time"] / 1e3
        json.dump(p, open(f"{HERE}/profile_calib_{tag}.json", "w"))
    else:
        entry["ms"] = None
    print(f"  {tag}: {entry}", flush=True)
    json.dump(res, open(f"{HERE}/aihub_calibrated.json", "w"), indent=1)
print("done", flush=True)
