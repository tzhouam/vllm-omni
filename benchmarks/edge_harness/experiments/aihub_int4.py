"""int4 weights on the NPU, calibrated.

int4 halves the weight traffic that dominates every matmul in the step, so it
is the largest remaining lever. The compile-path form (--quantize_full_type
w4a16) runs at 0.838 ms but is uncalibrated and scores -1.6 dB. The
quantize-job form carries calibration but failed to compile. This retries that
compile without --quantize_io, which is the one thing the failing jobs had in
common with an already-quantized input model.
"""
import argparse, json
import numpy as np
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default=f"{HERE}/onnx/ring3_sliding.onnx")
ap.add_argument("--acts", default=f"{HERE}/ring3_real_acts.npz")
ap.add_argument("--map", default="")
ap.add_argument("--tag", default="sliding")
ap.add_argument("--out", default=f"{HERE}/aihub_int4.json")
a = ap.parse_args()

import onnx
live = [t.name for t in onnx.load(a.onnx, load_external_data=False).graph.input]
mapping = dict(p.split("=") for p in a.map.split(",") if p)
d = np.load(a.acts)
n = d["x"].shape[0]
calib = {x: [d[mapping.get(x, x)][i] for i in range(n)] for x in live}
probe = {x: [d[mapping.get(x, x)][0]] for x in live}
ref = d["hidden_out"][0]


def snr(u, v):
    u, v = np.asarray(u).ravel(), np.asarray(v).ravel()
    return float(20 * np.log10(np.linalg.norm(u) / max(np.linalg.norm(u - v), 1e-12)))


src = hub.upload_model(a.onnx, name=f"spark25_{a.tag}_int4")
qj = hub.submit_quantize_job(model=src, calibration_data=calib,
                             weights_dtype=hub.QuantizeDtype.INT4,
                             activations_dtype=hub.QuantizeDtype.INT16,
                             name=f"spark {a.tag} int4 calib")
print("quantize", qj.job_id, flush=True)
st = qj.wait()
print("quantize", st.code, (st.message or "")[:200], flush=True)
res = {}
if st.success:
    qm = qj.get_target_model()
    # The failing attempts all passed --quantize_io on an already-quantized
    # model; try without it, and with the plain QNN DLC target as a fallback.
    VARIANTS = {
        "ctx_no_qio": "--target_runtime qnn_context_binary",
        "ctx_qio": "--target_runtime qnn_context_binary --quantize_io",
    }
    cj = {}
    for tag, opt in VARIANTS.items():
        try:
            cj[tag] = hub.submit_compile_job(model=qm, device=hub.Device(DEV),
                                             name=f"spark {a.tag} int4 {tag}",
                                             options=opt)
            print("compile", tag, cj[tag].job_id, flush=True)
        except Exception as e:
            print("submit failed", tag, str(e)[:160], flush=True)
    for tag, j in cj.items():
        s2 = j.wait()
        print("compile", tag, s2.code, (s2.message or "")[:200], flush=True)
        e = res.setdefault(tag, {})
        if not s2.success:
            e["ms"] = e["snr_db"] = None
            json.dump(res, open(a.out, "w"), indent=1)
            continue
        t = j.get_target_model()
        ij = hub.submit_inference_job(model=t, device=hub.Device(DEV), inputs=probe,
                                      name=f"spark {a.tag} int4 {tag}")
        pj = hub.submit_profile_job(model=t, device=hub.Device(DEV),
                                    name=f"spark {a.tag} int4 {tag}",
                                    options="--compute_unit npu")
        s3 = ij.wait()
        e["snr_db"] = round(snr(ref, np.asarray(list(ij.download_output_data().values())[0][0])), 2) if s3.success else None
        s4 = pj.wait()
        e["ms"] = pj.download_profile()["execution_summary"]["estimated_inference_time"] / 1e3 if s4.success else None
        print(f"  {a.tag} int4 {tag}: {e}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
print("done", flush=True)
