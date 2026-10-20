"""Calibrated quantisation + on-device parity for any exported graph."""
import argparse, json
import numpy as np
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
SCHEMES = {
    "w8a16": (hub.QuantizeDtype.INT8, hub.QuantizeDtype.INT16),
    "w16a16": (hub.QuantizeDtype.INT16, hub.QuantizeDtype.INT16),
}


def snr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(20 * np.log10(np.linalg.norm(a) / max(np.linalg.norm(a - b), 1e-12)))


ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--acts", required=True)
ap.add_argument("--map", default="", help="onnx_name=npz_name,...")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import onnx
g = onnx.load(a.onnx, load_external_data=False)
live = [t.name for t in g.graph.input]
mapping = dict(p.split("=") for p in a.map.split(",") if p)
d = np.load(a.acts)
n = d["x"].shape[0]
calib = {name: [d[mapping.get(name, name)][i] for i in range(n)] for name in live}
probe = {name: [d[mapping.get(name, name)][0]] for name in live}
ref_out = d["hidden_out"][0]

src = hub.upload_model(a.onnx, name=f"spark25_{a.tag}_src")
qj = {}
for tag, (w, act) in SCHEMES.items():
    qj[tag] = hub.submit_quantize_job(model=src, calibration_data=calib,
                                      weights_dtype=w, activations_dtype=act,
                                      name=f"spark {a.tag} {tag}")
    print("quantize", tag, qj[tag].job_id, flush=True)

res, cj = {}, {}
for tag, j in qj.items():
    st = j.wait()
    print("quantize", tag, st.code, (st.message or "")[:140], flush=True)
    if st.success:
        cj[tag] = hub.submit_compile_job(
            model=j.get_target_model(), device=hub.Device(DEV),
            name=f"spark {a.tag} {tag}",
            options="--target_runtime qnn_context_binary --quantize_io")
ij, pj = {}, {}
for tag, j in cj.items():
    st = j.wait()
    print("compile", tag, st.code, (st.message or "")[:140], flush=True)
    if st.success:
        t = j.get_target_model()
        ij[tag] = hub.submit_inference_job(model=t, device=hub.Device(DEV),
                                           inputs=probe, name=f"spark {a.tag} {tag}")
        pj[tag] = hub.submit_profile_job(model=t, device=hub.Device(DEV),
                                         name=f"spark {a.tag} {tag}",
                                         options="--compute_unit npu")
for tag in list(ij):
    e = res.setdefault(tag, {})
    st = ij[tag].wait()
    if st.success:
        out = ij[tag].download_output_data()
        e["snr_db"] = round(snr(ref_out, np.asarray(list(out.values())[0][0])), 2)
    else:
        e["snr_db"] = None
    st = pj[tag].wait()
    e["ms"] = (pj[tag].download_profile()["execution_summary"]
               ["estimated_inference_time"] / 1e3) if st.success else None
    print(f"  {a.tag} {tag}: {e}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
print("done", flush=True)
