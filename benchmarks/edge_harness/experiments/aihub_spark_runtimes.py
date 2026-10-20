"""Same Spark layer, three phone runtimes, one device.

Isolates runtime choice from export choice: if QNN, TFLite and ONNX Runtime
land close together, the export layout is what sets on-device speed, not the
engine. Run on the sliding layer, which is 21 of Spark's 28 layers.
"""
import json
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
PATH = f"{HERE}/onnx/spark_sliding_ctx1024.onnx"

# Each runtime spells precision differently: only QNN takes
# --quantize_full_type, TFLite and ONNX reject it outright, and a QNN context
# binary is NPU-only so the GPU leg has to go through TFLite.
RUNTIMES = [
    ("qnn_npu", "--target_runtime qnn_context_binary --quantize_full_type float16", "npu"),
    ("tflite_npu", "--target_runtime tflite", "npu"),
    ("onnx_npu", "--target_runtime onnx", "npu"),
    ("tflite_gpu", "--target_runtime tflite", "gpu"),
    ("tflite_cpu", "--target_runtime tflite", "cpu"),
]


def main():
    mdl = hub.upload_model(PATH, name="spark25_sliding_runtimes")
    pend = {}
    for name, opt, unit in RUNTIMES:
        try:
            cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                        name=f"spark sliding {name}", options=opt)
            pend[name] = (cj, unit)
            print("compile", name, cj.job_id, flush=True)
        except Exception as e:
            print("submit failed", name, str(e)[:160], flush=True)

    prof, res = {}, {}
    for name, (cj, unit) in pend.items():
        st = cj.wait()
        print("compile", name, st.code, (st.message or "")[:140], flush=True)
        if st.success:
            prof[name] = hub.submit_profile_job(
                model=cj.get_target_model(), device=hub.Device(DEV),
                name=f"spark sliding {name}", options=f"--compute_unit {unit}")
    for name, pj in prof.items():
        st = pj.wait()
        if st.success:
            p = pj.download_profile()
            res[name] = p["execution_summary"]["estimated_inference_time"] / 1e3
            print(f"  {name}: {res[name]:.3f} ms/layer", flush=True)
        else:
            res[name] = None
            print(f"  {name} FAILED: {(st.message or '')[:180]}", flush=True)
        json.dump(res, open(f"{HERE}/aihub_spark_runtimes.json", "w"), indent=1)
    print("done", flush=True)


main()
