"""The ring decode layers across every processor the S25 exposes.

llama.cpp cannot be run on an AI Hub device, so the honest phone-CPU
reference is the same graph compiled for TFLite and run on the CPU. That
gives an NPU-vs-CPU ratio on identical work, on identical silicon.
"""
import json
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
GRAPHS = ["ring_sliding_ctx1024", "ring_full_ctx1024"]
TARGETS = [
    ("qnn_npu", "--target_runtime qnn_context_binary --quantize_full_type float16", "npu"),
    ("tflite_npu", "--target_runtime tflite", "npu"),
    ("tflite_cpu", "--target_runtime tflite", "cpu"),
    ("tflite_gpu", "--target_runtime tflite", "gpu"),
    ("onnx_npu", "--target_runtime onnx", "npu"),
]

pend, res = {}, {}
for g in GRAPHS:
    mdl = hub.upload_model(f"{HERE}/onnx/{g}.onnx", name=f"spark25_{g}_proc")
    for name, opt, unit in TARGETS:
        try:
            cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                        name=f"spark {g} {name}", options=opt)
            pend[(g, name, unit)] = cj
            print("compile", g, name, cj.job_id, flush=True)
        except Exception as e:
            print("submit failed", g, name, str(e)[:140], flush=True)

prof = {}
for key, cj in pend.items():
    st = cj.wait()
    if st.success:
        prof[key] = hub.submit_profile_job(model=cj.get_target_model(),
                                           device=hub.Device(DEV),
                                           name=f"spark {key[0]} {key[1]}",
                                           options=f"--compute_unit {key[2]}")
    else:
        print("compile", key, "FAILED", (st.message or "")[:140], flush=True)
for key, pj in prof.items():
    st = pj.wait()
    tag = f"{key[0]}|{key[1]}"
    if st.success:
        p = pj.download_profile()
        res[tag] = p["execution_summary"]["estimated_inference_time"] / 1e3
        print(f"  {tag}: {res[tag]:.3f} ms", flush=True)
    else:
        res[tag] = None
        print(f"  {tag} FAILED: {(st.message or '')[:160]}", flush=True)
    json.dump(res, open(f"{HERE}/aihub_processors.json", "w"), indent=1)
print("done", flush=True)
