"""A quantized phone-CPU reference for the same layer.

llama.cpp cannot be run on an AI Hub device, and it would use 4-bit weights on
the CPU, so comparing our NPU path against an fp16 CPU run would flatter us.
This asks for the closest thing AI Hub offers: the same graph, int8, on the
S25's CPU.
"""
import json
import qai_hub as hub
HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
CAND = [
    ("tflite_int8_cpu", "--target_runtime tflite --quantize_full_type int8 --quantize_io", "cpu"),
    ("tflite_w8a8_cpu", "--target_runtime tflite --quantize_full_type w8a8 --quantize_io", "cpu"),
]
res, pend = {}, {}
mdl = hub.upload_model(f"{HERE}/onnx/ring3_sliding.onnx", name="spark25_ring3_cpu")
for name, opt, unit in CAND:
    try:
        cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                    name=f"spark {name}", options=opt)
        pend[(name, unit)] = cj; print("compile", name, cj.job_id, flush=True)
    except Exception as e:
        print("submit failed", name, str(e)[:180], flush=True)
prof = {}
for key, cj in pend.items():
    st = cj.wait(); print("compile", key[0], st.code, (st.message or "")[:180], flush=True)
    if st.success:
        prof[key] = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(DEV),
                                           name=f"spark {key[0]}", options=f"--compute_unit {key[1]}")
for key, pj in prof.items():
    st = pj.wait()
    if st.success:
        p = pj.download_profile()
        res[key[0]] = p["execution_summary"]["estimated_inference_time"]/1e3
        print(f"  {key[0]}: {res[key[0]]:.3f} ms", flush=True)
    else:
        res[key[0]] = None; print(f"  {key[0]} FAILED:", (st.message or "")[:180], flush=True)
    json.dump(res, open(f"{HERE}/aihub_cpu_int8.json","w"), indent=1)
print("done", flush=True)
