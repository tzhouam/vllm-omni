"""The architecture's repeating 4-layer block (3 sliding + 1 full) on a real S25.

Profiling one layer at a time pays the graph's Input/Output boundary once per
layer. A real decode step pays it once for all 28. Measuring 1 layer and 4
layers at the same context separates the fixed per-graph cost from the true
per-layer cost, so the 28-layer step can be extrapolated honestly.
"""
import json
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
OPT = {
    "fp16": "--target_runtime qnn_context_binary --quantize_full_type float16",
    "w4a16": "--target_runtime qnn_context_binary --quantize_full_type w4a16 --quantize_io",
}

res, pend = {}, {}
mdl = hub.upload_model(f"{HERE}/onnx/ring_block4_ctx1024.onnx", name="spark25_block4")
print("uploaded", flush=True)
for q in ("w4a16", "fp16"):
    try:
        cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                    name=f"spark block4 {q}", options=OPT[q])
        pend[q] = cj
        print("compile", q, cj.job_id, flush=True)
    except Exception as e:
        print("submit failed", q, str(e)[:200], flush=True)

prof = {}
for q, cj in pend.items():
    st = cj.wait()
    print("compile", q, st.code, (st.message or "")[:200], flush=True)
    if st.success:
        prof[q] = hub.submit_profile_job(model=cj.get_target_model(),
                                         device=hub.Device(DEV),
                                         name=f"spark block4 {q}",
                                         options="--compute_unit npu")
for q, pj in prof.items():
    st = pj.wait()
    if st.success:
        p = pj.download_profile()
        res[q] = p["execution_summary"]["estimated_inference_time"] / 1e3
        json.dump(p, open(f"{HERE}/profile_block4_{q}.json", "w"))
        print(f"  block4 {q}: {res[q]:.3f} ms", flush=True)
    else:
        res[q] = None
        print(f"  block4 {q} FAILED: {(st.message or '')[:250]}", flush=True)
    json.dump(res, open(f"{HERE}/aihub_block4.json", "w"), indent=1)
print("done", flush=True)
