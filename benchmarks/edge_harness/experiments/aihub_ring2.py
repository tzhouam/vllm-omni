"""Ring sliding layer, concat-free (online softmax), at w4a16 and fp16."""
import json
import qai_hub as hub
HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
OPT = {
    "w4a16": "--target_runtime qnn_context_binary --quantize_full_type w4a16 --quantize_io",
    "fp16": "--target_runtime qnn_context_binary --quantize_full_type float16",
}
mdl = hub.upload_model(f"{HERE}/onnx/ring2_sliding_ctx1024.onnx", name="spark25_ring2_sliding")
pend, res = {}, {}
for q, o in OPT.items():
    cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                name=f"spark ring2 sliding {q}", options=o)
    pend[q] = cj; print("compile", q, cj.job_id, flush=True)
prof = {}
for q, cj in pend.items():
    st = cj.wait(); print("compile", q, st.code, (st.message or "")[:180], flush=True)
    if st.success:
        prof[q] = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(DEV),
                                         name=f"spark ring2 sliding {q}",
                                         options="--compute_unit npu")
for q, pj in prof.items():
    st = pj.wait()
    if st.success:
        p = pj.download_profile()
        res[q] = p["execution_summary"]["estimated_inference_time"]/1e3
        json.dump(p, open(f"{HERE}/profile_ring2_sliding_{q}.json","w"))
        print(f"  ring2 sliding {q}: {res[q]:.3f} ms", flush=True)
    else:
        res[q] = None; print(f"  {q} FAILED:", (st.message or "")[:200], flush=True)
    json.dump(res, open(f"{HERE}/aihub_ring2.json","w"), indent=1)
print("done", flush=True)
