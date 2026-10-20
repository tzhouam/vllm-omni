"""int8 activations: does halving KV-cache traffic pay?

Input -- reading the layer's KV cache -- is 13.4% of a sliding layer, and at
w4a16 the cache crosses the boundary as int16. w8a8 halves that and uses int8
MACs. Prior work in this repo found automatic int8 numerically broken on two
other model families, so this measures speed first and only then asks whether
the fidelity is survivable.
"""
import json
import qai_hub as hub
HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
CAND = {
    "w8a8": "--target_runtime qnn_context_binary --quantize_full_type w8a8 --quantize_io",
    "w4a8": "--target_runtime qnn_context_binary --quantize_full_type w4a8 --quantize_io",
}
mdl = hub.upload_model(f"{HERE}/onnx/ring3_sliding.onnx", name="spark25_ring3_a8")
pend, res = {}, {}
for q, o in CAND.items():
    try:
        cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                    name=f"spark ring3 {q}", options=o)
        pend[q] = cj; print("compile", q, cj.job_id, flush=True)
    except Exception as e:
        print("submit failed", q, str(e)[:180], flush=True)
prof = {}
for q, cj in pend.items():
    st = cj.wait(); print("compile", q, st.code, (st.message or "")[:180], flush=True)
    if st.success:
        prof[q] = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(DEV),
                                         name=f"spark ring3 {q}", options="--compute_unit npu")
for q, pj in prof.items():
    st = pj.wait()
    if st.success:
        p = pj.download_profile()
        res[q] = p["execution_summary"]["estimated_inference_time"]/1e3
        json.dump(p, open(f"{HERE}/profile_ring3_{q}.json","w"))
        print(f"  ring3 {q}: {res[q]:.3f} ms", flush=True)
    else:
        res[q] = None; print(f"  {q} FAILED:", (st.message or "")[:180], flush=True)
    json.dump(res, open(f"{HERE}/aihub_w8a8.json","w"), indent=1)
print("done", flush=True)
