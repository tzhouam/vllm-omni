"""Is the cheaper GELU actually cheaper on the Hexagon NPU?

The erf form is 11-13% of every layer in the device profile. tanh holds 65.2 dB
against the reference with top-1/top-5 intact, so it is adoptable if it buys
time. This measures whether it does.
"""
import json
import qai_hub as hub
HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
OPT = "--target_runtime qnn_context_binary --quantize_full_type float16"
res = {}
mdl = hub.upload_model(f"{HERE}/onnx/ring_sliding_tanh.onnx", name="spark25_sliding_tanh")
cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                            name="spark sliding ring tanh fp16", options=OPT)
print("compile", cj.job_id, flush=True)
st = cj.wait(); print("compile", st.code, (st.message or "")[:160], flush=True)
if st.success:
    pj = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(DEV),
                                name="spark sliding ring tanh fp16",
                                options="--compute_unit npu")
    st2 = pj.wait()
    if st2.success:
        p = pj.download_profile()
        res["ring_sliding_tanh_fp16"] = p["execution_summary"]["estimated_inference_time"] / 1e3
        json.dump(p, open(f"{HERE}/profile_ring_sliding_tanh_fp16.json", "w"))
        print(f"  tanh: {res['ring_sliding_tanh_fp16']:.3f} ms", flush=True)
    else:
        print("  profile FAILED:", (st2.message or "")[:200], flush=True)
json.dump(res, open(f"{HERE}/aihub_gelu.json", "w"), indent=1)
print("done", flush=True)
