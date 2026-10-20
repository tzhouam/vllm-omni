"""w4a16 retry for the ring layout, after expressing the new-token term as a
matmul instead of a two-axis broadcast multiply (which QNN's context-binary
converter rejected at w4a16 while accepting it at fp16)."""
import json
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"
OPT = "--target_runtime qnn_context_binary --quantize_full_type w4a16 --quantize_io"
NAMES = ["ring_sliding_ctx1024", "ring_full_ctx1024", "ring_full_ctx4096",
         ]

pend, res = {}, {}
for n in NAMES:
    mdl = hub.upload_model(f"{HERE}/onnx/{n}.onnx", name=f"spark25_{n}_v3")
    try:
        cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                    name=f"spark {n} w4a16 v3", options=OPT)
        pend[n] = cj
        print("compile", n, cj.job_id, flush=True)
    except Exception as e:
        print("submit failed", n, str(e)[:200], flush=True)

prof = {}
for n, cj in pend.items():
    st = cj.wait()
    print("compile", n, st.code, (st.message or "")[:200], flush=True)
    if st.success:
        prof[n] = hub.submit_profile_job(model=cj.get_target_model(),
                                         device=hub.Device(DEV),
                                         name=f"spark {n} w4a16 v3",
                                         options="--compute_unit npu")
for n, pj in prof.items():
    st = pj.wait()
    if st.success:
        p = pj.download_profile()
        res[n] = p["execution_summary"]["estimated_inference_time"] / 1e3
        json.dump(p, open(f"{HERE}/profile_{n}_w4a16.json", "w"))
        print(f"  {n}: {res[n]:.3f} ms", flush=True)
    else:
        res[n] = None
        print(f"  {n} FAILED: {(st.message or '')[:250]}", flush=True)
    json.dump(res, open(f"{HERE}/aihub_ring_w4_v3.json", "w"), indent=1)
print("done", flush=True)
