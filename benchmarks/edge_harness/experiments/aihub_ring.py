"""Ring-buffer decode step on a real Galaxy S25, plus the output head.

The roll layout spent ~20% of a sliding layer writing the recomputed window
back out. The ring layout returns one K/V entry instead, so this measures
what that bought. The output head (2048 x 131072, tied to the embedding) was
missing from the earlier per-layer numbers entirely and is measured here.
"""
import json
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
DEV = "Samsung Galaxy S25"

JOBS = [
    ("ring_sliding_ctx1024", ["fp16", "w4a16"]),
    ("ring_full_ctx1024", ["fp16", "w4a16"]),
    ("ring_full_ctx4096", ["w4a16"]),
    ("lm_head", ["w4a16", "w8a16"]),
]
OPT = {
    "fp16": "--target_runtime qnn_context_binary --quantize_full_type float16",
    "w8a16": "--target_runtime qnn_context_binary --quantize_full_type w8a16 --quantize_io",
    "w4a16": "--target_runtime qnn_context_binary --quantize_full_type w4a16 --quantize_io",
}


def main():
    pend = {}
    for name, quants in JOBS:
        mdl = hub.upload_model(f"{HERE}/onnx/{name}.onnx", name=f"spark25_{name}")
        print("uploaded", name, flush=True)
        for q in quants:
            try:
                cj = hub.submit_compile_job(model=mdl, device=hub.Device(DEV),
                                            name=f"spark {name} {q}", options=OPT[q])
                pend[(name, q)] = cj
                print("compile", name, q, cj.job_id, flush=True)
            except Exception as e:
                print("submit failed", name, q, str(e)[:160], flush=True)

    prof, res = {}, {}
    for key, cj in pend.items():
        st = cj.wait()
        print("compile", key, st.code, (st.message or "")[:150], flush=True)
        if st.success:
            prof[key] = hub.submit_profile_job(
                model=cj.get_target_model(), device=hub.Device(DEV),
                name=f"spark {key[0]} {key[1]}", options="--compute_unit npu")
    for key, pj in prof.items():
        st = pj.wait()
        tag = f"{key[0]}|{key[1]}"
        if st.success:
            p = pj.download_profile()
            res[tag] = p["execution_summary"]["estimated_inference_time"] / 1e3
            json.dump(p, open(f"{HERE}/profile_{key[0]}_{key[1]}.json", "w"))
            print(f"  {tag}: {res[tag]:.3f} ms", flush=True)
        else:
            res[tag] = None
            print(f"  {tag} FAILED: {(st.message or '')[:200]}", flush=True)
        json.dump(res, open(f"{HERE}/aihub_ring.json", "w"), indent=1)
    print("done", flush=True)


main()
