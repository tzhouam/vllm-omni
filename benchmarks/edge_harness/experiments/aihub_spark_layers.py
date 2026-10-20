"""Profile Spark-X2.5 decoder layers on real Snapdragon hardware.

One sliding layer and one full-attention layer are profiled separately. The
sliding layer's cache is a fixed 512 entries at any context; the full layer's
grows, so it is profiled at 1024 and 4096 to show the slope. A whole decode
step is 21 sliding + 7 full layers plus the head.
"""
import json, sys
import qai_hub as hub

HERE = "/data/zhoutaichang/embedding_infer/analysis/experiments/spark_edge"
HID, HEADS, KV, D = 2048, 8, 2, 256
SW = 512

GRAPHS = [
    ("sliding", 1024, SW),
    ("full", 1024, 1024),
    ("full", 4096, 4096),
]
QUANTS = [
    ("fp16", "--target_runtime qnn_context_binary --quantize_full_type float16"),
    ("w8a16", "--target_runtime qnn_context_binary --quantize_full_type w8a16 --quantize_io"),
    ("w4a16", "--target_runtime qnn_context_binary --quantize_full_type w4a16 --quantize_io"),
]
DEVICES = ["Samsung Galaxy S25"]


def spec_for(ctx, cache_len):
    s = {
        "x": ((1, 1, HID), "float32"),
        "cos_sw": ((1, 1, 1, D), "float32"),
        "sin_sw": ((1, 1, 1, D), "float32"),
        "cos_full": ((1, 1, 1, D // 4), "float32"),
        "sin_full": ((1, 1, 1, D // 4), "float32"),
        "mask_sw": ((1, 1, 1, SW), "float32"),
        "mask_full": ((1, 1, 1, ctx + 1), "float32"),
        "k_cache_0": ((1, KV, cache_len, D), "float32"),
        "v_cache_0": ((1, KV, cache_len, D), "float32"),
    }
    return s


def main():
    results, pending = {}, {}
    for kind, ctx, cache_len in GRAPHS:
        path = f"{HERE}/onnx/spark_{kind}_ctx{ctx}.onnx"
        mdl = hub.upload_model(path, name=f"spark25_{kind}_ctx{ctx}")
        print("uploaded", kind, ctx, flush=True)
        for dev in DEVICES:
            for q, opt in QUANTS:
                key = (kind, ctx, dev, q)
                try:
                    # No input_specs: every shape is static, and the ONNX
                    # export prunes the inputs a given layer type never
                    # reads (a full layer has no sliding mask), so a
                    # hand-written spec would not match the graph.
                    cj = hub.submit_compile_job(
                        model=mdl, device=hub.Device(dev),
                        name=f"spark {kind} ctx{ctx} {q}", options=opt)
                    pending[key] = cj
                    print("compile submitted", key, cj.job_id, flush=True)
                except Exception as e:
                    print("submit failed", key, str(e)[:160], flush=True)

    prof = {}
    for key, cj in pending.items():
        st = cj.wait()
        print("compile", key, st.code, (st.message or "")[:160], flush=True)
        if st.success:
            prof[key] = hub.submit_profile_job(
                model=cj.get_target_model(), device=hub.Device(key[2]),
                name=f"spark {key[0]} ctx{key[1]} {key[3]}",
                options="--compute_unit npu")

    for key, pj in prof.items():
        st = pj.wait()
        kind, ctx, dev, q = key
        tag = f"{kind}|ctx{ctx}|{dev}|{q}"
        if st.success:
            p = pj.download_profile()
            ms = p["execution_summary"]["estimated_inference_time"] / 1e3
            results[tag] = ms
            json.dump(p, open(f"{HERE}/profile_spark_{kind}_{ctx}_{q}.json", "w"))
            print(f"  {tag}: {ms:.3f} ms/layer", flush=True)
        else:
            results[tag] = None
            print(f"  {tag} FAILED: {(st.message or '')[:200]}", flush=True)
        json.dump(results, open(f"{HERE}/aihub_spark_layers.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
