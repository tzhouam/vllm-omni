# Edge Agent download gate

`download.py` transports only the three reviewed, SHA-256-bound manifests when
`--candidate` names their exact catalog lineage. It checks **live available**
RAM, one discrete GPU's free VRAM, and destination disk before reading an HF
token or opening a download. Shared iGPU/NPU memory is not counted twice. A
Windows host RAM reading also limits WSL; if that reading fails, WSL admission
fails closed. The weight-size calculation is an **E lower bound**, not a model
loading or whole-request peak.

Normal network mode requires `--runtime-index` from `native_profile.py`. The
gate verifies the original config and lineage hashes, the current native
Windows server executable SHA-256, the local model/projector file SHA-256, the raw trace
SHA-256, the currently imported Omni runtime source hash, cold load, and at
least one measured, complete batch-1 Agent request.
A vision candidate must have a successful visual request in that trace. A
passing download gate does **not** qualify a default route, independent device
placement, latency, safety, or quality beyond that fixed request. Qwen3-30B's
catalog entry now uses the official pinned 18,556,685,824-byte GGUF and exact
artifact revision; its original 18.6 GB plan size is retained separately.
Its base checkpoint commit is still unknown.

This is not initial acquisition for all 22 catalog entries. Only three
manifests are reviewed and pinned here. A Gemma text-only native profile does
not prove its image projector, so it cannot authorize a fresh strict download
of the multimodal bundle. The strict tier also refuses claimed CPU/GPU
offload when the native trace lacks independent placement verification.
The browser profile was sealed before this catalog source change. Its raw
results remain historical evidence. Strict current-source matching requires
a new native profile under the changed runtime source hash.

For Qwen3-30B's first weights only, `--research-download PATH` accepts the
[bounded architecture diagnostic](experiments/GGUF_ARCHITECTURE.md) instead
of `--runtime-index`. It verifies the pinned one-file manifest, diagnostic
record hash, local DLL hash, and reproducible architecture string scan. The
result is labeled `quarantined_research_download_architecture_unverified`;
it establishes neither a successful load nor runtime operator support. Live
capacity and disk staging still must pass. No other candidate has a research
download path until an equally explicit diagnostic and manifest review exists.

`--check` remains an offline file/hash/disk diagnostic without `--candidate`
or runtime proof. With `--candidate`, it verifies the manifest binding but
reports `reviewed_binding_only_download_not_eligible` because live capacity
and runtime execution were not checked. Passing `--check` does not authorize
a network download.
`--reset-partial` remains local maintenance.

On 2026-10-06, a native Windows `--check` of the pinned Qwen3-30B manifest
returned `reviewed_binding_only_download_not_eligible` and left a missing
destination uncreated. A strict Gemma download attempt using the earlier
fixed-memory profile was refused before destination creation because the
catalog edit changed the Omni source hash. These are CLI boundary checks;
neither downloaded weights nor ran a model.

Example strict invocation:

```powershell
python -X utf8 -m benchmarks.edge_agent.download `
  --candidate gemma4-31b-qat-q4-0 `
  --manifest benchmarks/edge_agent/configs/gemma4_31b_qat_q4_download.json `
  --runtime-index C:\path\to\native_profile\index.json `
  --dest C:\path\to\models\gemma4
```

For first-weight Qwen3-30B research, replace the candidate/manifest values
and `--runtime-index` with `--research-download C:\path\to\probe.json`.
The diagnostic is intentionally weaker; the resulting files must be verified
and separately tested by the native runtime before any measurement claim.
