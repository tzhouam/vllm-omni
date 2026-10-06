# Bounded GGUF architecture diagnostic

`probe_gguf_architecture.py` checks the official Qwen3-30B-A3B Q4_K_M
candidate **before** downloading its 18,556,685,824-byte GGUF. Its manifest
pins [the artifact commit](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/tree/e4d4bafdfb96a411a163846265362aceb0b9c63a)
and the published file size and LFS SHA-256. The probe requests only byte range
`0..N-1` from that commit, requires a matching HTTP 206 response, reads no more
than `N` bytes, and parses the first `general.architecture` field. It then
hashes a *specified, already installed* local llama DLL and searches its bytes
for the exact architecture token. It never executes the DLL or downloads model
weights, and sends no authentication header.

From the repository root, with the native Windows Python environment and this
checkout on `PYTHONPATH`, set `LLAMA_CPP_DLL` to the installed `llama.dll` path:

```powershell
$env:PYTHONPATH = (Resolve-Path -LiteralPath (Get-Location)).Path
$llamaDll = (Resolve-Path -LiteralPath $env:LLAMA_CPP_DLL).Path
New-Item -ItemType Directory -Path 'benchmarks/edge_agent/results' -Force | Out-Null
python -X utf8 -m benchmarks.edge_agent.experiments.probe_gguf_architecture `
  --manifest benchmarks/edge_agent/configs/qwen3_30b_a3b_q4_k_m_download.json `
  --llama-dll $llamaDll `
  --output "benchmarks/edge_agent/results/qwen3_30b_architecture_$([guid]::NewGuid().ToString('N')).json"
```

The default range is 64 KiB and the hard maximum is 1 MiB. The pinned Qwen
header parsed within that 64 KiB range on the Windows test host; a 1 MiB
request returned an incomplete body and was correctly rejected. The output path must
be new; an existing evidence file is never replaced. Omit `--output` to print
JSON. If the server ignores the range, metadata is truncated before the
architecture field, or the response disagrees with the manifest, the command
fails without a success record. A redirect may point to a signed HTTPS Hub
storage endpoint; HTTP redirects are rejected. A misbehaving server can start
transmitting data before its non-206 response is closed, so the enforceable
client guarantee is that the probe does not *read* its body.

The record keeps the exact manifest hash, published whole-file hash (explicitly
**unverified**), requested and received range sizes and range SHA-256, GGUF
field offsets, DLL size and SHA-256, and marker presence. Parsing stops at the
first architecture field; later metadata, tensor tables, and duplicate fields
are not validated. A matching string in the DLL is only a diagnostic marker:
it does not establish that this build can load the GGUF, run all operators,
respect quantization, fit memory, or pass a complete Agent task. Those require
a separately pinned full download and paired native execution tests.

On 2026-10-06, the pinned Qwen range probe received 65,536 bytes (range
SHA-256 `fc48784321689200975bf1a003355da14d0b128d36b04c1fe77e01ade53b2502`),
parsed GGUF v3 `general.architecture=qwen3moe`, and found that exact token in
the installed Windows llama DLL (SHA-256
`69869a1932be0c3bd6002e679926f58c2768bfd7297091cabb919603df69e80c`).
The private diagnostic record is bound by SHA-256
`05ba66d3f92a4a645de18303603813c15a108b013f513769598f9becfed153fd`.
This is a header and binary-marker observation only; full load and inference
remain unverified.
