# Result artifacts

Raw output backing the published figures. Every file written by the current
harness carries a `provenance` block — model path, quantization, thread
affinity, build SHAs, host, environment — so a number can be traced to the run
that produced it and invalidated later if the run turns out to be wrong.

```bash
python -c "import json,sys; print(json.load(open(sys.argv[1]))['provenance'])" <file>.json
```

## Naming

| prefix | contents |
|---|---|
| `profile_*` | per-op decode profiles (`profile_cpu_step.py`) |
| `parity_*`, `export_parity_*` | numerical parity checks against a reference |
| `quality_*`, `cmp_*` | greedy-divergence fidelity |
| `aihub_*` | Qualcomm AI Hub on-device runs (Snapdragon) |
| `*_grid.jsonl` | interleaved A/B grids |
| `s25_*` | Galaxy S25 device estimates |

## Retracted files are kept, not deleted

Anything with `"retracted": true` names the reason and its replacement.
`profile_w4a8_g128_p{1,2}.json` are the clearest case: both show
`_C::onednn_mm` at ~2.97 ms/step — the signature of a checkpoint whose output
head was left in bf16 — and every W4A8 number derived from them is void. The
valid replacement is `profile_w4a8_g128_head_p1.json`.

They are kept because a deleted mistake teaches nothing, and because the older
reports still cite them.
