# `OMNI_EXEC_V1` native fields (v1, source-only)

All names below come from `observer.hpp.in`. Future compiled records must match
exactly; this file is not actual execution output. Max frame buffer is 32,768 B,
including prefix/newline/NUL space. Formatting overflow emits no partial JSON;
the absent required record makes the host observation incomplete.

Top-level object:

| Field | Type / rule |
| --- | --- |
| `schema` | exactly `strata-omni-exec-v1` |
| `native_request_seq` | positive integer, same owner-side counter as I/O V1; `< 2**62` |
| `snapshot_seq` | exact integer 0, 1, 2 in that order |
| `phase` | `request_start`, `prefill_end_decode_start`, `request_end` |
| `clock` | `{source:'Windows_QPC', ticks:positive int or null, frequency_hz:positive int or null}`; both null if unavailable; unavailable sets issue8192 |
| `counter_scope` | `process_lifetime_cumulative_phase_at_submission` |
| `snapshot_coherent` | false |
| `scope_status` | `partial_coverage` always |
| `request_terminal` | null at 0/1; `completed` or `cancelled` at 2; this is native terminal observation, not task quality |
| `boundary_complete` | false at 0/1/cancel; may be true at normal end only with no known sticky issue; does not prove whole coverage/quiescence/quality |
| `issues` | unsigned integer OR bitmask below; sticky across process lifetime |
| `observer` | fixed-overhead object below |
| `inflight` | `{cpu_methods_and_phases:nonnegative int, cuda_replays:nonnegative int}`; independent loads, unresolved end sets issue16 |
| `counters` | exact objects `cpu`, `cpu_phase`, `cuda`, arrays described below |
| `memory` | exact `ordinary_expert_cache` object below |
| `coverage` | literal instrumented/excluded lists from source and `entire_model_covered:false`; exact list identity is source-bound |
| `whole_model_placement` | null |
| `physical_ssd_read_bytes` | null |
| `aggregate_gpu_hard_cap_verified` | false |
| `aggregate_ram_hard_cap_verified` | false |

There is **no** native PID, creation time, process generation, GPU ordinal/UUID,
nonce, stage ID/request ID/epoch, artifact ID, runtime hash or claim of scope
ownership in the raw line. Those belong to the future verified adapter context.

`observer` exact keys:

- `state_bytes`: future compiled `sizeof(State)`, positive and <=16,384.
- `frame_buffer_bytes`: 32,768 exactly.
- `frame_state_bytes`: future compiled `sizeof(Frame)`, at least32,768.
- `ticket_bytes`: future compiled `sizeof(Ticket)`, positive.
- `concurrent_ticket_count_verified`: false.
- `root_registry_capacity`: 16.
- `overhead_scope`: `fixed_Cpp_state_and_one_owner_frame_plus_per_call_stack_ticket_excludes_CRT_stdio`.

All counter arrays are deterministic fixed order: phases
`outside_request`, `prefill`, `decode` (phase-at-submission, not current phase).
These fields are process-lifetime cumulative. Validate each field's exact integer
type/nonnegative range and monotonicity within the same owned native generation.
Do not demand `completed <= submitted` from independent snapshots, count equality
between families, or sum jobs/tasks/replays as one unit. Do not interpret omitted
instrumentation as zero.

`counters.cpu`: 12 records, phase-major then family `full`, `split`,
`split_multi`, `split_multi_native`. Each object has:

- `phase`, `family`, `unit:'CPU_expert_method'`.
- `submitted`, `completed`: method calls.
- `unique_jobs_submitted`, `unique_jobs_completed`: jobs unique within each call;
  repeated experts across calls remain repeated jobs.
- `token_expert_entries_submitted`, `token_expert_entries_completed`: actual
  single-call n or multi-call sum(nt), not a kernel/row/token count.

`counters.cpu_phase`: 18 records, phase-major then exact modes1..6. Each has
`phase`, `mode`, `unit:'CPU_row_partition_phase'`, `submitted`, `completed`,
`row_tasks_submitted`, `row_tasks_completed`. Existing modes1/2 single GU/down,
3/4 multi GU/down,5/6 native multi GU/down. Method completion and row-phase
completion overlap; never add them together as disjoint work.

`counters.cuda`: six records, phase-major then `session_run_token`, `Verifier_run`.
Each has `phase`, `family`,
`unit:'CUDA_graph_replay_including_host_copy_nodes'`, `launch_attempts`,
`submitted`, `completed_after_successful_existing_stream_fence`, `launch_errors`,
`fence_errors`, `last_launch_status`, `last_fence_status`. The five counts are
monotonic cumulative. Status values are the exact original integer CUDA return
value, or null before any such call. Completion is updated only for a successful
original covering stream fence; failed fence leaves in-flight work unresolved.
Capture without replay increments none of these counters.

`memory.ordinary_expert_cache` exact fields:

- `unit:'bytes'`, `accounting:'successful_owned_cudaMalloc_requested_bytes'`.
- Monotonic counts: `allocation_attempts`, `successful_allocations`,
  `allocation_errors`, `free_attempts`, `successful_frees`, `failed_frees`.
- `last_allocation_status`, `last_free_status`: original CUDA integer or null.
- `tracked_live_requested_bytes`: successful ordinary tracked roots less only
  successfully observed frees (gauge; can decrease).
- `lifetime_peak_requested_bytes`: max simultaneous ordinary tracked roots,
  monotonic lifetime high-water.
- `request_peak_requested_bytes`: seeded to current tracked live roots at start,
  nondecreasing within one request; may reset lower at next start.
- `untracked_cache_family_bits`: sticky OR mask 1=segmented,2=VMM attempted;
  those paths are not interpreted as zero requested/mapped/resident bytes.
- `physical_resident_bytes`, `vmm_reserved_bytes`,
  `vmm_mapped_committed_bytes`: null.

The registry has16 fixed private roots; failed frees retain uncertain bytes and
root identity. Alias views are never new tracked roots. No root pointer or serial
is emitted. These facts do not establish a physical GPU peak or allocator cap.

Issue bits (unknown bits must refuse this schema version):

| Bit | Meaning |
| --- | --- |
| 1 | counter saturation/overflow |
| 2 | invalid observed geometry |
| 4 | completion's immutable submission scope differs from current scope |
| 8 | invalid/replayed/out-of-order native boundary |
| 16 | unresolved observed work at boundary, or impossible completion retirement |
| 32 | fixed registry exhausted / unsupported invalid root geometry |
| 64 | successful free missing tracked ordinary root |
| 128 | same owner has another unreleased/uncertain root |
| 256 | ordinary cache allocation API error |
| 512 | ordinary cache free API error |
| 1024 | observed graph launch API error |
| 2048 | observed graph covering fence API error |
| 4096 | bounded frame formatting overflow (record omitted) |
| 8192 | native QPC unavailable |

Scope/clock/ownership failures are never repaired by an eventual OS drain. A
cancelled/missing-end/error request is partial. Sticky errors require a fresh
native generation for a clean subsequent observation. The absence of raw frames
from a legacy or non-opt-in runtime is unavailable, not an all-zero observation.
