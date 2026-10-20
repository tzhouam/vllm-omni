# Shell drivers

The exact invocations behind published grids — thread sweeps, interleaved A/Bs,
NUMA-node comparisons, fusion ablations.

They are kept verbatim, absolute paths and all, because a grid's *command* is
part of its result: which cores, which environment variables, which order. The
harness now records all of that into each artifact's `provenance` block, so new
work does not need a script like these; these document the runs that predate it.

Read them for what was done, not as a portable interface.
