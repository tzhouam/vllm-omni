"""Run benchmark arms so the numbers survive a shared machine.

Every wrong conclusion this study produced came from measurement, not from
code: a sequential grid that swung the llama.cpp control 86.7 -> 60.4 tok/s
mid-run, a 3.1 GB "saving" that was really `VmRSS` counting the mmap'd
checkpoint's pages, and a W4A8/W4A16 ordering that flipped between grids. The
discipline that fixed each of them is the same, so it lives here instead of
being re-typed into the next shell script:

* **Rotate the arm order every pass.** Drift hits arm 1 and arm 5 differently,
  and interleaving alone does not fix it: if every pass runs the arms in the
  same order, arm 1 is always the one measured first and carries whatever the
  machine was doing at the start of each pass. Pass ``p`` starts at arm
  ``(p - 1) mod n``.
* **Repeat, and report the spread.** A single number from a shared host is a
  claim about the host, not the code. An arm whose passes disagree by more
  than `--noise-threshold` is flagged, and the summary says so.
* **Record how busy the machine was, and refuse when it is.** A pass taken
  under load is kept and labelled rather than silently averaged in -- but load
  average is a whole-host number, and this host has 224 cores, so it says
  nothing about the 24 this benchmark is pinned to. The pre-flight reads
  per-core utilisation for the *target* cores, and a host lock keeps two
  benchmarks off the same cores in the first place. Two sessions benchmarking
  at once wrecked several measurements here, and one grid's control drifted
  30% mid-run.
* **Prefer allocation over memory.** `RssAnon` repeats to within 0.5 MB where
  `VmRSS` moved 1-2 GB between identical runs.
* **Know which direction is better.** ``best`` is the maximum for a rate and
  the minimum for memory, init time or latency. Reporting the maximum for every
  metric published the *worst* pass as the headline memory figure.
* **Stamp provenance.** A number whose model, build and thread set were not
  recorded cannot be retracted later, only doubted -- and this study has
  retracted four.
* **Give an arm its own compile cache when it changes the graph.** vLLM keys
  its AOT compile cache on the *config*, so two arms that differ only by an
  environment variable share a cached artifact -- and if that variable changes
  the traced graph, later passes load the wrong one. Three passes of a prefill
  grid died with ``KeyError: 'weight_halves'`` inside an AOT-loaded forward
  after the first pass saved an artifact for the other path. Set
  ``VLLM_CACHE_ROOT`` per arm, and note that the first pass of each arm then
  pays a cold compile, which is why ``init_s`` is not comparable across passes.
* **Gate on what the host can actually resolve.** ``--check`` compares against
  a checked-in expectations file, with a band per metric taken from that
  metric's *observed* spread: 0.2% for ``RssAnon``, ~5% for per-op self time,
  and nothing for end-to-end throughput, which swings ~30% here and cannot
  carry a gate at all. A band is an argument about what is measurable, not a
  tolerance someone picked.

The arms are subprocess invocations of the existing single-run benchmarks, so
the measurement code stays the one that was verified; this only decides what
to run, when, and how to read it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass, field


def load_average() -> float:
    return os.getloadavg()[0]


def parse_cores(spec: str) -> list[int]:
    """'0-23,48-71' -> [0..23, 48..71]"""
    cores: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cores.extend(range(int(lo), int(hi) + 1))
        else:
            cores.append(int(part))
    return sorted(set(cores))


@dataclass
class Arm:
    tag: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Observation:
    tag: str
    pass_no: int
    metrics: dict[str, float]
    load_before: float
    load_after: float
    wall_s: float
    ok: bool
    error: str = ""

    @property
    def busy(self) -> bool:
        return max(self.load_before, self.load_after) > BUSY_LOAD


BUSY_LOAD = 8.0
"""Above this 1-minute load average a pass is labelled contended.

A poor default on its own: load average is whole-host, and on a 224-core
machine a 24-thread benchmark contributes ~24 to it by itself, so a fixed 8
labels every pass contended and the label stops meaning anything. ``main()``
derives it from the target core count instead -- our own threads plus slack --
and the per-core pre-flight is the guard that actually decides whether to run.
"""

LOWER_IS_BETTER = (
    "rss", "mem", "mb", "init", "load_s", "latency", "ms", "_s", "time", "peak",
)
"""Substrings that mark a metric where smaller wins. Rates (`tg_tps`, `pp_tps`)
take the maximum; memory, init time and latency take the minimum."""


HOST_LOCK_PATH = os.environ.get("SPARK_EDGE_LOCK", "/tmp/spark_edge_bench.lock")
"""Host-level lock. It has to live somewhere every session can see, which a
session-private scratchpad by definition is not -- that is the whole point of
a host lock. Override with ``SPARK_EDGE_LOCK``."""

CORE_BUSY_PCT = 50.0
"""A target core sustained above this is a straggler. OpenMP barriers mean the
slowest thread gates the whole region, so one heavily-loaded core costs far
more than its share -- which is why this is a separate test from the aggregate
below rather than a single averaged number."""

STOLEN_PCT = 5.0
"""Refuse when this much of the target cores' aggregate capacity is going to
someone else, which bounds the bias on the measurement.

A flat per-core threshold does not work on this host. With 224 cores and
several users there is always a transient blip somewhere: a 20% rule over a
single 0.5 s window refused three consecutive launches on cores that were
idle seconds earlier, which makes the guard useless rather than strict. So
utilisation is sampled twice and the per-core minimum is taken -- a spike in
one window does not count as load -- and the decision is made on aggregate
stolen capacity plus the straggler rule."""


def _cpu_times() -> dict[int, tuple[int, int]]:
    """(busy, total) jiffies per core, from /proc/stat."""
    out: dict[int, tuple[int, int]] = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            try:
                cpu = int(parts[0][3:])
            except ValueError:
                continue
            values = [int(v) for v in parts[1:11]]
            idle = values[3] + values[4]           # idle + iowait
            total = sum(values)
            out[cpu] = (total - idle, total)
    return out


def core_utilization(cores: list[int], window_s: float = 0.5) -> dict[int, float]:
    """Percentage busy per core over ``window_s``."""
    before = _cpu_times()
    time.sleep(window_s)
    after = _cpu_times()
    usage: dict[int, float] = {}
    for cpu in cores:
        if cpu not in before or cpu not in after:
            continue
        busy = after[cpu][0] - before[cpu][0]
        total = after[cpu][1] - before[cpu][1]
        usage[cpu] = 0.0 if total <= 0 else busy / total * 100.0
    return usage


def preflight(cores: list[int], *, max_busy_pct: float = CORE_BUSY_PCT,
              max_stolen_pct: float = STOLEN_PCT,
              window_s: float = 0.75, samples: int = 2) -> dict:
    """Refuse to start when contention would bias the measurement.

    Two independent windows; a core's load is the *minimum* of them, so a
    transient spike is not mistaken for a busy core. Then two tests: aggregate
    stolen capacity (bounds the average bias) and a per-core straggler rule
    (one slow core gates every OpenMP barrier).
    """
    windows = [core_utilization(cores, window_s) for _ in range(max(samples, 1))]
    usage = {c: min(w.get(c, 0.0) for w in windows) for c in cores
             if any(c in w for w in windows)}
    stolen = (sum(usage.values()) / len(usage)) if usage else 0.0
    stragglers = {c: round(u, 1) for c, u in usage.items() if u > max_busy_pct}
    return {"cores": cores,
            "max_busy_pct": max_busy_pct, "max_stolen_pct": max_stolen_pct,
            "utilization": {c: round(u, 1) for c, u in usage.items()},
            "stolen_pct": round(stolen, 2),
            "busy_cores": stragglers,
            "clear": stolen <= max_stolen_pct and not stragglers}


class HostLock:
    """An advisory lock so two sessions do not benchmark the same cores.

    Cooperative, and says so: it excludes other runs that take it, not the
    unrelated job someone starts by hand. That is what the pre-flight is for.
    """

    def __init__(self, path: str = HOST_LOCK_PATH, note: str = ""):
        self.path, self.note, self._fd = path, note, None

    def __enter__(self) -> "HostLock":
        import fcntl

        self._fd = open(self.path, "a+")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fd.seek(0)
            holder = self._fd.read().strip()[-200:]
            self._fd.close()
            self._fd = None
            raise SystemExit(
                f"another benchmark holds {self.path}: {holder or 'unknown'}. "
                f"Wait for it, or set SPARK_EDGE_LOCK to a different path if "
                f"you really are on disjoint cores."
            )
        self._fd.seek(0)
        self._fd.truncate()
        self._fd.write(f"pid={os.getpid()} {time.strftime('%H:%M:%S')} {self.note}\n")
        self._fd.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


DEFAULT_BANDS_PCT = {
    "rss_anon_mb": 1.5,
    "peak_rss_anon_mb": 2.0,
    "rss_anon_at_context_mb": 2.0,
    "init_s": 12.0,
    "ready_s": 12.0,
    "import_s": 12.0,
    "step_ms": 6.0,
    "gemm_ms": 6.0,
}
"""Regression bands, per metric, from the spread each one actually shows on
this host. ``RssAnon`` repeats to 0.2% and is given 1.5% for safety; init time
is dominated by page cache and repeats far less tightly; per-op self time
reproduces to ~5%.

Throughput is deliberately absent. ``tg_tps`` swings ~30% run to run on this
shared host, so any band wide enough not to fire constantly is too wide to
catch a real regression. Gate on memory and per-op time instead; that is the
whole reason the per-op profiler exists."""

UNGATEABLE = {"tg_tps", "pp_tps", "tps"}
"""Metrics this host cannot resolve well enough to gate on."""


def check_expectations(summaries: list["ArmSummary"], expected: dict) -> list[dict]:
    """Compare measured medians against a checked-in expectations file.

    The median, not the best: `best` is one pass and a gate that fires on the
    luckiest run of a noisy metric is a coin toss.
    """
    bands = {**DEFAULT_BANDS_PCT, **(expected.get("bands_pct") or {})}
    arms = expected.get("arms") or {}
    results = []
    for s in summaries:
        want = (arms.get(s.tag) or {}).get(s.metric)
        if want is None:
            results.append({"arm": s.tag, "metric": s.metric, "status": "unlisted",
                            "measured": s.median, "expected": None})
            continue
        if s.metric in UNGATEABLE:
            results.append({"arm": s.tag, "metric": s.metric, "status": "ungateable",
                            "measured": s.median, "expected": want,
                            "why": "this host cannot resolve it"})
            continue
        band = bands.get(s.metric, 5.0)
        drift = abs(s.median - want) / want * 100.0 if want else 0.0
        worse = (s.median > want) if s.lower_is_better else (s.median < want)
        results.append({
            "arm": s.tag, "metric": s.metric,
            "measured": s.median, "expected": want,
            "drift_pct": round(drift, 2), "band_pct": band,
            "status": ("ok" if drift <= band else
                       "REGRESSION" if worse else "improved"),
        })
    return results


def format_check(results: list[dict]) -> str:
    if not results:
        return "nothing to check"
    width = max(len(r["arm"]) for r in results)
    lines = []
    for r in results:
        if r["status"] in ("unlisted", "ungateable"):
            lines.append(f"  {r['arm'].ljust(width)}  {r['metric']:22s} "
                         f"{r['measured']:9.2f}  {r['status']}"
                         + (f" ({r['why']})" if r.get("why") else ""))
        else:
            lines.append(f"  {r['arm'].ljust(width)}  {r['metric']:22s} "
                         f"{r['measured']:9.2f}  vs {r['expected']:9.2f}  "
                         f"{r['drift_pct']:+5.2f}% (band {r['band_pct']:.1f}%)  "
                         f"{r['status']}")
    return "\n".join(lines)


def metric_is_lower_better(metric: str) -> bool:
    name = metric.lower()
    if name.endswith(("_tps", "_tok_s", "tps")):
        return False
    return any(tok in name for tok in LOWER_IS_BETTER)


def provenance(extra: dict | None = None) -> dict:
    """Everything needed to decide later whether a number is still valid.

    Saved profiles in this tree carry no model path, context length or build,
    so two of them are indistinguishable from each other and one set silently
    describes a checkpoint that was later found wrong. This is the fix.
    """
    root = os.environ.get("EMBEDDING_INFER_ROOT", "/data/zhoutaichang/embedding_infer")

    def _sha(rel: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", os.path.join(root, rel),
                                  "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or None
        except Exception:
            return None

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []
    return {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "n_affinity": len(affinity),
        "affinity": affinity,
        "commits": {
            "embedding_infer": _sha("."),
            "vllm_omni": _sha("vllm-omni"),
            "vllm": _sha("vllm"),
        },
        "env": {k: v for k, v in sorted(os.environ.items())
                if k.startswith(("OMP_", "KMP_", "MKL_", "VLLM_", "LD_PRELOAD",
                                 "CUDA_VISIBLE_DEVICES"))},
        "loadavg": os.getloadavg(),
        **(extra or {}),
    }


def run_once(arm: Arm, pass_no: int, timeout: float, keys: list[str]) -> Observation:
    env = {**os.environ, **arm.env}
    load_before = load_average()
    t0 = time.perf_counter()
    proc = subprocess.run(arm.command, env=env, capture_output=True,
                          text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    load_after = load_average()
    if proc.returncode != 0:
        return Observation(arm.tag, pass_no, {}, load_before, load_after, wall,
                           ok=False, error=proc.stderr.strip()[-400:])
    metrics = _scrape(proc.stdout, keys)
    return Observation(arm.tag, pass_no, metrics, load_before, load_after, wall,
                       ok=bool(metrics))


def _scrape(stdout: str, keys: list[str]) -> dict[str, float]:
    """Pull metrics out of the last JSON object the run printed."""
    start = stdout.rfind("{")
    while start >= 0:
        try:
            payload = json.loads(stdout[start:stdout.rfind("}") + 1])
        except json.JSONDecodeError:
            start = stdout.rfind("{", 0, start)
            continue
        return {k: float(payload[k]) for k in keys
                if isinstance(payload.get(k), (int, float))}
    return {}


@dataclass
class ArmSummary:
    tag: str
    metric: str
    best: float
    median: float
    spread_pct: float
    n: int
    n_busy: int
    lower_is_better: bool = False
    noise_threshold: float = 8.0

    @property
    def noisy(self) -> bool:
        return self.spread_pct > self.noise_threshold


NOISE_THRESHOLD = 8.0


def summarize(observations: list[Observation], metric: str,
              noise_threshold: float = NOISE_THRESHOLD,
              lower_is_better: bool | None = None) -> list[ArmSummary]:
    """Collapse passes into one row per arm.

    ``lower_is_better`` defaults to reading the metric name, so `rss_anon_mb`
    and `init_s` take their minimum while `tg_tps` takes its maximum. Taking
    the maximum for every metric is how the *worst* memory pass came to be
    published as the best one.
    """
    if lower_is_better is None:
        lower_is_better = metric_is_lower_better(metric)
    by_tag: dict[str, list[Observation]] = {}
    for obs in observations:
        if obs.ok and metric in obs.metrics:
            by_tag.setdefault(obs.tag, []).append(obs)
    summaries = []
    for tag, rows in by_tag.items():
        values = [r.metrics[metric] for r in rows]
        lo, hi = min(values), max(values)
        spread = 0.0 if hi == 0 else (hi - lo) / hi * 100.0
        summaries.append(ArmSummary(
            tag=tag, metric=metric, best=lo if lower_is_better else hi,
            median=statistics.median(values), spread_pct=spread,
            n=len(values), n_busy=sum(1 for r in rows if r.busy),
            lower_is_better=lower_is_better, noise_threshold=noise_threshold,
        ))
    return sorted(summaries, key=lambda s: s.best if lower_is_better else -s.best)


def indistinguishable(summaries: list[ArmSummary]) -> str | None:
    """Say so when the gap between the top two arms is inside the noise.

    Per-op self time resolves ~5% on this host; end-to-end throughput resolves
    ~30%. An 8% difference in tg_tps is not a result, and publishing it as one
    is how the W4A8/W4A16 ordering came to flip between grids.
    """
    if len(summaries) < 2:
        return None
    first, second = summaries[0], summaries[1]
    if not first.best:
        return None
    gap = abs(first.best - second.best) / abs(first.best) * 100.0
    band = max(first.spread_pct, second.spread_pct,
               DEFAULT_BANDS_PCT.get(first.metric, first.noise_threshold))
    if gap > band:
        return None
    return (f"{first.tag} and {second.tag} differ by {gap:.1f}% on "
            f"{first.metric}, inside this measurement's own spread "
            f"({band:.1f}%). That is not a ranking. For a difference this "
            f"size use per-op self time, which reproduces to ~5%: "
            f"profile_cpu_step.py --model <build> --prompt-len <ctx>")


def format_summary(summaries: list[ArmSummary]) -> str:
    if not summaries:
        return "no successful observations"
    width = max(len(s.tag) for s in summaries)
    direction = "min" if summaries[0].lower_is_better else "max"
    lines = [f"{'arm'.ljust(width)}  {f'best({direction})':>9s} {'median':>9s} "
             f"{'spread':>7s}  n  notes"]
    for s in summaries:
        notes = []
        if s.noisy:
            notes.append(f"NOISY (>{s.noise_threshold:.0f}%)")
        if s.n_busy:
            notes.append(f"{s.n_busy} pass(es) under load")
        if s.n < 2:
            notes.append("single pass — not a result")
        lines.append(f"{s.tag.ljust(width)}  {s.best:9.2f} {s.median:9.2f} "
                     f"{s.spread_pct:6.1f}%  {s.n}  {'; '.join(notes)}")
    warning = indistinguishable(summaries)
    if warning:
        lines.append("")
        lines.append(f"  NOT A RANKING: {warning}")
    return "\n".join(lines)


def run_grid(arms: list[Arm], *, passes: int, metric: str, timeout: float,
             keys: list[str], settle_s: float = 0.0,
             busy_load: float = BUSY_LOAD) -> list[Observation]:
    """Run every arm each pass, rotating the starting arm between passes.

    Interleaving alone leaves arm 0 permanently first, so whatever the machine
    does at the start of a pass always lands on the same arm. Pass ``p`` starts
    at arm ``(p - 1) mod n``.
    """
    globals()["BUSY_LOAD"] = busy_load
    observations: list[Observation] = []
    for p in range(1, passes + 1):
        shift = (p - 1) % len(arms) if arms else 0
        for arm in arms[shift:] + arms[:shift]:
            if settle_s:
                time.sleep(settle_s)
            obs = run_once(arm, p, timeout, keys)
            observations.append(obs)
            status = "ok" if obs.ok else f"FAILED: {obs.error[:80]}"
            value = obs.metrics.get(metric)
            shown = f"{value:.2f}" if value is not None else "-"
            flag = "  [busy]" if obs.busy else ""
            print(f"  pass {p} {arm.tag:24s} {metric}={shown:>9s}  "
                  f"load {obs.load_before:.1f}->{obs.load_after:.1f}{flag}  {status}",
                  flush=True)
    return observations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True,
                    help="JSON file: {\"arms\": [{\"tag\":..., \"command\":[...], \"env\":{...}}]}")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--metric", default="tg_tps")
    ap.add_argument("--keys", default="tg_tps,pp_tps,init_s,rss_anon_mb,rss_mb")
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--busy-load", type=float, default=None,
                    help="load average above which a pass is labelled "
                         "contended; defaults to (target cores + 8), since "
                         "the run itself contributes about one per core")
    ap.add_argument("--noise-threshold", type=float, default=NOISE_THRESHOLD)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cores", default="",
                    help="target cores, e.g. '0-23'. Checked for other "
                         "people's work before the run starts.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if the target cores are already loaded "
                         "(the result is labelled, not trusted)")
    ap.add_argument("--check", default=None,
                    help="expectations JSON; exit non-zero on a regression "
                         "outside the metric's measured band")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip the host lock (only if you are on cores no "
                         "other benchmark uses)")
    a = ap.parse_args()

    spec = json.loads(open(a.spec).read())
    arms = [Arm(tag=x["tag"],
                command=x["command"] if isinstance(x["command"], list)
                else shlex.split(x["command"]),
                env={k: str(v) for k, v in (x.get("env") or {}).items()})
            for x in spec["arms"]]
    keys = [k.strip() for k in a.keys.split(",") if k.strip()]

    cores = parse_cores(a.cores) if a.cores else sorted(os.sched_getaffinity(0))
    # Our own run puts ~len(cores) on the load average, so a threshold below
    # that labels every pass contended and tells us nothing.
    busy_load = a.busy_load if a.busy_load is not None else len(cores) + 8.0
    check = preflight(cores)
    if not check["clear"]:
        detail = []
        if check["stolen_pct"] > check["max_stolen_pct"]:
            detail.append(f"{check['stolen_pct']}% of the cores' capacity is "
                          f"someone else's (limit {check['max_stolen_pct']}%)")
        if check["busy_cores"]:
            detail.append("sustained stragglers: "
                          + ", ".join(f"{c}={p}%"
                                      for c, p in check["busy_cores"].items()))
        msg = "contended target cores: " + "; ".join(detail)
        if not a.allow_busy:
            raise SystemExit(
                msg + "\n  Another job is on these cores; this run would "
                      "measure the contention, not the code. Wait, pick other "
                      "cores with --cores, or pass --allow-busy to label the "
                      "result instead of trusting it.")
        print(f"WARNING: {msg} (--allow-busy)")

    print(f"{len(arms)} arms x {a.passes} passes, rotating; metric={a.metric}; "
          f"{len(cores)} cores, pre-flight "
          f"{'clear' if check['clear'] else 'BUSY'} "
          f"({check['stolen_pct']}% stolen), "
          f"busy-label above load {busy_load:.0f}")

    lock = contextlib.nullcontext() if a.no_lock else HostLock(note=a.metric)
    with lock:
        observations = run_grid(arms, passes=a.passes, metric=a.metric,
                                timeout=a.timeout, keys=keys,
                                settle_s=a.settle_s, busy_load=busy_load)
    summaries = summarize(observations, a.metric, a.noise_threshold)
    print("\n" + format_summary(summaries))

    checks = []
    if a.check:
        checks = check_expectations(summaries, json.loads(open(a.check).read()))
        print("\nregression check\n" + format_check(checks))

    if a.out:
        with open(a.out, "w") as f:
            json.dump({
                "metric": a.metric, "passes": a.passes,
                "lower_is_better": metric_is_lower_better(a.metric),
                "provenance": provenance({
                    "arms": [{"tag": x.tag, "command": x.command, "env": x.env}
                             for x in arms],
                    "preflight": check,
                    "busy_load_threshold": busy_load,
                }),
                "observations": [vars(o) for o in observations],
                "summary": [vars(s) for s in summaries],
                "checks": checks,
            }, f, indent=2)
        print(f"\nwrote {a.out}")

    regressions = [c for c in checks if c["status"] == "REGRESSION"]
    if regressions:
        raise SystemExit(
            f"{len(regressions)} regression(s): "
            + ", ".join(f"{c['arm']}/{c['metric']} {c['drift_pct']:+.2f}%"
                        for c in regressions))


if __name__ == "__main__":
    main()
