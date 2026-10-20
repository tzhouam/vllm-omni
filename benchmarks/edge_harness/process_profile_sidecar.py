# SPDX-License-Identifier: Apache-2.0
"""Read-only, identity-bound process telemetry. No process control privileges."""
import argparse, datetime, hashlib, json, os, pathlib, time
import psutil


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sample_process(p, *, include_thread_count=False):
    with p.oneshot():
        return dict(pid=p.pid, created=p.create_time(), name=p.name(),
                    cpu=p.cpu_times()._asdict(), memory=p.memory_info()._asdict(),
                    threads=p.num_threads() if include_thread_count else None,
                    logical_io=p.io_counters()._asdict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--created', type=float, required=True)
    ap.add_argument('--run', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--include-thread-count', action='store_true',
                    help='Enable expensive per-process thread queries; disabled by default on all platforms')
    a = ap.parse_args()
    root = psutil.Process(a.pid)
    if abs(root.create_time() - a.created) > 0.001:
        raise RuntimeError('Root process identity mismatch')
    if str(a.run).lower() not in ' '.join(root.cmdline()).lower():
        raise RuntimeError('Root command does not identify the intended run')
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(schema=2, include_thread_count=a.include_thread_count, started=now(), root_pid=a.pid, root_created=a.created,
                    run=str(a.run), cadence_seconds=1, psutil=psutil.__version__,
                    observer_pid=os.getpid(), script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                    limits=['Starts mid-loading; does not cover complete cold startup',
                            'Process I/O is logical I/O, not attributable physical SSD reads',
                            'System disk counters include unrelated workloads and devices',
                            'Process RSS can double-count shared pages; do not sum as physical RAM',
                            'No GPU kernel timing, cache hit attribution or power instrumentation',
                            'Sampling has overhead; collection wall time and observer CPU are recorded',
                            'threads=null means disabled, not zero; use --include-thread-count for diagnostic runs'])
    (a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    previous = {}
    phase_position = 0
    last_phase = None
    seq = 0
    errors = 0
    self_process = psutil.Process()
    started = time.monotonic()
    with (a.output / 'samples.jsonl').open('x', encoding='utf-8') as f:
        while True:
            tick = time.monotonic()
            try:
                if abs(root.create_time() - a.created) > 0.001 or not root.is_running():
                    break
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                break
            observations = []
            current = {}
            for p in processes:
                try:
                    s = sample_process(p, include_thread_count=a.include_thread_count)
                    key = (p.pid, s['created'])
                    old = previous.get(key)
                    if old:
                        elapsed = tick - old[0]
                        s['interval_seconds'] = elapsed
                        s['cpu_seconds_delta'] = sum(s['cpu'][k]-old[1]['cpu'][k] for k in ('user', 'system'))
                        s['logical_io_delta'] = {k:s['logical_io'][k]-old[1]['logical_io'][k] for k in s['logical_io']}
                    current[key] = (tick, s)
                    observations.append(s)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    observations.append(dict(pid=p.pid, observation_error=type(e).__name__))
                    errors += 1
            previous = current
            try:
                with (a.run / 'runner-phases.jsonl').open('r', encoding='utf-8') as phases:
                    phases.seek(phase_position)
                    for line in iter(phases.readline, ''):
                        try:
                            last_phase = json.loads(line)
                        except ValueError:
                            break
                    phase_position = phases.tell()
            except OSError:
                pass
            disk = psutil.disk_io_counters()
            entry = dict(seq=seq, timestamp=now(), elapsed_seconds=tick-started,
                         latest_phase=last_phase, processes=observations,
                         system_available_ram_bytes=psutil.virtual_memory().available,
                         system_disk_cumulative= disk._asdict() if disk else None,
                         observer_cpu=self_process.cpu_times()._asdict(),
                         observer_rss_bytes=self_process.memory_info().rss,
                         collection_wall_seconds=time.monotonic()-tick)
            f.write(json.dumps(entry, separators=(',', ':'))+'\n')
            f.flush()
            seq += 1
            time.sleep(max(0, 1-(time.monotonic()-tick)))
    (a.output / 'completion.json').write_text(json.dumps(dict(ended=now(), samples=seq,
        observation_errors=errors, reason='bound root process exited',
        elapsed_seconds=time.monotonic()-started), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

