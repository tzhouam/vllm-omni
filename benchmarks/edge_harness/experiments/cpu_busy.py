"""Percent-busy of a core range over a short window.

Two engines sharing cores measure the scheduler, not the engine, so every run
in this directory records what else was on its cores before it started.
"""
import sys, time


def snap():
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            p = line.split()
            d[int(p[0][3:])] = (sum(map(int, p[1:])), int(p[4]))
    return d


def busy(lo, hi, window=2.0):
    a = snap(); time.sleep(window); b = snap()
    tot = idle = 0
    for c in range(lo, hi + 1):
        tot += b[c][0] - a[c][0]
        idle += b[c][1] - a[c][1]
    return 100.0 * (1 - idle / max(tot, 1))


if __name__ == "__main__":
    lo, hi = (int(x) for x in sys.argv[1].split("-"))
    print(f"{busy(lo, hi):.1f}")
