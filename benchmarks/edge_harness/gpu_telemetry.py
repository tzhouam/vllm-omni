# SPDX-License-Identifier: Apache-2.0
"""Low-rate, device-wide GPU telemetry for sustained local profiling."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class GpuTelemetry:
    def __init__(self, path: Path, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.path = path
        self.interval_s = interval_s
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            try:
                import psutil
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                try:
                    with self.path.open("w", encoding="utf-8") as output:
                        while not self._stop.is_set():
                            row: dict[str, Any] = {"unix": time.time()}

                            def read(name: str, getter: Any) -> None:
                                try:
                                    row[name] = getter()
                                except Exception:
                                    row[name] = None

                            read("gpu_temperature_c", lambda: int(pynvml.nvmlDeviceGetTemperature(
                                handle, pynvml.NVML_TEMPERATURE_GPU)))
                            read("gpu_power_mw", lambda: int(pynvml.nvmlDeviceGetPowerUsage(handle)))
                            read("gpu_clock_sm_mhz", lambda: int(pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_SM)))
                            read("gpu_clock_memory_mhz", lambda: int(pynvml.nvmlDeviceGetClockInfo(
                                handle, pynvml.NVML_CLOCK_MEM)))
                            read("gpu_throttle_reasons", lambda: int(
                                pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)))
                            read("gpu_utilization_pct", lambda: int(
                                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu))
                            read("gpu_memory_used_bytes", lambda: int(
                                pynvml.nvmlDeviceGetMemoryInfo(handle).used))
                            read("host_available_bytes", lambda: int(psutil.virtual_memory().available))
                            read("host_cpu_utilization_pct", lambda: float(psutil.cpu_percent(interval=None)))
                            output.write(json.dumps(row) + "\n")
                            output.flush()
                            self.samples += 1
                            self._stop.wait(self.interval_s)
                finally:
                    pynvml.nvmlShutdown()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            "path": self.path.name,
            "samples": self.samples,
            "interval_s": self.interval_s,
            "error": self.error,
            "scope": "whole GPU/host; not attributable to this process; sampled values may miss peaks",
        }
