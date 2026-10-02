#!/usr/bin/env python3
"""Capture the public Workbench attributes of the five matrix targets.

This is device-registration evidence only. It does not start inference jobs,
inspect a particular leased SKU's RAM, or establish an on-device Omni path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub


TARGETS = (
    "Samsung Galaxy S25",
    "Samsung Galaxy S24",
    "Snapdragon X Elite CRD",
    "SA8775P ADP",
    "Dragonwing RB3 Gen 2 Vision Kit",
)


def capture() -> dict:
    devices = []
    for name in TARGETS:
        exact = [device for device in hub.get_devices(name=name)
                 if device.name == name]
        devices.append({
            "requested_name": name,
            "exact_matches": len(exact),
            "registrations": [
                {"name": device.name, "os": device.os,
                 "attributes": list(device.attributes),
                 "ram_bytes": None,
                 "ram_source": "not disclosed by qai_hub Device fields or attributes"}
                for device in exact
            ],
        })
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "qai_hub_sdk_version": importlib.metadata.version("qai-hub"),
        "depth": "D: hosted device registration only",
        "batch_size": None,
        "active_requests": None,
        "devices": devices,
        "limitations": [
            "This call does not lease a device or reveal exact RAM SKU.",
            "No model component or complete request was executed.",
            "Capacity refusals require exact artifact bytes and target usable RAM."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "exact_matches": {row["requested_name"]: row["exact_matches"]
                          for row in report["devices"]},
    }, indent=2))


if __name__ == "__main__":
    main()
