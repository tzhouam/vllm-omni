# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``python -m vllm_omni.windows report`` -- print the audit as JSON (exit 1 if a wheel site is missing)."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vllm_omni.windows")
    ap.add_argument("command", choices=["report"], nargs="?", default="report")
    ap.add_argument("--json", dest="json_path", help="also write the report here")
    args = ap.parse_args(argv)

    from vllm_omni.windows import activate, report

    activate()
    rep = report()
    text = json.dumps(rep, indent=2, default=str)
    print(text)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    sites = (rep.get("vllm") or {}).get("wheel_sites") or {}
    missing = sorted(k for k, v in sites.items() if v in ("wheel_required", "unknown_upstream"))
    if missing:
        print(f"wheel_required: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
