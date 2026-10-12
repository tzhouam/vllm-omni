#!/usr/bin/env python3
"""Upload and submit Spark blocks at one fixed context sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--last", type=int, default=7)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--context", type=int, default=128)
    args = ap.parse_args()
    submit = Path(__file__).with_name("submit_spark_hub_block.py")
    for block in range(args.first, args.last + 1):
        manifests = list(args.exports.glob(f"spark_block{block}_*_ctx{args.context}.json"))
        if len(manifests) != 1:
            raise ValueError(f"expected one context-{args.context} manifest for block {block}")
        command = [
            sys.executable, str(submit), "--manifest", str(manifests[0]),
            "--evidence", str(args.evidence), "--device", "Samsung Galaxy S25",
            "--quantize", "float16", "--runtime", "qnn_dlc",
        ]
        if args.wait:
            command.append("--wait")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
