#!/usr/bin/env python3
"""Verify 12×5 summary-table coverage and its stated E2E depth counts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("readme", type=Path)
    args = parser.parse_args()
    text = args.readme.read_text(encoding="utf-8")
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines)
                  if line.startswith("| Device | Spark-X2.5 | Qwen3-TTS"))
    rows = []
    coverage = None
    for line in lines[header + 2:]:
        if not line.startswith("| "):
            break
        cells = line.strip().strip("|").strip().split(" | ")
        if len(cells) != 6:
            raise ValueError(f"matrix row has {len(cells)} columns: {line[:80]}")
        if cells[0] == "Coverage across 12 configurations":
            coverage = cells
            break
        rows.append(cells)
    if len(rows) != 12 or len({row[0] for row in rows}) != 12:
        raise ValueError("matrix must contain twelve distinct device configurations")
    complete = synthetic = unverified = 0
    complete_by_model = [0] * 5
    synthetic_by_model = [0] * 5
    for row in rows:
        for model, cell in enumerate(row[1:]):
            if cell.startswith(("scoped E2E", "experimental scoped E2E")):
                complete += 1
                complete_by_model[model] += 1
            elif model == 4 and cell.startswith(("synthetic policy pass",
                                                  "experimental synthetic Omni policy")):
                synthetic += 1
                synthetic_by_model[model] += 1
            elif cell.startswith("NOT E2E"):
                unverified += 1
            else:
                raise ValueError(f"unrecognized depth for {row[0]}: {cell[:100]}")
    if coverage is None:
        raise ValueError("matrix coverage row is absent")
    for model in range(4):
        match = re.fullmatch(r"(\d+) scoped E2E.*; (\d+) not E2E", coverage[model + 1])
        if (match is None or tuple(map(int, match.groups())) !=
                (complete_by_model[model], 12 - complete_by_model[model])):
            raise ValueError(f"coverage row disagrees for model column {model + 1}")
    vla = re.match(r"(\d+) synthetic-policy passes; (\d+) without a complete policy", coverage[5])
    if vla is None or tuple(map(int, vla.groups())) != (synthetic_by_model[4], 12 - synthetic_by_model[4]):
        raise ValueError("InternVLA coverage row disagrees with policy depth")
    claim = re.search(
        r"\*\*(\d+) scoped complete-request E2E paths, "
        r"(\d+) synthetic InternVLA policy-only paths, and "
        r"(\d+) paths without a verified complete workload\*\*", text)
    if claim is None or tuple(map(int, claim.groups())) != (complete, synthetic, unverified):
        raise ValueError("opening summary count disagrees with the configuration matrix")
    print(f"verified {len(rows) * 5} cells: {complete} scoped E2E, "
          f"{synthetic} synthetic policy-only, {unverified} without E2E")


if __name__ == "__main__":
    main()
