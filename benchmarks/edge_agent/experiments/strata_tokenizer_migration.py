# SPDX-License-Identifier: Apache-2.0
"""Compare two pinned tokenizer implementations on one extracted real vocabulary.

No weight shard or neural backend is opened. Exact token IDs and round trips are
checked in both cold and cached calls, with protected special-token spans. This
does not certify the model, pack format, chat template rendering or runtime ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import random
import sys
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str, path: Path, expected: str):
    if sha(path) != expected:
        raise ValueError(f"{name} source digest differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def corpus():
    fixed = [
        "",
        "Hello, world!",
        "你好，世界。请只返回 JSON。",
        "日本語と한국어",
        "مرحبا بالعالم",
        "e\u0301\u0301 😀🚀🇨🇳",
        "\x00\x01\x7f control",
        "\r\n\t  spaces\u00a0\u2028\u2029",
        "def f(x):\n\treturn x ** 2\n",
        '{"tool":"browser_screenshot","args":{}}',
        "<think>literal</think><|im_start|>assistant\n<tool_call>{}</tool_call><|im_end|>",
        "x" * 5000,
        "上下文 memory 1234567890 " * 128,
    ]
    rng = random.Random(20261013)
    pieces = [
        "a",
        "你",
        "e\u0301",
        "🙂",
        "\n",
        " ",
        "123",
        "<think>",
        "</think>",
        "<|im_start|>",
        "<|im_end|>",
        "<tool_call>",
        "</tool_call>",
        '"',
        "{}",
        "\\",
    ]
    return fixed + ["".join(rng.choices(pieces, k=rng.randint(1, 48))) for _ in range(256)]


def compare(args):
    harness_sha256 = sha(Path(__file__))
    pack = args.tokenizer_dir.resolve(strict=True)
    files = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")
    identities = {name: sha(pack / name) for name in files}
    config = json.loads((pack / "tokenizer.json").read_bytes())
    vocab = json.loads((pack / "vocab.json").read_bytes())
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("vocabulary IDs are not contiguous and unique")
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    merges = (pack / "merges.txt").read_text(encoding="utf-8").splitlines()
    types = json.loads((pack / "token_type.json").read_bytes())
    old = load("strata_tokenizer_baseline", args.baseline, args.baseline_sha256)
    new = load("strata_tokenizer_candidate", args.candidate, args.candidate_sha256)
    instances = [
        m.Tokenizer(tokens, merges, types, pre=config["pre"], special_ids=config["special_ids"]) for m in (old, new)
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "samples.jsonl"
    checks = failures = 0
    with raw.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(corpus()):
            spans = [(), [(0, len(text))], [(len(text) // 2, len(text)), (0, len(text) // 3)]]
            for protected in spans:
                for special in (False, True):
                    for repeat in range(2):
                        outputs = [t.encode(text, parse_special=special, plain=protected) for t in instances]
                        decoded = [t.decode(ids) for t, ids in zip(instances, outputs)]
                        passed = outputs[0] == outputs[1] and decoded == [text, text]
                        record = dict(
                            case=index,
                            text=text,
                            protected=protected,
                            parse_special=special,
                            repeat=repeat,
                            baseline_ids=outputs[0],
                            candidate_ids=outputs[1],
                            passed=passed,
                        )
                        stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                        checks += 1
                        failures += not passed
    stable = identities == {name: sha(pack / name) for name in files}
    stable = stable and sha(args.baseline) == args.baseline_sha256 and sha(args.candidate) == args.candidate_sha256
    stable = stable and sha(Path(__file__)) == harness_sha256
    result = dict(
        schema="strata-tokenizer-migration-v1",
        status="passed" if failures == 0 and stable else "failed",
        checks=checks,
        failures=failures,
        input_files_stable=stable,
        harness_sha256=harness_sha256,
        python_version=sys.version,
        regex_version=importlib.metadata.version("regex"),
        metadata_sha256=identities,
        baseline_sha256=args.baseline_sha256,
        candidate_sha256=args.candidate_sha256,
        vocab_size=len(tokens),
        merges=len(merges),
        raw_sha256=sha(raw),
        candidate_piece_cache_entries=len(instances[1]._piece_ids),
        candidate_piece_cache_entry_limit=instances[1].PIECE_CACHE_MAX,
        scope="same real vocabulary; exact IDs and round trips; generated corpus, no neural execution",
        model_loaded=False,
        runtime_qualified=False,
        limitations=[
            "Not model-output parity or complete pack compatibility.",
            "Cache entry count is not a byte-bound memory budget.",
            "Both versions can share errors; no independent tokenizer oracle was run.",
        ],
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "tokenizer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    result = compare(parser.parse_args())
    print(json.dumps(result, indent=2))
    raise SystemExit(result["status"] != "passed")
