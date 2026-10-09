"""Private source-only combined observer verifier; not imported or tested.

This checks archived bytes and mutually bound build evidence. It does not
execute a compiler, establish current OS identity, or authorize a live owner.
"""

from __future__ import annotations

import copy
import hashlib
import json
import ntpath
import os
import re
import stat
import struct
import types
from pathlib import Path, PurePosixPath

BASE = "d5ea7133741e67743c0e886bb426c0ce8d69cf6c"
DEPENDENCY = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
DEPENDENCY_TREE = "d255198f04f9b8349f1dff24501d513f83cbfeda"
PATCHES = (
    ("existing_io_patch", "31b98e5e02e21a6289d0713097036334a59aef2d15ad9e951ded80f59f4bfdcc"),
    ("incremental_execution_patch", "884ce375cfab94756834f1e97211bfbef6471674e50e24c0bfcc03a9eca96358"),
    ("boundary_safety_v2", "4689571b0c15873dba9e232baf3834db925c1cfb6d9f5929ef70f95c773ab583"),
)
HEADER = "include/strata/core/omni_execution_observer.hpp"
HEADER_V1 = "1aab06f18a6c3068ae3de389acc3f32347092bc3dd18c415343cb8b0e8a8d81b"
HEADER_V2 = "148b2d3de4b3f62e8f5ebff9211d04a89d488bd00d9256cc26ef41cc5bfa2f55"
SCHEMA_SHA = "d7ff209a7533f85e182e2be9954805c1e08d505e253a7e116c405542ea729cfc"
PATCH_MANIFEST_SHA = "b6a0546fc281491fb6171c5e201c46dad714bf7183f3b0ccbd96b872968ee98f"
PARSER_SHA = "c653d6eb422ed1f5ca970d5db1d0d3c8d08139a1e085eabd171497490a7663bf"
BUILD_RECORDER_SHA = "ae8e7d19eadf58ca53d0acadb57d0854343de71816fd72fb2882a56adfc13f76"
SOURCE_RECORDER_SHA = "e13dd43105a56c0a0fc05fac95805bd922efee5f0b48c429fff15dfa2aa74da1"
POST_SOURCE_RECORDER_SHA = "7a8fb2572e6a0220ec18a825390e9e12ca94f619aeed0177e358aecfe1c1076f"
SUPPLEMENT_RECORDER_SHA = "cc686578cb063d429506d029b7bf2dcdf0a6969a49c099ce1290c72be96ea5dc"
GRAPH_RECORDER_SHA = "119f281f304b1e8685e115b48b2cd426ed0c20efc5de157da95d7f5c566a6dae"
ORIGINAL_FAILED_BUILD_SHA = "60d12283b50c7786fcfeeda128e644fda1d4cffc7593fa1f38a32114e1adfede"
BUILD_QUIESCENCE_SHA = "4bd1fee727852354b3ecaaf8a69c510a7d6f85ea20cc53dbfe232f0f58283201"
SUPPLEMENT_DESCRIPTOR_KEYS = {"build_closure_supplement_file", "closure_supplement_recorder_file", "ninja_graph_recorder_file", "dependency_post_source_recorder_file"}
ALIAS_DESCRIPTOR_KEYS = {"external_header_alias_witness_file", "external_header_alias_collector_file"}
ALIAS_WITNESS_SHA = "483ca972a331c6017808d5bdb68bed6561fd601fb1a7859b7e1b9d68ca6f92d6"
ALIAS_COLLECTOR_SHA = "248a2c877f32bf2626bd0410a5f280632a4cc0535277b23e7de894fdf2902344"
ALIAS_DIAGNOSIS_SHA = "a4bf0d93f0be79cb36a724d00f6640dd9fa8dde17b569762dc830dfa85e57d6f"
FIXTURE_SOURCE_SHA = "ff6e9b06a31b9aca702889296c736741f703e1019d3789e02288837ba75ee286"
FIXTURE_RECORDER_SHA = "ddc2f71c85af1a9ba475f14fac61e9b34efb07d91172f0bc19efc147341b05b3"
FIXTURE_CASES = (
    "allocator-success-memset-equivalent", "allocator-failed-allocation", "allocator-failed-free",
    "allocator-same-owner", "allocator-registry-exhaustion", "allocator-baseline-gauge",
    "allocator-request-transient", "allocator-untracked-families", "cpu-counts", "cpu-invalid-geometry",
    "cpu-stale-phase", "cpu-stale-sequence", "cuda-launch-error", "cuda-successful-fence", "cuda-fence-error",
    "cuda-stale-phase", "boundary-labels", "boundary-order", "clock-unavailable", "cancelled",
    "cancelled-pending-fence", "missing-end", "counter-overflow", "formatting-overflow", "worst-case-frames",
)
MAX_JSON = 2 << 20
# Only the receipt-bound target compile metadata has a separate byte envelope.
MAX_TARGET_COMPILE_JSON = 4 << 20
MAX_MANIFEST_BYTES = 16 << 20
MAX_MANIFEST_NODES = 300000
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 250000
MAX_TEXT = 16 << 20
MAX_MEMBER = 512 << 20
MAX_TOTAL = 4 << 30
MAX_FILES = 65536
MAX_DIRS = 8192
MAX_GIT_OBJECT = 64 << 20
STRATA_SOURCE_AUXILIARY = {
    "path": "data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf",
    "size_bytes": 483520,
    "sha256": "ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b",
    "git_blob_id": "5c10294a9981a76ea83f51aadd15816d3911285f",
    "mode": "100644",
}
# Exact tokenizer fixtures from the separately pinned dependency Git tree.
# Only the two descriptor-bound dependency source snapshots may contain them.
DEPENDENCY_VOCABULARY = {'models/ggml-vocab-aquila.gguf': {'size_bytes': 4825676, 'sha256': '7c53c3c516ac67c7ca12977b9690fdea3d2ef13bbaed6378f98191a13ef5ca00'}, 'models/ggml-vocab-baichuan.gguf': {'size_bytes': 1340998, 'sha256': '4f5b955697f3bd3108070b1d5936c7eb9fc542b81c6932e59abddec75bca1963'}, 'models/ggml-vocab-bert-bge.gguf': {'size_bytes': 627549, 'sha256': 'fbcbe22278fb302694d5f4a41bfe48c5f90e8e3554eab1c0435387dff654a854'}, 'models/ggml-vocab-command-r.gguf': {'size_bytes': 10874545, 'sha256': 'a2f8cfea952ef7c391a6d92a1c309d0bd32e36384d9b9230569a7425732f27d9'}, 'models/ggml-vocab-deepseek-coder.gguf': {'size_bytes': 1156067, 'sha256': '91cb1379f2e33af1c4866b194622b7a0e12e8f0c9dba7ba2f10d55978730bec1'}, 'models/ggml-vocab-deepseek-llm.gguf': {'size_bytes': 3970167, 'sha256': '867f77537b54565f0d81d508c04edc41aa1d4ffc1a92745f225b4c1b02755f76'}, 'models/ggml-vocab-falcon.gguf': {'size_bytes': 2287728, 'sha256': '9f0bf8b0733680398b72e652e90f260f43782f326e75545fc0e49611a5ba35ad'}, 'models/ggml-vocab-gemma-4.gguf': {'size_bytes': 15776467, 'sha256': '58b1ba0b57f3b4d7c468ba4ffd91ad85190346a3d7ad7e71d1cabaae8a14bb65'}, 'models/ggml-vocab-gpt-2.gguf': {'size_bytes': 1766807, 'sha256': 'cedc56ca6e2e89f63e781696d1fd76b4b1d49e6720dee86463e915f6e90016ac'}, 'models/ggml-vocab-gpt-neox.gguf': {'size_bytes': 1771431, 'sha256': 'ae593a7f9b8bb174ed4f5019e41530463e4dac7aa06e42dee8aa650d2bdac53d'}, 'models/ggml-vocab-llama-bpe.gguf': {'size_bytes': 7818140, 'sha256': '97272e430d53bc7688f52d5e0ad8ea8f163ede9f1bbd1694feaa504797d5d96e'}, 'models/ggml-vocab-llama-spm.gguf': {'size_bytes': 723869, 'sha256': '16c3724582d59aa8bf84711894e833f916ee46a31d80e21312759c48bf8d0e69'}, 'models/ggml-vocab-mpt.gguf': {'size_bytes': 1771393, 'sha256': '59dc382612866d1fc6c11ea531318d327598f3412d9c8f8600607cdf3030898f'}, 'models/ggml-vocab-nomic-bert-moe.gguf': {'size_bytes': 6821877, 'sha256': '90a6746926454784a98389ad36a36d89bc9cfc81db9cb0f33c941bcc959fe5f9'}, 'models/ggml-vocab-phi-3.gguf': {'size_bytes': 726019, 'sha256': '967d7190d11c4842eab697079d98d56c2116e10eb617be355a2733bfc132e326'}, 'models/ggml-vocab-qwen2.gguf': {'size_bytes': 5928681, 'sha256': '44c2f46b715f585c6ab513970e8a006bfa5badd6108560054921cf598d154d8c'}, 'models/ggml-vocab-qwen35.gguf': {'size_bytes': 5928682, 'sha256': '63ed952ff338996cf0bdf24a7b10015124273f75c6dc9bb427356aa3f67ec62c'}, 'models/ggml-vocab-refact.gguf': {'size_bytes': 1720710, 'sha256': 'ac3ceda902fed91ccf74312b305d9b86c37e4f8e35fa9cc6ef3ce34fca7d4678'}, 'models/ggml-vocab-starcoder.gguf': {'size_bytes': 1719346, 'sha256': 'fedb892b4e1bd3c1f2fcdae356440b14fb458f4264d586e5c987ed93df4e174d'}}
REQUIRED_CACHE = {
    "CMAKE_BUILD_TYPE": "Release", "CMAKE_CUDA_ARCHITECTURES": "120",
    "CMAKE_CUDA_RUNTIME_LIBRARY": "Static", "STRATA_ENABLE_CUDA": "ON",
    "STRATA_PORTABLE": "ON", "STRATA_NATIVE_EXPERTS": "ON",
    "STRATA_MMQ_KQUANTS": "OFF", "STRATA_BUILD_TESTS": "OFF",
}
VERSIONS = {"nvcc": "13.4.59", "cmake": "3.31.6", "ninja": "1.12.1", "cl": "19.44.35229"}
# Each tuple is actual base, after I/O, after execution-v1, final boundary-v2.
# The generate.cpp overlap is one distinct source and has all four preimages.
SOURCE_CHAIN = {
    "include/strata/core/expert_source.hpp": (
        "32f9bc3e7e72c2713d9323dd7353ad70e6b58604daa4e4c18a7e20e1c9929f1f",
        "b6e40e35eafa70c5a7affc44293b7d088516e4419a6d7c39d590221e7063c081"),
    "include/strata/kernels/ngram.hpp": (
        "edc6d36eaf6062e8c377577de787ed9e5576ede19e04c5964dc7e6b83b8d01ac",
        "638db1f925019030a7e91316dd788ccb19839122808df214e34d25013043c3c4"),
    "include/strata/ngram/ple_reader.hpp": (
        "d19b99a681a533f47144293bb170913c1496b5e0c344aed097ad1418375f7498",
        "fc64ba52686cda814e47846462c3a02f15d33edb247a653c38c85a773619037c"),
    "src/core/expert_source.cpp": (
        "2f0e14294ba463fd4fbcfb58829865c5f33712bcc72c6856ebbe2ff595b6e208",
        "97ce82cc3ca476f0b647617ed73dee7f5791bd277401bc9640207db7c6942ad0"),
    "src/kernels/ngram.cpp": (
        "e609caf1bb0812f366d96a5304c0f98e2864c1c0d22a0bef7b35d818e09debd5",
        "cabe072a32d9499fb054c0438780ab68a1e05387f16452ff7dd09965cac584bb"),
    "src/ngram/ple_reader.cpp": (
        "ddec6dbcc8afd8186c724403d5aaa54b70b9fbd551fea733ce422974131ede9e",
        "ca5d63ee37d739f6f08273577bde182987e2b3bb5da856e59cb8e54bec4d517c"),
    "src/program/generate.cpp": (
        "f4eda3a268133016a74b80d61dca99d6e4ce0df99dae4862fd3f6c1026a11d56",
        "d6cec7139469ded5e949025e145cbbd1328212d9e151f10d1f118b522cd15cb8",
        "d0c92501d8fb57751b0f25aeffdaf8de4b9ce69b99f824e99ffc51dab49a204a"),
    "src/kernels/cpu/pool.cpp": (
        "e54f0c2fe6a297c4aa49a3a9efabab3166d8644f05385b9f7e2fa5e5d34f8eff",
        "e54f0c2fe6a297c4aa49a3a9efabab3166d8644f05385b9f7e2fa5e5d34f8eff",
        "0aea96e3f050b1e72563c9939224328422ecdec323b2a02ae6d5ff3fbdb32975"),
    "src/core/session.cpp": (
        "9ce0156024e2a923df5baf913c7474f1b1462b8fec20e6d12775b2e0c992bb00",
        "9ce0156024e2a923df5baf913c7474f1b1462b8fec20e6d12775b2e0c992bb00",
        "761a324da5842095bd0d08b5cefab6cb8ba2256a5ace6dcd13f9a100b2f90035"),
    "src/core/verify.cpp": (
        "8ad078d10468c28f95b29bb996e78b3600a61c9d5408219324db125c9b7a2472",
        "8ad078d10468c28f95b29bb996e78b3600a61c9d5408219324db125c9b7a2472",
        "9d923524df6b4accba027f1476f12567bf81ee64a6e6f0c7f7680eb94352c312"),
    "src/core/expert_cache.cpp": (
        "583f090a5d013d5e52289d1c844f5a9877e7768886dc775aa575d5d333b95051",
        "583f090a5d013d5e52289d1c844f5a9877e7768886dc775aa575d5d333b95051",
        "84bfc1ae38cc8c8f9afe649613716ca397b506b656216a48a0d9781e3f32c387"),
    HEADER: (None, None, HEADER_V1, HEADER_V2),
}
SOURCE_CHAIN = {path: (*values, *(values[-1] for _ in range(4 - len(values))))
                for path, values in SOURCE_CHAIN.items()}


class EvidenceError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def object_(value, keys, code):
    require(type(value) is dict and set(value) == set(keys), code)
    return value


def uint(value, maximum=MAX_TOTAL, positive=False):
    require(type(value) is int and int(positive) <= value <= maximum, "integer_bound")
    return value


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "digest_shape")
    return value


def relative(value):
    require(type(value) is str and 0 < len(value) <= 512 and value.isascii(), "member_path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and p.as_posix() == value and not any(
        part in ("", ".", "..") or ":" in part or "\\" in part or part.endswith((".", " "))
        or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)
        for part in p.parts), "member_path")
    return value


def bounded_json_structure(value, node_limit=None):
    """Bound post-parse traversal independently of CPython recursion settings."""
    node_limit = MAX_JSON_NODES if node_limit is None else node_limit
    stack, nodes = [(iter((value,)), 0)], 0
    while stack:
        iterator, depth = stack[-1]
        try:
            node = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        require(nodes <= node_limit, "json_node_bound")
        if type(node) in (dict, list):
            require(depth < MAX_JSON_DEPTH, "json_depth_bound")
            stack.append((iter(node.values() if type(node) is dict else node), depth + 1))
    return value


def json_(raw, *, maximum=None, node_limit=None):
    maximum = MAX_JSON if maximum is None else maximum
    require(type(raw) is bytes and len(raw) <= maximum, "json_bound")
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate_json_key")
            out[key] = value
        return out
    def reject(_):
        raise EvidenceError("non_integer_json_number")
    try:
        return bounded_json_structure(json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                                                 parse_float=reject, parse_constant=reject), node_limit)
    except EvidenceError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError):
        raise EvidenceError("invalid_json") from None


def file_bytes(path, maximum):
    """Bound read with no symlink/junction traversal and stable open-file identity."""
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "file_bound")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(len(data) <= maximum and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "file_changed")
    return data


def contained_file(root, name):
    relative(name)
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(), "reparse_member")
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), "member_escape")
    return resolved


def member_manifest_json(raw):
    value = object_(json_(raw, maximum=MAX_MANIFEST_BYTES, node_limit=MAX_MANIFEST_NODES),
                    {"schema", "files"}, "manifest_shape")
    require(value["schema"] == "omni-strata-combined-members-v2", "manifest_schema")
    require(type(value["files"]) is list and 0 < len(value["files"]) <= MAX_FILES, "manifest_count")
    return value


def dependency_source_roots(descriptor):
    stages = object_(descriptor["source_stage_dirs"],
        {"base", "after_io", "after_exec_v1", "after_boundary_v2", "post_build", "dependency-base", "dependency-post_build"},
        "stage_roots")
    roots = tuple(relative(stages[key]) + "/source" for key in ("dependency-base", "dependency-post_build"))
    require(len(set(roots)) == 2 and all(not a.startswith(b + "/") for a in roots for b in roots if a != b),
            "dependency_source_root_alias")
    return roots


def dependency_vocabulary_member(name, row, roots):
    for root in roots:
        prefix = root + "/"
        if name.startswith(prefix):
            record = DEPENDENCY_VOCABULARY.get(name[len(prefix):])
            return record is not None and type(row["size_bytes"]) is int and row["size_bytes"] == record["size_bytes"] \
                and row["sha256"] == record["sha256"]
    return False


def strata_source_roots(descriptor):
    stages = object_(
        descriptor["source_stage_dirs"],
        {
            "base",
            "after_io",
            "after_exec_v1",
            "after_boundary_v2",
            "post_build",
            "dependency-base",
            "dependency-post_build",
        },
        "stage_roots",
    )
    roots = tuple(
        relative(stages[key]) + "/source"
        for key in ("base", "after_io", "after_exec_v1", "after_boundary_v2", "post_build")
    )
    all_roots = roots + dependency_source_roots(descriptor)
    require(
        len(set(all_roots)) == 7 and all(not a.startswith(b + "/") for a in all_roots for b in all_roots if a != b),
        "strata_source_root_alias",
    )
    return roots

def strata_source_auxiliary_member(name, row, roots):
    # Preliminary manifest admission only. Full Git closure and all six actual
    # byte preimages are independently required before static identity returns.
    allowed = {STRATA_SOURCE_AUXILIARY["path"]} | {root + "/" + STRATA_SOURCE_AUXILIARY["path"] for root in roots}
    return (
        len(roots) == 5
        and name in allowed
        and type(row["size_bytes"]) is int
        and row["size_bytes"] == STRATA_SOURCE_AUXILIARY["size_bytes"]
        and row["sha256"] == STRATA_SOURCE_AUXILIARY["sha256"]
    )


class Bundle:
    """No caller supplied verified Boolean or verified-files set is trusted."""

    def __init__(self, root, manifest_file, *, dependency_source_roots=(), strata_source_roots=()):
        self.root = Path(root).resolve(strict=True)
        require(self.root.is_dir(), "runtime_root")
        self.manifest_file = relative(manifest_file)
        require(
            type(dependency_source_roots) is tuple
            and len(dependency_source_roots) in (0, 2)
            and len(set(dependency_source_roots)) == len(dependency_source_roots),
            "dependency_source_roots_shape",
        )
        roots = tuple(relative(root) for root in dependency_source_roots)
        require(
            type(strata_source_roots) is tuple
            and len(strata_source_roots) in (0, 5)
            and len(set(strata_source_roots)) == len(strata_source_roots),
            "strata_source_roots_shape",
        )
        strata_roots = tuple(relative(root) for root in strata_source_roots)
        raw = file_bytes(self.path(manifest_file, listed=False), MAX_MANIFEST_BYTES)
        manifest = member_manifest_json(raw)
        rows = manifest["files"]
        require(type(rows) is list and 0 < len(rows) <= MAX_FILES, "manifest_count")
        self.rows = {}
        folded = set()
        total = 0
        for row in rows:
            object_(row, {"path", "size_bytes", "sha256"}, "manifest_row")
            name = relative(row["path"])
            require(
                name not in self.rows and name.casefold() not in folded and name != manifest_file, "manifest_duplicate"
            )
            require(
                Path(name).suffix.lower() not in (".gguf", ".safetensors")
                or Path(name).suffix.lower() == ".gguf"
                and (
                    dependency_vocabulary_member(name, row, roots)
                    or strata_source_auxiliary_member(name, row, strata_roots)
                ),
                "model_member_forbidden",
            )
            size = uint(row["size_bytes"], MAX_MEMBER)
            digest(row["sha256"])
            total += size
            require(total <= MAX_TOTAL, "manifest_total")
            self.rows[name] = row
            folded.add(name.casefold())
        self.manifest_sha256 = sha(raw)
        del raw, manifest, rows, folded
        # Bounded enumeration refuses unlisted injections, including pycache.
        found, pending, directories = set(), [self.root], 0
        while pending:
            directory = pending.pop()
            directories += 1
            require(directories <= MAX_DIRS, "directory_count")
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    require(
                        not entry.is_symlink() and not getattr(path, "is_junction", lambda: False)(), "reparse_member"
                    )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    else:
                        require(entry.is_file(follow_symlinks=False), "non_regular_member")
                        found.add(path.relative_to(self.root).as_posix())
                        require(len(found) <= MAX_FILES + 1, "member_count")
        require(found == set(self.rows) | {manifest_file}, "unlisted_or_missing_member")
        for name, row in self.rows.items():
            path = self.path(name)
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                require(before.st_size == row["size_bytes"], "member_size")
                h = hashlib.sha256()
                count = 0
                for block in iter(lambda: stream.read(1 << 20), b""):
                    count += len(block)
                    require(count <= row["size_bytes"], "member_changed")
                    h.update(block)
                after = os.fstat(stream.fileno())
            require(
                count == row["size_bytes"]
                and h.hexdigest() == row["sha256"]
                and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                "member_hash_or_change",
            )

    def path(self, name, listed=True):
        relative(name)
        if listed:
            require(name in self.rows, "unbound_member")
        return contained_file(self.root, name)

    def read(self, name, maximum=MAX_TEXT):
        path = self.path(name)
        data = file_bytes(path, maximum)
        require(
            self.path(name) == path
            and len(data) == self.rows[name]["size_bytes"]
            and sha(data) == self.rows[name]["sha256"],
            "member_reread_changed",
        )
        return data

    def load(self, name):
        return json_(self.read(name, MAX_JSON))

    def expected(self, name, expected, *, read=True):
        require(self.rows[relative(name)]["sha256"] == expected, "reviewed_source_hash")
        return self.read(name) if read else self.path(name)


def inventory(bundle, name, prefix):
    value = bundle.load(name)
    if type(value) is list:
        rows, producer_shape = value, True
    else:
        value = object_(value, {"schema", "files"}, "inventory_shape")
        require(value["schema"] == "omni-strata-source-inventory-v2", "inventory_schema")
        rows, producer_shape = value["files"], False
    require(type(rows) is list and 0 < len(rows) <= MAX_FILES, "inventory_count")
    result = {}
    for row in rows:
        object_(row, {"path", "size_bytes", "sha256", "archive_path"} if producer_shape else
                {"path", "size_bytes", "sha256"}, "inventory_row")
        path = relative(row["path"])
        if producer_shape:
            require(row["archive_path"] == "source/" + path, "source_stage_archive_path")
        require(path not in result, "inventory_duplicate")
        member = relative(prefix + "/" + path)
        require(bundle.rows.get(member) == {"path": member, "size_bytes": uint(row["size_bytes"], MAX_MEMBER),
                                           "sha256": digest(row["sha256"])}, "inventory_actual_bytes")
        result[path] = {key: row[key] for key in ("path", "size_bytes", "sha256")}
    require(list(result) == sorted(result), "inventory_order")
    return result


def fixed_source_afterimage(raw, canonical_blob, expected, canonical_base=False):
    if canonical_base:
        require(sha(canonical_blob) == expected, "git_fixed_IO_base_canonical_pin")
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeError:
            raise EvidenceError("git_fixed_IO_base_encoding") from None
        require(b"\0" not in raw and (raw == canonical_blob or raw.replace(b"\r\n", b"\n") == canonical_blob),
                "git_fixed_IO_base_checkout_transform")
    else:
        require(expected is not None and sha(raw) == expected, "git_fixed_patch_afterimage")


IO_BASE_CANONICAL_PATHS = frozenset((
    "include/strata/core/expert_source.hpp", "include/strata/kernels/ngram.hpp",
    "include/strata/ngram/ple_reader.hpp", "src/core/expert_source.cpp",
    "src/kernels/ngram.cpp", "src/ngram/ple_reader.cpp", "src/program/generate.cpp",
))


def git_source(bundle, proof, expected_commit, expected_tree=None, overrides=None, canonical_base_paths=frozenset()):
    """Verify raw Git commit/tree/blob preimages, not a reported HEAD Boolean."""
    object_(proof, {"commit_file", "tree_objects", "inventory_file", "archived_root", "gitlinks"}, "git_proof_shape")
    require(canonical_base_paths == frozenset() or canonical_base_paths == IO_BASE_CANONICAL_PATHS
            and expected_commit == BASE and overrides is not None, "IO_base_scope")
    commit = bundle.read(proof["commit_file"], MAX_GIT_OBJECT)
    require(hashlib.sha1(b"commit " + str(len(commit)).encode() + b"\0" + commit).hexdigest() == expected_commit,
            "git_commit_preimage")
    match = re.match(rb"tree ([0-9a-f]{40})\n", commit)
    require(match is not None, "git_commit_tree")
    tree_id = match[1].decode("ascii")
    require(expected_tree is None or tree_id == expected_tree, "git_tree_identity")
    records = proof["tree_objects"]
    require(type(records) is list and 0 < len(records) <= MAX_DIRS, "git_tree_count")
    trees = {}
    for row in records:
        object_(row, {"oid", "file"}, "git_tree_record")
        require(type(row["oid"]) is str and re.fullmatch(r"[0-9a-f]{40}", row["oid"])
                and row["oid"] not in trees, "git_oid")
        raw = bundle.read(row["file"], MAX_GIT_OBJECT)
        require(hashlib.sha1(b"tree " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == row["oid"], "git_tree_preimage")
        trees[row["oid"]] = raw
    links = proof["gitlinks"]
    require(type(links) is dict and all(type(k) is str and type(v) is str for k, v in links.items()), "gitlinks")
    require(not links or links == {"third_party/llama.cpp": DEPENDENCY}, "unreviewed_gitlink")
    files, actual_links, visited, canonical_hashes = {}, {}, set(), {}
    def walk(oid, prefix, depth):
        require(depth <= 64 and oid in trees, "git_tree_missing_or_depth")
        visited.add(oid)
        data, offset = trees[oid], 0
        while offset < len(data):
            end_mode = data.find(b" ", offset)
            end_name = data.find(b"\0", end_mode + 1)
            require(end_mode > offset and end_name > end_mode and end_name + 21 <= len(data), "git_tree_encoding")
            mode = data[offset:end_mode]
            try:
                name = data[end_mode + 1:end_name].decode("ascii")
            except UnicodeError:
                raise EvidenceError("git_path_encoding") from None
            path = relative(prefix + name)
            require("/" not in name, "git_path_component")
            child = data[end_name + 1:end_name + 21].hex()
            offset = end_name + 21
            if mode == b"40000":
                walk(child, path + "/", depth + 1)
            elif mode == b"160000":
                require(path not in actual_links, "git_duplicate")
                actual_links[path] = child
            else:
                require(mode in (b"100644", b"100755") and path not in files, "git_mode_or_duplicate")
                require(len(files) < MAX_FILES, "git_file_count")
                canonical_member = (PurePosixPath(proof["commit_file"]).parent / (child + ".blob")).as_posix()
                canonical_blob = bundle.read(canonical_member, MAX_GIT_OBJECT)
                require(hashlib.sha1(b"blob " + str(len(canonical_blob)).encode() + b"\0" + canonical_blob).hexdigest() == child,
                        "git_blob_preimage")
                raw = bundle.read(proof["archived_root"] + "/" + path, MAX_TEXT)
                if overrides is not None and path in overrides:
                    fixed_source_afterimage(raw, canonical_blob, overrides[path], path in canonical_base_paths)
                elif raw != canonical_blob:
                    # Checked-out CRLF is recorded as a transformation, never
                    # confused with canonical Git bytes. Binary filters refuse.
                    try:
                        raw.decode("utf-8", errors="strict")
                    except UnicodeError:
                        raise EvidenceError("git_binary_checkout_transform") from None
                    require(b"\0" not in raw and raw.replace(b"\r\n", b"\n") == canonical_blob,
                            "git_checkout_differs_beyond_CRLF")
                canonical_hashes[path] = {"git_blob_id": child, "mode": mode.decode("ascii"),
                                          "canonical_size_bytes": len(canonical_blob), "canonical_sha256": sha(canonical_blob),
                                          "checkout_sha256": sha(raw), "checkout_matches_canonical": raw == canonical_blob}
                files[path] = {"path": path, "size_bytes": len(raw), "sha256": sha(raw)}
    walk(tree_id, "", 0)
    if overrides is not None:
        # New fixed source files do not occur in the original Git tree.
        for path, expected in overrides.items():
            if path not in files and expected is not None:
                raw = bundle.read(proof["archived_root"] + "/" + path, MAX_TEXT)
                require(sha(raw) == expected, "git_new_fixed_patch_afterimage")
                files[path] = {"path": path, "size_bytes": len(raw), "sha256": expected}
    require(visited == set(trees) and actual_links == links, "git_tree_closure")
    declared = inventory(bundle, proof["inventory_file"], proof["archived_root"])
    require(files == declared, "git_inventory_closure")
    return {"revision": expected_commit, "tree": tree_id, "files": files, "canonical_source_hashes": canonical_hashes}


def recorded_file(bundle, row, member, code):
    """Bind a raw native-path receipt to contained bytes without reading that path."""
    object_(row, {"path", "size_bytes", "sha256"}, code + "_shape")
    winpath(row["path"])
    require(bundle.rows.get(member) == {"path": member, "size_bytes": uint(row["size_bytes"], MAX_MEMBER),
                                       "sha256": digest(row["sha256"])}, code + "_bytes")
    return member


def verify_checkout_controls(bundle, folder, controls, dependency):
    common = {"core_autocrlf", "source_rewritten"}
    extra = {"header_path", "attributes_sha256", "attributes_archive_path", "attributes_size_bytes",
             "check_attr_sha256", "check_attr_archive_path"}
    object_(controls, common if dependency else common | extra, "checkout_control_shape")
    require(controls["core_autocrlf"] == "true" and controls["source_rewritten"] is False, "checkout_control_scope")
    result = {"scope": "archived_checkout_rule_and_report_not_current_Git_configuration_attestation",
              "core_autocrlf": "true", "source_rewritten": False}
    if not dependency:
        expected_attributes = (HEADER + " text eol=lf\n").encode("ascii")
        expected_report = (HEADER + ": text: set\n" + HEADER + ": eol: lf\n").encode("ascii")
        require(controls["header_path"] == HEADER
                and controls["attributes_archive_path"] == "checkout-controls/git-info-attributes"
                and controls["check_attr_archive_path"] == "checkout-controls/git-check-attr.txt"
                and type(controls["attributes_size_bytes"]) is int
                and controls["attributes_size_bytes"] == len(expected_attributes), "header_checkout_control_identity")
        for key, expected in (("attributes", expected_attributes), ("check_attr", expected_report)):
            name = folder + "/" + controls[key + "_archive_path"]
            require(controls[key + "_sha256"] == sha(expected), "header_checkout_control_hash")
            bundle.expected(name, sha(expected), read=False)
            require(bundle.read(name, 4096) == expected, "header_checkout_control_preimage")
            result[key + "_sha256"] = sha(expected)
    return result


def stage_source_recorder(key, descriptor):
    amended_post = key == "dependency-post_build" and "dependency_post_source_recorder_file" in descriptor
    return ((descriptor["dependency_post_source_recorder_file"], POST_SOURCE_RECORDER_SHA) if amended_post
            else (descriptor["source_recorder_file"], SOURCE_RECORDER_SHA))


def source_stages(bundle, descriptor, base, dependency):
    """Reconcile the recorder's actual seven full snapshots and raw Git objects."""
    roots = object_(descriptor["source_stage_dirs"], {"base", "after_io", "after_exec_v1", "after_boundary_v2", "post_build",
                                                   "dependency-base", "dependency-post_build"}, "stage_roots")
    results = {}
    for key, folder in roots.items():
        folder = relative(folder)
        dep = key.startswith("dependency-")
        stage = key.removeprefix("dependency-") if dep else key
        original = dependency if dep else base
        revision = DEPENDENCY if dep else BASE
        raw = bundle.load(folder + "/receipt.json")
        object_(raw, {"schema", "status", "kind", "stage", "source_root", "revision", "limits", "tree", "files",
                      "final_patched_sources", "git_status_porcelain", "full_inventory_sha256", "canonical_leaves_sha256",
                      "git_executable_sha256", "helper_sha256", "checkout_controls"}, "actual_stage_shape")
        require((raw["schema"], raw["status"], raw["kind"], raw["stage"], raw["revision"], raw["tree"])
                == ("omni-strata-execution-source-stage-v1", "source_stage_archived_not_built",
                    "dependency" if dep else "strata", stage, revision, original["tree"]), "actual_stage_identity")
        require(raw["limits"] == {"max_file_bytes": 64 << 20, "max_total_bytes": 1 << 30, "max_files": 10000}, "actual_stage_limits")
        require(type(raw["git_status_porcelain"]) is str and len(raw["git_status_porcelain"]) <= MAX_TEXT
                and (not (dep or stage == "base") or raw["git_status_porcelain"] == ""), "actual_stage_clean_base")
        recorder_file, expected_recorder = stage_source_recorder(key, descriptor)
        require(raw["helper_sha256"] == expected_recorder, "actual_stage_recorder_pin")
        checkout = verify_checkout_controls(bundle, folder, raw["checkout_controls"], dep)
        bundle.expected(recorder_file, expected_recorder)
        bundle.expected(descriptor["tool_binary_files"]["git"], digest(raw["git_executable_sha256"]), read=False)
        inv_name, leaves_name = folder + "/full-inventory.json", folder + "/canonical-leaves.json"
        bundle.expected(inv_name, digest(raw["full_inventory_sha256"]))
        bundle.expected(leaves_name, digest(raw["canonical_leaves_sha256"]))
        require(bundle.load(inv_name) == raw["files"], "actual_stage_inventory_preimage")
        index = 3 if stage == "post_build" else ("base", "after_io", "after_exec_v1", "after_boundary_v2").index(stage)
        overrides = {} if dep else {path: values[index] for path, values in SOURCE_CHAIN.items()}
        proof = {"commit_file": folder + "/git-objects/" + revision + ".commit",
                 "tree_objects": [{"oid": PurePosixPath(path).stem, "file": path} for path in sorted(bundle.rows)
                                  if path.startswith(folder + "/git-objects/") and path.endswith(".tree")],
                 "inventory_file": inv_name, "archived_root": folder + "/source", "gitlinks": {}}
        canonical_base_paths = IO_BASE_CANONICAL_PATHS if not dep and stage == "base" else frozenset()
        checked = git_source(bundle, proof, revision, original["tree"], overrides, canonical_base_paths)
        expected = copy.deepcopy(original["files"])
        for path, expected_hash in overrides.items():
            if expected_hash is None:
                require(path not in checked["files"], "actual_stage_absent_header")
            else:
                expected[path] = checked["files"][path]
        require(checked["files"] == expected, "actual_stage_no_other_source_changes")
        leaves = bundle.load(leaves_name)
        expected_leaves = [{"path": path, "git_blob_id": record["git_blob_id"], "mode": record["mode"],
                            "canonical_archive_path": "git-objects/" + record["git_blob_id"] + ".blob",
                            "canonical_sha256": record["canonical_sha256"], "canonical_size_bytes": record["canonical_size_bytes"]}
                           for path, record in sorted(original["canonical_source_hashes"].items())]
        require(leaves == expected_leaves, "actual_stage_canonical_leaves")
        expected_patched = [] if dep else [next((row for row in raw["files"] if row["path"] == path), {"path": path, "absent": True})
                                          for path in sorted(SOURCE_CHAIN)]
        require(raw["final_patched_sources"] == expected_patched, "actual_stage_twelve_record_set")
        results[key] = {"folder": folder, "source_root": winpath(raw["source_root"]), "files": checked["files"],
                        "receipt_sha256": bundle.rows[folder + "/receipt.json"]["sha256"], "tree": raw["tree"],
                        "checkout_controls": checkout}
    require(results["after_boundary_v2"]["files"] == results["post_build"]["files"]
            and results["dependency-base"]["files"] == results["dependency-post_build"]["files"], "actual_build_source_stability")
    require(descriptor["final_sources_dir"] == results["after_boundary_v2"]["folder"] + "/source"
            and descriptor["base_source_provenance"]["archived_root"] == results["base"]["folder"] + "/source"
            and descriptor["dependency_source_provenance"]["archived_root"] == results["dependency-base"]["folder"] + "/source",
            "actual_stage_archive_bindings")
    for row in descriptor["source_states"]:
        require(row["states"] == [None if value is None else results[stage]["folder"] + "/source/" + row["path"]
                                  for stage, value in zip(("base", "after_io", "after_exec_v1", "after_boundary_v2"), SOURCE_CHAIN[row["path"]])],
                "actual_intermediate_source_path_binding")
    return results


def verify_base_source_pins(base):
    for path, expected in SOURCE_CHAIN.items():
        if expected[0] is None:
            require(path not in base["files"], "new_header_in_base")
        elif path in IO_BASE_CANONICAL_PATHS:
            require(base["canonical_source_hashes"].get(path, {}).get("canonical_sha256") == expected[0],
                    "base_IO_canonical_source_preimage")
        else:
            require(base["files"].get(path, {}).get("sha256") == expected[0], "base_source_preimage")


def source_chain(bundle, rows, base=None):
    require(type(rows) is list and len(rows) == 12, "twelve_source_records")
    actual = {}
    for row in rows:
        object_(row, {"path", "states"}, "source_state_record")
        name = relative(row["path"])
        require(name in SOURCE_CHAIN and name not in actual, "source_state_identity")
        require(type(row["states"]) is list and len(row["states"]) == 4, "source_state_count")
        actual[name] = []
        for index, (path, expected) in enumerate(zip(row["states"], SOURCE_CHAIN[name])):
            if expected is None:
                require(path is None, "new_source_must_be_absent")
                actual[name].append(None)
            else:
                if index == 0 and name in IO_BASE_CANONICAL_PATHS:
                    require(type(base) is dict and base["canonical_source_hashes"][name]["canonical_sha256"] == expected,
                            "source_IO_base_canonical_proof")
                    expected_raw = base["files"][name]["sha256"]
                    bundle.expected(path, expected_raw)
                    record = {"path": path, "size_bytes": bundle.rows[path]["size_bytes"], "sha256": expected_raw,
                              "canonical_sha256": expected, "pin_scope": "base_IO_canonical_Git_blob_raw_checkout_separately_bound"}
                else:
                    bundle.expected(path, expected)
                    record = {"path": path, "size_bytes": bundle.rows[path]["size_bytes"], "sha256": expected}
                actual[name].append(record)
    require(set(actual) == set(SOURCE_CHAIN), "source_state_set")
    return actual


def winpath(value, directory=None):
    require(type(value) is str and 0 < len(value) <= 32768 and not any(x in value for x in ("\0", "\r", "\n")), "native_evidence_path")
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if not ntpath.isabs(value):
        require(directory is not None, "native_relative_evidence_path")
        value = ntpath.join(directory, value)
    return ntpath.normcase(ntpath.normpath(value))



def ninja_path_tokens(text, variables, expanding=()):
    """Parse bounded Ninja path escapes; never execute rule commands."""
    require(type(text) is str and len(text) <= 1 << 20 and len(expanding) <= 32, "Ninja_path_bound")
    words, word, index = [], [], 0
    while index < len(text):
        char = text[index]
        if char == "$":
            index += 1
            require(index < len(text), "Ninja_unterminated_escape")
            char = text[index]
            if char in " $:":
                word.append(char)
            else:
                if char == "{":
                    end = text.find("}", index + 1)
                    require(end >= 0, "Ninja_unterminated_variable")
                    name, index = text[index + 1:end], end
                else:
                    match = re.match(r"[A-Za-z0-9_]+", text[index:])
                    require(match is not None, "Ninja_path_escape")
                    name = match.group()
                    index += len(name) - 1
                require(name in variables and name not in expanding, "Ninja_unknown_or_cyclic_variable")
                values = ninja_path_tokens(variables[name], variables, (*expanding, name))
                require(len(values) == 1, "Ninja_path_variable_not_singleton")
                word.append(values[0])
        elif char.isspace():
            if word:
                words.append("".join(word)); word = []
        elif char in ":|":
            if word:
                words.append("".join(word)); word = []
            if char == "|" and index + 1 < len(text) and text[index + 1] in "|@":
                index += 1; char += text[index]
            words.append(char)
        else:
            require(32 <= ord(char) <= 126, "Ninja_path_encoding")
            word.append(char)
        index += 1
    if word: words.append("".join(word))
    require(len(words) <= 10000 and all(len(value) <= 32768 for value in words), "Ninja_token_bound")
    return words


def ninja_logical_lines(raw):
    require(type(raw) is bytes and len(raw) <= 8 << 20, "Ninja_graph_bytes")
    lines, result, index = raw.decode("latin1").splitlines(), [], 0
    require(len(lines) <= 100000, "Ninja_graph_line_bound")
    while index < len(lines):
        start, text = index + 1, lines[index]
        while (len(text) - len(text.rstrip("$"))) & 1:
            index += 1
            require(index < len(lines), "Ninja_continuation_unterminated")
            text = text[:-1] + lines[index].lstrip()
            require(len(text) <= 1 << 20, "Ninja_logical_line_bound")
        result.append((start, text)); index += 1
    return result


def verify_archived_ninja_graph(bundle, rows, prefix, build_root):
    """Independently derive the selected target from archived graph/includes."""
    require(type(rows) is list and 0 < len(rows) <= 64, "Ninja_graph_member_count")
    members, total = {}, 0
    for row in rows:
        object_(row, {"path", "size_bytes", "sha256", "archive_path"}, "Ninja_graph_member_shape")
        archive = relative(row["archive_path"])
        require(archive.startswith("graph/"), "Ninja_graph_archive_path")
        name = relative(archive[len("graph/"):])
        require(name not in members and name.casefold() not in {x.casefold() for x in members}, "Ninja_graph_member_alias")
        member = prefix + "/" + archive
        require(winpath(row["path"]) == winpath(ntpath.join(build_root, name)), "Ninja_graph_native_path")
        recorded_file(bundle, {k: row[k] for k in ("path", "size_bytes", "sha256")}, member, "Ninja_graph_actual_bytes")
        total += uint(row["size_bytes"], 8 << 20)
        require(total <= 8 << 20, "Ninja_graph_total_bytes")
        members[name] = bundle.read(member, 8 << 20)
    require("build.ninja" in members, "Ninja_root_graph_missing")
    seen, visiting, by_output, edges, rules, edge_bindings = set(), set(), {}, [], {}, {}
    def read(name, variables, depth):
        name = relative(name.replace("\\", "/"))
        require(depth <= 32 and name in members and name not in visiting and name not in seen, "Ninja_include_missing_cycle_or_duplicate")
        visiting.add(name); seen.add(name)
        active_rule, active_edge = None, None
        for number, line in ninja_logical_lines(members[name]):
            if not line or line.startswith("#"): continue
            if line.startswith((" ", "\t")):
                if active_rule is not None:
                    match = re.fullmatch(r"([A-Za-z0-9_]+) = (.*)", line.strip())
                    require(match is not None and match[1] not in rules[active_rule]["bindings"], "Ninja_rule_binding_shape")
                    rules[active_rule]["bindings"][match[1]] = match[2]
                elif active_edge is not None:
                    match = re.fullmatch(r"([A-Za-z0-9_]+) = (.*)", line.strip())
                    require(match is not None and match[1] not in edge_bindings[active_edge], "Ninja_edge_binding_shape")
                    edge_bindings[active_edge][match[1]] = match[2]
                continue
            active_rule, active_edge = None, None
            if line.startswith(("include ", "subninja ")):
                kind, text = line.split(" ", 1)
                values = ninja_path_tokens(text, variables)
                require(len(values) == 1 and not ntpath.isabs(values[0]), "Ninja_include_path")
                # Ninja includes are addressed relative to the build working directory.
                read(values[0], variables if kind == "include" else dict(variables), depth + 1)
            elif line.startswith("build "):
                values = ninja_path_tokens(line[6:], variables)
                require(values.count(":") == 1, "Ninja_build_statement")
                split = values.index(":")
                outputs = [x for x in values[:split] if x != "|"]
                require(outputs and split + 1 < len(values) and len(edges) < 10000, "Ninja_edge_bound")
                rule, inputs, mode = values[split + 1], {"explicit": [], "implicit": [], "order_only": [], "validation": []}, "explicit"
                for value in values[split + 2:]:
                    if value in ("|", "||", "|@"):
                        mode = {"|": "implicit", "||": "order_only", "|@": "validation"}[value]
                    else: inputs[mode].append(value)
                edge = {"outputs": outputs, "rule": rule, "inputs": inputs, "origin_file": name, "origin_line": number, "attributes": {}}
                for value in outputs:
                    key = winpath(value, build_root)
                    require(key not in by_output, "Ninja_duplicate_output")
                    by_output[key] = edge
                edges.append(edge)
                active_edge = (name, number)
                edge_bindings[active_edge] = edge["attributes"]
            elif line.startswith("rule "):
                rule = line[5:]
                require(re.fullmatch(r"[A-Za-z0-9_.-]+", rule) is not None and rule not in rules, "Ninja_rule_name_or_duplicate")
                rules[rule] = {"origin_file": name, "origin_line": number, "bindings": {}}
                active_rule = rule
            elif re.match(r"^[A-Za-z0-9_]+ = ", line):
                key, value = line.split(" = ", 1); variables[key] = value
            elif not line.startswith(("pool ", "default ")):
                raise EvidenceError("Ninja_unknown_top_level_syntax")
        visiting.remove(name)
    read("build.ninja", {}, 0)
    require(seen == set(members), "Ninja_unreferenced_graph_member")
    require(all(edge["rule"] == "phony" or edge["rule"] in rules for edge in edges), "Ninja_unknown_build_rule")
    reachable, pending = {}, set()
    def visit(name, kind, depth):
        key = winpath(name, build_root)
        require(depth <= 512 and key not in pending and len(reachable) <= 10000, "Ninja_target_cycle_or_bound")
        if key in reachable:
            reachable[key]["input_kinds"].add(kind); return
        edge = by_output.get(key)
        reachable[key] = {"name": name, "path": key, "rule": None if edge is None else edge["rule"], "input_kinds": {kind}, "edge": edge}
        if edge is not None:
            pending.add(key)
            for input_kind, names in edge["inputs"].items():
                for child in names: visit(child, input_kind, depth + 1)
            pending.remove(key)
    visit("strata", "target", 0)
    require(reachable[winpath("strata", build_root)]["rule"] == "phony"
            and winpath("strata.exe", build_root) in reachable, "Ninja_selected_engine_alias")
    for row in reachable.values(): row["input_kinds"] = sorted(row["input_kinds"])
    return {"members": tuple(members), "edges": edges, "by_output": by_output, "rules": rules, "reachable": reachable, "edge_bindings": edge_bindings}



def verified_dependency_command(configured, actual, edge, rule, build_root):
    """Only exact graph-declared dependency flags differ; no command repair."""
    require(type(configured.get("command")) is str and type(configured.get("output")) is str
            and type(actual.get("command")) is str, "target_command_shape")
    template, bindings = rule["bindings"].get("command", ""), rule["bindings"]
    if bindings.get("deps") == "msvc" and " /showIncludes " in template:
        removed, mode = " /showIncludes", "graph_rule_msvc_showIncludes"
    elif bindings.get("deps") == "gcc" and " -MD -MT $out -MF $DEP_FILE " in template:
        output, depfile = configured["output"], edge["attributes"].get("DEP_FILE")
        require(depfile == output + ".d" and bindings.get("depfile") == "$DEP_FILE", "target_CUDA_depfile_rule")
        removed, mode = " -MD -MT " + output + " -MF " + depfile, "graph_rule_cuda_MD_MT_MF"
    else:
        raise EvidenceError("target_dependency_command_rule")
    observed = actual["command"]
    require(len(observed) <= 1 << 20 and observed.count(removed) == 1
            and observed.replace(removed, "", 1) == configured["command"], "target_command_extra_difference")
    require(winpath(actual["file"], build_root) == winpath(configured["file"], build_root)
            and winpath(actual["output"], build_root) == winpath(configured["output"], build_root)
            and winpath(actual["directory"]) == winpath(build_root), "target_compdb_identity")
    return {"mode": mode, "removed_exact_substring": removed, "configured_command": configured["command"],
            "actual_command": observed, "rule": edge["rule"], "rule_origin_file": rule["origin_file"], "rule_origin_line": rule["origin_line"]}


def actual_target_dependencies(raw, build_root):
    groups, active = [], None
    for line in raw.decode("utf-8").splitlines():
        if not line: active = None
        elif not line.startswith((" ", "\t")):
            require(": #deps " in line, "target_dependency_grammar")
            name, metadata = line.split(": #deps ", 1)
            match = re.fullmatch(r"([0-9]+), deps mtime ([0-9]+) \((VALID|STALE)\)", metadata)
            require(match is not None and len(groups) < 10000, "target_dependency_count_or_metadata")
            active = {"output": winpath(name, build_root), "metadata": metadata, "valid_record": match[3] == "VALID",
                      "declared_count": int(match[1]), "inputs": []}
            groups.append(active)
        else:
            require(active is not None and len(active["inputs"]) < 10000, "target_dependency_orphan_or_bound")
            active["inputs"].append(winpath(line, build_root))
    require(groups and all(len(row["inputs"]) == row["declared_count"] for row in groups), "target_dependency_actual_count")
    return groups


class _WitnessDecimal(str):
    """A witness-only decimal token; never a generic JSON number allowance."""


def alias_witness_json(raw):
    require(type(raw) is bytes and len(raw) <= MAX_JSON, "alias_witness_json_bound")
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate_json_key")
            out[key] = value
        return out
    def reject(_):
        raise EvidenceError("alias_witness_nonfinite")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_float=_WitnessDecimal, parse_constant=reject)
        bounded_json_structure(value)
        require(type(value) is dict and "elapsed_s" in value, "alias_witness_elapsed_missing")
        elapsed = value["elapsed_s"]
        require(type(elapsed) in (int, _WitnessDecimal) and 0 <= float(elapsed) <= 60,
                "alias_witness_elapsed_invalid")
        # Keep the exact decimal lexeme as metadata, without admitting floats
        # anywhere else or changing generic json_ / canonical number policy.
        value["elapsed_s"] = str(elapsed)
        pending = [value]
        while pending:
            item = pending.pop()
            require(type(item) is not _WitnessDecimal, "alias_witness_unexpected_decimal")
            if type(item) is dict: pending.extend(item.values())
            elif type(item) is list: pending.extend(item)
        return value
    except EvidenceError:
        raise
    except (UnicodeError, ValueError, TypeError, RecursionError, OverflowError):
        raise EvidenceError("invalid_alias_witness_json") from None


def alias_descriptor_keys(descriptor):
    present = ALIAS_DESCRIPTOR_KEYS & set(descriptor)
    require(not present or present == ALIAS_DESCRIPTOR_KEYS, "alias_descriptor_pair_required")
    require(not present or "build_closure_supplement_file" in descriptor, "alias_requires_target_supplement")
    return present


def verified_external_header_aliases(bundle, descriptor, original, compile_inputs, prefix):
    """Validate an archived current-OS witness; never attest build-time reads."""
    if not alias_descriptor_keys(descriptor):
        return {}, None
    member = relative(descriptor["external_header_alias_witness_file"])
    folder = PurePosixPath(member).parent.as_posix()
    collector = relative(descriptor["external_header_alias_collector_file"])
    require(PurePosixPath(member).name == "receipt.json" and collector == folder + "/collect.py",
            "alias_witness_member_paths")
    collector_source = bundle.expected(collector, ALIAS_COLLECTOR_SHA).decode("utf-8")
    captured_roots = re.findall(r'^RUNTIME = Path\("([^"\r\n]+)"\)$', collector_source, re.MULTILINE)
    require(len(captured_roots) == 1, "alias_collector_captured_runtime_root")
    captured_root = winpath(captured_roots[0])
    witness = alias_witness_json(bundle.expected(member, ALIAS_WITNESS_SHA))
    diagnosis = json_(bundle.expected(folder + "/diagnosis.json", ALIAS_DIAGNOSIS_SHA))
    object_(witness, {"all_owned_file_handles_closed", "api_geometry", "build_execution_attested",
        "collector_creation_filetime_100ns", "collector_pid", "collector_sha256", "compile_inputs_sha256",
        "diagnosis_sha256", "elapsed_s", "finished_utc", "full_static_eligibility", "kernel32_loaded_module_path",
        "models_loaded", "ninja_deps_sha256", "open_policy", "os_version", "pairs", "python_executable",
        "python_version", "raw_preimage_bytes", "runtime_installed", "schema", "scope", "source_stable",
        "started_utc", "status", "unique_header_bytes"}, "alias_witness_shape")
    require(witness["schema"] == "omni-strata-current-Windows-CUDA-header-alias-witness-v1"
            and witness["status"] == "current_exact_eleven_alias_pairs_and_archived_bytes_verified"
            and witness["scope"] == "current Windows held-handle namespace and content witness; not build-time execution attestation"
            and witness["collector_sha256"] == ALIAS_COLLECTOR_SHA and witness["diagnosis_sha256"] == ALIAS_DIAGNOSIS_SHA,
            "alias_witness_identity")
    require(all(witness[key] is False for key in ("build_execution_attested", "full_static_eligibility", "models_loaded", "runtime_installed"))
            and witness["all_owned_file_handles_closed"] is True and witness["source_stable"] is True, "alias_witness_scope")
    require(canonical(witness["api_geometry"]) == b'{"BY_HANDLE_FILE_INFORMATION_bytes":52,"FILE_ID_INFO_bytes":24}'
            and canonical(witness["open_policy"]) == b'{"access":"GENERIC_READ","disposition":"OPEN_EXISTING","share":"FILE_SHARE_READ_only"}',
            "alias_witness_api_policy")
    uint(witness["collector_pid"], (1 << 32) - 1, positive=True)
    uint(witness["collector_creation_filetime_100ns"], (1 << 64) - 1, positive=True)
    require(type(witness["os_version"]) is list and len(witness["os_version"]) == 5
            and all(type(x) is int and 0 <= x <= (1 << 32) - 1 for x in witness["os_version"][:4])
            and type(witness["os_version"][4]) is str and len(witness["os_version"][4]) <= 256, "alias_witness_OS_record")
    for key in ("python_version", "started_utc", "finished_utc"):
        require(type(witness[key]) is str and 0 < len(witness[key]) <= 1024, "alias_witness_text_record")
    winpath(witness["kernel32_loaded_module_path"])
    python = object_(witness["python_executable"], {"path", "size_bytes", "sha256"}, "alias_python_record")
    winpath(python["path"]); uint(python["size_bytes"], 8 << 20, positive=True); digest(python["sha256"])
    require(diagnosis["schema"] == "omni-strata-selected-dependency-path-projection-diagnosis-v1"
            and diagnosis["original_compile_inputs_sha256"] == witness["compile_inputs_sha256"]
            == bundle.rows[prefix + "/compile-inputs.json"]["sha256"]
            and diagnosis["original_ninja_deps_sha256"] == witness["ninja_deps_sha256"]
            == bundle.rows[descriptor["build_evidence_dir"] + "/ninja_deps.txt"]["sha256"], "alias_original_preimages")
    require(type(witness["pairs"]) is list and type(diagnosis["pairs"]) is list
            and len(witness["pairs"]) == len(diagnosis["pairs"]) == 11, "alias_exact_eleven_pairs")
    inputs = {row["archive_path"]: row for row in compile_inputs["inputs"]}
    source_roots = [winpath(original["commands"][0][2]), winpath(original["commands"][1][2])]
    source_roots += [winpath(x.split("=", 1)[1]) for x in original["commands"][0] if x.startswith("-DSTRATA_GGML_DIR=")]
    aliases, total = {}, 0
    def identity(row, size):
        object_(row, {"creation_filetime_100ns", "file_id_128_hex", "last_write_filetime_100ns",
            "legacy_file_index", "legacy_volume_serial_number", "size_bytes", "volume_serial_number"}, "alias_file_identity_shape")
        for key in ("creation_filetime_100ns", "last_write_filetime_100ns", "legacy_file_index", "volume_serial_number"):
            uint(row[key], (1 << 64) - 1)
        uint(row["legacy_volume_serial_number"], (1 << 32) - 1)
        require(uint(row["size_bytes"], 2 << 20, positive=True) == size and type(row["file_id_128_hex"]) is str
                and re.fullmatch(r"[0-9a-f]{32}", row["file_id_128_hex"]) is not None, "alias_file_identity_values")
    for index, (row, fixed) in enumerate(zip(witness["pairs"], diagnosis["pairs"])):
        base_keys = {"raw_ninja_path", "resolved_supplement_path", "archive_path", "family", "size_bytes", "sha256"}
        object_(fixed, base_keys, "alias_fixed_pair_shape")
        object_(row, base_keys | {"GetFinalPathNameByHandleW", "GetLongPathNameW", "archived_input_identity_after",
            "archived_input_identity_before", "preimages", "short_and_long_identity_after", "short_and_long_identity_before"}, "alias_pair_shape")
        require(canonical({key: row[key] for key in base_keys}) == canonical(fixed), "alias_pair_not_reviewed")
        short, long = winpath(row["raw_ninja_path"]), winpath(row["resolved_supplement_path"])
        require(short == row["raw_ninja_path"] and long == row["resolved_supplement_path"] and short != long
                and short not in aliases and long not in aliases.values()
                and not any(path == root or path.startswith(root + "\\") for path in (short, long) for root in source_roots),
                "alias_only_exact_external_paths")
        original_input = inputs.get(row["archive_path"])
        require(row["family"] == "external_toolchain_or_system_input" and type(original_input) is dict
                and original_input["family"] == row["family"] and winpath(original_input["path"]) == long
                and type(original_input["size_bytes"]) is int and original_input["size_bytes"] == row["size_bytes"]
                and original_input["sha256"] == row["sha256"], "alias_original_input_binding")
        size = uint(row["size_bytes"], 2 << 20, positive=True); total += size
        require(total <= 2 << 20, "alias_unique_header_total")
        for key in ("short_and_long_identity_before", "short_and_long_identity_after",
                    "archived_input_identity_before", "archived_input_identity_after"):
            identity(row[key], size)
        require(canonical(row["short_and_long_identity_before"]) == canonical(row["short_and_long_identity_after"])
                and canonical(row["archived_input_identity_before"]) == canonical(row["archived_input_identity_after"]), "alias_held_identity_changed")
        for field, keys in (("GetFinalPathNameByHandleW", {"short", "long", "archived"}), ("GetLongPathNameW", {"short", "long"})):
            names = object_(row[field], keys, "alias_resolution_shape")
            for key in keys:
                name = names[key]
                require(type(name) is str, "alias_resolution_type")
                expected_path = winpath(ntpath.join(captured_root, "c", row["archive_path"])) if key == "archived" else long
                require(winpath(name[4:] if name.startswith("\\\\?\\") else name) == expected_path, "alias_resolution_value")
        preimages = object_(row["preimages"], {"short", "long", "archived"}, "alias_preimage_shape")
        reference = bundle.read(prefix + "/" + relative(row["archive_path"]), 2 << 20)
        require(len(reference) == size and sha(reference) == digest(row["sha256"]), "alias_original_input_bytes")
        for kind, image in preimages.items():
            object_(image, {"path", "size_bytes", "sha256"}, "alias_preimage_record")
            require(image["path"] == "evidence/pair-" + str(index).zfill(2) + "/" + kind + ".bin"
                    and type(image["size_bytes"]) is int and image["size_bytes"] == size and image["sha256"] == row["sha256"], "alias_preimage_binding")
            require(bundle.read(folder + "/" + relative(image["path"]), 2 << 20) == reference, "alias_preimage_bytes")
        aliases[short] = long
    require(type(witness["unique_header_bytes"]) is int and witness["unique_header_bytes"] == total
            and type(witness["raw_preimage_bytes"]) is int and witness["raw_preimage_bytes"] == 3 * total, "alias_witness_byte_totals")
    return aliases, {"schema": "omni-strata-archived-header-alias-reconciliation-v1", "witness_sha256": ALIAS_WITNESS_SHA,
        "collector_sha256": ALIAS_COLLECTOR_SHA, "diagnosis_sha256": ALIAS_DIAGNOSIS_SHA, "exact_pair_count": 11,
        "scope": "archived_current_Windows_namespace_and_equal_header_preimages_not_build_time_attestation",
        "current_OS_identity_verified_by_static_verifier": False, "build_execution_attested_by_verifier": False,
        "system_toolchain_inputs_pinned": False}


def verify_target_compile_inputs(bundle, descriptor, original, supplement, graph, prefix):
    member = recorded_file(bundle, supplement["compile_closure"], prefix + "/compile-inputs.json", "target_closure_file")
    value = json_(bundle.read(member, MAX_TARGET_COMPILE_JSON), maximum=MAX_TARGET_COMPILE_JSON)
    object_(value, {"schema", "target", "translation_units", "outside_target_translation_units", "dependency_records", "inputs",
        "phony_target_inputs", "original_physical_target_inputs", "directory_order_dependencies", "observer_sources_covered",
        "candidate_closure_complete", "independent_verification_required", "system_toolchain_inputs_pinned", "scope_change",
        "system_unpinned_scope", "limits"}, "target_compile_shape")
    require(value["schema"] == "omni-strata-native-target-compile-inputs-v2" and value["target"] == "strata"
            and value["independent_verification_required"] is True and value["system_toolchain_inputs_pinned"] is False
            and value["limits"] == {"max_file_bytes": 64 << 20, "max_total_bytes": 1 << 30, "max_files": 10000}, "target_compile_scope")
    aliases, alias_identity = verified_external_header_aliases(bundle, descriptor, original, value, prefix)
    build_root = original["commands"][1][2]
    evidence = descriptor["build_evidence_dir"]
    configured = bundle.load(evidence + "/compile_commands.json")
    compdb = bundle.load(prefix + "/reports/compdb.json")
    require(type(configured) is list and 0 < len(configured) <= 10000
            and type(compdb) is list and 0 < len(compdb) <= 10000, "target_compile_TU_count")
    actual_outputs = {}
    for index, row in enumerate(compdb):
        require(type(row) is dict and {"directory", "file", "output", "command"} <= set(row), "target_compdb_row")
        actual_outputs.setdefault(winpath(row["output"], build_root), []).append((index, row))
    groups = actual_target_dependencies(bundle.read(evidence + "/ninja_deps.txt"), build_root)
    used_aliases = set()
    for row in groups:
        used_aliases.update(path for path in row["inputs"] if path in aliases)
        row["inputs"] = [aliases.get(path, path) for path in row["inputs"]]
    require(used_aliases == set(aliases), "alias_witness_pairs_not_all_in_actual_dependencies")
    declared_groups = value["dependency_records"]
    require(type(declared_groups) is list and len(declared_groups) == len(groups), "target_dependency_projection_count")
    by_output = {}
    for index, (row, declared) in enumerate(zip(groups, declared_groups)):
        object_(declared, set(row), "target_dependency_projection_shape")
        checked = {**declared, "output": winpath(declared["output"]), "inputs": [winpath(x) for x in declared["inputs"]]}
        require(canonical(checked) == canonical(row), "target_dependency_projection_differs")
        by_output.setdefault(row["output"], []).append(index)
    commands = bundle.read(evidence + "/ninja_commands.txt").decode("utf-8").splitlines()
    require(0 < len(commands) <= 20000 and all(len(x) <= 1 << 20 for x in commands), "target_command_bounds")
    command_counts = {}
    for command in commands: command_counts[command.strip()] = command_counts.get(command.strip(), 0) + 1
    declared_units, outside, selected, physical, covered = value["translation_units"], [], set(), set(), set()
    require(type(declared_units) is list and len(declared_units) == len(configured), "target_all_configured_TUs_retained")
    for index, (entry, declared) in enumerate(zip(configured, declared_units)):
        require(type(entry) is dict and {"directory", "file", "output", "command"} <= set(entry)
                and winpath(entry["directory"]) == winpath(build_root), "target_configured_TU_shape")
        unit, output = winpath(entry["file"], build_root), winpath(entry["output"], build_root)
        edge, actual_rows = graph["by_output"].get(output), actual_outputs.get(output, [])
        require(edge is not None and edge["rule"] in graph["rules"] and len(actual_rows) == 1, "target_unique_graph_compdb_TU")
        compdb_index, actual = actual_rows[0]
        binding = verified_dependency_command(entry, actual, edge, graph["rules"][edge["rule"]], build_root)
        member, occurrences = output in graph["reachable"], command_counts.get(actual["command"].strip(), 0)
        dependency_index = None
        if member:
            require(output not in selected and len(by_output.get(output, [])) == 1 and occurrences == 1, "target_unique_built_TU")
            dependency_index = by_output[output][0]
            require(groups[dependency_index]["valid_record"] is True, "target_VALID_dependencies_required")
            selected.add(output); physical.update((unit, output)); physical.update(groups[dependency_index]["inputs"])
            covered.add(unit); covered.update(groups[dependency_index]["inputs"])
        else:
            require(occurrences == 0 and output not in by_output, "outside_target_execution_evidence")
            outside.append({"compile_command_index": index, "path": unit, "output": output, "rule": edge["rule"]})
        expected = {"compile_command_index": index, "path": unit, "output": output, "target_member": member,
            "built_dependency_record_verified": member, "command_occurrences": occurrences, "dependency_record_index": dependency_index,
            "rule": edge["rule"], "ninja_compdb_index": compdb_index, "command_binding": binding}
        object_(declared, set(expected), "target_TU_projection_shape")
        checked = {**declared, "path": winpath(declared["path"]), "output": winpath(declared["output"])}
        require(canonical(checked) == canonical(expected), "target_TU_projection_differs")
    require(set(by_output) == selected, "target_all_and_only_built_dependencies")
    declared_outside = value["outside_target_translation_units"]
    require(type(declared_outside) is list and len(declared_outside) == len(outside), "outside_target_TU_count")
    for actual, declared in zip(outside, declared_outside):
        object_(declared, set(actual), "outside_target_TU_shape")
        require(canonical({**declared, "path": winpath(declared["path"]), "output": winpath(declared["output"])}) == canonical(actual), "outside_target_TU_binding")
    phony, target_inputs, directories = [], [], []
    for name in bundle.read(evidence + "/ninja_inputs.txt").decode("utf-8").splitlines():
        key = winpath(name, build_root)
        require(key in graph["reachable"], "target_original_input_not_in_graph")
        row = graph["reachable"][key]
        if row["rule"] == "phony":
            phony.append({"name": name, "path": key, "origin_file": row["edge"]["origin_file"], "origin_line": row["edge"]["origin_line"], "rule": "phony"})
        else: physical.add(key); target_inputs.append(key)
    for key, row in graph["reachable"].items():
        if row["rule"] is None and key == winpath(build_root):
            require(row["input_kinds"] == ["order_only"], "target_directory_input_not_order_only")
            directories.append({"name": row["name"], "path": key, "input_kinds": row["input_kinds"], "scope": "directory timestamp ordering only"})
        elif row["rule"] != "phony": physical.add(key)
    physical.update(winpath(name, build_root) for name in graph["members"])
    declared_physical = value["original_physical_target_inputs"]
    require(type(declared_physical) is list and all(type(x) is str for x in declared_physical)
            and declared_physical == sorted(declared_physical)
            and len(declared_physical) == len(target_inputs)
            and {winpath(x) for x in declared_physical} == set(target_inputs), "target_original_physical_input_set")
    require(canonical(value["phony_target_inputs"]) == canonical(phony)
            and canonical(value["directory_order_dependencies"]) == canonical(directories), "target_physical_phony_directory_projection")
    source_root = winpath(original["commands"][0][2])
    dep_arg = [x for x in original["commands"][0] if x.startswith("-DSTRATA_GGML_DIR=")]
    require(len(dep_arg) == 1, "target_dependency_source_root")
    roots = ((source_root, "strata_source"), (winpath(dep_arg[0].split("=", 1)[1]), "dependency_source"), (winpath(build_root), "generated_build_input"))
    inputs, mapped, total = value["inputs"], {}, 0
    require(type(inputs) is list and 0 < len(inputs) <= 10000, "target_physical_input_count")
    for row in inputs:
        object_(row, {"path", "family", "relative_path", "size_bytes", "sha256", "archive_path"}, "target_physical_input_shape")
        native, size = winpath(row["path"]), uint(row["size_bytes"], 64 << 20)
        require(native not in mapped, "target_duplicate_physical_input")
        total += size; require(total <= 1 << 30, "target_physical_total_bytes")
        member = prefix + "/" + relative(row["archive_path"])
        require(bundle.rows.get(member) == {"path": member, "size_bytes": size, "sha256": digest(row["sha256"])}, "target_archived_physical_input_bytes")
        family, relative_path = "external_toolchain_or_system_input", None
        for origin, candidate in roots:
            if native.startswith(origin + "\\"):
                family, relative_path = candidate, native[len(origin) + 1:].replace("\\", "/"); break
        require(row["family"] == family and (row["relative_path"] is None if relative_path is None else type(row["relative_path"]) is str
                and row["relative_path"].casefold() == relative_path.casefold()), "target_input_family")
        if family in ("strata_source", "dependency_source"):
            origin = descriptor["final_sources_dir"] if family == "strata_source" else descriptor["dependency_source_provenance"]["archived_root"]
            matches = [x for name, x in bundle.rows.items() if name.casefold() == (origin + "/" + relative_path).casefold()]
            require(len(matches) == 1 and matches[0]["sha256"] == row["sha256"] and matches[0]["size_bytes"] == size, "target_source_snapshot_byte_match")
        mapped[native] = member
    require(set(mapped) == physical and not (set(mapped) & {x["output"] for x in outside}), "target_complete_physical_closure_and_no_outside_object")
    require(all(winpath(path, source_root) in covered for path in SOURCE_CHAIN), "target_all_twelve_observer_sources")
    expected_observer = {path: winpath(path, source_root) in covered for path in ("src/program/generate.cpp", "src/kernels/cpu/pool.cpp",
        "src/core/session.cpp", "src/core/verify.cpp", "src/core/expert_cache.cpp", HEADER)}
    require(canonical(value["observer_sources_covered"]) == canonical(expected_observer) and value["candidate_closure_complete"] is True,
            "target_observer_projection_or_complete_disagrees")
    counts = {"configured_translation_units": len(configured), "target_translation_units": len(selected), "outside_target_translation_units": len(outside),
              "physical_inputs": len(mapped), "physical_input_bytes": total, "phony_target_inputs": len(phony)}
    require(canonical(supplement["counts"]) == canonical(counts), "target_count_projection")
    return {"schema": "omni-strata-verified-target-compile-inputs-v3", **counts, "all_twelve_sources_covered": True,
            "scope": "strata_target_and_recursive_dependencies_all_configured_TUs_retained_outside_target_not_executed",
            "system_toolchain_inputs_pinned": False, "source_receipt_sha256": bundle.rows[prefix + "/compile-inputs.json"]["sha256"],
            **({"external_header_alias_witness": alias_identity} if alias_identity is not None else {})}


def verify_failed_build_supplement(bundle, original, descriptor, stages):
    member = relative(descriptor["build_closure_supplement_file"])
    prefix = PurePosixPath(member).parent.as_posix()
    require(PurePosixPath(member).name == "receipt.json" and bundle.rows[descriptor["build_receipt_file"]]["sha256"] == ORIGINAL_FAILED_BUILD_SHA,
            "exact_preserved_failed_build_required")
    value = bundle.load(member)
    keys = {"schema", "status", "started_utc", "finished_utc", "original_wrapper_failure_preserved", "rebuild_performed", "models_loaded",
        "runtime_installed", "engine_ABI_verified", "live_modules_verified", "whole_model_placement", "memory_hard_caps_verified", "physical_ssd_read_bytes",
        "compiler_full_descendant_container_coverage_verified", "original_build_receipt", "original_error_sha256", "original_native_configure_build_exit_codes",
        "quiescence", "post_build_source", "closed_original_files", "ninja_binary", "graph_files", "graph_queries", "target_graph", "compile_closure", "counts",
        "source_files", "source_files_before", "source_stable"}
    object_(value, keys, "closure_supplement_shape")
    require(value["schema"] == "omni-strata-execution-build-closure-supplement-v1"
            and value["status"] == "closure_collected_independent_static_verification_pending"
            and value["original_wrapper_failure_preserved"] is True and value["source_stable"] is True, "closure_supplement_status")
    require(all(value[key] is False for key in ("rebuild_performed", "models_loaded", "runtime_installed", "engine_ABI_verified", "live_modules_verified",
        "memory_hard_caps_verified", "compiler_full_descendant_container_coverage_verified"))
        and value["whole_model_placement"] is None and value["physical_ssd_read_bytes"] is None, "closure_supplement_no_runtime_grant")
    require(value["original_error_sha256"] == sha(original["error"].encode("utf-8"))
            and canonical(value["original_native_configure_build_exit_codes"]) == b"[0,0]", "original_failed_wrapper_reason_and_native_exits")
    recorded_file(bundle, value["original_build_receipt"], descriptor["build_receipt_file"], "supplement_original_receipt")
    require(value["closed_original_files"] == original["archived_build_files"] and value["ninja_binary"] == original["tools"]["ninja"]["binary"], "supplement_original_closed_files")
    require(value["source_files_before"] == value["source_files"], "supplement_recorder_changed")
    object_(value["source_files"], {"supplement.py", "ninja_graph.py"}, "supplement_recorders_shape")
    for name, key, pin in (("supplement.py", "closure_supplement_recorder_file", SUPPLEMENT_RECORDER_SHA), ("ninja_graph.py", "ninja_graph_recorder_file", GRAPH_RECORDER_SHA)):
        recorded_file(bundle, value["source_files"][name], descriptor[key], "supplement_recorder_bytes")
        bundle.expected(descriptor[key], pin)
    quiet_ref = value["quiescence"]
    # The archived quiescence observation is scoped, never current OS attestation.
    quiet_name = prefix + "/compiler-quiescence.json"
    recorded_file(bundle, quiet_ref, quiet_name, "supplement_quiescence")
    bundle.expected(quiet_name, BUILD_QUIESCENCE_SHA)
    quiet = bundle.load(quiet_name)
    require(quiet.get("build_receipt_sha256") == ORIGINAL_FAILED_BUILD_SHA and quiet.get("original_wrapper_pid_absent") is True
            and quiet.get("build_wrapper_exit_code") == 1 and quiet.get("named_compiler_processes_absent") is True
            and quiet.get("process_container_or_full_descendant_coverage_verified") is False, "scoped_build_quiescence_required")
    object_(value["post_build_source"], {"strata", "dependency"}, "supplement_post_source_shape")
    for kind, stage in (("strata", "post_build"), ("dependency", "dependency-post_build")):
        row = object_(value["post_build_source"][kind], {"receipt", "inventory", "canonical_leaves", "source_bytes_reverified_by_frozen_recorder", "full_source_byte_verification_in_supplement"}, "supplement_post_source_record")
        require(row["source_bytes_reverified_by_frozen_recorder"] is True and row["full_source_byte_verification_in_supplement"] is False, "supplement_source_scope")
        for field, name in (("receipt", "receipt.json"), ("inventory", "full-inventory.json"), ("canonical_leaves", "canonical-leaves.json")):
            recorded_file(bundle, row[field], stages[stage]["folder"] + "/" + name, "supplement_actual_post_source")
    build_root = original["commands"][1][2]
    graph = verify_archived_ninja_graph(bundle, value["graph_files"], prefix, build_root)
    graph_name = recorded_file(bundle, value["target_graph"], prefix + "/target-graph.json", "supplement_target_graph")
    declared = object_(bundle.load(graph_name), {"schema", "target", "reachable_nodes", "source_files", "queries"}, "supplement_target_graph_shape")
    require(declared["schema"] == "omni-strata-actual-ninja-target-graph-v1" and declared["target"] == "strata"
            and declared["source_files"] == value["graph_files"] and declared["queries"] == value["graph_queries"]
            and canonical(declared["reachable_nodes"]) == canonical(list(graph["reachable"].values())), "supplement_target_graph_projection")
    phony = [row for row in graph["reachable"].values() if row["rule"] == "phony"]
    queries = value["graph_queries"]
    require(type(queries) is list and len(queries) == len(phony) + 2, "supplement_query_count")
    ninja = original["tools"]["ninja"]["binary"]["path"]
    expected = [(["targets", "all"], "targets-all.txt"), (["compdb"], "compdb.json")]
    expected += [(["query", row["name"]], "query-" + str(index).zfill(3) + ".txt") for index, row in enumerate(phony)]
    for query, (arguments, filename) in zip(queries, expected):
        object_(query, {"command", "cwd", "exit_code", "log"}, "supplement_query_shape")
        require(query["command"] == [ninja, "-t", *arguments] and winpath(query["cwd"]) == winpath(build_root)
                and type(query["exit_code"]) is int and query["exit_code"] == 0, "supplement_query_success")
        recorded_file(bundle, query["log"], prefix + "/reports/" + filename, "supplement_query_raw")
    targets = {}
    for line in bundle.read(prefix + "/reports/targets-all.txt").decode("utf-8").splitlines():
        name, marker, rule = line.rpartition(": ")
        require(marker and name and rule and not any(x.isspace() for x in rule), "supplement_targets_grammar")
        key = winpath(name, build_root)
        require(key not in targets or targets[key] == rule, "supplement_targets_duplicate")
        targets[key] = rule
    require(all(row["rule"] is None or targets.get(key) == row["rule"] for key, row in graph["reachable"].items()), "supplement_actual_target_rules")
    for index, row in enumerate(phony):
        lines = bundle.read(prefix + "/reports/query-" + str(index).zfill(3) + ".txt").decode("utf-8").splitlines()
        require(len(lines) >= 3 and lines[:2] == [row["name"] + ":", "  input: phony"] and "  outputs:" in lines, "supplement_phony_query_grammar")
        actual = []
        for line in lines[2:lines.index("  outputs:")]:
            require(line.startswith("    "), "supplement_query_child_grammar")
            name, kind = line[4:], "explicit"
            if name.startswith("|| "): name, kind = name[3:], "order_only"
            elif name.startswith("| "): name, kind = name[2:], "implicit"
            actual.append((kind, winpath(name, build_root)))
        expected_children = [(kind, winpath(name, build_root)) for kind, names in row["edge"]["inputs"].items() for name in names]
        require(actual == expected_children, "supplement_phony_query_children")
    closure = verify_target_compile_inputs(bundle, descriptor, original, value, graph, prefix)
    return {**closure, "supplement_sha256": bundle.rows[member]["sha256"], "original_wrapper_status": "failed",
            "original_wrapper_failure_preserved": True, "original_build_receipt_sha256": ORIGINAL_FAILED_BUILD_SHA,
            "native_configure_and_link_exit_codes": [0, 0], "compiled_engine_ABI_verified": False,
            "compiler_retirement_scope": "archived_exact_wrapper_and_named_tools_observation_not_full_descendant_container_or_current_OS_attestation"}


def verify_recipe_compile_closure(bundle, descriptor, receipt, commands_file):
    """Recompute actual Ninja/TU membership; candidate booleans are not proof."""
    ref = object_(receipt["compile_closure"], {"receipt", "candidate_closure_complete", "input_count", "unverified_translation_unit_count"}, "recipe_compile_reference")
    require(type(ref["unverified_translation_unit_count"]) is int and ref["unverified_translation_unit_count"] == 0,
            "recipe_unverified_TU_count")
    file_ref = object_(ref["receipt"], {"path", "size_bytes", "sha256"}, "recipe_compile_file")
    evidence_root = PurePosixPath(descriptor["build_evidence_dir"]).parent.as_posix()
    name = evidence_root + "/compile-inputs.json"
    require(bundle.rows.get(name, {}).get("sha256") == file_ref["sha256"]
            and bundle.rows[name]["size_bytes"] == file_ref["size_bytes"], "recipe_compile_reference_bytes")
    value = object_(bundle.load(name), {"schema", "translation_units", "dependency_records", "inputs", "unmatched_dependency_outputs",
             "observer_sources_covered", "candidate_closure_complete", "unverified_translation_units", "system_toolchain_inputs_pinned", "raw_dependency_log_file",
             "raw_dependency_log_sha256", "system_unpinned_scope", "limits", "independent_verification_required"}, "recipe_compile_shape")
    require(value["schema"] == "omni-strata-native-compile-inputs-v1" and value["system_toolchain_inputs_pinned"] is False
            and value["independent_verification_required"] is True and value["unmatched_dependency_outputs"] == []
            and value["unverified_translation_units"] == [], "recipe_compile_scope")
    require(value["limits"] == {"max_file_bytes": 64 << 20, "max_total_bytes": 1 << 30, "max_files": 10000}, "recipe_compile_limits")
    command_rows = json_(bundle.read(commands_file, MAX_JSON))
    require(type(command_rows) is list and 0 < len(command_rows) <= 10000, "recipe_compile_commands")
    build_root = receipt["steps"][1]["command"][2]
    ninja_commands_raw = bundle.read(descriptor["build_evidence_dir"] + "/ninja_commands.txt")
    try:
        ninja_commands = ninja_commands_raw.decode("utf-8").splitlines()
    except UnicodeError:
        raise EvidenceError("recipe_Ninja_command_encoding") from None
    require(0 < len(ninja_commands) <= 20000 and all(0 < len(line) <= 1 << 20 for line in ninja_commands),
            "recipe_Ninja_command_bounds")
    units, outputs = [], {}
    for index, row in enumerate(command_rows):
        require(type(row) is dict and {"directory", "file"} <= set(row), "recipe_compile_command_row")
        require(type(row.get("command")) is str and 0 < len(row["command"]) <= 1 << 20
                and row["command"].strip() in ninja_commands, "recipe_TU_command_in_actual_Ninja_report")
        require(winpath(row["directory"]) == winpath(build_root), "recipe_TU_build_directory")
        unit = winpath(row["file"], row["directory"])
        output = row.get("output")
        if output is None:
            command = row.get("command", "")
            require(type(command) is str and len(command) <= 1 << 20, "recipe_compile_command_text")
            match = re.search(r'(?:^|\s)(?:-o\s+|/Fo)(?:"([^"]+)"|([^\s]+))', command)
            if match:
                output = match.group(1) or match.group(2)
        require(output is not None, "recipe_compile_output_unavailable")
        output = winpath(output, row["directory"])
        require(output not in outputs, "recipe_compile_output_duplicate")
        outputs[output] = index
        units.append({"compile_command_index": index, "path": unit, "output": output, "built_dependency_record_verified": True})
    declared_units = value["translation_units"]
    require(type(declared_units) is list and len(declared_units) == len(units), "recipe_all_translation_units")
    for actual, declared in zip(units, declared_units):
        object_(declared, {"compile_command_index", "path", "output", "built_dependency_record_verified"}, "recipe_TU_record")
        require(type(declared["compile_command_index"]) is int and declared["compile_command_index"] == actual["compile_command_index"]
                and winpath(declared["path"]) == actual["path"] and winpath(declared["output"]) == actual["output"]
                and declared["built_dependency_record_verified"] is True, "recipe_actual_TU_binding")
    deps_file = evidence_root + "/" + relative(value["raw_dependency_log_file"])
    deps_raw = bundle.expected(deps_file, digest(value["raw_dependency_log_sha256"]))
    try:
        deps_lines = deps_raw.decode("utf-8").splitlines()
    except UnicodeError:
        raise EvidenceError("recipe_Ninja_encoding") from None
    groups, active = [], None
    for line in deps_lines:
        if not line:
            active = None
        elif not line.startswith((" ", "\t")):
            require(": #deps " in line, "recipe_Ninja_deps_format")
            output, metadata = line.split(": #deps ", 1)
            match = re.fullmatch(r"([0-9]+), deps mtime [0-9]+ \(VALID\)", metadata)
            require(match is not None, "recipe_Ninja_deps_invalid")
            active = {"output": winpath(output, build_root), "metadata": metadata, "inputs": [], "expected_count": int(match[1])}
            require(active["output"] in outputs and len(groups) < 10000, "recipe_Ninja_unmatched_output")
            groups.append(active)
        else:
            require(active is not None, "recipe_Ninja_orphan_input")
            active["inputs"].append(winpath(line, build_root))
    require(len(groups) == len(units) and len({x["output"] for x in groups}) == len(units), "recipe_all_built_TU_dependencies")
    declared_groups = value["dependency_records"]
    require(type(declared_groups) is list and len(declared_groups) == len(groups), "recipe_dependency_count")
    required_inputs = {row["path"] for row in units}
    covered = set()
    for actual, declared in zip(groups, declared_groups):
        object_(declared, {"output", "metadata", "valid_record", "inputs", "compile_command_indices"}, "recipe_dependency_record")
        require(len(actual["inputs"]) == actual["expected_count"] and winpath(declared["output"]) == actual["output"]
                and declared["metadata"] == actual["metadata"] and declared["valid_record"] is True
                and type(declared["inputs"]) is list and [winpath(x) for x in declared["inputs"]] == actual["inputs"]
                and declared["compile_command_indices"] == [outputs[actual["output"]]]
                and type(declared["compile_command_indices"][0]) is int, "recipe_actual_dependency_binding")
        required_inputs.update(actual["inputs"])
        covered.add(units[outputs[actual["output"]]]["path"])
        covered.update(actual["inputs"])
    ninja_inputs = bundle.read(descriptor["build_evidence_dir"] + "/ninja_inputs.txt").decode("utf-8").splitlines()
    required_inputs.update(winpath(line, build_root) for line in ninja_inputs if line)
    inputs = value["inputs"]
    require(type(inputs) is list and len(inputs) <= 10000 and type(ref["input_count"]) is int
            and ref["input_count"] == len(inputs), "recipe_input_count")
    source_root = winpath(receipt["commands"][0][receipt["commands"][0].index("-S") + 1])
    dep_args = [x for x in receipt["commands"][0] if x.startswith("-DSTRATA_GGML_DIR=")]
    require(len(dep_args) == 1, "recipe_dependency_root")
    dependency_root = winpath(dep_args[0].split("=", 1)[1])
    roots = ((source_root, "strata_source"), (dependency_root, "dependency_source"), (winpath(build_root), "generated_build_input"))
    mapped, total = {}, 0
    for row in inputs:
        object_(row, {"path", "family", "relative_path", "size_bytes", "sha256", "archive_path"}, "recipe_input_record")
        native = winpath(row["path"])
        require(native not in mapped, "recipe_duplicate_input")
        size = uint(row["size_bytes"], 64 << 20)
        total += size
        require(total <= 1 << 30, "recipe_input_total")
        member = evidence_root + "/" + relative(row["archive_path"])
        require(bundle.rows.get(member) == {"path": member, "size_bytes": size, "sha256": digest(row["sha256"])}, "recipe_input_actual_bytes")
        family, rel = "external_toolchain_or_system_input", None
        for origin, candidate_family in roots:
            if native.startswith(origin + "\\"):
                family, rel = candidate_family, native[len(origin) + 1:].replace("\\", "/")
                break
        require(row["family"] == family and (row["relative_path"] is None if rel is None else
                type(row["relative_path"]) is str and row["relative_path"].casefold() == rel.casefold()), "recipe_input_family")
        if family in ("strata_source", "dependency_source"):
            prefix = descriptor["final_sources_dir"] if family == "strata_source" else descriptor["dependency_source_provenance"]["archived_root"]
            candidates = [record for path, record in bundle.rows.items() if path.casefold() == (prefix + "/" + rel).casefold()]
            require(len(candidates) == 1 and candidates[0]["sha256"] == row["sha256"]
                    and candidates[0]["size_bytes"] == size, "recipe_input_matches_source_snapshot")
        mapped[native] = member
    require(set(mapped) == required_inputs, "recipe_complete_actual_input_set")
    observed = {path: winpath(ntpath.join(source_root, path)) in covered for path in SOURCE_CHAIN}
    require(all(observed.values()), "recipe_all_twelve_sources_in_actual_build_closure")
    declared_observer = value["observer_sources_covered"]
    expected_observer = {path: observed[path] for path in ("src/program/generate.cpp", "src/kernels/cpu/pool.cpp",
                        "src/core/session.cpp", "src/core/verify.cpp", "src/core/expert_cache.cpp", HEADER)}
    require(declared_observer == expected_observer and all(type(x) is bool for x in declared_observer.values()), "recipe_observer_closure_projection")
    require(value["candidate_closure_complete"] is True and ref["candidate_closure_complete"] is True,
            "recipe_projection_disagrees_with_verified_closure")
    return {"schema": "omni-strata-verified-native-compile-inputs-v2", "translation_units": len(units),
            "actual_input_count": len(mapped), "actual_input_bytes": total, "all_twelve_sources_covered": True,
            "system_toolchain_inputs_pinned": False, "source_receipt_sha256": bundle.rows[name]["sha256"]}


def cmake_cache_values(cache, key):
    """Read exact values; accept LF/CRLF without stripping value characters."""
    require(type(cache) is str and len(cache) <= MAX_TEXT, "build_cache_text_bound")
    require(re.search(r"\r(?!\n)", cache) is None, "build_cache_line_endings")
    return re.findall(r"(?m)^" + re.escape(key) + r":[^=\r\n]+=([^\r\n]*)(?:\r?\n|\Z)", cache)


def validate_build(bundle, receipt, descriptor, base, dependency, states):
    """Accept the frozen recorder's raw schema, never a hand-written normalized grant."""
    required = {"schema", "status", "started_utc", "finished_utc", "base_revision", "dependency_revision", "dependency_tree",
                "parallelism", "tools", "steps", "evidence_steps", "compiler_environment", "abi", "runtime_installed",
                "models_loaded", "live_modules_verified", "compiler_process_retirement_independently_verified",
                "whole_model_placement", "memory_hard_caps_verified", "physical_ssd_read_bytes", "source_identity",
                "dependency_identity", "patch_chain", "commands", "files", "compile_closure", "archived_build_files", "recorder"}
    supplemented = "build_closure_supplement_file" in descriptor
    object_(receipt, (required - {"compile_closure"}) | {"error"} if supplemented else required, "build_receipt_shape")
    require(receipt["schema"] == "omni-strata-execution-private-build-v1"
            and (receipt["status"] == "failed" and type(receipt["error"]) is str and 0 < len(receipt["error"]) <= 65536
                 if supplemented else receipt["status"] == "built_static_verification_pending_not_installed_not_neurally_qualified"), "successful_native_build_or_exact_failed_wrapper_supplement_required")
    require((receipt["base_revision"], receipt["dependency_revision"], receipt["dependency_tree"])
            == (BASE, DEPENDENCY, DEPENDENCY_TREE), "build_source_identity")
    require(type(receipt["parallelism"]) is int and receipt["parallelism"] == 2, "build_parallelism")
    for key in ("runtime_installed", "models_loaded", "live_modules_verified", "memory_hard_caps_verified",
                "compiler_process_retirement_independently_verified"):
        require(receipt[key] is False, "build_only_scope")
    require(receipt["whole_model_placement"] is None and receipt["physical_ssd_read_bytes"] is None
            and receipt["abi"] == {"compiled_fixture_ABI_reference": None, "engine_ABI_verified": False}, "build_no_live_or_ABI_grant")
    for key in ("started_utc", "finished_utc"):
        require(type(receipt[key]) is str and 0 < len(receipt[key]) <= 128, "build_time_record")
    evidence = relative(descriptor["build_evidence_dir"])
    evidence_root = PurePosixPath(evidence).parent.as_posix()
    require(descriptor["build_receipt_file"] == evidence_root + "/build-receipt.json", "build_receipt_archive_location")
    recorded_file(bundle, receipt["recorder"], descriptor["build_recorder_file"], "actual_build_recorder")
    bundle.expected(descriptor["build_recorder_file"], BUILD_RECORDER_SHA)
    stages = source_stages(bundle, descriptor, base, dependency)
    for key, stage_key in (("source_identity", "after_boundary_v2"), ("dependency_identity", "dependency-base")):
        ref = object_(receipt[key], {"receipt", "inventory", "canonical_leaves", "tree", "count"}, "build_stage_reference")
        stage = stages[stage_key]
        require(ref["tree"] == stage["tree"] and type(ref["count"]) is int and ref["count"] == len(stage["files"]), "build_stage_identity")
        for field, filename in (("receipt", "receipt.json"), ("inventory", "full-inventory.json"), ("canonical_leaves", "canonical-leaves.json")):
            recorded_file(bundle, ref[field], stage["folder"] + "/" + filename, "build_actual_stage_" + field)
    chain = receipt["patch_chain"]
    require(type(chain) is list and len(chain) == 3, "build_patch_chain_count")
    for row, expected_kind, declared in zip(chain, ("io_v1", "execution_v1", "boundary_v2"), descriptor["patch_chain"]):
        object_(row, {"kind", "file"}, "build_patch_chain_record")
        require(row["kind"] == expected_kind, "build_patch_order")
        recorded_file(bundle, row["file"], declared["file"], "build_actual_patch")
    environment = object_(receipt["compiler_environment"], {"CL", "_CL_", "INCLUDE", "LIB", "VCToolsVersion",
                  "WindowsSDKVersion", "VisualStudioVersion", "CUDA_PATH", "PATH"}, "compiler_environment")
    require(all(value is None or type(value) is str and len(value) <= 65536 for value in environment.values()), "compiler_environment_bound")
    require(environment["CL"] in (None, "") and environment["_CL_"] in (None, "")
            and type(environment["VCToolsVersion"]) is str and environment["VCToolsVersion"].startswith("14.44.")
            and type(environment["WindowsSDKVersion"]) is str and environment["WindowsSDKVersion"].rstrip("\\/") == "10.0.26100.0",
            "compiler_environment_values")
    archived = receipt["archived_build_files"]
    require(type(archived) is dict and 0 < len(archived) <= 256, "actual_build_archive")
    for name, row in archived.items():
        require(relative(name) == name and "/" not in name, "actual_build_archive_name")
        object_(row, {"path", "size_bytes", "sha256", "archive_path"}, "actual_build_archive_record")
        require(row["archive_path"] == "build/" + name, "actual_build_archive_relative")
        recorded_file(bundle, {key: row[key] for key in ("path", "size_bytes", "sha256")}, evidence + "/" + name, "actual_archived_build_file")
    files = object_(receipt["files"], {"strata.exe", "CMakeCache.txt", "compile_commands.json", "build.ninja"}, "build_outputs")
    for name, row in files.items():
        require(name in archived and row == {key: archived[name][key] for key in ("path", "size_bytes", "sha256")}, "raw_build_archive_binding")
        recorded_file(bundle, row, evidence + "/" + name, "actual_build_output")
    require(bundle.rows.get(descriptor["engine_file"], {}).get("sha256") == files["strata.exe"]["sha256"]
            and bundle.rows[descriptor["engine_file"]]["size_bytes"] == files["strata.exe"]["size_bytes"], "installed_member_is_actual_build_output")
    cache = bundle.read(evidence + "/CMakeCache.txt").decode("utf-8")
    for key, value in REQUIRED_CACHE.items():
        require(cmake_cache_values(cache, key) == [value], "build_cache_flags")
    tools = object_(receipt["tools"], set(VERSIONS), "compiler_tools")
    object_(descriptor["tool_binary_files"], set(VERSIONS) | {"git"}, "archived_tool_binary_map")
    for name, version in VERSIONS.items():
        tool = object_(tools[name], {"command", "cwd", "exit_code", "log", "binary"}, "actual_compiler_tool")
        require(type(tool["exit_code"]) is int and tool["exit_code"] in ((0, 2) if name == "cl" else (0,))
                and tool["cwd"] is None and tool["command"] == [tool["binary"]["path"], "/Bv" if name == "cl" else "--version"], "compiler_probe_exit_command")
        recorded_file(bundle, tool["binary"], descriptor["tool_binary_files"][name], "actual_compiler_binary")
        recorded_file(bundle, tool["log"], evidence + "/" + name + "-version.log", "actual_compiler_probe")
        log = bundle.read(evidence + "/" + name + "-version.log")
        require(version.encode("ascii") in log, "compiler_version")
    commands, steps = receipt["commands"], receipt["steps"]
    require(type(commands) is list and len(commands) == 2 and type(steps) is list and len(steps) == 2, "build_steps")
    for command in commands:
        require(type(command) is list and all(type(arg) is str and 0 < len(arg) <= 32768 for arg in command), "build_command_bound")
    configure, build = commands
    require(len(configure) >= 8 and configure[:2] == [tools["cmake"]["binary"]["path"], "-S"]
            and configure[3:4] == ["-B"] and configure[5:7] == ["-G", "Ninja"], "configure_command_prefix")
    definitions = [arg[2:].split("=", 1) for arg in configure[7:]]
    require(all(arg.startswith("-D") and "=" in arg for arg in configure[7:])
            and len({pair[0] for pair in definitions}) == len(definitions), "configure_no_extra_or_duplicate_flags")
    expected_definitions = dict(REQUIRED_CACHE, CMAKE_EXPORT_COMPILE_COMMANDS="ON",
                                CMAKE_MAKE_PROGRAM=tools["ninja"]["binary"]["path"], CMAKE_CUDA_COMPILER=tools["nvcc"]["binary"]["path"])
    expected_definitions["STRATA_GGML_DIR"] = bundle.load(stages["dependency-base"]["folder"] + "/receipt.json")["source_root"]
    require(dict(definitions) == expected_definitions and winpath(configure[2]) == stages["after_boundary_v2"]["source_root"], "actual_configure_controls")
    for key, expected_path in (("CMAKE_HOME_DIRECTORY", configure[2]), ("CMAKE_MAKE_PROGRAM", tools["ninja"]["binary"]["path"]),
                               ("CMAKE_CUDA_COMPILER", tools["nvcc"]["binary"]["path"]), ("STRATA_GGML_DIR", expected_definitions["STRATA_GGML_DIR"])):
        values = cmake_cache_values(cache, key)
        require(len(values) == 1 and winpath(values[0]) == winpath(expected_path), "actual_cache_source_tool_paths")
    require(build == [tools["cmake"]["binary"]["path"], "--build", configure[4], "--target", "strata", "--parallel", "2"], "actual_build_controls")
    for index, step in enumerate(steps):
        object_(step, {"command", "cwd", "exit_code", "log"}, "actual_build_step")
        require(step["command"] == commands[index] and step["cwd"] is None and type(step["exit_code"]) is int
                and step["exit_code"] == 0, "actual_build_step_success")
        recorded_file(bundle, step["log"], evidence + "/step-" + str(index) + ".log", "actual_build_step_log")
    evidence_steps = receipt["evidence_steps"]
    expected_reports = (("ninja_deps", ["deps"]), ("ninja_inputs", ["inputs", "strata"]), ("ninja_commands", ["commands", "strata"]))
    require(type(evidence_steps) is list and len(evidence_steps) == 3, "actual_ninja_report_steps")
    for step, (name, arguments) in zip(evidence_steps, expected_reports):
        object_(step, {"command", "cwd", "exit_code", "log"}, "actual_ninja_report_step")
        require(step["command"] == [tools["ninja"]["binary"]["path"], "-t", *arguments]
                and winpath(step["cwd"]) == winpath(configure[4]) and type(step["exit_code"]) is int and step["exit_code"] == 0,
                "actual_ninja_report_success")
        recorded_file(bundle, step["log"], evidence + "/" + name + ".txt", "actual_ninja_report_bytes")
    closure = (verify_failed_build_supplement(bundle, receipt, descriptor, stages) if supplemented
               else verify_recipe_compile_closure(bundle, descriptor, receipt, evidence + "/compile_commands.json"))
    return {"build_receipt_sha256": bundle.rows[descriptor["build_receipt_file"]]["sha256"],
            "original_wrapper_status": receipt["status"], "original_wrapper_failure_preserved": supplemented,
            "native_configure_and_link_exit_codes": [step["exit_code"] for step in steps],
            "source_file_count": len(stages["after_boundary_v2"]["files"]), "dependency_file_count": len(dependency["files"]),
            "source_stage_receipts": {key: value["receipt_sha256"] for key, value in stages.items()}, "compile_closure": closure,
            "source_checkout_controls": {key: value["checkout_controls"] for key, value in stages.items()},
            "tools": {name: {"binary_sha256": tool["binary"]["sha256"]} for name, tool in tools.items()},
            "compiler_environment": {key: environment[key] for key in ("VCToolsVersion", "WindowsSDKVersion")},
            "compiler_process_retirement_independently_verified": False, "system_toolchain_inputs_pinned": False}


def verify_combined_runtime(descriptor_file, runtime_root):
    """Check static bytes only. Missing ABI/PE/build preimages refuse, never opt in."""
    root = Path(runtime_root).resolve(strict=True)
    name = relative(descriptor_file)
    raw = file_bytes(contained_file(root, name), MAX_JSON)
    descriptor = json_(raw)
    require(type(descriptor) is dict, "descriptor_shape")
    extra = SUPPLEMENT_DESCRIPTOR_KEYS if "build_closure_supplement_file" in descriptor else set()
    extra = extra | alias_descriptor_keys(descriptor)
    descriptor = object_(
        descriptor,
        {
            "schema",
            "base_revision",
            "dependency_revision",
            "dependency_tree",
            "manifest_file",
            "engine_file",
            "patch_chain",
            "combined_patch_manifest_file",
            "native_schema_file",
            "parser_file",
            "base_source_provenance",
            "dependency_source_provenance",
            "source_states",
            "final_sources_dir",
            "build_receipt_file",
            "build_evidence_dir",
            "build_recorder_file",
            "source_recorder_file",
            "source_stage_dirs",
            "tool_binary_files",
            "pe_receipt_file",
            "abi_receipt_file",
        }
        | extra,
        "descriptor_shape",
    )
    require(descriptor["schema"] == "omni-strata-combined-observation-runtime-v2", "combined_descriptor_required")
    require(
        (descriptor["base_revision"], descriptor["dependency_revision"], descriptor["dependency_tree"])
        == (BASE, DEPENDENCY, DEPENDENCY_TREE),
        "combined_revision",
    )
    bundle = Bundle(
        root,
        descriptor["manifest_file"],
        dependency_source_roots=dependency_source_roots(descriptor),
        strata_source_roots=strata_source_roots(descriptor),
    )
    require(bundle.read(name, MAX_JSON) == raw, "descriptor_member_binding")
    object_(descriptor["tool_binary_files"], set(VERSIONS) | {"git"}, "archived_tool_binary_map")
    chain = descriptor["patch_chain"]
    require(type(chain) is list and len(chain) == 3, "ordered_patch_count")
    for row, (kind, expected) in zip(chain, PATCHES):
        object_(row, {"kind", "file", "sha256"}, "patch_record")
        require(row["kind"] == kind and row["sha256"] == expected, "ordered_patch_identity")
        bundle.expected(row["file"], expected)
    bundle.expected(descriptor["combined_patch_manifest_file"], PATCH_MANIFEST_SHA)
    bundle.expected(descriptor["native_schema_file"], SCHEMA_SHA)
    bundle.expected(descriptor["parser_file"], PARSER_SHA)
    base = git_source(bundle, descriptor["base_source_provenance"], BASE)
    dependency = git_source(bundle, descriptor["dependency_source_provenance"], DEPENDENCY, DEPENDENCY_TREE)
    states = source_chain(bundle, descriptor["source_states"], base)
    verify_base_source_pins(base)
    receipt = bundle.load(descriptor["build_receipt_file"])
    build = validate_build(bundle, receipt, descriptor, base, dependency, states)
    verify_strata_source_auxiliary(bundle, descriptor, base)
    # Neither receipt Booleans nor their digests replace PE/emitted-frame bytes.
    pe = verify_pe_evidence(bundle, descriptor["pe_receipt_file"], descriptor["engine_file"])
    abi = verify_abi_evidence(
        bundle, descriptor["abi_receipt_file"], descriptor["engine_file"], descriptor["parser_file"], build
    )
    identity = {
        "schema": "omni-strata-combined-static-runtime-identity-v2",
        "status": "static_archived_bytes_and_build_records_verified_not_live_runtime_eligible",
        "base_revision": BASE,
        "base_tree": base["tree"],
        "dependency_revision": DEPENDENCY,
        "dependency_tree": DEPENDENCY_TREE,
        "native_io_schema": "strata-omni-io-v1",
        "native_execution_schema": "strata-omni-exec-v1",
        "patch_chain": copy.deepcopy(chain),
        "combined_patch_manifest_sha256": PATCH_MANIFEST_SHA,
        "header_sha256": HEADER_V2,
        "schema_sha256": SCHEMA_SHA,
        "parser_sha256": PARSER_SHA,
        "runtime_manifest_sha256": bundle.manifest_sha256,
        "descriptor_sha256": sha(raw),
        "native_executable_sha256": bundle.rows[descriptor["engine_file"]]["sha256"],
        "source_hashes": states,
        "build": build,
        "pe": pe,
        "ABI_reference": abi,
        "observer_layout": abi["observer_layout"],
        "observer_layout_scope": "compiled_standalone_fixture_reference_only",
        "compiled_engine_ABI_verified": False,
        "runtime_binding": None,
        "live_loaded_module_paths_verified": False,
        "current_OS_identity_verified": False,
        "owner_adapter_verified": False,
        "build_execution_attested_by_verifier": False,
        "runtime_qualification": False,
        "default_eligible": False,
        "aggregate_gpu_hard_cap_verified": False,
        "aggregate_ram_hard_cap_verified": False,
        "physical_ssd_read_bytes": None,
    }
    return identity | {"identity_sha256": sha(canonical(identity))}


def pe_imports(path):
    """Read-only normal/delay x64 PE scan, adapted from current strata_vision.py.

    System imports are classified names only; no current System32 identity is
    established. Explicit LoadLibrary remains for the separate live verifier.
    """
    import mmap
    require(64 <= Path(path).stat().st_size <= MAX_MEMBER, "PE_size")
    with Path(path).open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        def unpack(format_, offset):
            require(0 <= offset and offset + struct.calcsize(format_) <= len(data), "PE_extent")
            return struct.unpack_from(format_, data, offset)
        require(data[:2] == b"MZ", "PE_DOS_signature")
        pe = unpack("<I", 60)[0]
        require(pe >= 64 and data[pe:pe + 4] == b"PE\0\0", "PE_signature")
        machine, sections = unpack("<HH", pe + 4)
        optional_size = unpack("<H", pe + 20)[0]
        optional = pe + 24
        require(machine == 0x8664 and unpack("<H", optional)[0] == 0x20B
                and 1 <= sections <= 96 and optional_size >= 112, "PE_x64")
        image_base = unpack("<Q", optional + 24)[0]
        headers = unpack("<I", optional + 60)[0]
        directory_count = unpack("<I", optional + 108)[0]
        require(optional_size >= 112 + 8 * min(directory_count, 16) and headers <= len(data), "PE_optional_extent")
        ranges = []
        for index in range(sections):
            virtual_size, address, size, offset = unpack("<IIII", optional + optional_size + index * 40 + 8)
            require(offset + size <= len(data), "PE_section_extent")
            ranges.append((address, max(virtual_size, size), offset, size))
        def raw(rva, size):
            require(0 < rva < 2**32 and size > 0, "PE_RVA")
            if rva < headers and rva + size <= headers:
                return rva
            matches = [(offset + rva - address, extent - (rva - address))
                       for address, virtual, offset, extent in ranges if address <= rva < address + virtual]
            require(len(matches) == 1 and size <= matches[0][1], "PE_raw_extent")
            return matches[0][0]
        def dll_name(rva):
            result = bytearray()
            for index in range(256):
                value = data[raw(rva + index, 1)]
                if value == 0:
                    break
                result.append(value)
            else:
                raise EvidenceError("PE_name_terminator")
            try:
                name = result.decode("ascii").lower()
            except UnicodeError:
                raise EvidenceError("PE_name_encoding") from None
            require(re.fullmatch(r"[a-z0-9_-][a-z0-9_.-]{0,126}\.dll", name) is not None and ".." not in name, "PE_name")
            return name
        result = {}
        for kind, directory, row_bytes in (("normal", 1, 20), ("delay", 13, 32)):
            names = set()
            if directory >= directory_count:
                result[kind] = []
                continue
            rva, extent = unpack("<II", optional + 112 + directory * 8)
            if rva == extent == 0:
                result[kind] = []
                continue
            require(rva > 0 and row_bytes <= extent <= 1 << 20, "PE_import_extent")
            for index in range(min(extent // row_bytes, 4096)):
                row = unpack("<" + "I" * (row_bytes // 4), raw(rva + index * row_bytes, row_bytes))
                if not any(row):
                    break
                if kind == "normal":
                    name_rva = row[3]
                else:
                    require(row[0] in (0, 1), "PE_delay_mode")
                    name_rva = row[1] if row[0] else row[1] - image_base
                names.add(dll_name(name_rva))
            else:
                raise EvidenceError("PE_import_terminator")
            result[kind] = sorted(names)
        return result


def verify_strata_source_auxiliary(bundle, descriptor, base):
    """Reconcile retained runtime bytes only after all source Git closures pass."""
    pin = STRATA_SOURCE_AUXILIARY
    require(
        base.get("revision") == BASE and base.get("tree") == "332979d72ea7c5fae7f00bec6f2c79234292bc3b",
        "strata_auxiliary_base_git_identity",
    )
    record = base.get("canonical_source_hashes", {}).get(pin["path"])
    require(
        type(record) is dict
        and canonical(record)
        == canonical(
            {
                "git_blob_id": pin["git_blob_id"],
                "mode": pin["mode"],
                "canonical_size_bytes": pin["size_bytes"],
                "canonical_sha256": pin["sha256"],
                "checkout_sha256": pin["sha256"],
                "checkout_matches_canonical": True,
            }
        ),
        "strata_auxiliary_canonical_git_record",
    )
    require(
        canonical(base.get("files", {}).get(pin["path"]))
        == canonical({"path": pin["path"], "size_bytes": pin["size_bytes"], "sha256": pin["sha256"]}),
        "strata_auxiliary_base_checkout",
    )
    blob_name = (
        PurePosixPath(descriptor["base_source_provenance"]["commit_file"]).parent / (pin["git_blob_id"] + ".blob")
    ).as_posix()
    canonical_blob = bundle.read(blob_name, MAX_GIT_OBJECT)
    require(
        len(canonical_blob) == pin["size_bytes"]
        and sha(canonical_blob) == pin["sha256"]
        and hashlib.sha1(b"blob " + str(len(canonical_blob)).encode() + b"\0" + canonical_blob).hexdigest()
        == pin["git_blob_id"],
        "strata_auxiliary_git_blob_preimage",
    )
    names = [pin["path"], *(root + "/" + pin["path"] for root in strata_source_roots(descriptor))]
    for name in names:
        require(
            canonical(bundle.rows.get(name))
            == canonical({"path": name, "size_bytes": pin["size_bytes"], "sha256": pin["sha256"]}),
            "strata_auxiliary_six_required_members",
        )
        require(bundle.read(name, MAX_TEXT) == canonical_blob, "strata_auxiliary_actual_copy_differs")
    return {
        "path": pin["path"],
        "git_blob_id": pin["git_blob_id"],
        "copies": names,
        "scope": "base_Git_bound_source_auxiliary_only_not_a_deployed_model_or_neural_result",
    }


def verify_pe_evidence(bundle, receipt_file, engine_file):
    receipt = object_(bundle.load(receipt_file), {"schema", "engine_file", "files", "raw_import_logs"}, "PE_receipt")
    require(receipt["schema"] == "omni-strata-combined-pe-evidence-v2" and receipt["engine_file"] == engine_file, "PE_receipt_identity")
    require(type(receipt["files"]) is list and 0 < len(receipt["files"]) <= 128, "PE_closure_count")
    declared = {}
    for row in receipt["files"]:
        object_(row, {"path", "size_bytes", "sha256", "normal", "delay", "non_system", "system"}, "PE_record")
        require(row["path"] not in declared, "PE_duplicate")
        require(bundle.rows.get(row["path"]) == {key: row[key] for key in ("path", "size_bytes", "sha256")}, "PE_member_identity")
        declared[row["path"]] = row
    engine_dir = PurePosixPath(engine_file).parent
    paths_by_casefold = {}
    for path in bundle.rows:
        folded = path.casefold()
        require(folded not in paths_by_casefold, "PE_case_alias")
        paths_by_casefold[folded] = path
    system_names = {"kernel32.dll", "advapi32.dll", "ntdll.dll", "user32.dll", "bcrypt.dll", "shell32.dll",
                    "ole32.dll", "oleaut32.dll", "ws2_32.dll", "secur32.dll", "crypt32.dll", "version.dll"}
    queue, observed = [engine_file], {}
    while queue:
        name = queue.pop()
        if name in observed:
            continue
        require(name in declared and PurePosixPath(name).parent == engine_dir, "PE_appdir_closure")
        imports = pe_imports(bundle.path(name))
        node = {**bundle.rows[name], **imports, "non_system": [], "system": []}
        for library in sorted(set(imports["normal"] + imports["delay"])):
            local = paths_by_casefold.get((engine_dir / library).as_posix().casefold())
            if local is not None:
                node["non_system"].append(library)
                queue.append(local)
            else:
                require(library in system_names or library.startswith(("api-ms-win-", "ext-ms-win-")), "PE_unknown_external_DLL")
                node["system"].append(library)
        require(node == declared[name], "PE_actual_imports")
        observed[name] = node
    require(set(observed) == set(declared)
            and {"cublas64_13.dll", "cublaslt64_13.dll"} <= {PurePosixPath(path).name.casefold() for path in observed}, "PE_full_closure")
    logs = receipt["raw_import_logs"]
    require(type(logs) is list and len(logs) == len(observed), "PE_raw_log_count")
    seen = set()
    for row in logs:
        object_(row, {"binary_file", "log_file", "log_sha256", "exit_code", "tool_file", "tool_sha256"}, "PE_log_record")
        require(row["binary_file"] in observed and row["binary_file"] not in seen
                and type(row["exit_code"]) is int and row["exit_code"] == 0, "PE_log_owner")
        bundle.expected(row["tool_file"], digest(row["tool_sha256"]), read=False)
        bundle.expected(row["log_file"], digest(row["log_sha256"]))
        seen.add(row["binary_file"])
    identity = {"schema": "omni-strata-static-PE-closure-v2", "files": sorted(observed.values(), key=lambda x: x["path"]),
                "all_dynamic_loads_covered": False, "current_system_dependencies_verified": False}
    return identity | {"identity_sha256": sha(canonical(identity))}


def reviewed_parser(bundle, parser_file):
    """Future execution loads only the exact independently reviewed pure parser.

    Merely preparing this function does not import/execute the parser. The
    static verifier never imports vLLM, Strata stages or an arbitrary module.
    """
    source = bundle.expected(parser_file, PARSER_SHA)
    module = types.ModuleType("_byte_bound_private_strata_exec_parser")
    exec(compile(source, "<byte-bound-reviewed-strata-exec-parser>", "exec"), module.__dict__)
    return module


def check_fixture_frames(parser, case, raw, expected_layout=None):
    """Independently parse actual header frames; no manual passed flag suffices."""
    require(type(raw) is bytes and len(raw) <= 6 * 32768, "fixture_raw_bound")
    lines = raw.splitlines(keepends=True)
    expected_count = 0 if case == "disabled" else 2 if case == "missing-end" else 6 if case == "cpu-stale-sequence" else 3
    require(len(lines) == expected_count, "fixture_raw_count")
    snapshots = []
    for index, line in enumerate(lines):
        try:
            row = parser.parse_native_line(line)
        except ValueError:
            raise EvidenceError("fixture_actual_native_schema") from None
        require(row["native_request_seq"] == index // 3 + 1 and row["snapshot_seq"] == index % 3, "fixture_native_order")
        if case == "clock-unavailable" and index == 0:
            require(row["clock"]["ticks"] is None and row["clock"]["frequency_hz"] is None, "fixture_unavailable_clock")
        else:
            require(row["clock"]["ticks"] == 100 * (index + 1) and row["clock"]["frequency_hz"] == 1000, "fixture_emitted_clock")
        if expected_layout is None:
            expected_layout = row["observer"]
        require(canonical(row["observer"]) == canonical(expected_layout), "fixture_layout_variation")
        snapshots.append(row)
    if not snapshots:
        return expected_layout, snapshots
    final = snapshots[-1]
    if case in ("cancelled", "cancelled-pending-fence"):
        require(final["request_terminal"] == "cancelled" and final["boundary_complete"] is False, "fixture_cancel_terminal")
    if case != "missing-end" and final["issues"]:
        require(final["boundary_complete"] is False, "fixture_sticky_issue_terminal")
    issue_bits = {
        "allocator-failed-allocation": 256, "allocator-failed-free": 512, "allocator-same-owner": 128 | 512,
        "allocator-registry-exhaustion": 32 | 64, "cpu-invalid-geometry": 2,
        "cpu-stale-phase": 4 | 16, "cpu-stale-sequence": 4 | 16,
        "cuda-launch-error": 1024, "cuda-fence-error": 2048 | 16, "cuda-stale-phase": 4 | 16,
        "boundary-labels": 8, "boundary-order": 8, "clock-unavailable": 8192,
        "cancelled-pending-fence": 16, "counter-overflow": 1, "formatting-overflow": 4096,
    }
    require(final["issues"] == issue_bits.get(case, 0), "fixture_expected_issue_bits")
    memory_expected = {
        "allocator-success-memset-equivalent": (1,1,0,1,1,0,0,1280,0,0,0,0),
        "allocator-failed-allocation": (1,0,1,0,0,0,0,0,0,0,2,None),
        "allocator-failed-free": (1,1,0,1,0,1,100,100,100,0,0,2),
        "allocator-same-owner": (2,2,0,2,1,1,100,140,100,0,0,0),
        "allocator-registry-exhaustion": (17,17,0,1,1,0,1024,1024,1024,0,0,0),
        "allocator-baseline-gauge": (3,3,0,3,3,0,0,300,250,0,0,0),
        "allocator-request-transient": (1,1,0,1,1,0,0,1280,1280,0,0,0),
        "allocator-untracked-families": (0,0,0,0,0,0,0,0,0,3,None,None),
    }
    memory = final["memory"]["ordinary_expert_cache"]
    memory_fields = (*parser.MEMORY_COUNTS, "tracked_live_requested_bytes", "lifetime_peak_requested_bytes",
                     "request_peak_requested_bytes", "untracked_cache_family_bits", "last_allocation_status", "last_free_status")
    if case in memory_expected:
        require(tuple(memory[k] for k in memory_fields) == memory_expected[case], "fixture_memory_values")
    if case == "allocator-baseline-gauge":
        initial = snapshots[0]["memory"]["ordinary_expert_cache"]
        require(initial["tracked_live_requested_bytes"] == initial["request_peak_requested_bytes"] == 200, "fixture_baseline_gauge")
    if case == "counter-overflow":
        require(final["counters"]["cpu"][0]["submitted"] == (1 << 64) - 1, "fixture_saturated_counter")
    if case == "worst-case-frames":
        for row in snapshots:
            for family, fields in (("cpu", parser.CPU_COUNTS), ("cpu_phase", parser.CPU_PHASE_COUNTS), ("cuda", parser.CUDA_COUNTS)):
                require(all(counter[k] == (1 << 64) - 1 for counter in row["counters"][family] for k in fields), "fixture_max_counters")
            require(all(counter["last_launch_status"] == -(1 << 31) and counter["last_fence_status"] == (1 << 31) - 1
                        for counter in row["counters"]["cuda"]), "fixture_max_status")
    if case == "cpu-counts":
        for index in (4, 7):
            cpu = final["counters"]["cpu"][index]
            require(tuple(cpu[k] for k in parser.CPU_COUNTS) == (1,1,2,2,5 if index == 7 else 2,5 if index == 7 else 2), "fixture_CPU_counts")
        require(all(tuple(row[k] for k in parser.CPU_PHASE_COUNTS) == (1,1,9,9)
                    for row in final["counters"]["cpu_phase"][6:12]), "fixture_CPU_row_counts")
    if case in ("cpu-stale-phase", "cpu-stale-sequence"):
        require(final["counters"]["cpu"][4]["completed"] == 1 and final["counters"]["cpu"][8]["completed"] == 0, "fixture_immutable_CPU_phase")
    if case in ("cuda-launch-error", "cuda-successful-fence", "cuda-fence-error", "cuda-stale-phase", "cancelled-pending-fence"):
        index = 4 if case == "cancelled-pending-fence" else 3 if case in ("cuda-successful-fence", "cuda-fence-error") else 2
        counter = final["counters"]["cuda"][index]
        expected = (1, 0 if case == "cuda-launch-error" else 1, 1 if case in ("cuda-successful-fence", "cuda-stale-phase") else 0,
                    1 if case == "cuda-launch-error" else 0, 1 if case == "cuda-fence-error" else 0)
        require(tuple(counter[k] for k in parser.CUDA_COUNTS) == expected, "fixture_CUDA_counts")
        require(counter["last_launch_status"] == (9 if case == "cuda-launch-error" else 0)
                and counter["last_fence_status"] == (7 if case == "cuda-fence-error" else 0 if case in ("cuda-successful-fence", "cuda-stale-phase") else None), "fixture_CUDA_status")
    return copy.deepcopy(expected_layout), snapshots


def verify_abi_evidence(bundle, receipt_file, engine_file, parser_file, build_receipt):
    """Accept only actual closed fixture-v3 output as a reference, never engine ABI."""
    receipt = object_(bundle.load(receipt_file), {"schema", "engine_file", "engine_sha256", "header_sha256",
              "emitter_kind", "fixture_receipt_file", "fixture_source_file", "fixture_recorder_file", "fixture_header_file",
              "fixture_compiler_file", "fixture_executable_file", "fixture_evidence_dir"}, "ABI_receipt")
    require(receipt["schema"] == "omni-strata-compiled-observer-abi-reference-v1" and receipt["engine_file"] == engine_file
            and receipt["engine_sha256"] == bundle.rows[engine_file]["sha256"] and receipt["header_sha256"] == HEADER_V2
            and receipt["emitter_kind"] == "standalone_fixture_v3_reference_not_engine_ABI", "ABI_identity")
    bundle.expected(receipt["fixture_source_file"], FIXTURE_SOURCE_SHA)
    bundle.expected(receipt["fixture_recorder_file"], FIXTURE_RECORDER_SHA)
    bundle.expected(receipt["fixture_header_file"], HEADER_V2)
    fixture = bundle.load(receipt["fixture_receipt_file"])
    keys = {"schema", "status", "scope", "compiler", "inputs", "environment_scope", "compile_exit_code", "case_results",
            "cuda_model_executed", "original_runtime_or_production_modified", "toolchain_environment", "native_exit_preference",
            "inputs_after", "small_inputs_stable", "selected_header_include_resolution_verified", "include_trace_scope",
            "fixture_exe_sha256", "case_processes", "all_owned_case_invocations_returned", "closed_files"}
    object_(fixture, keys, "fixture_receipt_shape")
    require(fixture["schema"] == "omni-standalone-observer-fixture-build-v3"
            and fixture["status"] == "standalone_contract_checks_passed" and type(fixture["compile_exit_code"]) is int
            and fixture["compile_exit_code"] == 0 and type(fixture["case_processes"]) is int
            and fixture["case_processes"] == 29 and fixture["all_owned_case_invocations_returned"] is True
            and fixture["cuda_model_executed"] is False and fixture["original_runtime_or_production_modified"] is False,
            "fixture_completed_scope")
    require(fixture["small_inputs_stable"] is True and fixture["inputs_after"] == fixture["inputs"], "fixture_inputs_stable")
    preference = object_(fixture["native_exit_preference"], {"inherited_present", "inherited_value", "owned_invocation_value",
                         "restore_scope"}, "fixture_native_exit_preference")
    require(type(preference["inherited_present"]) is bool and preference["owned_invocation_value"] is False
            and preference["restore_scope"] == "prior script binding restored or removed; inherited outer binding unchanged"
            and (type(preference["inherited_value"]) is bool if preference["inherited_present"] else preference["inherited_value"] is None),
            "fixture_native_exit_policy")
    input_map = object_(fixture["inputs"], {"source", "header", "recorder"}, "fixture_inputs")
    for name, expected in (("source", FIXTURE_SOURCE_SHA), ("header", HEADER_V2), ("recorder", FIXTURE_RECORDER_SHA)):
        row = object_(input_map[name], {"path", "sha256"}, "fixture_input_record")
        require(type(row["path"]) is str and 0 < len(row["path"]) <= 32768 and row["sha256"] == expected, "fixture_input_identity")
    compiler = object_(fixture["compiler"], {"path", "sha256", "file_version", "product_version", "args", "toolchain"}, "fixture_compiler")
    require(compiler["toolchain"] == "msvc" and compiler["sha256"] == build_receipt["tools"]["cl"]["binary_sha256"], "fixture_same_compiler")
    bundle.expected(receipt["fixture_compiler_file"], compiler["sha256"], read=False)
    bundle.expected(receipt["fixture_executable_file"], digest(fixture["fixture_exe_sha256"]), read=False)
    env = fixture["toolchain_environment"]
    require(type(env) is dict and set(env) == {"CL", "_CL_", "INCLUDE", "LIB", "VCToolsVersion", "WindowsSDKVersion", "VisualStudioVersion"}, "fixture_environment")
    require(env["CL"] in (None, "") and env["_CL_"] in (None, "")
            and env["VCToolsVersion"] == build_receipt["compiler_environment"]["VCToolsVersion"]
            and env["WindowsSDKVersion"] == build_receipt["compiler_environment"]["WindowsSDKVersion"], "fixture_toolchain_binding")
    args = compiler["args"]
    require(type(args) is list and len(args) == 10 and all(type(x) is str for x in args)
            and args[:6] == ["/showIncludes", "/nologo", "/std:c++17", "/EHsc", "/Od", "/W4"]
            and args[6].startswith("/I") and args[7] == input_map["source"]["path"]
            and args[8].startswith("/Fe:") and args[9].startswith("/Fo:"), "fixture_compile_args")
    prefix = relative(receipt["fixture_evidence_dir"])
    require(receipt["fixture_receipt_file"] == prefix + "/receipt.json", "fixture_receipt_location")
    closed = fixture["closed_files"]
    require(type(closed) is list and 0 < len(closed) <= 256, "fixture_closed_count")
    archived = {}
    for row in closed:
        object_(row, {"path", "size_bytes", "sha256"}, "fixture_closed_row")
        require(type(row["path"]) is str, "fixture_closed_path")
        local = relative(row["path"].replace("\\", "/"))
        require(local not in archived and local != "receipt.json", "fixture_closed_duplicate")
        member = prefix + "/" + local
        require(bundle.rows.get(member) == {"path": member, "size_bytes": uint(row["size_bytes"], MAX_MEMBER),
                                           "sha256": digest(row["sha256"])}, "fixture_closed_actual_bytes")
        archived[local] = member
    expected_members = {path[len(prefix) + 1:] for path in bundle.rows if path.startswith(prefix + "/")}
    require(set(archived) | {"receipt.json"} == expected_members, "fixture_full_closed_inventory")
    require(receipt["fixture_executable_file"] == prefix + "/observer_fixture.exe", "fixture_EXE_location")
    include = bundle.read(prefix + "/compiler-stdout.txt") + bundle.read(prefix + "/compiler-stderr.txt")
    require(fixture["selected_header_include_resolution_verified"] is True
            and input_map["header"]["path"].replace("\\", "/").lower().encode("utf-8")
            in include.replace(b"\\", b"/").lower(), "fixture_actual_include_trace")
    rows = fixture["case_results"]
    expected = [(case, case, "1") for case in FIXTURE_CASES]
    expected += [("disabled-" + label, "disabled", value) for label, value in (("unset", None), ("0", "0"), ("true", "true"), ("11", "11"))]
    require(type(rows) is list and len(rows) == len(expected) == 29, "fixture_exact_29_cases")
    parser = reviewed_parser(bundle, parser_file)
    layout, observations = None, []
    for row, (label, case, environment) in zip(rows, expected):
        object_(row, {"label", "case", "observer_env", "exit_code", "passed", "raw_frames_present"}, "fixture_case_record")
        require((row["label"], row["case"], row["observer_env"]) == (label, case, environment)
                and type(row["exit_code"]) is int and row["exit_code"] == 0 and row["passed"] is True
                and row["raw_frames_present"] is True, "fixture_case_completion")
        stdout = object_(json_(bundle.read(prefix + "/" + label + "/stdout.txt", MAX_JSON)),
                         {"schema", "case", "passed", "records", "real_cuda_or_model_executed"}, "fixture_stdout")
        raw = bundle.read(prefix + "/" + label + "/native-frames.txt", 6 * 32768)
        layout, snapshots = check_fixture_frames(parser, case, raw, layout)
        require(stdout["schema"] == "omni-standalone-observer-fixture-v1" and stdout["case"] == case
                and stdout["passed"] is True and type(stdout["records"]) is int and stdout["records"] == len(snapshots)
                and stdout["real_cuda_or_model_executed"] is False
                and bundle.read(prefix + "/" + label + "/stderr.txt") == b"", "fixture_actual_stdout_stderr")
        observations.append({"label": label, "raw_sha256": sha(raw), "native_frames": len(snapshots)})
    require(layout is not None, "fixture_no_emitted_layout")
    return {"schema": "omni-strata-compiled-fixture-ABI-reference-v1", "observer_layout": copy.deepcopy(layout),
            "observer_layout_scope": "compiled_standalone_fixture_reference_only", "compiled_engine_ABI_verified": False,
            "fixture_receipt_sha256": bundle.rows[receipt["fixture_receipt_file"]]["sha256"],
            "fixture_source_sha256": FIXTURE_SOURCE_SHA, "fixture_recorder_sha256": FIXTURE_RECORDER_SHA,
            "fixture_executable_sha256": fixture["fixture_exe_sha256"], "case_processes": 29,
            "actual_frame_observations": observations, "system_toolchain_include_closure_verified": False,
            "runtime_binding": None, "runtime_qualified": False}


def parser_runtime_binding(*args, **kwargs):
    """An old I/O identity or a static receipt must never authorize OMNI_EXEC."""
    raise EvidenceError("live_runtime_owner_module_and_adapter_verifier_required")


def record_verification_attempt(descriptor_file, runtime_root, output_file):
    """Future opt-in audit retains failures outside the immutable input bundle.

    This function has not been imported or executed. It performs no compiler,
    process, GPU or model operation; it only invokes the static byte verifier.
    """
    root = Path(runtime_root).resolve(strict=False)
    output = Path(output_file).absolute()
    require(not output.resolve(strict=False).is_relative_to(root), "audit_output_inside_runtime")
    require(output.parent.is_dir() and not output.exists(), "audit_output_must_be_fresh")
    result = {"schema": "omni-private-combined-runtime-verification-attempt-v1",
              "verifier_source_sha256": sha(file_bytes(Path(__file__), MAX_JSON)),
              "descriptor_file": None, "status": "failed", "identity": None,
              "runtime_eligible": False, "runtime_qualification": False, "native_execution": False}
    try:
        result["descriptor_file"] = relative(descriptor_file)
        result["identity"] = verify_combined_runtime(descriptor_file, root)
        result["status"] = "static_records_verified_live_eligibility_unavailable"
    except EvidenceError as error:
        result["reason_code"] = error.code
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
        result["reason_code"] = "static_evidence_unavailable_or_invalid"
        result["failure_type"] = type(error).__name__
    encoded = canonical(result) + b"\n"
    require(len(encoded) <= MAX_JSON, "audit_result_bound")
    with output.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return copy.deepcopy(result)
