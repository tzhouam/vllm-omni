# SPDX-License-Identifier: Apache-2.0
"""Production file-read contract, extracted without runtime/model initialization."""
import ast
import io
import stat
import types
import unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[2] / "vllm_omni/engine/backends/strata_execution/strata_exec_runtime.py")


class EvidenceError(ValueError):
    pass


def require(value, reason):
    if not value:
        raise EvidenceError(reason)


class Stream(io.BytesIO):
    def fileno(self):
        return 71


def read_fixture(payload, maximum, *, after_change=None, short=False, regular=True, error=None):
    tree = ast.parse(SOURCE.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "file_bytes")
    before = dict(st_mode=stat.S_IFREG if regular else stat.S_IFDIR, st_size=len(payload),
                  st_dev=4, st_ino=31, st_mtime_ns=123)
    after = dict(before, **(after_change or {}))
    observations = iter((types.SimpleNamespace(**before), types.SimpleNamespace(**after)))
    stream = Stream(payload)
    normal_read = stream.read
    def read(amount):
        if error:
            raise error
        data = normal_read(amount)
        return data[:-1] if short and data else data
    stream.read = read
    namespace = {
        "Path": lambda _: types.SimpleNamespace(open=lambda mode: stream),
        "os": types.SimpleNamespace(fstat=lambda fd: next(observations)),
        "stat": stat, "require": require,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    try:
        return namespace["file_bytes"]("fixture", maximum)
    finally:
        assert stream.closed, "opened stream must close on success and failure"


class Contract(unittest.TestCase):
    def test_complete_bytes_at_empty_small_and_boundary_sizes(self):
        for size in (0, 1, 4095, 4096, 4097, 65537):
            with self.subTest(size=size):
                payload = (bytes(range(256)) * ((size + 255) // 256))[:size]
                self.assertEqual(read_fixture(payload, max(size, 1 << 20)), payload)

    def test_oversize_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "file_bound"):
            read_fixture(b"abcd", 3)

    def test_nonregular_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "file_bound"):
            read_fixture(b"abcd", 4, regular=False)

    def test_identity_size_or_timestamp_change_rejected(self):
        for field, value in (("st_dev", 5), ("st_ino", 32), ("st_size", 3), ("st_size", 5), ("st_mtime_ns", 124)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(EvidenceError, "file_changed"):
                    read_fixture(b"abcd", 100, after_change={field: value})

    def test_incomplete_read_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "file_changed"):
            read_fixture(b"abcd", 100, short=True)

    def test_read_failure_propagates_and_closes(self):
        with self.assertRaisesRegex(OSError, "injected read failure"):
            read_fixture(b"abcd", 100, error=OSError("injected read failure"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
