# SPDX-License-Identifier: Apache-2.0
import importlib.util
import struct
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / 'benchmarks/edge_harness/compare_strata_sve.py'
_spec = importlib.util.spec_from_file_location('sve_compare_test_subject', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def sve(path, values, shape=(2, 2, 1, 1)):
    path.write_bytes(struct.pack('<5i', 0x31455653, *shape) + struct.pack('<' + 'f' * len(values), *values))
    return path


def test_exact_and_numeric_difference(tmp_path):
    a = sve(tmp_path / 'a', [3, 4])
    b = sve(tmp_path / 'b', [3, 4])
    result = _module.compare(a, b)
    assert result['exact_tensor_equal'] and result['relative_l2'] == 0
    sve(b, [3, 5])
    result = _module.compare(a, b)
    assert result['relative_l2'] == pytest.approx(0.2)
    assert result['max_absolute_error'] == 1
    assert not result['release_qualified']


@pytest.mark.parametrize('values', [[float('nan'), 0], [float('inf'), 1]])
def test_nonfinite_rejected(tmp_path, values):
    with pytest.raises(ValueError, match='nonfinite'):
        _module.read_sve(sve(tmp_path / 'a', values))


def test_shape_and_length_refused(tmp_path):
    a = sve(tmp_path / 'a', [1, 2])
    b = sve(tmp_path / 'b', [1, 2], (1, 1, 1, 2))
    with pytest.raises(ValueError, match='shapes differ'):
        _module.compare(a, b)
    b.write_bytes(b.read_bytes()[:-1])
    with pytest.raises(ValueError, match='size mismatch'):
        _module.read_sve(b)
    with pytest.raises(ValueError, match='bound'):
        _module.read_sve(a, max_bytes=20)


def test_zero_reference_is_explicit(tmp_path):
    a = sve(tmp_path / 'a', [0, 0])
    b = sve(tmp_path / 'b', [1, 0])
    result = _module.compare(a, b)
    assert result['relative_l2'] is None and result['cosine_similarity'] is None
