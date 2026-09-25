import numpy as np
import pytest

from benchmarks.edge_harness.experiments.minicpmo_kv_site.minicpmo_kv_patch import (
    run_tiled_projection,
)


def test_kv_projection_tiles_preserve_crop_and_token_order():
    hidden = np.zeros((3, 1035, 1152), dtype=np.float32)
    hidden[..., 0] = np.arange(3105, dtype=np.float32).reshape(3, 1035)
    hidden[..., 1] = np.arange(3105, dtype=np.float32).reshape(3, 1035) * 2
    calls = []

    def project(tile, index, valid):
        calls.append((index, valid, tile.copy()))
        return tile[..., :2] + np.array([3, -7], dtype=np.float32)

    observed = run_tiled_projection(hidden, project, output_width=2)
    np.testing.assert_array_equal(observed, hidden[..., :2] + [3, -7])
    assert [(index, valid) for index, valid, _ in calls] == [
        (0, 1024), (1, 1024), (2, 1024), (3, 33)]
    assert not np.any(calls[-1][2][0, 33:])


def test_kv_projection_tiles_refuse_unbounded_or_wrong_precision():
    oversized = np.broadcast_to(np.zeros((1, 1, 1152), dtype=np.float32), (1, 4097, 1152))
    with pytest.raises(ValueError, match="exceed 4096"):
        run_tiled_projection(oversized, lambda *_: None)
    with pytest.raises(ValueError, match="float32"):
        run_tiled_projection(np.zeros((1, 1, 1152), dtype=np.float16), lambda *_: None)
