# SPDX-License-Identifier: Apache-2.0
"""Compare bound SVE tensors. Metrics do not confer model/task qualification."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def read_sve(path, max_bytes=256 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes or len(raw) < 20:
        raise ValueError('SVE file exceeds bound or lacks header')
    magic, n, nx, ny, width = struct.unpack('<5i', raw[:20])
    if magic != 0x31455653 or min(n, nx, ny, width) <= 0 or nx * ny != n:
        raise ValueError('Invalid SVE magic or dimensions')
    if len(raw) != 20 + n * width * 4:
        raise ValueError('SVE payload size mismatch')
    values = np.frombuffer(raw, dtype='<f4', offset=20).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('SVE contains nonfinite values')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw),
            'shape': [n, nx, ny, width]}, values


def compare(reference, observed):
    ri, rv = read_sve(reference)
    oi, ov = read_sve(observed)
    if ri['shape'] != oi['shape']:
        raise ValueError('Reference and observed SVE shapes differ')
    delta = ov - rv
    rn, on, dn = (float(np.linalg.norm(v)) for v in (rv, ov, delta))
    return {'schema': 'omni-sve-comparison-v1', 'reference': ri, 'observed': oi,
            'elements': int(rv.size), 'max_absolute_error': float(np.max(np.abs(delta))),
            'mean_absolute_error': float(np.mean(np.abs(delta))),
            'rmse': float(np.sqrt(np.mean(delta * delta))),
            'relative_l2': dn / rn if rn else (0.0 if dn == 0 else None),
            'cosine_similarity': float(np.dot(rv, ov) / (rn * on)) if rn and on else None,
            'reference_l2': rn, 'observed_l2': on,
            'exact_tensor_equal': bool(np.array_equal(rv, ov)),
            'release_qualified': False,
            'scope': 'Tensor diagnostics only; no inherited checkpoint or task quality qualification'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--observed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.reference, a.observed)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
