"""Small, deterministic helpers shared by geometry objects."""
import hashlib
import json

import numpy as np


def fingerprint(metadata, *arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)
