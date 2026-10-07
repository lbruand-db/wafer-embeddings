"""Dataset-agnostic wafer helpers: cell states, hashing, dedup, splits.

Shared by the MixedWM38 and WM-811K data layers. All functions operate on individual
wafer maps (any H x W), so they work for fixed- and variable-size datasets alike.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Cell states (SPECS.md §2); ``no_die`` cells are dropped at tokenization time.
NO_DIE, PASS, FAIL = 0, 1, 2


def content_hash(wafer_map: np.ndarray) -> str:
    """Stable hash of a wafer map's exact contents (shape-sensitive)."""
    a = np.ascontiguousarray(wafer_map, dtype=np.uint8)
    h = hashlib.sha1(repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def dedup_indices(maps) -> np.ndarray:
    """Indices of the first occurrence of each distinct map (exact-content dedup).

    Accepts a 3-D array or a list of variable-size 2-D maps.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for i, m in enumerate(maps):
        hid = content_hash(m)
        if hid not in seen:
            seen.add(hid)
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def _unit_hash(wafer_map: np.ndarray, salt: str) -> float:
    """Map content -> deterministic float in [0, 1) via a salted hash."""
    h = hashlib.sha1(salt.encode())
    h.update(content_hash(wafer_map).encode())
    return int(h.hexdigest()[:8], 16) / 0x100000000


def split_assignments(
    maps,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "wafer-embeddings/v1",
) -> np.ndarray:
    """Assign each map to 'train'/'val'/'test' by content hash.

    Leakage-safe: identical maps always land in the same split, so duplicates (or a
    wafer that also appears in pretraining) cannot straddle the boundary (SPECS.md §8.7).
    Accepts a 3-D array or a list of variable-size 2-D maps.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    train_f, val_f, _ = fractions
    maps_list = list(maps)
    names = np.empty(len(maps_list), dtype=object)
    for i, m in enumerate(maps_list):
        u = _unit_hash(m, salt)
        if u < train_f:
            names[i] = "train"
        elif u < train_f + val_f:
            names[i] = "val"
        else:
            names[i] = "test"
    return names
