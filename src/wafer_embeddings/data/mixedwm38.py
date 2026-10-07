"""MixedWM38 parsing and dataset hygiene.

Dataset format (SPECS.md §2): an ``.npz`` with
  ``arr_0`` : (N, 52, 52) uint8 wafer maps, cell in {0=no-die, 1=pass, 2=fail}
  ``arr_1`` : (N, 8)      multi-hot of the 8 base defect types
The 38 classes are the observed combinations of those 8 base defects (+ Normal).

Everything here is a pure function over numpy arrays so it is testable without the
real file; ``load_npz`` is the thin I/O wrapper used by the ingest job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafer_embeddings.data.common import (  # re-exported for back-compat
    FAIL,
    NO_DIE,
    PASS,
    content_hash,
    dedup_indices,
    split_assignments,
)

__all__ = [
    "FAIL",
    "NO_DIE",
    "PASS",
    "BASE_DEFECTS",
    "content_hash",
    "dedup_indices",
    "split_assignments",
    "pattern_key",
    "pattern_name",
    "ParsedWafers",
    "parse_arrays",
    "load_npz",
]

# Base defect names. NOTE: the exact column order of ``arr_1`` must be confirmed
# against the MixedWM38 README before trusting pattern *names* (the mechanics below
# don't depend on the order). SPECS.md §16 item 21.
BASE_DEFECTS: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Scratch",
    "Random",
)
N_BASE = len(BASE_DEFECTS)


def pattern_key(label_row: np.ndarray) -> tuple[int, ...]:
    """Active base-defect indices (sorted). Empty tuple == Normal (defect-free)."""
    idx = np.flatnonzero(np.asarray(label_row).astype(bool))
    return tuple(int(i) for i in idx)


def pattern_name(key: tuple[int, ...]) -> str:
    if not key:
        return "Normal"
    return "+".join(BASE_DEFECTS[i] for i in key)


@dataclass(frozen=True)
class ParsedWafers:
    maps: np.ndarray  # (N, H, W) uint8
    labels: np.ndarray  # (N, 8) uint8 multi-hot
    pattern_names: list[str]  # len N
    split: np.ndarray  # (N,) of 'train'/'val'/'test'


def parse_arrays(
    maps: np.ndarray,
    labels: np.ndarray,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "wafer-embeddings/v1",
) -> ParsedWafers:
    """Validate shapes, dedup, derive pattern names + leakage-safe splits."""
    maps = np.asarray(maps, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.uint8)
    if maps.ndim != 3:
        raise ValueError(f"maps must be (N, H, W), got {maps.shape}")
    if labels.shape[:1] != maps.shape[:1] or labels.shape[1] != N_BASE:
        raise ValueError(f"labels must be (N, {N_BASE}), got {labels.shape}")
    if not set(np.unique(maps)).issubset({NO_DIE, PASS, FAIL}):
        raise ValueError("maps may only contain {0, 1, 2}")

    keep = dedup_indices(maps)
    maps, labels = maps[keep], labels[keep]
    names = [pattern_name(pattern_key(row)) for row in labels]
    split = split_assignments(maps, fractions, salt)
    return ParsedWafers(maps=maps, labels=labels, pattern_names=names, split=split)


def load_npz(path: str):  # pragma: no cover - thin I/O wrapper
    """Load MixedWM38 arrays from an ``.npz`` (run ``-I`` on untrusted data)."""
    with np.load(path) as z:
        return z["arr_0"], z["arr_1"]
