"""WM-811K parsing and dataset hygiene (variable-size, single-label).

WM-811K ships as ``LSWMD.pkl`` — a pickled pandas DataFrame with columns
``waferMap`` (variable H x W, cells in {0=no-die, 1=pass, 2=fail}), ``failureType``
(one of 9 classes or empty = unlabeled), and ``trianTestLabel``. Most maps are
unlabeled — fine for label-free DINO; labels are used only for evaluation.

Pure functions over numpy/lists so they are testable without the real pickle.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import numpy as np

from wafer_embeddings.data.common import (
    FAIL,
    NO_DIE,
    PASS,
    content_hash,
    split_for,
)

# The 9 WM-811K classes ("none" = labeled defect-free; unlabeled maps map to id -1).
CLASS_NAMES: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
UNLABELED = -1

# Normalized (lowercased, de-punctuated) -> canonical name, for messy label strings.
_CANON = {name.lower().replace("-", "").replace(" ", ""): name for name in CLASS_NAMES}


def clean_label(raw) -> str | None:
    """Normalize a raw ``failureType`` cell to a canonical class name, or None.

    WM-811K stores labels inconsistently — nested object arrays like
    ``array([['Center']])``, empty arrays, ``'[]'``, or plain strings. Anything not
    recognized (incl. empty / unlabeled) returns None.
    """
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return None
        raw = raw.flatten()[0]
    s = str(raw).strip()
    if not s or s == "[]":
        return None
    return _CANON.get(s.lower().replace("-", "").replace(" ", ""))


def label_to_id(name: str | None) -> int:
    return CLASS_NAMES.index(name) if name in CLASS_NAMES else UNLABELED


@dataclass(frozen=True)
class Record:
    """One cleaned, deduped wafer with its single label and split."""

    wafer_map: np.ndarray  # (H, W) uint8
    label_name: str | None  # None = unlabeled
    label_id: int  # class id or -1
    split: str  # 'train' | 'val' | 'test'


@dataclass(frozen=True)
class ParsedWM811K:
    maps: list[np.ndarray]  # N variable-size (H, W) uint8 maps
    label_ids: np.ndarray  # (N,) int: class id or -1 (unlabeled)
    label_names: list[str | None]  # len N
    split: np.ndarray  # (N,) of 'train'/'val'/'test'


def _valid_map(m) -> np.ndarray | None:
    try:
        a = np.asarray(m, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    if not set(np.unique(a)).issubset({NO_DIE, PASS, FAIL}):
        return None
    return a


def iter_records(
    maps,
    raw_labels,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "wafer-embeddings/wm811k/v1",
    skip_invalid: bool = True,
    seen: set[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[Record]:
    """Stream cleaned, deduped ``Record``s one at a time (constant memory).

    Dedup uses a running ``seen`` hash set (shared across chunks); splits are per-map
    content hashes, so no global pass is needed. Invalid maps are skipped (or raise if
    ``skip_invalid`` is False). ``progress(done, total)`` is called per scanned input.
    """
    if seen is None:
        seen = set()
    total = len(maps) if hasattr(maps, "__len__") else 0
    for i, (m, raw) in enumerate(zip(maps, raw_labels), start=1):
        a = _valid_map(m)
        if a is None:
            if not skip_invalid:
                raise ValueError("invalid wafer map (need 2-D with cells in {0,1,2})")
        else:
            h = content_hash(a)
            if h not in seen:
                seen.add(h)
                name = clean_label(raw)
                yield Record(a, name, label_to_id(name), split_for(a, fractions, salt))
        if progress is not None:
            progress(i, total)


def parse_records(
    maps,
    raw_labels,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "wafer-embeddings/wm811k/v1",
    skip_invalid: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> ParsedWM811K:
    """Eager wrapper over :func:`iter_records` (materializes all records).

    Convenient for in-memory/testing use; the ingest job uses ``iter_records`` directly
    so it never holds every row at once.
    """
    recs = list(iter_records(maps, raw_labels, fractions, salt, skip_invalid, None, progress))
    return ParsedWM811K(
        maps=[r.wafer_map for r in recs],
        label_ids=np.array([r.label_id for r in recs], dtype=np.int64),
        label_names=[r.label_name for r in recs],
        split=np.array([r.split for r in recs], dtype=object),
    )
