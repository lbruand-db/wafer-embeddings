"""WM-811K parsing and dataset hygiene (variable-size, single-label).

WM-811K ships as ``LSWMD.pkl`` — a pickled pandas DataFrame with columns
``waferMap`` (variable H x W, cells in {0=no-die, 1=pass, 2=fail}), ``failureType``
(one of 9 classes or empty = unlabeled), and ``trianTestLabel``. Most maps are
unlabeled — fine for label-free DINO; labels are used only for evaluation.

Pure functions over numpy/lists so they are testable without the real pickle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafer_embeddings.data.common import FAIL, NO_DIE, PASS, dedup_indices, split_assignments

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


def parse_records(
    maps,
    raw_labels,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "wafer-embeddings/wm811k/v1",
    skip_invalid: bool = True,
) -> ParsedWM811K:
    """Validate/clean maps + labels, dedup, and assign leakage-safe splits.

    With ``skip_invalid`` (default), maps that aren't 2-D {0,1,2} are dropped (robust
    to real-world oddities); otherwise a bad map raises.
    """
    cleaned: list[np.ndarray] = []
    names: list[str | None] = []
    for m, raw in zip(maps, raw_labels):
        a = _valid_map(m)
        if a is None:
            if skip_invalid:
                continue
            raise ValueError("invalid wafer map (need 2-D with cells in {0,1,2})")
        cleaned.append(a)
        names.append(clean_label(raw))

    keep = dedup_indices(cleaned)
    maps_out = [cleaned[i] for i in keep]
    names_out = [names[i] for i in keep]
    label_ids = np.array([label_to_id(n) for n in names_out], dtype=np.int64)
    split = split_assignments(maps_out, fractions, salt)
    return ParsedWM811K(maps=maps_out, label_ids=label_ids, label_names=names_out, split=split)
