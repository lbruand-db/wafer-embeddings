"""Helpers over wafer-map row dicts (the Delta / parquet layout: ``height``, ``width``,
flattened ``wafer_map``), shared by the jobs and the tools."""

from __future__ import annotations

import numpy as np


def row_map(r: dict) -> np.ndarray:
    """The (height, width) uint8 map of one row."""
    return np.asarray(r["wafer_map"], dtype=np.uint8).reshape(int(r["height"]), int(r["width"]))


def rows_to_maps(rows) -> list[np.ndarray]:
    return [row_map(r) for r in rows]


def shape_groups(rows) -> np.ndarray:
    """Map shape ``"<h>x<w>"`` per row: the device proxy behind the cross-device metrics
    (devices span lots, so same-shape neighbours share labels far above chance)."""
    return np.array([f"{r['height']}x{r['width']}" for r in rows], dtype=object)
