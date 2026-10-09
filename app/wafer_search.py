"""Pure helpers for the wafer-search app: map codec, rendering, SQL (CI-tested).

Self-contained (numpy only) because the app ships as its own source directory; a test
pins ``decode_map`` to the package's ``wafer_embeddings.serving.encode_map`` format.
"""

from __future__ import annotations

import base64

import numpy as np

TABLE = "wafer_embeddings.wafer_map_embeddings_pg"  # synced table: <uc schema>.<uc table>
# no-die, pass, fail -> background, light grey, red
PALETTE = np.array([[255, 255, 255], [205, 205, 205], [214, 39, 40]], dtype=np.uint8)


def decode_map(s: str) -> np.ndarray:
    """Decode ``"<h>x<w>:<base64 2-bit cells>"`` into an (h, w) uint8 map."""
    shape, b64 = s.split(":", 1)
    h, w = (int(x) for x in shape.split("x"))
    packed = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    cells = np.stack([(packed >> k) & 3 for k in (0, 2, 4, 6)], axis=1).reshape(-1)
    return cells[: h * w].reshape(h, w).astype(np.uint8)


def render(wafer_map: np.ndarray, scale: int = 4) -> np.ndarray:
    """RGB image of a wafer map, each die drawn as a ``scale`` x ``scale`` block."""
    rgb = PALETTE[np.clip(wafer_map, 0, 2)]
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def auto_scale(wafer_map: np.ndarray, target_px: int = 160) -> int:
    """Block size so maps of any grid size render at roughly ``target_px`` wide."""
    return max(1, target_px // max(wafer_map.shape))


def vector_literal(embedding) -> str:
    """pgvector text literal ``[x1,x2,...]`` for a bind parameter."""
    return "[" + ",".join(f"{float(x):.7g}" for x in embedding) + "]"


def neighbours_sql(table: str = TABLE, exclude_same_lot: bool = False) -> str:
    """Top-k cosine neighbours via the lakebase_ann index (SPECS.md §9).

    Binds: %(q)s query vector literal, %(qid)s query wafer id (excluded), %(lot)s its lot,
    %(k)s result count. ``ORDER BY embedding <=> q`` is what the ANN index serves.
    """
    lot = "AND (lot IS DISTINCT FROM %(lot)s) " if exclude_same_lot else ""
    return (
        f"SELECT id, label, split, lot, height, width, map_code, "
        f"1 - (embedding <=> %(q)s::vector) AS cosine_similarity "
        f"FROM {table} WHERE id <> %(qid)s {lot}"
        f"ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s"
    )


def sample_sql(table: str = TABLE, with_label: bool = False) -> str:
    """A random wafer (optionally of one class) to use as the query."""
    where = "WHERE label = %(label)s " if with_label else "WHERE label IS NOT NULL "
    return (
        f"SELECT id, label, split, lot, height, width, map_code, embedding::text "
        f"FROM {table} {where}ORDER BY random() LIMIT 1"
    )


def by_id_sql(table: str = TABLE) -> str:
    return (
        f"SELECT id, label, split, lot, height, width, map_code, embedding::text "
        f"FROM {table} WHERE id = %(qid)s"
    )


def parse_vector(text: str) -> list[float]:
    """Parse a pgvector ``[x,y,...]`` text value."""
    return [float(x) for x in text.strip().strip("[]").split(",") if x]


def endpoint_request(wafer_map: np.ndarray) -> dict:
    """Serving-endpoint body for one map (Delta row layout, see the pyfunc signature)."""
    h, w = wafer_map.shape
    return {
        "dataframe_split": {
            "columns": ["height", "width", "wafer_map"],
            "data": [[int(h), int(w), [int(x) for x in wafer_map.reshape(-1)]]],
        }
    }
