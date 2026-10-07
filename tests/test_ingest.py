import numpy as np

from wafer_embeddings.data import wm811k as wm
from wafer_embeddings.jobs.ingest import record_to_row


def test_record_to_row_roundtrips_variable_shape_and_label():
    m = np.random.default_rng(0).integers(0, 3, size=(10, 12), dtype=np.uint8)
    rec = wm.Record(m, "Scratch", wm.label_to_id("Scratch"), "train")
    row = record_to_row(rec, row_id=7)
    assert set(row) == {"id", "height", "width", "wafer_map", "label", "label_id", "split"}
    assert row["id"] == 7 and row["height"] == 10 and row["width"] == 12
    recon = np.array(row["wafer_map"], dtype=np.uint8).reshape(row["height"], row["width"])
    np.testing.assert_array_equal(recon, m)
    assert row["label"] == "Scratch" and row["label_id"] == wm.label_to_id("Scratch")


def test_record_to_row_unlabeled():
    m = np.zeros((4, 4), dtype=np.uint8)
    row = record_to_row(wm.Record(m, None, wm.UNLABELED, "test"), 0)
    assert row["label"] is None and row["label_id"] == wm.UNLABELED


def _vmap(h, w, seed):
    return np.random.default_rng(seed).integers(0, 3, size=(h, w), dtype=np.uint8)


def test_iter_records_streams_and_dedups():
    a = _vmap(8, 8, 1)
    recs = list(wm.iter_records([a, a.copy(), _vmap(8, 8, 2)], ["Center", "Center", "Loc"]))
    assert len(recs) == 2  # exact duplicate dropped
    assert all(isinstance(r, wm.Record) for r in recs)


def test_iter_records_shared_seen_dedups_across_chunks():
    a = _vmap(6, 6, 3)
    seen: set[str] = set()
    first = list(wm.iter_records([a], ["Loc"], seen=seen))
    second = list(wm.iter_records([a.copy()], ["Loc"], seen=seen))  # same content, next chunk
    assert len(first) == 1 and len(second) == 0  # cross-chunk dedup via shared seen
