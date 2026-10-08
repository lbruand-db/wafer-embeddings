import numpy as np

from wafer_embeddings.data import wm811k as wm
from wafer_embeddings.jobs.ingest import record_to_row


def test_record_to_row_roundtrips_variable_shape_and_label():
    m = np.random.default_rng(0).integers(0, 3, size=(10, 12), dtype=np.uint8)
    rec = wm.Record(m, "Scratch", wm.label_to_id("Scratch"), "train")
    row = record_to_row(rec, row_id=7)
    assert set(row) == {"id", "height", "width", "wafer_map", "label", "label_id", "split", "lot"}
    assert row["lot"] is None  # lot defaults to unknown
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


def test_iter_records_lot_grouped_split_keeps_each_lot_in_one_split():
    n_lots, per_lot = 60, 5
    maps = [_vmap(7, 9, s) for s in range(n_lots * per_lot)]
    lots = [f"lot{i // per_lot}" for i in range(len(maps))]
    recs = list(wm.iter_records(maps, ["Center"] * len(maps), lots=lots))
    by_lot: dict[str | None, set[str]] = {}
    for r in recs:
        by_lot.setdefault(r.lot, set()).add(r.split)
    assert len(by_lot) == n_lots
    assert all(len(s) == 1 for s in by_lot.values())  # no lot straddles a split
    assert {next(iter(s)) for s in by_lot.values()} == {"train", "val", "test"}


def test_iter_records_lot_split_is_deterministic_and_carried_to_rows():
    maps = [_vmap(5, 5, s) for s in range(20)]
    lots = [f"lot{i % 4}" for i in range(20)]
    a = list(wm.iter_records(maps, [""] * 20, lots=lots))
    b = list(wm.iter_records(maps, [""] * 20, lots=lots))
    assert [r.split for r in a] == [r.split for r in b]
    assert record_to_row(a[0], 0)["lot"] == "lot0"


def test_iter_records_missing_lot_falls_back_to_per_map_split():
    from wafer_embeddings.data.common import split_for

    m = _vmap(6, 7, 11)
    (rec,) = wm.iter_records([m], ["Loc"], lots=[float("nan")])
    assert rec.lot is None
    assert rec.split == split_for(m, salt="wafer-embeddings/wm811k/v1")


def test_clean_lot_normalizes_raw_cells():
    assert wm.clean_lot("lot47542 ") == "lot47542"
    assert wm.clean_lot(np.array([["lot1"]])) == "lot1"
    for raw in (None, float("nan"), "", np.array([])):
        assert wm.clean_lot(raw) is None
