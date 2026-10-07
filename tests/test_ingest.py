import numpy as np

from wafer_embeddings.data import wm811k as wm
from wafer_embeddings.jobs.ingest import wm811k_rows


def test_wm811k_rows_roundtrips_variable_sizes_and_labels():
    rng = np.random.default_rng(0)
    maps = [
        rng.integers(0, 3, size=(10, 12), dtype=np.uint8),
        rng.integers(0, 3, size=(26, 26), dtype=np.uint8),
    ]
    parsed = wm.parse_records(maps, ["Scratch", ""])
    rows = wm811k_rows(parsed)

    assert len(rows) == len(parsed.maps)
    r0 = rows[0]
    assert set(r0) == {"id", "height", "width", "wafer_map", "label", "label_id", "split"}
    # Flattened map reconstructs the original (variable) shape.
    recon = np.array(r0["wafer_map"], dtype=np.uint8).reshape(r0["height"], r0["width"])
    np.testing.assert_array_equal(recon, parsed.maps[0])
    # Labeled vs unlabeled carried through.
    assert rows[0]["label"] == "Scratch" and rows[0]["label_id"] == wm.label_to_id("Scratch")
    assert rows[1]["label"] is None and rows[1]["label_id"] == wm.UNLABELED
    assert all(r["split"] in {"train", "val", "test"} for r in rows)


def test_wm811k_rows_invokes_progress():
    maps = [np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]
    parsed = wm.parse_records(maps, ["Loc", "Center"])
    calls = []
    wm811k_rows(parsed, progress=lambda done, total: calls.append((done, total)))
    assert calls[-1] == (len(parsed.maps), len(parsed.maps))
