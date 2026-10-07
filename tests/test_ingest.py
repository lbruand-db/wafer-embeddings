import numpy as np

from wafer_embeddings.data import mixedwm38 as mw
from wafer_embeddings.jobs.ingest import parsed_to_rows


def test_parsed_to_rows_roundtrips_shape_and_fields():
    rng = np.random.default_rng(0)
    maps = rng.integers(0, 3, size=(5, 8, 8), dtype=np.uint8)
    labels = np.zeros((5, 8), dtype=np.uint8)
    labels[0, 0] = 1  # Center
    parsed = mw.parse_arrays(maps, labels)
    rows = parsed_to_rows(parsed)

    assert len(rows) == parsed.maps.shape[0]
    r = rows[0]
    assert set(r) == {"id", "height", "width", "wafer_map", "labels", "pattern", "split"}
    assert r["height"] == 8 and r["width"] == 8
    assert len(r["wafer_map"]) == 8 * 8  # flattened map
    assert len(r["labels"]) == 8
    assert r["split"] in {"train", "val", "test"}
    # Flattened map reconstructs the original parsed map.
    recon = np.array(r["wafer_map"], dtype=np.uint8).reshape(8, 8)
    np.testing.assert_array_equal(recon, parsed.maps[0])
