import numpy as np
import pytest

from wafer_embeddings.data import wm811k as wm


def test_clean_label_handles_messy_forms():
    assert wm.clean_label(np.array([["Center"]], dtype=object)) == "Center"
    assert wm.clean_label("Edge-Loc") == "Edge-Loc"
    assert wm.clean_label("edgeloc") == "Edge-Loc"
    assert wm.clean_label("near full") == "Near-full"
    assert wm.clean_label("none") == "none"
    assert wm.clean_label(np.array([], dtype=object)) is None
    assert wm.clean_label("[]") is None
    assert wm.clean_label("") is None
    assert wm.clean_label("Bogus") is None


def test_label_to_id():
    assert wm.label_to_id("Center") == 0
    assert wm.label_to_id("none") == len(wm.CLASS_NAMES) - 1
    assert wm.label_to_id(None) == wm.UNLABELED


def _vmap(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(h, w), dtype=np.uint8)


def test_parse_records_variable_sizes_labels_and_splits():
    maps = [_vmap(10, 12, 1), _vmap(26, 26, 2), _vmap(8, 15, 3)]
    raw = [np.array([["Scratch"]], dtype=object), "", "none"]
    parsed = wm.parse_records(maps, raw)
    assert len(parsed.maps) == 3
    assert [m.shape for m in parsed.maps] == [(10, 12), (26, 26), (8, 15)]  # variable sizes
    assert parsed.label_ids.tolist() == [
        wm.label_to_id("Scratch"),
        wm.UNLABELED,  # empty -> unlabeled
        wm.label_to_id("none"),
    ]
    assert set(np.unique(parsed.split)).issubset({"train", "val", "test"})


def test_parse_records_dedup():
    a = _vmap(10, 10, 4)
    maps = [a, a.copy(), _vmap(10, 10, 5)]
    parsed = wm.parse_records(maps, ["Center", "Center", "Loc"])
    assert len(parsed.maps) == 2  # exact duplicate dropped


def test_parse_records_invokes_progress():
    maps = [_vmap(6, 6, i) for i in range(5)]
    calls = []
    wm.parse_records(maps, [""] * 5, progress=lambda done, total: calls.append((done, total)))
    assert calls[-1] == (5, 5)  # reports completion over all inputs
    assert all(t == 5 for _, t in calls)


def test_parse_records_skips_or_raises_on_invalid():
    good = _vmap(6, 6, 6)
    bad_val = np.full((5, 5), 7, dtype=np.uint8)  # illegal cell value
    one_d = np.array([0, 1, 2], dtype=np.uint8)  # not 2-D
    parsed = wm.parse_records([good, bad_val, one_d], ["Loc", "Center", "Donut"])
    assert len(parsed.maps) == 1  # only the good map survives
    with pytest.raises(ValueError):
        wm.parse_records([bad_val], ["Center"], skip_invalid=False)
