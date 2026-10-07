import numpy as np
import pytest

from wafer_embeddings.data import mixedwm38 as mw


def _rng_maps(n, h=52, w=52, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=(n, h, w), dtype=np.uint8)


def test_pattern_key_and_name():
    assert mw.pattern_key(np.zeros(8)) == ()
    assert mw.pattern_name(()) == "Normal"
    row = np.zeros(8, dtype=np.uint8)
    row[0] = 1  # Center
    row[6] = 1  # Scratch
    assert mw.pattern_key(row) == (0, 6)
    assert mw.pattern_name((0, 6)) == "Center+Scratch"


def test_content_hash_stable_and_discriminative():
    a = _rng_maps(1, seed=1)[0]
    b = a.copy()
    c = _rng_maps(1, seed=2)[0]
    assert mw.content_hash(a) == mw.content_hash(b)
    assert mw.content_hash(a) != mw.content_hash(c)


def test_dedup_removes_exact_duplicates():
    base = _rng_maps(3, seed=3)
    maps = np.concatenate([base, base[:1]], axis=0)  # duplicate row 0
    keep = mw.dedup_indices(maps)
    assert keep.tolist() == [0, 1, 2]  # the 4th (dup of 0) dropped


def test_split_is_deterministic_and_leakage_safe():
    maps = _rng_maps(200, seed=4)
    s1 = mw.split_assignments(maps)
    s2 = mw.split_assignments(maps)
    assert np.array_equal(s1, s2)  # deterministic
    # Identical content -> same split, even at different positions.
    dup = np.concatenate([maps, maps[:5]], axis=0)
    sd = mw.split_assignments(dup)
    assert np.array_equal(sd[:5], sd[200:205])
    # Fractions roughly honored.
    frac_train = (s1 == "train").mean()
    assert 0.7 < frac_train < 0.9


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        mw.split_assignments(_rng_maps(2), fractions=(0.5, 0.3, 0.3))


def test_parse_arrays_happy_path():
    maps = _rng_maps(50, seed=5)
    labels = np.zeros((50, 8), dtype=np.uint8)
    labels[::2, 0] = 1  # half have Center
    parsed = mw.parse_arrays(maps, labels)
    assert parsed.maps.shape[0] == len(parsed.pattern_names) == len(parsed.split)
    assert set(np.unique(parsed.split)).issubset({"train", "val", "test"})


@pytest.mark.parametrize(
    "maps,labels",
    [
        (np.zeros((4, 52)), np.zeros((4, 8))),  # maps not 3-D
        (np.zeros((4, 8, 8)), np.zeros((4, 7))),  # labels wrong width
        (np.full((2, 8, 8), 5), np.zeros((2, 8))),  # illegal cell value
    ],
)
def test_parse_arrays_rejects_bad_shapes(maps, labels):
    with pytest.raises(ValueError):
        mw.parse_arrays(maps.astype(np.uint8), labels.astype(np.uint8))
