import numpy as np

from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE, PASS
from wafer_embeddings.tokenize import tokenizer as tk
from wafer_embeddings.tokenize import augment as aug


def _disk(h=52, w=52, seed=0):
    """A circular wafer of pass/fail dies inside a square grid (rest no-die)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= (min(h, w) / 2) ** 2
    m = np.full((h, w), NO_DIE, dtype=np.uint8)
    m[inside] = PASS
    fail = inside & (rng.random((h, w)) < 0.2)
    m[fail] = FAIL
    return m, int(inside.sum()), int(fail.sum())


def test_tokenize_drops_no_die_and_counts_match():
    m, n_on, n_fail = _disk()
    t = tk.tokenize(m)
    assert t.n_tokens == n_on  # one token per on-wafer die, no-die dropped
    assert int((t.state_ids == tk.STATE_FAIL).sum()) == n_fail


def test_tokenize_coords_centered_and_unit_radius():
    m, _, _ = _disk()
    t = tk.tokenize(m)
    assert abs(t.coords[:, 0].mean()) < 1e-5  # centroid at origin
    assert abs(t.coords[:, 1].mean()) < 1e-5
    assert abs(t.coords[:, 2].max() - 1.0) < 1e-5  # radius normalized to 1
    np.testing.assert_allclose(t.coords[:, 2], np.hypot(t.coords[:, 0], t.coords[:, 1]), atol=1e-5)


def test_tokenize_empty_wafer():
    m = np.full((10, 10), NO_DIE, dtype=np.uint8)
    t = tk.tokenize(m)
    assert t.n_tokens == 0 and t.coords.shape == (0, 3)


def test_collate_shapes_and_mask():
    a = tk.tokenize(_disk(10, 10, 1)[0])
    b = tk.tokenize(_disk(52, 52, 2)[0])
    coords, state_ids, mask = tk.collate([a, b])
    lmax = max(a.n_tokens, b.n_tokens)
    assert coords.shape == (2, lmax, 3)
    assert state_ids.shape == (2, lmax) and mask.shape == (2, lmax)
    assert int(mask[0].sum()) == a.n_tokens
    assert int(mask[1].sum()) == b.n_tokens
    assert mask.sum().item() == a.n_tokens + b.n_tokens


def test_rotation_is_isometry_preserving_radius_and_count():
    t = tk.tokenize(_disk(52, 52, 3)[0])
    rot = aug.rotate_coords(t.coords, np.pi / 3)
    assert rot.shape == t.coords.shape
    np.testing.assert_allclose(rot[:, 2], t.coords[:, 2], atol=1e-5)  # r preserved
    full = aug.rotate_coords(t.coords, 2 * np.pi)
    np.testing.assert_allclose(full, t.coords, atol=1e-5)  # 2*pi ~ identity


def test_flip_preserves_radius_and_count():
    t = tk.tokenize(_disk(30, 30, 4)[0])
    f = aug.flip_coords(t.coords, "u")
    assert f.shape == t.coords.shape
    np.testing.assert_allclose(f[:, 2], t.coords[:, 2], atol=1e-6)
    np.testing.assert_allclose(f[:, 0], -t.coords[:, 0], atol=1e-6)


def test_die_noise_bounds():
    t = tk.tokenize(_disk(30, 30, 5)[0])
    rng = np.random.default_rng(0)
    assert np.array_equal(aug.toggle_die_noise(t.state_ids, 0.0, rng), t.state_ids)
    allflip = aug.toggle_die_noise(t.state_ids, 1.0, rng)
    assert np.array_equal(allflip, 1 - t.state_ids)
    assert allflip.shape == t.state_ids.shape


def test_crop_is_subset_with_unchanged_coords():
    t = tk.tokenize(_disk(52, 52, 6)[0])
    rng = np.random.default_rng(1)
    c, s = aug.crop_window(t.coords, t.state_ids, 0.5, rng)
    assert 0 < c.shape[0] <= t.n_tokens
    assert c.shape[0] == s.shape[0]
    orig = {tuple(np.round(row, 6)) for row in t.coords}
    assert all(tuple(np.round(row, 6)) in orig for row in c)  # coords unchanged


def test_cutout_never_empty_and_subset():
    t = tk.tokenize(_disk(40, 40, 7)[0])
    rng = np.random.default_rng(2)
    c, s = aug.cutout(t.coords, t.state_ids, 2, 0.3, rng)
    assert 0 < c.shape[0] <= t.n_tokens and c.shape[0] == s.shape[0]


def test_cap_tokens_subsamples_and_preserves():
    t = tk.tokenize(_disk(52, 52, 9)[0])
    rng = np.random.default_rng(0)
    cap = t.n_tokens // 3
    capped = tk.cap_tokens(t, cap, rng)
    assert capped.n_tokens == cap
    assert capped.coords.shape == (cap, 3) and capped.state_ids.shape == (cap,)
    # kept tokens are a subset of the originals (coords unchanged)
    orig = {tuple(np.round(r, 6)) for r in t.coords}
    assert all(tuple(np.round(r, 6)) in orig for r in capped.coords)
    # no-op when already under the cap
    assert tk.cap_tokens(t, t.n_tokens + 10, rng) is t


def test_random_view_both_modes_valid():
    t = tk.tokenize(_disk(52, 52, 8)[0])
    rng = np.random.default_rng(3)
    for oi in (True, False):
        v = aug.random_view(t, rng, orientation_invariant=oi)
        assert v.coords.shape[0] == v.state_ids.shape[0]
        assert v.n_tokens <= t.n_tokens  # crop/cutout only remove
        assert np.isfinite(v.coords).all()
