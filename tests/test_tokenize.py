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


def test_cap_tokens_keeps_all_fail_dies():
    # Sparse defects must survive the cap: a uniform random cap would decimate them and
    # destroy the defect pattern (the class signal). cap_tokens keeps every FAIL die.
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:60, 0:60]
    inside = (yy - 29.5) ** 2 + (xx - 29.5) ** 2 <= 30**2
    m = np.full((60, 60), NO_DIE, dtype=np.uint8)
    m[inside] = PASS
    m[inside & (rng.random((60, 60)) < 0.02)] = FAIL  # ~2% sparse defects
    t = tk.tokenize(m)
    n_fail = int((t.state_ids == tk.STATE_FAIL).sum())
    cap = n_fail + 200  # budget comfortably above the fail count
    assert cap < t.n_tokens  # ensure the cap actually triggers
    capped = tk.cap_tokens(t, cap, rng)
    assert capped.n_tokens == cap
    assert int((capped.state_ids == tk.STATE_FAIL).sum()) == n_fail  # every FAIL kept

    # pathological: budget below the fail count -> still respects the cap
    tiny = tk.cap_tokens(t, max(n_fail // 2, 1), rng)
    assert tiny.n_tokens == max(n_fail // 2, 1)


def test_random_view_both_modes_valid():
    t = tk.tokenize(_disk(52, 52, 8)[0])
    rng = np.random.default_rng(3)
    for oi in (True, False):
        v = aug.random_view(t, rng, orientation_invariant=oi)
        assert v.coords.shape[0] == v.state_ids.shape[0]
        assert v.n_tokens <= t.n_tokens  # crop/cutout only remove
        assert np.isfinite(v.coords).all()


def test_fail_dropout_only_turns_fails_to_pass_at_rate():
    rng = np.random.default_rng(0)
    s = np.array([tk.STATE_FAIL] * 5000 + [tk.STATE_PASS] * 5000)
    out = aug.fail_dropout(s, 0.4, rng)
    assert np.all(out[5000:] == tk.STATE_PASS)  # PASS dies never become FAIL
    kept = float((out[:5000] == tk.STATE_FAIL).mean())
    assert abs(kept - 0.6) < 0.03
    assert np.array_equal(aug.fail_dropout(s, 0.0, rng), s)


def test_token_dropout_subset_rate_and_never_empty():
    rng = np.random.default_rng(1)
    coords = rng.random((4000, 3)).astype(np.float32)
    states = rng.integers(0, 2, 4000)
    c, s = aug.token_dropout(coords, states, 0.5, rng)
    assert abs(len(c) / 4000 - 0.5) < 0.03
    rows = {tuple(r) for r in coords.tolist()}
    assert all(tuple(r) in rows for r in c.tolist())  # kept dies keep their positions
    c1, _ = aug.token_dropout(coords[:1], states[:1], 0.999999, rng)
    assert len(c1) == 1


def test_random_view_with_strong_aug_differs_more_between_views():
    m, _, _ = _disk(31, 31, seed=2)  # ~20% FAIL dies
    w = tk.tokenize(m)
    rng = np.random.default_rng(3)
    base = aug.random_view(w, rng, orientation_invariant=False, crop_scale=1.0, cutout_regions=0)
    strong = aug.random_view(
        w,
        rng,
        orientation_invariant=False,
        crop_scale=1.0,
        cutout_regions=0,
        token_drop_p=0.5,
        fail_drop_p=0.5,
    )
    n_fail = int((w.state_ids == tk.STATE_FAIL).sum())
    assert base.n_tokens == w.n_tokens
    assert strong.n_tokens < 0.6 * w.n_tokens
    assert int((strong.state_ids == tk.STATE_FAIL).sum()) < 0.4 * n_fail
