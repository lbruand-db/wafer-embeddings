import numpy as np

from wafer_embeddings.eval.baselines import (
    pixel_grid_features,
    pixel_pca_embedding,
    polar_fail_embedding,
    polar_fail_features,
)
from wafer_embeddings.tokenize.augment import rotate_coords
from wafer_embeddings.tokenize.tokenizer import TokenizedWafer, tokenize
from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE, PASS
from wafer_embeddings.train import g1_metrics


def _disk(n=31):
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    inside = (yy - c) ** 2 + (xx - c) ** 2 <= (n / 2) ** 2
    m = np.full((n, n), NO_DIE, dtype=np.uint8)
    m[inside] = PASS
    return m, yy, xx, c, inside


def _center(n=31):
    m, yy, xx, c, inside = _disk(n)
    m[inside & ((yy - c) ** 2 + (xx - c) ** 2 <= (n / 8) ** 2)] = FAIL
    return tokenize(m)


def _edge_loc(n=31, side=1):
    m, yy, xx, c, inside = _disk(n)
    m[inside & ((yy - c) ** 2 + (xx - c) ** 2 >= (0.4 * n) ** 2) & (side * (xx - c) > 0)] = FAIL
    return tokenize(m)


def test_features_are_rotation_invariant():
    w = _edge_loc()
    rot = TokenizedWafer(rotate_coords(w.coords, np.pi / 2), w.state_ids)  # 3 sectors
    np.testing.assert_allclose(polar_fail_features(w), polar_fail_features(rot), atol=1e-9)


def test_features_shape_and_empty_wafer():
    f = polar_fail_features(_center(), n_r=4, n_a=8)
    assert f.shape == (4 * 5 + 1,)
    empty = TokenizedWafer(np.zeros((0, 3), np.float32), np.zeros((0,), np.int64))
    assert np.all(polar_fail_features(empty, n_r=4, n_a=8) == 0)


def test_features_ignore_map_size():
    # same pattern on a small and a large grid -> nearly the same descriptor
    a, b = polar_fail_features(_center(31)), polar_fail_features(_center(61))
    assert np.linalg.norm(a - b) < 0.1 * np.linalg.norm(a)


def test_embedding_separates_center_from_edge_patterns():
    wafers = [_center(31), _center(41), _edge_loc(31, 1), _edge_loc(41, -1)] * 2
    labels = np.array([0, 0, 1, 1] * 2)
    splits = np.array(["train"] * 4 + ["test"] * 4, dtype=object)
    emb = polar_fail_embedding(wafers)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-9)
    assert g1_metrics(emb, labels, splits, n_classes=2, knn_k=2)["knn_acc"] == 1.0


def test_pixel_grid_locates_fails_and_ignores_map_size():
    f = pixel_grid_features(_center(31), size=8).reshape(8, 8)
    assert f[3:5, 3:5].min() > 0.5 and f[0, :].max() == 0  # centre fails, edge passes
    a, b = pixel_grid_features(_center(31), 8), pixel_grid_features(_center(61), 8)
    assert np.linalg.norm(a - b) < 0.25 * np.linalg.norm(a)  # cells coarser than dies
    empty = TokenizedWafer(np.zeros((0, 3), np.float32), np.zeros((0,), np.int64))
    assert pixel_grid_features(empty, size=4).tolist() == [0.0] * 16


def test_pixel_pca_embedding_separates_patterns():
    wafers = [_center(31), _center(41), _edge_loc(31, 1), _edge_loc(41, 1)] * 2
    labels = np.array([0, 0, 1, 1] * 2)
    splits = np.array(["train"] * 4 + ["test"] * 4, dtype=object)
    emb = pixel_pca_embedding(wafers, dim=4)
    assert emb.shape == (8, 4)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)  # float32
    assert g1_metrics(emb, labels, splits, n_classes=2, knn_k=2)["knn_acc"] == 1.0


def test_wafer_raster_and_image():
    from wafer_embeddings.eval.frozen import IMAGENET_MEAN, PALETTE, wafer_image, wafer_raster

    r = wafer_raster(_center(31), size=8)
    assert r[0, 0] == 0  # outside the disk: no die
    assert r[4, 4] == 2 and r[1, 4] == 1  # centre fails, mid-ring passes
    img = wafer_image(_center(31), size=16, cells=8)
    assert img.shape == (3, 16, 16) and img.dtype == np.float32
    white = (PALETTE[0] - IMAGENET_MEAN) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(img[:, 0, 0], white, rtol=1e-5)  # nearest upsampling of cell 0
    empty = TokenizedWafer(np.zeros((0, 3), np.float32), np.zeros((0,), np.int64))
    assert wafer_raster(empty, size=4).sum() == 0
