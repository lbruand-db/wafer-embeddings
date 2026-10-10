"""Descriptor-guided positives (GAPS 1.1): neighbour search and its use in DINO steps."""

import numpy as np
import pytest
import torch

from wafer_embeddings.data.mixedwm38 import FAIL, PASS
from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
from wafer_embeddings.model.encoder import PerDieViT
from wafer_embeddings.tokenize import tokenizer as tk
from wafer_embeddings.train import fit_dino, train_step
from wafer_embeddings.train.positives import (
    descriptor_neighbours,
    label_neighbours,
    sample_neighbours,
)


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_neighbours_are_nearest_by_cosine_and_never_self():
    f = _unit([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99], [0.7, 0.7]])
    nn = descriptor_neighbours(f, k=1, chunk=2)  # chunking must not change the answer
    assert nn[:, 0].tolist() == [1, 0, 3, 2, nn[4, 0]]
    assert nn[4, 0] in (1, 3)  # the diagonal one is closest to the near-diagonal points
    nn2 = descriptor_neighbours(f, k=2)
    assert all(i not in row for i, row in enumerate(nn2))


def test_neighbours_match_brute_force():
    f = _unit(np.random.default_rng(0).normal(size=(50, 8)))
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    want = np.argsort(-sim, axis=1)[:, :3]
    got = descriptor_neighbours(f, k=3, chunk=7)
    assert all(set(a) == set(b) for a, b in zip(got, want))


def test_neighbours_validate_k():
    with pytest.raises(ValueError):
        descriptor_neighbours(_unit(np.eye(3)), k=3)


def test_sample_neighbours_picks_from_each_row():
    nn = np.array([[1, 2], [0, 2], [0, 1]])
    out = sample_neighbours(nn, np.array([0, 0, 2]), np.random.default_rng(0))
    assert out[0] in (1, 2) and out[1] in (1, 2) and out[2] in (0, 1)


def _wafers(n=6, size=12):
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        m = np.full((size, size), PASS, dtype=np.uint8)
        m[rng.random((size, size)) < 0.2] = FAIL
        out.append(tk.tokenize(m))
    return out


def _tiny():
    torch.manual_seed(0)
    enc = PerDieViT(embed_dim=16, width=32, depth=2, n_heads=4)
    head = DINOHead(16, out_dim=64, hidden=32, bottleneck=8)
    dino = DinoModel(enc, head)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    return dino, DINOLoss(out_dim=64), opt


def test_train_step_adds_one_student_view_per_positive_batch(monkeypatch):
    dino, loss_fn, opt = _tiny()
    seen = []
    real = dino.student_views
    monkeypatch.setattr(dino, "student_views", lambda v: seen.append(len(v)) or real(v))
    w = _wafers()
    aug = {"n_local": 2, "local_crop_area": (0.05, 0.4)}
    loss = train_step(dino, loss_fn, opt, w[:3], np.random.default_rng(0), aug=aug)
    loss_p = train_step(
        dino, loss_fn, opt, w[:3], np.random.default_rng(0), aug=aug, positives=w[3:]
    )
    assert seen == [4, 5]  # 2 global + 2 local, then + 1 neighbour view
    assert np.isfinite(loss) and np.isfinite(loss_p)


def test_fit_dino_with_neighbours_trains():
    dino, loss_fn, opt = _tiny()
    w = _wafers()
    nn = descriptor_neighbours(_unit(np.random.default_rng(1).normal(size=(6, 4))), k=2)
    losses = fit_dino(
        dino, loss_fn, opt, w, steps=3, batch_size=3, rng=np.random.default_rng(0), neighbours=nn
    )
    assert len(losses) == 3 and all(np.isfinite(losses))


def test_label_neighbours_are_same_class_or_self():
    labels = np.array([0, 0, 1, -1, 1, 1, 2])
    nn = label_neighbours(labels, 4, np.random.default_rng(0))
    assert nn.shape == (7, 4)
    for i, row in enumerate(nn):
        if labels[i] < 0 or (labels == labels[i]).sum() < 2:
            assert (row == i).all()  # unlabeled / singleton class: plain DINO
        else:
            assert all(labels[j] == labels[i] and j != i for j in row)
