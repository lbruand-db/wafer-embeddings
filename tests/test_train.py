import numpy as np
import torch

from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE, PASS
from wafer_embeddings.model.dino import DinoModel, DINOHead
from wafer_embeddings.model.encoder import PerDieViT
from wafer_embeddings.tokenize import tokenizer as tk
from wafer_embeddings.train import embed_all, fit_dino, g1_metrics, wafers_from_rows
from wafer_embeddings.model.dino import DINOLoss


def _disk(seed, h=20, w=20):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 <= (h / 2) ** 2
    m = np.full((h, w), NO_DIE, dtype=np.uint8)
    m[inside] = PASS
    m[inside & (rng.random((h, w)) < 0.2)] = FAIL
    return m


def test_wafers_from_rows_reconstructs_and_aligns():
    m = _disk(0, 12, 15)
    rows = [
        {
            "height": 12,
            "width": 15,
            "wafer_map": m.reshape(-1).tolist(),
            "label_id": 3,
            "split": "train",
        },
        {
            "height": 12,
            "width": 15,
            "wafer_map": m.reshape(-1).tolist(),
            "label_id": -1,
            "split": "test",
        },
    ]
    wafers, labels, splits = wafers_from_rows(rows)
    assert len(wafers) == 2
    # tokenization matches a direct tokenize of the reconstructed map
    assert wafers[0].n_tokens == tk.tokenize(m).n_tokens
    assert labels.tolist() == [3, -1]
    assert splits.tolist() == ["train", "test"]


def _tiny_dino(embed_dim=16):
    torch.manual_seed(0)
    enc = PerDieViT(embed_dim=embed_dim, width=32, depth=2, n_heads=4)
    return DinoModel(enc, DINOHead(embed_dim, out_dim=64, hidden=32, bottleneck=8))


def test_fit_dino_runs_and_embed_all_shapes():
    wafers = [tk.tokenize(_disk(i)) for i in range(6)]
    dino = _tiny_dino(embed_dim=16)
    opt = torch.optim.SGD(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=0.3
    )
    losses = fit_dino(
        dino,
        DINOLoss(out_dim=64),
        opt,
        wafers,
        steps=3,
        batch_size=4,
        rng=np.random.default_rng(0),
        log_every=1,
    )
    assert len(losses) == 3 and all(np.isfinite(losses))
    emb = embed_all(dino, wafers, batch_size=4)
    assert emb.shape == (6, 16)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_g1_metrics_on_separable_embeddings():
    # 3 classes, embeddings = class basis vectors; train bank + test queries.
    embs, labels, splits = [], [], []
    for c in range(3):
        for split in ("train", "train", "test", "test"):
            embs.append(np.eye(3)[c])
            labels.append(c)
            splits.append(split)
    m = g1_metrics(
        np.array(embs, dtype=np.float64),
        np.array(labels),
        np.array(splits, dtype=object),
        n_classes=3,
    )
    assert m["knn_acc"] == 1.0
    assert m["map@10"] == 1.0
    assert m["cluster_nmi"] > 0.9
