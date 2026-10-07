import numpy as np
import torch

from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE, PASS
from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
from wafer_embeddings.model.encoder import PerDieViT
from wafer_embeddings.tokenize import tokenizer as tk
from wafer_embeddings.train.trainer import train_step


def _disk(seed, h=24, w=24):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 <= (h / 2) ** 2
    m = np.full((h, w), NO_DIE, dtype=np.uint8)
    m[inside] = PASS
    m[inside & (rng.random((h, w)) < 0.2)] = FAIL
    return tk.tokenize(m)


def _tiny_dino(out_dim=128):
    torch.manual_seed(0)
    enc = PerDieViT(embed_dim=16, width=32, depth=2, n_heads=4)
    head = DINOHead(16, out_dim=out_dim, hidden=32, bottleneck=8)
    return DinoModel(enc, head, teacher_momentum=0.9)


def test_head_shape():
    head = DINOHead(16, out_dim=256, hidden=32, bottleneck=8)
    out = head(torch.randn(5, 16))
    assert out.shape == (5, 256)


def test_loss_nonnegative_and_center_moves():
    loss_fn = DINOLoss(out_dim=64)
    s = [torch.randn(4, 64), torch.randn(4, 64)]
    t = [torch.randn(4, 64), torch.randn(4, 64)]
    before = loss_fn.center.clone()
    val = loss_fn(s, t)
    assert torch.isfinite(val) and val.item() >= 0.0 and val.ndim == 0
    assert not torch.allclose(before, loss_fn.center)  # centering updated


def test_teacher_is_frozen_and_ema_moves_after_step():
    dino = _tiny_dino()
    assert all(not p.requires_grad for p in dino.teacher_enc.parameters())
    loss_fn = DINOLoss(out_dim=128)
    opt = torch.optim.SGD(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=0.5
    )
    wafers = [_disk(i) for i in range(4)]
    rng = np.random.default_rng(0)
    t_before = next(iter(dino.teacher_enc.parameters())).clone()
    loss = train_step(dino, loss_fn, opt, wafers, rng)
    t_after = next(iter(dino.teacher_enc.parameters()))
    assert np.isfinite(loss)
    assert not torch.allclose(t_before, t_after)  # EMA pulled teacher toward student


def test_embed_is_normalized_384_default():
    enc = PerDieViT(depth=2).eval()
    head = DINOHead(384, out_dim=64, hidden=64, bottleneck=16)
    dino = DinoModel(enc, head)
    coords, states, mask = tk.collate([_disk(1), _disk(2)])
    with torch.no_grad():
        emb = dino.embed(coords, states, mask)
    assert emb.shape == (2, 384)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_multi_step_training_stays_finite():
    dino = _tiny_dino()
    loss_fn = DINOLoss(out_dim=128)
    opt = torch.optim.SGD(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=0.3
    )
    wafers = [_disk(i) for i in range(6)]
    rng = np.random.default_rng(1)
    losses = [train_step(dino, loss_fn, opt, wafers, rng) for _ in range(5)]
    assert all(np.isfinite(losses)) and all(x >= 0 for x in losses)
