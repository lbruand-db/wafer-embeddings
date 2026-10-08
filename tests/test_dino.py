from typing import cast

import numpy as np
import torch
import torch.nn.functional as F

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


def test_head_logits_are_cosine_and_non_degenerate():
    # With unit-normalized prototypes, logits are cosine in [-1,1] and span a real
    # range — the fix for the ln(K) uniform collapse (plain Linear left logits ~0).
    torch.manual_seed(0)
    head = DINOHead(16, out_dim=128, hidden=32, bottleneck=8)
    out = head(torch.randn(4, 16))
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-4  # bounded cosine
    assert (out.max(dim=1).values - out.min(dim=1).values).min() > 0.1  # not uniform


def test_default_head_logits_are_sharpenable_not_uniform():
    # Regression guard for the uniform collapse: the DEFAULT head must emit cosine logits
    # with enough per-sample spread (~1/sqrt(bottleneck)) that the teacher temperature can
    # sharpen them into a non-uniform target. bottleneck=256 gave spread ~0.06 (too flat →
    # teacher stays uniform → loss pinned at ln K); the default dropped to 64 (~0.125).
    torch.manual_seed(0)
    out_dim = 1024
    head = DINOHead(48, out_dim=out_dim)  # defaults (bottleneck=64)
    x = F.normalize(torch.randn(64, 48), dim=-1)
    logits = head(x)
    assert logits.abs().max() <= 1.0 + 1e-4  # cosine
    per_sample_std = logits.std(dim=-1).mean().item()
    assert per_sample_std > 0.09, per_sample_std  # 64 -> ~0.125; 256 would be ~0.06
    # teacher sharpening (temp 0.04, no centering yet) must beat uniform by a wide margin
    teacher = F.softmax(logits / 0.04, dim=-1)
    assert teacher.max(dim=-1).values.mean().item() > 20.0 / out_dim


def test_loss_nonnegative_and_center_moves():
    loss_fn = DINOLoss(out_dim=64)
    s = [torch.randn(4, 64), torch.randn(4, 64)]
    t = [torch.randn(4, 64), torch.randn(4, 64)]
    before = cast(torch.Tensor, loss_fn.center).clone()
    val = loss_fn(s, t)
    assert torch.isfinite(val) and val.item() >= 0.0 and val.ndim == 0
    assert not torch.allclose(before, cast(torch.Tensor, loss_fn.center))  # centering updated


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


def test_freeze_last_holds_prototypes_fixed():
    # DINO stabilizer: with freeze_last=True the prototype layer must not move, while the
    # encoder still updates; without it, prototypes move.
    rng = np.random.default_rng(0)
    wafers = [_disk(i) for i in range(4)]
    for freeze in (True, False):
        dino = _tiny_dino()
        loss_fn = DINOLoss(out_dim=128)
        opt = torch.optim.SGD(
            list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=0.5
        )
        proto_before = dino.student_head.prototypes.weight.clone()
        enc_before = next(iter(dino.student_enc.parameters())).clone()
        train_step(dino, loss_fn, opt, wafers, rng, freeze_last=freeze)
        proto_moved = not torch.allclose(proto_before, dino.student_head.prototypes.weight)
        enc_moved = not torch.allclose(enc_before, next(iter(dino.student_enc.parameters())))
        assert proto_moved == (not freeze)  # frozen -> unchanged; unfrozen -> changed
        assert enc_moved  # encoder always trains


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
