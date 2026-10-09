"""Reference-faithful DINO recipe pieces (ref [1]): crops, multi-crop, WD, LR, teacher."""

import numpy as np
import torch
import torch.nn.functional as F

from wafer_embeddings.data.mixedwm38 import NO_DIE, PASS
from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
from wafer_embeddings.model.encoder import PerDieViT
from wafer_embeddings.tokenize import augment as aug
from wafer_embeddings.tokenize import tokenizer as tk
from wafer_embeddings.train import (
    build_views,
    embed_all,
    fit_dino,
    param_groups,
    scaled_lr,
    train_step,
)


def _square(n=40):
    m = np.full((n, n), PASS, dtype=np.uint8)  # full grid: area fraction == token fraction
    m[0, 0] = NO_DIE
    return tk.tokenize(m)


def _tiny_dino():
    torch.manual_seed(0)
    enc = PerDieViT(embed_dim=16, width=32, depth=2, n_heads=4)
    return DinoModel(enc, DINOHead(16, out_dim=64, hidden=32, bottleneck=8))


def test_crop_area_keeps_the_requested_area_fraction():
    w, rng = _square(), np.random.default_rng(0)

    def n(area):
        v = aug.random_view(
            w, rng, orientation_invariant=False, cutout_regions=0, die_noise_p=0.0, crop_area=area
        )
        return v.n_tokens

    glob = [n((0.4, 1.0)) for _ in range(200)]
    loc = [n((0.05, 0.4)) for _ in range(200)]
    g, lc = np.array(glob) / w.n_tokens, np.array(loc) / w.n_tokens
    assert g.min() > 0.35 and g.max() <= 1.0  # global views: 40-100% of the dies
    assert lc.min() > 0.03 and lc.max() < 0.45  # local views: 5-40%
    assert np.std(g) > 0.05  # the size is drawn per view, not fixed


def test_build_views_global_then_local():
    wafers = [_square(20) for _ in range(3)]
    views = build_views(
        wafers, np.random.default_rng(0), n_views=2, n_local=4, crop_area=(0.4, 1.0)
    )
    assert len(views) == 6
    n_valid = [int(m.sum()) for (_, _, m) in views]
    assert min(n_valid[:2]) > max(n_valid[2:])  # local crops are smaller than global ones


def test_dino_loss_multicrop_matches_manual_cross_entropy():
    torch.manual_seed(0)
    k = 64
    teacher = [torch.randn(4, k) for _ in range(2)]
    student = [torch.randn(4, k) for _ in range(5)]  # 2 global + 3 local
    loss_fn = DINOLoss(out_dim=k)
    got = loss_fn(student, teacher)
    total, n = torch.zeros(()), 0
    for ti, t in enumerate(teacher):
        p = F.softmax(t / loss_fn.teacher_temp, dim=-1)  # center starts at zero
        for si, s in enumerate(student):
            if si == ti:
                continue
            total = total + (-(p * F.log_softmax(s / loss_fn.student_temp, -1)).sum(-1).mean())
            n += 1
    assert n == 2 * 5 - 2
    assert torch.allclose(got, total / n, atol=1e-6)


def test_train_step_teacher_sees_only_global_views():
    dino, seen = _tiny_dino(), {}
    orig_t, orig_s = dino.teacher_views, dino.student_views
    dino.teacher_views = lambda v: (seen.__setitem__("t", len(v)), orig_t(v))[1]
    dino.student_views = lambda v: (seen.__setitem__("s", len(v)), orig_s(v))[1]
    opt = torch.optim.SGD(param_groups(dino.student_enc, dino.student_head), lr=0.1)
    wafers = [_square(12) for _ in range(4)]
    loss = train_step(dino, DINOLoss(64), opt, wafers, np.random.default_rng(0), aug={"n_local": 3})
    assert np.isfinite(loss) and seen == {"t": 2, "s": 5}


def test_param_groups_decay_weights_only():
    dino = _tiny_dino()
    reg, noreg = param_groups(dino.student_enc, dino.student_head)
    trainable = [p for m in (dino.student_enc, dino.student_head) for p in m.parameters()]
    assert len(reg["params"]) + len(noreg["params"]) == len(trainable)
    assert all(p.ndim >= 2 for p in reg["params"]) and reg["wd_schedule"]
    assert all(p.ndim == 1 for p in noreg["params"]) and noreg["weight_decay"] == 0.0


def test_fit_dino_follows_weight_decay_schedule():
    dino = _tiny_dino()
    opt = torch.optim.AdamW(param_groups(dino.student_enc, dino.student_head), lr=1e-3)
    fit_dino(
        dino,
        DINOLoss(64),
        opt,
        [_square(10) for _ in range(4)],
        steps=3,
        batch_size=2,
        rng=np.random.default_rng(0),
        weight_decay=(0.04, 0.4),
    )
    assert abs(opt.param_groups[0]["weight_decay"] - 0.4) < 1e-9  # cosine ends at `end`
    assert opt.param_groups[1]["weight_decay"] == 0.0  # biases / norms never decayed


def test_scaled_lr_rule():
    assert scaled_lr(256) == 5e-4
    assert abs(scaled_lr(16) - 3.125e-5) < 1e-12


def test_embed_all_teacher_vs_student():
    import pytest

    dino = _tiny_dino()
    wafers = [_square(10) for _ in range(3)]
    s0, t0 = embed_all(dino, wafers), embed_all(dino, wafers, which="teacher")
    np.testing.assert_allclose(s0, t0, atol=1e-6)  # teacher starts as a copy
    opt = torch.optim.SGD(param_groups(dino.student_enc, dino.student_head), lr=0.5)
    train_step(dino, DINOLoss(64), opt, wafers, np.random.default_rng(0))
    s1, t1 = embed_all(dino, wafers), embed_all(dino, wafers, which="teacher")
    assert not np.allclose(s1, t1)  # EMA teacher lags the student
    with pytest.raises(ValueError):
        embed_all(dino, wafers, which="ema")


def test_fit_dino_track_fn_is_log_only():
    wafers = [_square(10) for _ in range(4)]
    seen: list[int] = []

    def run(track: bool):
        dino = _tiny_dino()
        opt = torch.optim.SGD(param_groups(dino.student_enc, dino.student_head), lr=0.1)
        fit_dino(
            dino,
            DINOLoss(64),
            opt,
            wafers,
            steps=5,
            batch_size=2,
            rng=np.random.default_rng(0),
            track_fn=seen.append if track else None,
            track_every=2 if track else 0,
        )
        return dino

    plain, tracked = run(False), run(True)
    assert seen == [0, 2, 4, 5]  # start, every 2 steps, and the end
    for pa, pb in zip(plain.parameters(), tracked.parameters()):
        assert torch.equal(pa, pb)  # tracking never changes training
