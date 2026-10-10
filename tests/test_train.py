import numpy as np
import torch

from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE, PASS
from wafer_embeddings.model.dino import DinoModel, DINOHead
from wafer_embeddings.model.encoder import PerDieViT
from wafer_embeddings.tokenize import tokenizer as tk
from wafer_embeddings.train import (
    embed_all,
    fit_dino,
    g1_metrics,
    stratified_indices,
    wafers_from_rows,
)
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
    assert m["knn_macro_recall"] == 1.0
    assert m["majority_baseline"] == 1 / 3  # 3 balanced classes in the queries
    assert m["map@10"] == 1.0
    assert m["cluster_nmi"] > 0.9
    assert m["knn_recall_0"] == m["knn_recall_1"] == m["knn_recall_2"] == 1.0

    light = g1_metrics(
        np.array(embs, dtype=np.float64),
        np.array(labels),
        np.array(splits, dtype=object),
        n_classes=3,
        clustering=False,
    )
    assert not any(k.startswith("cluster_") for k in light)  # the KMeans fit is skipped
    assert {k: v for k, v in m.items() if not k.startswith("cluster_")} == light


def test_g1_metrics_per_class_recall_uses_names_and_exposes_failures():
    # class 2's queries sit on class 0's embedding -> its recall is 0, others perfect.
    embs = [np.eye(3)[c] for c in (0, 1, 2)] * 2 + [np.eye(3)[0], np.eye(3)[1], np.eye(3)[0]]
    labels = [0, 1, 2] * 2 + [0, 1, 2]
    splits = ["train"] * 6 + ["test"] * 3
    m = g1_metrics(
        np.array(embs, dtype=np.float64),
        np.array(labels),
        np.array(splits, dtype=object),
        n_classes=3,
        knn_k=1,
        class_names=("Center", "Donut", "Scratch"),
    )
    assert m["knn_recall_Center"] == 1.0 and m["knn_recall_Donut"] == 1.0
    assert m["knn_recall_Scratch"] == 0.0
    assert abs(m["knn_macro_recall"] - 2 / 3) < 1e-9


def _imbalanced_labels():
    # WM-811K-like skew: one dominant class, one mid, one rare.
    return np.array([0] * 1000 + [1] * 50 + [2] * 5)


def test_stratified_indices_balanced_water_fills():
    labels = _imbalanced_labels()
    pos = stratified_indices(labels, 60, np.random.default_rng(0), balanced=True)
    assert len(pos) == 60 and len(np.unique(pos)) == 60  # exact size, no duplicates
    counts = np.bincount(labels[pos], minlength=3)
    assert counts[2] == 5  # the rare class contributes everything it has
    assert abs(int(counts[0]) - int(counts[1])) <= 1  # leftover shared equally


def test_stratified_indices_proportional_and_deterministic():
    labels = _imbalanced_labels()
    a = stratified_indices(labels, 211, np.random.default_rng(7), balanced=False)
    b = stratified_indices(labels, 211, np.random.default_rng(7), balanced=False)
    assert np.array_equal(a, b)  # same seed -> same eval set across runs
    counts = np.bincount(labels[a], minlength=3)
    assert counts.sum() == 211
    expected = np.array([1000, 50, 5]) * 211 / 1055
    assert np.all(np.abs(counts - expected) <= 1)


def test_stratified_indices_returns_all_when_n_exceeds():
    labels = _imbalanced_labels()
    pos = stratified_indices(labels, 10_000, np.random.default_rng(0))
    assert np.array_equal(pos, np.arange(len(labels)))


def _fit_tiny(steps=4, **kw):
    wafers = [tk.tokenize(_disk(i)) for i in range(6)]
    dino = _tiny_dino(embed_dim=16)
    opt = torch.optim.SGD(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=0.3
    )
    seen: list[tuple[int, dict]] = []
    losses = fit_dino(
        dino,
        DINOLoss(out_dim=64),
        opt,
        wafers,
        steps=steps,
        batch_size=4,
        rng=np.random.default_rng(0),
        monitor=lambda step, st: seen.append((step, dict(st))),
        **kw,
    )
    return dino, losses, seen


def test_fit_dino_monitor_reports_collapse_stats():
    _, _, seen = _fit_tiny(steps=4, log_every=2, clip_grad=1.0)
    steps = [s for s, st in seen if "loss" in st]
    assert steps == [2, 4]
    st = seen[0][1]
    for k in ("loss", "lr", "teacher_temp", "t_entropy", "t_maxp", "emb_std", "grad_norm"):
        assert k in st and np.isfinite(st[k])
    assert 0.0 <= st["t_entropy"] <= np.log(64) + 1e-6
    assert 1 / 64 - 1e-6 <= st["t_maxp"] <= 1.0
    assert st["emb_std"] >= 0.0


def test_collapse_stats_emb_std_is_zero_when_all_embeddings_identical():
    from wafer_embeddings.train import collapse_stats

    dino = _tiny_dino(embed_dim=16)
    w = tk.tokenize(_disk(0))
    view = tk.collate([w, w, w])  # three copies -> identical embeddings
    st = collapse_stats(dino, DINOLoss(out_dim=64), torch.zeros(3, 64), view)
    assert st["emb_std"] < 1e-5
    assert abs(st["t_entropy"] - np.log(64)) < 1e-4  # zero logits -> uniform teacher
    assert abs(st["t_maxp"] - 1 / 64) < 1e-6


def test_clip_grad_bounds_the_update():
    # with a tiny clip, one SGD step can move the params by at most lr * clip in L2
    wafers = [tk.tokenize(_disk(i)) for i in range(4)]
    dino = _tiny_dino(embed_dim=16)
    params = list(dino.student_enc.parameters()) + list(dino.student_head.parameters())
    before = torch.cat([p.detach().flatten().clone() for p in params])
    opt = torch.optim.SGD(params, lr=1.0)
    from wafer_embeddings.train import train_step

    train_step(dino, DINOLoss(out_dim=64), opt, wafers, np.random.default_rng(0), clip_grad=1e-3)
    after = torch.cat([p.detach().flatten() for p in params])
    assert float((after - before).norm()) <= 1e-3 + 1e-6


def test_fit_dino_restores_best_selected_weights():
    # a selection score that peaks at step 2 -> the step-2 weights are restored
    scores = iter([0.1, 0.2, 0.9, 0.3, 0.4])  # steps 0, 1, 2, 3, 4
    snaps: dict[int, torch.Tensor] = {}
    holder: dict = {}

    def select():
        w = holder["dino"].student_enc.state_dict()
        key = next(iter(w))
        snaps[len(snaps)] = w[key].detach().clone()
        return next(scores)

    wafers = [tk.tokenize(_disk(i)) for i in range(6)]
    dino = _tiny_dino(embed_dim=16)
    holder["dino"] = dino
    opt = torch.optim.SGD(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=0.3
    )
    seen = []
    fit_dino(
        dino,
        DINOLoss(out_dim=64),
        opt,
        wafers,
        steps=4,
        batch_size=4,
        rng=np.random.default_rng(0),
        select_fn=select,
        select_every=1,
        monitor=lambda s, st: seen.append(st),
    )
    key = next(iter(dino.student_enc.state_dict()))
    assert torch.equal(dino.student_enc.state_dict()[key], snaps[2])
    assert seen[-1]["best_step"] == 2.0 and seen[-1]["best_score"] == 0.9


def test_selection_score_uses_only_given_maps():
    from wafer_embeddings.train import selection_score

    labels = np.array([0, 0, 1, 1, 2, 2] * 3)
    emb = np.eye(3)[labels].astype(np.float64)
    assert selection_score(emb, labels, n_classes=3) == 1.0


def test_g1_xgroup_metrics_remove_the_same_group_shortcut():
    # embeddings encode only the group; labels follow the group except across groups.
    # Group A: label 0 in bank and queries; group B: label 1. A "group detector" scores
    # perfectly on plain kNN but has no cross-group signal.
    groups = np.array(["A"] * 4 + ["B"] * 4 + ["A"] * 2 + ["B"] * 2, dtype=object)
    labels = np.array([0] * 4 + [1] * 4 + [0] * 2 + [1] * 2)
    splits = np.array(["train"] * 8 + ["test"] * 4, dtype=object)
    emb = np.array([[1.0, 0.0] if g == "A" else [0.0, 1.0] for g in groups])
    m = g1_metrics(emb, labels, splits, n_classes=2, knn_k=3, groups=groups)
    assert m["knn_acc"] == 1.0
    assert m["xgroup_knn_acc"] == 0.0  # only other-group neighbours -> always wrong
    assert m["xgroup_precision@10"] == 0.0


def test_g1_xgroup_metrics_keep_real_defect_signal():
    # embeddings encode the label; groups are unrelated -> cross-group stays perfect
    labels = np.array([0, 1, 0, 1] * 2 + [0, 1, 0, 1])
    groups = np.array(["A", "A", "B", "B"] * 3, dtype=object)
    splits = np.array(["train"] * 8 + ["test"] * 4, dtype=object)
    emb = np.eye(2)[labels].astype(np.float64)
    m = g1_metrics(emb, labels, splits, n_classes=2, knn_k=3, groups=groups)
    assert m["xgroup_knn_acc"] == 1.0 and m["xgroup_knn_macro_recall"] == 1.0
    # each query has only 2 cross-group queries (one per label) -> 1 of 2 is a hit
    assert m["xgroup_precision@10"] == 0.5


def test_selection_score_with_groups_ignores_the_group_shortcut():
    from wafer_embeddings.train import selection_score

    groups = np.array(["A", "A", "B", "B"] * 4, dtype=object)
    labels = np.array([0, 0, 1, 1] * 4)  # label == group
    emb = np.array([[1.0, 0.0] if g == "A" else [0.0, 1.0] for g in groups])
    assert selection_score(emb, labels, n_classes=2) == 1.0
    assert selection_score(emb, labels, n_classes=2, groups=groups) == 0.0
