"""The diagnostics in tools/ (run by hand): their pure helpers, so they can't silently rot."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import baseline_check  # noqa: E402
import nuisance_probe  # noqa: E402
import ref_recipe  # noqa: E402
import score_test  # noqa: E402

from wafer_embeddings.jobs.train import _args, train_recipe  # noqa: E402
from wafer_embeddings.tokenize.tokenizer import TokenizedWafer  # noqa: E402


def test_shape_leak_counts_only_query_bank_pairs():
    shapes = ["a", "a", "b", "b"]
    labels = [0, 0, 1, 0]
    splits = ["val", "train", "val", "train"]
    lots = ["L1", "L1", None, "L2"]
    out = baseline_check.shape_leak(shapes, labels, splits, lots)
    # query x bank pairs: (0,1) same shape same label, (0,3), (2,1), (2,3) same shape
    assert out["frac_pairs_same_shape"] == 0.5
    assert out["p_same_label"] == 0.5  # (0,1) and (0,3)
    assert out["p_same_label_given_same_shape"] == 0.5  # (0,1) yes, (2,3) no
    assert out["pairs_same_lot"] == 1  # (0,1); the unknown lot never matches


def test_shape_onehot():
    oh = baseline_check.shape_onehot(np.array(["2x3", "4x4", "2x3"], dtype=object))
    assert oh.tolist() == [[1, 0], [0, 1], [1, 0]]


def test_nuisance_probe_on_a_shape_clustered_embedding():
    # two tight clusters that coincide with both shape and label
    emb = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]], dtype=float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    shapes = np.array(["a", "a", "b", "b"], dtype=object)
    labels = np.array([0, 0, 1, 1])
    out = nuisance_probe.nuisance_probe(emb, labels, shapes, np.log([10, 10, 100, 100.0]), k=1)
    assert out == {"shape@10": 1.0, "label@10": 1.0, "logdies_r2": 1.0}


def test_jittered_view_moves_coords_but_keeps_states():
    rng = np.random.default_rng(0)
    coords = rng.uniform(-0.5, 0.5, size=(50, 3)).astype(np.float32)
    w = TokenizedWafer(coords, np.ones(50, dtype=np.int64))
    v = nuisance_probe.jittered_view(1.0)(w, np.random.default_rng(1))
    v0 = nuisance_probe.jittered_view(0.0)(w, np.random.default_rng(1))
    assert v.n_tokens == v0.n_tokens
    assert not np.allclose(v.coords[:, :2], v0.coords[:, :2])
    np.testing.assert_allclose(v.coords[:, 2], np.hypot(v.coords[:, 0], v.coords[:, 1]), 1e-5)


def test_r1_is_the_train_job_default_recipe():
    job = _args(["--catalog", "c", "--schema", "s"])
    r1 = ref_recipe.recipe_args("R1")
    assert train_recipe(r1) == train_recipe(job)
    keep = ("embed_dim", "depth", "heads", "attention", "pre_norm", "out_dim", "bottleneck")
    assert all(getattr(r1, k) == getattr(job, k) for k in keep)
    assert (r1.steps, r1.max_train) == (3000, 20000)  # CPU budget


def test_recipe_ablations():
    assert ref_recipe.recipe_args("R2").pre_norm is False
    assert "n_local" not in train_recipe(ref_recipe.recipe_args("R3"))["aug"]
    r0 = ref_recipe.recipe_args("R0")
    assert (r0.embed_dim, r0.depth, r0.attention, r0.pre_norm) == (128, 6, "isab", False)
    assert ref_recipe.recipe_args("R1", "--steps 10").steps == 10
    with pytest.raises(ValueError):
        ref_recipe.recipe_args("R9")


def test_score_test_needs_one_source():
    a = score_test._args(["data", "--model", "c.s.m", "--version", "3"])
    assert (a.model, a.version, a.checkpoint) == ("c.s.m", "3", None)
    assert score_test._args(["data", "--checkpoint", "e.pt"]).checkpoint == "e.pt"
    for bad in (
        ["data"],
        ["data", "--version", "3"],
        ["data", "--checkpoint", "e", "--version", "3"],
    ):
        with pytest.raises(SystemExit):
            score_test._args(bad)


def test_score_test_headline_keeps_report_keys():
    m = {"xgroup_knn_macro_recall": 0.38851, "map@10": 0.3, "knn_recall_Loc": 0.0712, "x": 1.0}
    assert score_test.headline(m) == {
        "xgroup_knn_macro_recall": 0.3885,
        "map@10": 0.3,
        "knn_recall_Loc": 0.071,
    }


def test_compare_runs_band_and_metric_parsing():
    import compare_runs

    assert compare_runs.band([0.4]) == (0.4, 0.0)
    mean, std = compare_runs.band([0.40, 0.42, 0.44])
    assert abs(mean - 0.42) < 1e-12 and abs(std - 0.02) < 1e-12
    with pytest.raises(ValueError):
        compare_runs.band([])
    flat = {
        "val/student/xgroup_knn_macro_recall": 0.41,
        "val/polar/map@10": 0.34,
        "val/student/rankme": 12.0,  # not a compared metric
        "test/student/map@10": 0.3,  # other split
        "train/student/rankme": 9.0,
    }
    assert compare_runs.final_metrics(flat) == {
        "student": {"xgroup_knn_macro_recall": 0.41},
        "polar": {"map@10": 0.34},
    }
