"""Parquet sampling in the train job (jobs/train.py) — no Databricks, no GPU."""

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from wafer_embeddings.jobs.train import load_eval_table, load_train_table


def _write_parts(tmp_path, n_parts=4, rows_per_part=50):
    """Several parquet files, like the Spark export; each part holds a distinct id range."""
    rng = np.random.default_rng(0)
    for p in range(n_parts):
        ids = np.arange(p * rows_per_part, (p + 1) * rows_per_part, dtype=np.int64)
        splits = rng.choice(["train", "val", "test"], size=rows_per_part, p=[0.6, 0.2, 0.2])
        labels = np.where(rng.random(rows_per_part) < 0.5, rng.integers(0, 3, rows_per_part), -1)
        tbl = pa.table(
            {
                "id": ids,
                "height": np.full(rows_per_part, 2, dtype=np.int32),
                "width": np.full(rows_per_part, 2, dtype=np.int32),
                "wafer_map": [[1, 2, 1, 1]] * rows_per_part,
                "label_id": labels.astype(np.int32),
                "split": splits.tolist(),
            }
        )
        pq.write_table(tbl, tmp_path / f"part-{p:05d}.parquet")
    return ds.dataset(str(tmp_path), format="parquet"), rows_per_part


def test_load_train_table_is_seeded_random_not_first_file(tmp_path):
    dset, per_part = _write_parts(tmp_path)
    a = load_train_table(dset, 40, seed=0)
    b = load_train_table(dset, 40, seed=0)
    c = load_train_table(dset, 40, seed=1)
    ids = sorted(a.column("id").to_pylist())
    assert a.num_rows == 40
    assert set(a.column("split").to_pylist()) == {"train"}
    assert ids == sorted(b.column("id").to_pylist())  # same seed -> same sample
    assert ids != sorted(c.column("id").to_pylist())  # different seed -> different sample
    # the old head(N) drew everything from part 0; a random sample spans the files
    assert len({i // per_part for i in ids}) > 1


def test_load_train_table_zero_means_all(tmp_path):
    dset, _ = _write_parts(tmp_path)
    n_train = dset.to_table(filter=ds.field("split") == "train").num_rows
    assert load_train_table(dset, 0, seed=0).num_rows == n_train
    assert load_train_table(dset, 10_000, seed=0).num_rows == n_train  # cap above size


def test_load_eval_table_stratified_and_split_separated(tmp_path):
    dset, _ = _write_parts(tmp_path)
    t = load_eval_table(dset, 12, seed=0, balanced=True)
    t2 = load_eval_table(dset, 12, seed=0, balanced=True)
    assert sorted(t.column("id").to_pylist()) == sorted(t2.column("id").to_pylist())
    labels = np.array(t.column("label_id").to_pylist())
    splits = np.array(t.column("split").to_pylist())
    assert (labels >= 0).all()  # labeled only
    assert set(splits) == {"val", "train"}  # development protocol: test never loaded
    queries = splits == "val"
    assert queries.sum() == 12 and (splits == "train").sum() == 12  # eval_cap per side
    counts = np.bincount(labels[queries], minlength=3)
    assert int(np.max(counts)) - int(np.min(counts)) <= 1  # balanced across the 3 classes


def test_load_eval_table_test_split_only_when_asked(tmp_path):
    import pytest

    dset, _ = _write_parts(tmp_path)
    t = load_eval_table(dset, 8, seed=0, query_split="test")
    assert set(t.column("split").to_pylist()) == {"test", "train"}
    with pytest.raises(ValueError):
        load_eval_table(dset, 8, seed=0, query_split="val+test")


def test_train_defaults_are_the_r1_recipe():
    # the recipe behind wafer_encoder v2 (PLAN.md P1), no flags needed
    from wafer_embeddings.jobs.train import _args, train_recipe

    a = _args(["--catalog", "c", "--schema", "s"])
    assert (a.embed_dim, a.depth, a.heads, a.attention, a.pre_norm) == (128, 6, 4, "isab", True)
    assert (a.max_tokens, a.batch_size, a.steps, a.max_train) == (512, 32, 10000, 100000)
    assert (a.out_dim, a.bottleneck, a.freeze_last_frac, a.clip_grad) == (2048, 256, 0.01, 3.0)
    assert train_recipe(a) == {
        "lr": 5e-4 * 32 / 256,  # LR rule
        "aug": {
            "die_noise_p": 0.03,
            "crop_area": (0.4, 1.0),
            "n_local": 6,
            "local_crop_area": (0.05, 0.4),
        },
        "weight_decay": (0.04, 0.4),
    }


def test_legacy_recipe_flags_reproduce_the_old_behaviour():
    from wafer_embeddings.jobs.train import LEGACY_RECIPE, _args, train_recipe

    a = _args(["--catalog", "c", "--schema", "s"] + LEGACY_RECIPE.split())
    r = train_recipe(a)
    assert r["lr"] == 5e-4 and r["weight_decay"] == (0.0, 0.0)  # constant 0 = no decay
    assert r["aug"] == {"die_noise_p": 0.005, "crop_area": (0.81, 0.81)}  # fixed 0.9 side
    assert (a.embed_dim, a.depth, a.attention, a.pre_norm) == (384, 12, "full", False)


def test_train_recipe_reference_dino_flags():
    from wafer_embeddings.jobs.train import _args, train_recipe

    a = _args(
        "--catalog c --schema s --lr 0 --batch-size 64 --weight-decay 0.04 "
        "--weight-decay-end 0.4 --n-local 6 --global-crop-area 0.4 1.0 "
        "--local-crop-area 0.1 0.4 --die-noise 0.03 --pre-norm --bottleneck 256".split()
    )
    r = train_recipe(a)
    assert r["lr"] == 5e-4 * 64 / 256  # linear scaling rule
    assert r["weight_decay"] == (0.04, 0.4)
    assert r["aug"] == {
        "die_noise_p": 0.03,
        "crop_area": (0.4, 1.0),
        "n_local": 6,
        "local_crop_area": (0.1, 0.4),
    }
    assert a.pre_norm and a.bottleneck == 256


def test_track_every_is_validated():
    import pytest

    from wafer_embeddings.jobs.train import _args

    base = ["--catalog", "c", "--schema", "s"]
    assert _args(base + ["--track-every", "100"]).track_every == 100
    with pytest.raises(SystemExit):
        _args(base + ["--track-every", "-3"])
    with pytest.raises(SystemExit):  # no training curve on the test split
        _args(base + ["--track-every", "100", "--eval-split", "test"])
    assert _args(base + ["--eval-split", "test"]).track_every == 0  # default: off on test
    assert _args(base).track_every == 1000  # default: on for val development
    assert _args(base).track_clustering is True  # full metrics unless the light path is asked
    assert _args(base + ["--no-track-clustering"]).track_clustering is False


def test_eval_seed_defaults_to_seed_and_can_be_fixed():
    from wafer_embeddings.jobs.train import _args

    base = ["--catalog", "c", "--schema", "s"]
    assert _args(base + ["--seed", "3"]).eval_seed == 3
    a = _args(base + ["--seed", "3", "--eval-seed", "0"])  # training noise only
    assert (a.seed, a.eval_seed) == (3, 0)


def test_eval_metric_keys_namespace_by_split_and_model():
    from wafer_embeddings.jobs.train import eval_metric_keys

    got = eval_metric_keys("val", "teacher", {"xgroup_precision@10": 0.3, "knn_acc": 1})
    assert got == {"val/teacher/xgroup_precision@10": 0.3, "val/teacher/knn_acc": 1.0}
    assert all(isinstance(v, float) for v in got.values())
    assert list(eval_metric_keys("test", "polar", {"map@10": 0.1})) == ["test/polar/map@10"]


def test_beats_polar_needs_both_cross_device_metrics():
    from wafer_embeddings.jobs.train import beats_polar

    polar = {"xgroup_knn_macro_recall": 0.44, "xgroup_precision@10": 0.36}
    assert beats_polar({"xgroup_knn_macro_recall": 0.45, "xgroup_precision@10": 0.37}, polar)
    assert not beats_polar({"xgroup_knn_macro_recall": 0.45, "xgroup_precision@10": 0.35}, polar)
    assert not beats_polar({"xgroup_knn_macro_recall": 0.45}, polar)  # missing metric
    assert not beats_polar({"xgroup_knn_macro_recall": 0.9, "xgroup_precision@10": 0.9}, {})


def test_descriptor_and_orientation_flags():
    from wafer_embeddings.jobs.train import _args

    base = ["--catalog", "c", "--schema", "s"]
    a = _args(base)
    assert (a.nn_positives, a.nn_descriptor, a.orientation_invariant) == (0, "pixel", True)
    a = _args(
        base + ["--nn-positives", "5", "--nn-descriptor", "polar", "--no-orientation-invariant"]
    )
    assert (a.nn_positives, a.nn_descriptor, a.orientation_invariant) == (5, "polar", False)
