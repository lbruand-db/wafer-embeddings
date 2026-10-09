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
