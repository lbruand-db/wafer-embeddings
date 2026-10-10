import pytest

from wafer_embeddings.jobs.teardown import _args, plan


def test_serving_scope_removes_rebuildable_things_in_dependency_order():
    steps = [(s.kind, s.target) for s in plan("c", "s")]
    assert steps == [
        ("endpoint", "wafer-encoder"),
        ("synced_table", "c.s.wafer_map_embeddings_pg"),  # before the project it lives in
        ("lakebase_project", "wafer-embeddings"),
        ("table", "c.s.wafer_map_embeddings"),  # after the synced table that reads it
        ("volume_dir", "/Volumes/c/s/raw/wafer_map_embeddings_parquet"),
    ]
    kinds = [k for k, _ in steps]
    assert "model" not in kinds and "schema" not in kinds  # source data + model kept


def test_all_scope_adds_data_model_volume_schema_never_catalog():
    steps = plan("c", "s", scope="all")
    kinds = [s.kind for s in steps]
    assert kinds[-1] == "schema" and kinds[-2] == "volume"  # containers last
    assert ("model", "c.s.wafer_encoder") in [(s.kind, s.target) for s in steps]
    assert ("table", "c.s.wafer_maps") in [(s.kind, s.target) for s in steps]
    assert all(s.target != "c" for s in steps)  # the catalog is never dropped
    with pytest.raises(ValueError):
        plan("c", "s", scope="everything")


def test_dry_run_is_the_default():
    a = _args(["--catalog", "c", "--schema", "s"])
    assert a.yes is False and a.scope == "serving"
    assert _args(["--catalog", "c", "--schema", "s", "--yes"]).yes is True
