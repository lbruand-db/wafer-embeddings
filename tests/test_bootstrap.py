import pytest

from wafer_embeddings.jobs.bootstrap import build_bootstrap_sql


def test_bootstrap_sql_is_idempotent_and_ordered():
    sql = build_bootstrap_sql("mmf_mlops_demo_catalog", "wafer_embeddings", "raw")
    assert sql == [
        "USE CATALOG mmf_mlops_demo_catalog",
        "CREATE SCHEMA IF NOT EXISTS mmf_mlops_demo_catalog.wafer_embeddings",
        "CREATE VOLUME IF NOT EXISTS mmf_mlops_demo_catalog.wafer_embeddings.raw",
    ]
    # Every create is guarded so re-runs are no-ops.
    assert all("IF NOT EXISTS" in s for s in sql if s.startswith("CREATE"))
    # The catalog is never created here (assumed to pre-exist).
    assert not any(s.startswith("CREATE CATALOG") for s in sql)


@pytest.mark.parametrize("bad", ["", "has space", "a.b", "drop;table", "1leading"])
def test_bootstrap_rejects_bad_identifiers(bad):
    with pytest.raises(ValueError):
        build_bootstrap_sql(bad, "wafer_embeddings", "raw")
    with pytest.raises(ValueError):
        build_bootstrap_sql("cat", bad, "raw")
    with pytest.raises(ValueError):
        build_bootstrap_sql("cat", "sch", bad)
