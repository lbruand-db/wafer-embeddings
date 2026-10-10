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
    # An existing catalog is not re-created (CREATE CATALOG needs a metastore privilege).
    assert not any(s.startswith("CREATE CATALOG") for s in sql)


@pytest.mark.parametrize("bad", ["", "has space", "a.b", "drop;table", "1leading"])
def test_bootstrap_rejects_bad_identifiers(bad):
    with pytest.raises(ValueError):
        build_bootstrap_sql(bad, "wafer_embeddings", "raw")
    with pytest.raises(ValueError):
        build_bootstrap_sql("cat", bad, "raw")
    with pytest.raises(ValueError):
        build_bootstrap_sql("cat", "sch", bad)


def test_catalog_is_created_only_when_missing():
    # existing catalog: no CREATE CATALOG (it would fail without the metastore privilege)
    assert not any("CREATE CATALOG" in s for s in build_bootstrap_sql("cat", "sch", "vol"))
    stmts = build_bootstrap_sql("cat", "sch", "vol", catalog_exists=False)
    assert stmts[0] == "CREATE CATALOG IF NOT EXISTS cat"
    assert stmts[1] == "USE CATALOG cat"  # then the usual schema / volume DDL


def test_catalog_exists_checks_show_catalogs():
    from wafer_embeddings.jobs.bootstrap import catalog_exists

    class FakeSpark:
        def __init__(self, names):
            self.names, self.sql_seen = names, []

        def sql(self, stmt):
            self.sql_seen.append(stmt)
            rows = [(n,) for n in self.names]
            return type("R", (), {"collect": lambda _self: rows})()

    spark = FakeSpark(["cat"])
    assert catalog_exists(spark, "cat") and spark.sql_seen == ["SHOW CATALOGS LIKE 'cat'"]
    assert not catalog_exists(FakeSpark([]), "cat")
