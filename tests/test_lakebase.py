import pytest

from wafer_embeddings.jobs.lakebase import setup_sql, verify_sql


def test_setup_sql_creates_cosine_lakebase_ann_index():
    stmts = setup_sql("wafer_embeddings.wafer_map_embeddings_pg", "wafer_ann")
    assert stmts[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert "lakebase_vector CASCADE" in stmts[1]
    idx = stmts[2]
    assert "USING lakebase_ann (embedding vector_cosine_ops)" in idx
    assert "IF NOT EXISTS wafer_ann ON wafer_embeddings.wafer_map_embeddings_pg" in idx
    assert "build_mode = 'standard'" in idx and stmts[3].startswith("ANALYZE")


def test_verify_sql_checks_count_index_and_plan():
    count, info, plan = verify_sql("s.t", "ix")
    assert count == "SELECT count(*) FROM s.t"
    assert "lakebase_ann_index_info('ix'::regclass)" in info
    assert plan.startswith("EXPLAIN") and "ORDER BY embedding <=>" in plan


@pytest.mark.parametrize("bad", ["t; DROP TABLE x", "a.b.c", "1table", "t--"])
def test_identifiers_are_validated(bad):
    with pytest.raises(ValueError):
        setup_sql(bad, "ix")
    with pytest.raises(ValueError):
        verify_sql("t", bad)


def test_build_mode_validated():
    assert "build_mode = 'quality'" in setup_sql("t", "ix", "quality")[2]
    with pytest.raises(ValueError):
        setup_sql("t", "ix", "fast")
