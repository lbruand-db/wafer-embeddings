import pytest

from wafer_embeddings.jobs.lakebase import (
    LakebaseConfig,
    _args,
    extension_sql,
    grant_sql,
    index_sql,
    owner_role,
    plan_uses_index,
    project_spec,
    synced_table_action,
    synced_table_spec,
    verify_sql,
)


def test_extension_is_lakebase_vector_never_pgvector_directly():
    (stmt,) = extension_sql()
    assert stmt == "CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE"
    assert "EXTENSION IF NOT EXISTS vector" not in stmt


def test_index_sql_creates_cosine_lakebase_ann_index():
    idx, analyze = index_sql("wafer_embeddings.wafer_map_embeddings_pg", "wafer_ann")
    assert "USING lakebase_ann (embedding vector_cosine_ops)" in idx
    assert "IF NOT EXISTS wafer_ann ON wafer_embeddings.wafer_map_embeddings_pg" in idx
    assert "build_mode = 'standard'" in idx and analyze.startswith("ANALYZE")
    assert "build_mode = 'quality'" in index_sql("t", "ix", "quality")[0]
    with pytest.raises(ValueError):
        index_sql("t", "ix", "fast")


def test_verify_sql_and_plan_check():
    count, info, plan = verify_sql("s.t", "ix")
    assert count == "SELECT count(*) FROM s.t"
    assert "lakebase_ann_index_info('s.ix'::regclass)" in info  # schema-qualified
    assert "('ix'::regclass)" in verify_sql("t", "ix")[1]
    assert plan.startswith("EXPLAIN") and "ORDER BY embedding <=>" in plan
    ann = ["  ->  Index Scan using ix on t", "        Order By: (embedding <=> $1)"]
    assert plan_uses_index(["Limit"] + ann, "ix")
    # a synced (partitioned) table names the child index; the pkey scan doesn't count
    part = [
        "Limit",
        '  ->  Index Scan using "tmp_pkey" on partition_1 t_1',
        "  ->  Index Scan using partition_1_embedding_idx on partition_1 t",
        "        Order By: (embedding <=> (InitPlan 1).col1)",
    ]
    assert plan_uses_index(part, "ix")
    assert not plan_uses_index(["Limit", "  ->  Sort", "  ->  Seq Scan on t"], "ix")
    assert not plan_uses_index(['  ->  Index Scan using "tmp_pkey" on t', "Filter: x"], "ix")


@pytest.mark.parametrize("bad", ["t; DROP TABLE x", "a.b.c", "1table", "t--"])
def test_identifiers_are_validated(bad):
    with pytest.raises(ValueError):
        index_sql(bad, "ix")
    with pytest.raises(ValueError):
        verify_sql("t", bad)


def test_grant_sql_quotes_service_principal_and_rejects_injection():
    sp = "0b2a6a23-b099-4d66-af1d-f2b83727234f"
    stmts = grant_sql("wafer_embeddings", sp)
    assert stmts[0] == f'GRANT USAGE ON SCHEMA wafer_embeddings TO "{sp}"'
    assert all(f'"{sp}"' in s for s in stmts) and "DEFAULT PRIVILEGES" in stmts[2]
    with pytest.raises(ValueError):
        grant_sql("wafer_embeddings", 'x"; DROP TABLE t; --')


@pytest.mark.parametrize(
    "state,action",
    [
        (None, "create"),
        ("SYNCED_TABLE_PROVISIONING_PIPELINE_RESOURCES", "wait"),
        ("SYNCED_TABLE_PROVISIONING_INITIAL_SNAPSHOT", "wait"),
        ("SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE", "ok"),
        ("SYNCED_TABLE_ONLINE", "ok"),
        ("SYNCED_TABLE_OFFLINE_FAILED", "recreate"),
        # serving data with a failed last refresh: keep it, never delete
        ("SYNCED_TABLE_ONLINE_PIPELINE_FAILED", "ok"),
        ("SYNCED_TABLE_ONLINE_TRIGGERED_UPDATE", "ok"),
        ("SYNCED_TABLE_OFFLINE", "wait"),
    ],
)
def test_synced_table_action(state, action):
    assert synced_table_action(state) == action


def test_synced_table_action_after_our_create_waits_instead_of_reposting():
    assert synced_table_action(None, created=True) == "wait"
    assert synced_table_action(None, created=False) == "create"


def test_drop_table_and_preload_detection():
    from wafer_embeddings.jobs.lakebase import drop_table_sql, is_preload_not_ready

    assert drop_table_sql("s.t") == ["DROP TABLE IF EXISTS s.t CASCADE"]
    with pytest.raises(ValueError):
        drop_table_sql("s.t; DROP DATABASE x")
    assert is_preload_not_ready(
        RuntimeError("[lakebase_vector] must be loaded via shared_preload_libraries.")
    )
    assert not is_preload_not_ready(RuntimeError('role "x" does not exist'))


def test_synced_table_spec_types_embedding_as_vector_dim():
    cfg = LakebaseConfig(dim=64)
    s = synced_table_spec(cfg)["spec"]
    assert s["source_table_full_name"] == cfg.source_table
    assert s["primary_key_columns"] == ["id"] and s["scheduling_policy"] == "SNAPSHOT"
    assert s["branch"] == "projects/wafer-embeddings/branches/production"
    assert s["type_overrides"] == [
        {"column_name": "embedding", "pg_type": "PG_SPECIFIC_TYPE_VECTOR", "size": 64}
    ]
    assert s["new_pipeline_spec"] == {
        "storage_catalog": "mmf_mlops_demo_catalog",
        "storage_schema": "wafer_embeddings",
    }


def test_config_names_and_project_spec():
    cfg = LakebaseConfig()
    assert cfg.pg_table == "wafer_embeddings.wafer_map_embeddings_pg"
    assert cfg.pg_schema == "wafer_embeddings"
    assert project_spec(cfg)["spec"]["pg_version"] >= 16  # Lakebase Search needs PG 16+


def test_owner_role_finds_the_job_identity():
    roles = [
        {
            "name": "projects/p/branches/b/roles/dbrx-apps-x",
            "role_id": "dbrx-apps-x",
            "status": {"postgres_role": "0b2a"},
        },
        {
            "name": "projects/p/branches/b/roles/lucas-bruand",
            "role_id": "lucas-bruand",
            "status": {"postgres_role": "lucas.bruand@databricks.com"},
        },
    ]
    role = owner_role(roles, "lucas.bruand@databricks.com")
    assert role is not None and role.endswith("/lucas-bruand")
    assert owner_role(roles, "nobody@x.com") is None


def test_args_build_config():
    cfg = _args(["--project", "p2", "--dim", "384", "--build-mode", "quality"])
    assert cfg.project == "p2" and cfg.dim == 384 and cfg.build_mode == "quality"
    assert cfg.branch_name == "projects/p2/branches/production"
    assert cfg.refresh is False and _args(["--refresh"]).refresh is True
