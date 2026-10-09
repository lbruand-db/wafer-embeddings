"""Lakebase Search setup on the synced embeddings table (SPECS.md §9, §11.6).

Run after the UC synced table (Delta ``wafer_map_embeddings`` -> Postgres
``<schema>.<table>``, ``embedding vector(D)``) is online. Lakebase Search must be enabled
on the project first (UI-only today; it preloads ``lakebase_vector``). Idempotent.

The SQL is built by pure functions (tested); ``main`` connects with an OAuth credential.
"""

from __future__ import annotations

import argparse
import re

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _check_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def setup_sql(table: str, index: str, build_mode: str = "standard") -> list[str]:
    """Statements: extensions, then the cosine ``lakebase_ann`` index (§9)."""
    _check_ident(table)
    _check_ident(index)
    if build_mode not in ("standard", "quality"):
        raise ValueError("build_mode must be 'standard' or 'quality'")
    return [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE",
        f"CREATE INDEX IF NOT EXISTS {index} ON {table} USING lakebase_ann "
        f"(embedding vector_cosine_ops) WITH (build_mode = '{build_mode}')",
        f"ANALYZE {table}",
    ]


def verify_sql(table: str, index: str) -> list[str]:
    """Checks: row count, index metadata, and that a top-k query uses the ANN index."""
    _check_ident(table)
    _check_ident(index)
    probe = f"(SELECT embedding FROM {table} ORDER BY id LIMIT 1)"
    return [
        f"SELECT count(*) FROM {table}",
        f"SELECT lakebase_ann_index_info('{index}'::regclass)",
        f"EXPLAIN SELECT id FROM {table} ORDER BY embedding <=> {probe} LIMIT 10",
    ]


def _args(argv=None):
    p = argparse.ArgumentParser(description="Create the lakebase_ann index on the synced table.")
    p.add_argument(
        "--endpoint", default="projects/wafer-embeddings/branches/production/endpoints/primary"
    )
    p.add_argument("--database", default="wafer_embeddings")
    p.add_argument("--table", default="wafer_embeddings.wafer_map_embeddings_pg")
    p.add_argument("--index", default="wafer_map_embeddings_ann")
    p.add_argument("--build-mode", default="standard", choices=["standard", "quality"])
    return p.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover - needs a Lakebase project
    import psycopg  # ty: ignore[unresolved-import]
    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]

    a = _args(argv)
    w = WorkspaceClient()
    host = w.postgres.get_endpoint(a.endpoint).status.hosts.host
    token = w.postgres.generate_database_credential(endpoint=a.endpoint).token
    with psycopg.connect(
        host=host,
        dbname=a.database,
        user=w.current_user.me().user_name,
        password=token,
        sslmode="require",
        autocommit=True,
    ) as conn:
        for stmt in setup_sql(a.table, a.index, a.build_mode) + verify_sql(a.table, a.index):
            cur = conn.execute(stmt)
            print(f"-- {stmt}")
            if cur.description:
                for row in cur.fetchall():
                    print("  ", *row)


if __name__ == "__main__":  # pragma: no cover
    main()
