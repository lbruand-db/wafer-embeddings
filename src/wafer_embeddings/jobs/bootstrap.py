"""Idempotent UC schema + volume bootstrap (SPECS.md §11.8 / N7).

The SQL is produced by a pure, unit-tested function (`build_bootstrap_sql`); `main`
only wires it to a Spark session so the DDL logic is testable without Databricks.
"""

from __future__ import annotations

import argparse
import re

# Unity Catalog identifiers we accept: start with a letter/underscore, then
# letters/digits/underscores. We reject anything else rather than quote-escaping,
# since these come from bundle config, not end users.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(kind: str, value: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"invalid {kind} identifier: {value!r}")
    return value


def build_bootstrap_sql(catalog: str, schema: str, volume: str) -> list[str]:
    """Return idempotent DDL to ensure the schema + volume exist.

    Does NOT create the catalog: it is assumed to pre-exist at the metastore level
    (SPECS.md §11.8). Every statement is ``IF NOT EXISTS`` so re-runs are no-ops.
    """
    cat = _check_ident("catalog", catalog)
    sch = _check_ident("schema", schema)
    vol = _check_ident("volume", volume)
    return [
        f"USE CATALOG {cat}",
        f"CREATE SCHEMA IF NOT EXISTS {cat}.{sch}",
        f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{vol}",
    ]


def _get_spark():  # pragma: no cover - requires a Databricks runtime
    try:
        from databricks.sdk.runtime import spark  # type: ignore

        return spark
    except Exception:
        from pyspark.sql import SparkSession  # ty: ignore[unresolved-import]

        return SparkSession.builder.getOrCreate()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs Spark
    parser = argparse.ArgumentParser(description="Bootstrap UC schema + volume.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    args = parser.parse_args(argv)

    from wafer_embeddings.obs import get_logger

    log = get_logger("wafer_embeddings.bootstrap")
    spark = _get_spark()
    for stmt in build_bootstrap_sql(args.catalog, args.schema, args.volume):
        log.info(stmt)
        spark.sql(stmt)
    log.info("done")


if __name__ == "__main__":  # pragma: no cover
    main()
