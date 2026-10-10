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


def build_bootstrap_sql(
    catalog: str, schema: str, volume: str, catalog_exists: bool = True
) -> list[str]:
    """Return idempotent DDL to ensure the catalog, schema and volume exist.

    ``CREATE CATALOG IF NOT EXISTS`` is emitted **only when the catalog is missing**:
    Unity Catalog checks the metastore ``CREATE CATALOG`` privilege before existence, so
    the statement fails (PERMISSION_DENIED) even as a no-op for users who don't hold it.
    The caller checks existence first (``catalog_exists``). Every other statement is
    ``IF NOT EXISTS`` so re-runs are no-ops (SPECS.md §11.8).
    """
    cat = _check_ident("catalog", catalog)
    sch = _check_ident("schema", schema)
    vol = _check_ident("volume", volume)
    create = [] if catalog_exists else [f"CREATE CATALOG IF NOT EXISTS {cat}"]
    return create + [
        f"USE CATALOG {cat}",
        f"CREATE SCHEMA IF NOT EXISTS {cat}.{sch}",
        f"CREATE VOLUME IF NOT EXISTS {cat}.{sch}.{vol}",
    ]


def catalog_exists(spark, catalog: str) -> bool:
    """True if ``catalog`` is visible to the caller (no privilege needed beyond USE)."""
    cat = _check_ident("catalog", catalog)
    return any(row[0] == cat for row in spark.sql(f"SHOW CATALOGS LIKE '{cat}'").collect())


def _get_spark():  # pragma: no cover - requires a Databricks runtime
    try:
        from databricks.sdk.runtime import spark  # type: ignore

        return spark
    except Exception:
        from pyspark.sql import SparkSession  # ty: ignore[unresolved-import]

        return SparkSession.builder.getOrCreate()


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs Spark
    parser = argparse.ArgumentParser(description="Bootstrap UC catalog + schema + volume.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--volume", required=True)
    args = parser.parse_args(argv)

    from wafer_embeddings.obs import get_logger

    log = get_logger("wafer_embeddings.bootstrap")
    spark = _get_spark()
    exists = catalog_exists(spark, args.catalog)
    log.info(f"catalog {args.catalog}: {'exists' if exists else 'missing -> will create'}")
    for stmt in build_bootstrap_sql(args.catalog, args.schema, args.volume, exists):
        log.info(stmt)
        spark.sql(stmt)
    log.info("done")


if __name__ == "__main__":  # pragma: no cover
    main()
