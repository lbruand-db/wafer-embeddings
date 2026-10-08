"""Ingest WM-811K into Delta (SPECS.md §7, §16 item 21).

Downloads the WM-811K archive, loads ``LSWMD.pkl`` (a pickled pandas DataFrame), then
**streams** cleaned/deduped records to a Delta table in bounded chunks (first chunk
overwrite, rest append). Streaming keeps peak memory flat — building all ~700K rows (plus
a one-shot pandas->Spark conversion) on top of the in-memory pickle OOMs the driver.

Row-building is pure and tested; download / unzip / pickle-load / Spark write happen
Databricks-side in ``main``.

SECURITY: ``LSWMD.pkl`` is a Python pickle (arbitrary-code on load) and the archive is
untrusted data — fetched and read only on the Databricks side, never locally.
"""

from __future__ import annotations

import argparse

import numpy as np

from wafer_embeddings.data import wm811k as wm
from wafer_embeddings.obs import get_logger, periodic, stage

WM811K_URL = "http://mirlab.org/dataSet/public/MIR-WM811K.zip"


def record_to_row(rec: wm.Record, row_id: int) -> dict:
    """Flatten one Record into a Delta row dict (map stored flattened)."""
    m = rec.wafer_map
    return {
        "id": int(row_id),
        "height": int(m.shape[0]),
        "width": int(m.shape[1]),
        "wafer_map": m.reshape(-1).astype(np.uint8).tolist(),
        "label": rec.label_name,  # None for unlabeled
        "label_id": int(rec.label_id),  # -1 for unlabeled
        "split": rec.split,
        "lot": rec.lot,  # None if unknown
    }


def _ensure_pickle(source_url: str, vol: str, log) -> str:  # pragma: no cover - IO
    """Download + unzip the WM-811K archive into the volume; return the .pkl path."""
    import os
    import urllib.request
    import zipfile

    zip_path = f"{vol}/MIR-WM811K.zip"
    if not os.path.exists(zip_path):
        log.info(f"downloading {source_url} -> {zip_path}")
        urllib.request.urlretrieve(source_url, zip_path)  # noqa: S310
    else:
        log.info(f"using existing archive {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        pkls = [m for m in zf.namelist() if m.lower().endswith(".pkl")]
        if not pkls:
            raise RuntimeError(f"no .pkl in archive; contents={zf.namelist()}")
        log.info(f"extracting {pkls[0]}")
        zf.extract(pkls[0], vol)
        return f"{vol}/{pkls[0]}"


def _delta_schema():  # pragma: no cover - needs pyspark
    from pyspark.sql.types import (  # ty: ignore[unresolved-import]
        ArrayType,
        IntegerType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField("id", LongType(), False),
            StructField("height", IntegerType(), False),
            StructField("width", IntegerType(), False),
            StructField("wafer_map", ArrayType(IntegerType()), False),
            StructField("label", StringType(), True),
            StructField("label_id", IntegerType(), False),
            StructField("split", StringType(), False),
            StructField("lot", StringType(), True),
        ]
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs Spark
    p = argparse.ArgumentParser(description="Ingest WM-811K into Delta (streaming).")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="raw")
    p.add_argument("--table", default="wafer_maps")
    p.add_argument("--source-url", default=WM811K_URL)
    p.add_argument("--pkl", default=None, help="Path to an already-extracted LSWMD.pkl.")
    p.add_argument("--limit", type=int, default=0, help="Rows to ingest (0 = all).")
    p.add_argument("--chunk-size", type=int, default=25000, help="Rows per Delta write.")
    args = p.parse_args(argv)

    import pandas as pd  # ty: ignore[unresolved-import]
    from databricks.sdk.runtime import spark  # type: ignore

    log = get_logger("wafer_embeddings.ingest")
    vol = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}"
    fqn = f"{args.catalog}.{args.schema}.{args.table}"
    schema = _delta_schema()
    cols = [f.name for f in schema.fields]

    with stage(log, "ensure dataset"):
        pkl = args.pkl or _ensure_pickle(args.source_url, vol, log)
    with stage(log, f"read_pickle {pkl}"):
        df = pd.read_pickle(pkl)
    if args.limit:
        df = df.iloc[: args.limit]
    log.info(f"{len(df)} raw rows; columns={list(df.columns)}")

    def flush(chunk: list[dict], mode: str) -> None:
        tuples = [tuple(r[c] for c in cols) for r in chunk]
        spark.createDataFrame(tuples, schema=schema).write.mode(mode).option(
            "overwriteSchema", "true"
        ).saveAsTable(fqn)

    seen: set[str] = set()
    chunk: list[dict] = []
    row_id = 0
    written = 0
    labeled = 0
    mode = "overwrite"
    prog = periodic(log.info, "scan")
    with stage(log, f"stream -> {fqn}"):
        # Lot-grouped splits: one lot = one device, often one defect, so a per-map split
        # would put near-identical wafers on both sides of the eval boundary.
        if "lotName" not in df.columns:
            raise RuntimeError(f"LSWMD.pkl has no lotName column; columns={list(df.columns)}")
        records = wm.iter_records(
            df["waferMap"], df["failureType"], seen=seen, progress=prog, lots=df["lotName"]
        )
        for rec in records:
            chunk.append(record_to_row(rec, row_id))
            row_id += 1
            labeled += int(rec.label_id >= 0)
            if len(chunk) >= args.chunk_size:
                flush(chunk, mode)
                written += len(chunk)
                mode = "append"
                chunk = []
                log.info(f"written {written} rows")
        if chunk:
            flush(chunk, mode)
            written += len(chunk)
    log.info(f"wrote {written} rows ({labeled} labeled) to {fqn}")
    split_lots = spark.sql(
        f"SELECT split, count(*) AS n, count(DISTINCT lot) AS lots, "
        f"count_if(label_id >= 0) AS labeled FROM {fqn} GROUP BY split ORDER BY split"
    ).collect()
    log.info(f"lot-grouped splits: {[r.asDict() for r in split_lots]}")
    straddle = spark.sql(
        f"SELECT count(*) AS n FROM (SELECT lot FROM {fqn} WHERE lot IS NOT NULL "
        f"GROUP BY lot HAVING count(DISTINCT split) > 1)"
    ).collect()[0]["n"]
    if straddle:
        raise RuntimeError(f"{straddle} lots straddle splits; lot-grouped split is broken")

    # Parquet export to the volume: AI Runtime serverless GPU has no Spark, so the
    # training job reads these files directly (SPECS.md §11.3).
    pq = f"{vol}/wafer_maps_parquet"
    with stage(log, f"export parquet -> {pq}"):
        spark.read.table(fqn).write.mode("overwrite").parquet(pq)


if __name__ == "__main__":  # pragma: no cover
    main()
