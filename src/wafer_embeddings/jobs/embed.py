"""Batch-embed every wafer map with the registered encoder -> Delta (SPECS.md §10).

Reads ``<catalog>.<schema>.wafer_maps``, embeds each map with the UC model (default alias
``@champion``) and writes ``<catalog>.<schema>.wafer_map_embeddings`` with the embedding,
the metadata the search app filters/displays on, and a compact ``map_code`` for drawing
the map. Change data feed is enabled so the table can be synced into Lakebase (§9).

The per-batch work is the pure, CI-tested :func:`embed_frames`; ``main`` is Spark/UC glue.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator

import numpy as np

OUTPUT_COLUMNS = (
    "id",
    "embedding",
    "map_code",
    "label",
    "label_id",
    "split",
    "lot",
    "height",
    "width",
    "n_dies",
    "fail_frac",
    "model_version",
)


def embed_frames(frames: Iterator, encoder, model_version: str, batch_size: int = 256):
    """Map pandas batches of wafer rows to embedding rows (one output frame per input)."""
    import pandas as pd

    from wafer_embeddings.data.common import FAIL, NO_DIE
    from wafer_embeddings.serving.encoder import encode_map

    for pdf in frames:
        maps = [
            np.asarray(f, dtype=np.uint8).reshape(int(h), int(w))
            for h, w, f in zip(pdf["height"], pdf["width"], pdf["wafer_map"])
        ]
        emb = encoder.embed_maps(maps, batch_size=batch_size) if maps else []
        dies = [int((m != NO_DIE).sum()) for m in maps]
        fails = [int((m == FAIL).sum()) for m in maps]
        yield pd.DataFrame(
            {
                "id": pdf["id"].astype("int64").to_numpy(),
                "embedding": [row.astype(np.float32).tolist() for row in emb],
                "map_code": [encode_map(m) for m in maps],
                "label": pdf["label"].to_numpy(),
                "label_id": pdf["label_id"].astype("int32").to_numpy(),
                "split": pdf["split"].to_numpy(),
                "lot": pdf["lot"].to_numpy(),
                "height": pdf["height"].astype("int32").to_numpy(),
                "width": pdf["width"].astype("int32").to_numpy(),
                "n_dies": np.array(dies, dtype=np.int32),
                "fail_frac": np.array(
                    [f / d if d else 0.0 for f, d in zip(fails, dies)], dtype=np.float32
                ),
                "model_version": model_version,
            },
            columns=list(OUTPUT_COLUMNS),
        )


def _args(argv=None):
    p = argparse.ArgumentParser(description="Batch-embed wafer maps with the UC encoder.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--source", default="wafer_maps")
    p.add_argument("--target", default="wafer_map_embeddings")
    p.add_argument("--model", default="wafer_encoder")
    p.add_argument("--alias", default="champion")
    p.add_argument("--partitions", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="Rows to embed (0 = all).")
    return p.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover - needs Spark + UC
    import glob

    import mlflow  # ty: ignore[unresolved-import]
    from databricks.sdk.runtime import spark  # type: ignore
    from pyspark.sql import types as T  # ty: ignore[unresolved-import]

    from wafer_embeddings.obs import get_logger, stage

    a = _args(argv)
    log = get_logger("wafer_embeddings.embed")
    mlflow.set_registry_uri("databricks-uc")
    fqn_model = f"{a.catalog}.{a.schema}.{a.model}"
    version = mlflow.MlflowClient().get_model_version_by_alias(fqn_model, a.alias).version
    with stage(log, f"download {fqn_model} v{version}"):
        local = mlflow.artifacts.download_artifacts(f"models:/{fqn_model}/{version}")
        (ckpt,) = glob.glob(f"{local}/**/*.pt", recursive=True)
        ckpt_bytes = open(ckpt, "rb").read()  # shipped to executors in the UDF closure
    log.info(f"checkpoint {len(ckpt_bytes) / 1e6:.1f} MB from {ckpt}")

    def _udf(frames):
        import io

        import torch

        from wafer_embeddings.serving.encoder import WaferEncoder

        torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
        ck = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu", weights_only=False)
        enc = WaferEncoder(ck["state_dict"], ck["config"])
        yield from embed_frames(frames, enc, str(version))

    schema = T.StructType(
        [
            T.StructField("id", T.LongType(), False),
            T.StructField("embedding", T.ArrayType(T.FloatType()), False),
            T.StructField("map_code", T.StringType(), False),
            T.StructField("label", T.StringType(), True),
            T.StructField("label_id", T.IntegerType(), False),
            T.StructField("split", T.StringType(), False),
            T.StructField("lot", T.StringType(), True),
            T.StructField("height", T.IntegerType(), False),
            T.StructField("width", T.IntegerType(), False),
            T.StructField("n_dies", T.IntegerType(), False),
            T.StructField("fail_frac", T.FloatType(), False),
            T.StructField("model_version", T.StringType(), False),
        ]
    )
    src = spark.read.table(f"{a.catalog}.{a.schema}.{a.source}")
    if a.limit:
        src = src.limit(a.limit)
    target = f"{a.catalog}.{a.schema}.{a.target}"
    with stage(log, f"embed -> {target}"):
        (
            src.repartition(a.partitions)
            .mapInPandas(_udf, schema=schema)
            .write.mode("overwrite")
            .option("overwriteSchema", "true")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(target)
        )
    spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    n = spark.read.table(target).count()
    log.info(f"wrote {n} embeddings (model v{version}) to {target}")


if __name__ == "__main__":  # pragma: no cover
    main()
