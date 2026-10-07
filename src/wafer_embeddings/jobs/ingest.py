"""Ingest MixedWM38 into Delta (SPECS.md §7, §16 item 21).

Row-building is a pure, tested function; download + Spark write are Databricks-side
(``main``). The dataset archive is untrusted data: it is written to the UC volume and
only ever read as arrays — never executed (environment note).
"""

from __future__ import annotations

import argparse

import numpy as np

from wafer_embeddings.data import mixedwm38 as mw


def parsed_to_rows(parsed: mw.ParsedWafers) -> list[dict]:
    """Flatten parsed wafers into Delta-ready row dicts (map stored flattened)."""
    rows = []
    for i in range(parsed.maps.shape[0]):
        m = parsed.maps[i]
        rows.append(
            {
                "id": int(i),
                "height": int(m.shape[0]),
                "width": int(m.shape[1]),
                "wafer_map": m.reshape(-1).astype(np.uint8).tolist(),
                "labels": parsed.labels[i].astype(np.uint8).tolist(),
                "pattern": parsed.pattern_names[i],
                "split": str(parsed.split[i]),
            }
        )
    return rows


def _download(source_url: str, dest_path: str) -> str:  # pragma: no cover - IO
    """Fetch the dataset archive into the UC volume (idempotent)."""
    import os
    import urllib.request

    if os.path.exists(dest_path):
        print(f"[ingest] {dest_path} already present; skipping download")
        return dest_path
    print(f"[ingest] downloading {source_url} -> {dest_path}")
    urllib.request.urlretrieve(source_url, dest_path)  # noqa: S310 (trusted FE source)
    return dest_path


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs Spark
    p = argparse.ArgumentParser(description="Ingest MixedWM38 into Delta.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="raw")
    p.add_argument("--table", default="wafer_maps")
    p.add_argument(
        "--source-url",
        required=True,
        help="MixedWM38 .npz URL (or pre-place the file in the volume and point --npz).",
    )
    p.add_argument("--npz", default=None, help="Path to an already-downloaded .npz.")
    args = p.parse_args(argv)

    from databricks.sdk.runtime import spark  # type: ignore

    vol = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}"
    npz = args.npz or _download(args.source_url, f"{vol}/MixedWM38.npz")
    maps, labels = mw.load_npz(npz)
    parsed = mw.parse_arrays(maps, labels)
    rows = parsed_to_rows(parsed)

    import pandas as pd

    sdf = spark.createDataFrame(pd.DataFrame(rows))
    fqn = f"{args.catalog}.{args.schema}.{args.table}"
    sdf.write.mode("overwrite").saveAsTable(fqn)
    print(f"[ingest] wrote {len(rows)} rows to {fqn}")


if __name__ == "__main__":  # pragma: no cover
    main()
