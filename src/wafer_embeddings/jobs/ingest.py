"""Ingest WM-811K into Delta (SPECS.md §7, §16 item 21).

Downloads the WM-811K archive, loads ``LSWMD.pkl`` (a pickled pandas DataFrame),
parses variable-size maps + single-label classes (dedup + leakage-safe splits), and
writes a Delta table. Row-building is a pure, tested function; download / unzip /
pickle-load / Spark write happen Databricks-side in ``main``.

SECURITY: ``LSWMD.pkl`` is a Python pickle (arbitrary-code on load) and the archive is
untrusted data — it is fetched and read only on the Databricks side, never locally.
"""

from __future__ import annotations

import argparse

import numpy as np

from wafer_embeddings.data import wm811k as wm

WM811K_URL = "http://mirlab.org/dataSet/public/MIR-WM811K.zip"


def wm811k_rows(parsed: wm.ParsedWM811K) -> list[dict]:
    """Flatten parsed WM-811K wafers into Delta-ready row dicts."""
    rows = []
    for i, m in enumerate(parsed.maps):
        rows.append(
            {
                "id": int(i),
                "height": int(m.shape[0]),
                "width": int(m.shape[1]),
                "wafer_map": m.reshape(-1).astype(np.uint8).tolist(),
                "label": parsed.label_names[i],  # None for unlabeled
                "label_id": int(parsed.label_ids[i]),  # -1 for unlabeled
                "split": str(parsed.split[i]),
            }
        )
    return rows


def _ensure_pickle(source_url: str, vol: str) -> str:  # pragma: no cover - IO
    """Download + unzip the WM-811K archive into the volume; return the .pkl path."""
    import os
    import urllib.request
    import zipfile

    zip_path = f"{vol}/MIR-WM811K.zip"
    if not os.path.exists(zip_path):
        print(f"[ingest] downloading {source_url} -> {zip_path}")
        urllib.request.urlretrieve(source_url, zip_path)  # noqa: S310
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        pkls = [m for m in members if m.lower().endswith(".pkl")]
        if not pkls:
            raise RuntimeError(f"no .pkl in archive; contents={members}")
        zf.extract(pkls[0], vol)
        return f"{vol}/{pkls[0]}"


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs Spark
    p = argparse.ArgumentParser(description="Ingest WM-811K into Delta.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="raw")
    p.add_argument("--table", default="wafer_maps")
    p.add_argument("--source-url", default=WM811K_URL)
    p.add_argument("--pkl", default=None, help="Path to an already-extracted LSWMD.pkl.")
    p.add_argument("--limit", type=int, default=0, help="Rows to ingest (0 = all).")
    args = p.parse_args(argv)

    import pandas as pd  # ty: ignore[unresolved-import]
    from databricks.sdk.runtime import spark  # type: ignore

    vol = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}"
    pkl = args.pkl or _ensure_pickle(args.source_url, vol)
    print(f"[ingest] reading {pkl}")
    df = pd.read_pickle(pkl)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"[ingest] {len(df)} raw rows; columns={list(df.columns)}")

    parsed = wm.parse_records(df["waferMap"].to_list(), df["failureType"].to_list())
    rows = wm811k_rows(parsed)
    labeled = int((parsed.label_ids >= 0).sum())
    print(f"[ingest] parsed {len(rows)} unique maps ({labeled} labeled)")

    sdf = spark.createDataFrame(pd.DataFrame(rows))
    fqn = f"{args.catalog}.{args.schema}.{args.table}"
    sdf.write.mode("overwrite").saveAsTable(fqn)
    print(f"[ingest] wrote {len(rows)} rows to {fqn}")


if __name__ == "__main__":  # pragma: no cover
    main()
