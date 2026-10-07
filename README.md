# wafer-embeddings

Per-die Vision-Transformer + DINO embedding model for semiconductor wafer maps
(reference dataset: [MixedWM38](https://github.com/Junliangwangdhu/WaferMap)), served into
Lakebase Search. See [`SPEC/SPECS.md`](SPEC/SPECS.md) for the design and
[`SPEC/PLAN.md`](SPEC/PLAN.md) for the risk-first build plan.

## Status

Build in progress — **P0 (reproducible substrate) + P1 (does-it-learn) scaffolding**.
Everything is created programmatically and idempotently (SPECS.md §11.8 / N7); no
click-ops. GPU training is **not** run yet (paused before the first billable GPU run).

## Layout

```
pyproject.toml            # uv project + wheel (hatchling)
databricks.yml            # Databricks Asset Bundle (target: dev → fevm-mmf-mlops-demo)
resources/                # bundle job/endpoint/resource definitions
src/wafer_embeddings/
  config.py               # typed config (model, tokenizer, schemes, eval)
  jobs/                   # Databricks job entry points (bootstrap, ingest, ...)
  data/                   # MixedWM38 parsing, dedup, splits
  tokenize/               # native per-die tokenizer + augmentations
  model/                  # per-die ViT (patch=1, ISAB), DINO
  eval/                   # exact-search retrieval + clustering + health metrics
tests/                    # local unit tests (no Databricks, no GPU)
```

## Local dev

```bash
uv sync --all-extras        # create .venv and install deps
uv run pytest               # run unit tests (CPU-only, no Databricks)
databricks bundle validate -t dev --profile mmf   # validate the bundle config
```

Target workspace: `fevm-mmf-mlops-demo.cloud.databricks.com` · catalog
`mmf_mlops_demo_catalog` · schema `wafer_embeddings`.
