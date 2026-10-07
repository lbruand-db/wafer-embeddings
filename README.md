# wafer-embeddings

[![CI](https://github.com/lbruand-db/wafer-embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand-db/wafer-embeddings/actions/workflows/ci.yml)

Per-die Vision-Transformer + DINO embedding model for semiconductor wafer maps
(reference dataset: [MixedWM38](https://github.com/Junliangwangdhu/WaferMap)), served into
Lakebase Search. See [`SPEC/SPECS.md`](SPEC/SPECS.md) for the design and
[`SPEC/PLAN.md`](SPEC/PLAN.md) for the risk-first build plan.

## Status

**P0 (reproducible substrate) + P1 (does-it-learn) scaffolding — code-complete,
unit-tested on CPU (54 tests), nothing run on Databricks yet.** Paused before the first
billable action (SPECS.md §16 gates). Everything is created programmatically and
idempotently (§11.8 / N7); no click-ops.

Built and tested:
- `data/` — MixedWM38 parse, dedup, leakage-safe splits
- `tokenize/` — native per-die tokenizer + die-preserving augmentations
- `model/` — per-die ViT (patch=1), full-attention + ISAB, DINO head/loss/EMA
- `train/` — device-agnostic DINO training step (runs on CPU)
- `eval/` — exact-search retrieval, clustering, collapse/alignment-uniformity
- `jobs/` — idempotent `bootstrap` + MixedWM38 `ingest` (bundle jobs, not yet run)

Next (each billable → confirm first): `bundle deploy`, run `bootstrap`, set
`var.mixedwm38_url` + run `ingest`, then the first GPU DINO training run (Gate G1).

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
uv sync                     # create .venv and install deps (+ dev: black, ty, pytest)
uv run black --check .      # formatting
uv run ty check             # type checking (Astral ty)
uv run pytest               # unit tests (CPU-only, no Databricks)
databricks bundle validate -t dev --profile mmf   # validate the bundle config
```

CI (`.github/workflows/ci.yml`) runs black + ty + pytest on every push/PR.

Target workspace: `fevm-mmf-mlops-demo.cloud.databricks.com` · catalog
`mmf_mlops_demo_catalog` · schema `wafer_embeddings`.
