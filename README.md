# wafer-embeddings

[![CI](https://github.com/lbruand-db/wafer-embeddings/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand-db/wafer-embeddings/actions/workflows/ci.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![types: ty](https://img.shields.io/badge/types-ty-261230.svg)](https://github.com/astral-sh/ty)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Databricks Asset Bundle](https://img.shields.io/badge/Databricks-Asset%20Bundle-FF3621?logo=databricks&logoColor=white)](databricks.yml)
[![AI Runtime](https://img.shields.io/badge/Databricks%20AI%20Runtime-serverless%20GPU-FF3621?logo=databricks&logoColor=white)](SPEC/SPECS.md#11-databricks-implementation-stack)
[![Lakebase Search](https://img.shields.io/badge/Lakebase%20Search-planned-FF3621?logo=databricks&logoColor=white)](SPEC/SPECS.md#9-lakebase-search-integration)

Per-die Vision-Transformer + DINO self-supervised embedding model for semiconductor
wafer maps, trained on Databricks AI Runtime serverless GPU and served into Lakebase
Search. Primary dataset:
[WM-811K](http://mirlab.org/dataSet/public/MIR-WM811K.zip) (~811K real maps, variable
die-grid sizes, 9 defect classes on the labeled subset). See
[`SPEC/SPECS.md`](SPEC/SPECS.md) for the design and [`SPEC/PLAN.md`](SPEC/PLAN.md) for
the risk-first build plan.

## Why per-die tokens

Wafer maps vary wildly in size (a handful of dies up to tens of thousands) and every
die matters, so instead of resizing to a fixed image we treat **each on-wafer die as one
token** (patch=1). The encoder is then a Set Transformer over dies: coordinate-based
positional encoding (center/radius-normalized Fourier features, generalizes to unseen
grid sizes), a learned per-die state embedding, and permutation/padding-invariant
attention. Full self-attention is O(N²); for WM-811K's giant maps we use **ISAB** induced
attention (O(N·m)) plus a per-wafer token cap to bound GPU memory. Training is **DINO**
self-distillation (no labels); labels are used only for Gate-G1 evaluation.

## Status

**P0 (reproducible substrate) + P1 (does-it-learn) — code-complete, 73 unit tests green
on CPU, and the GPU training path is proven end-to-end on AI Runtime** (CUDA A10, parquet
read from UC volume, DINO train → embed → Gate-G1 eval → MLflow). Everything is created
programmatically and idempotently (SPECS.md §11.8 / N7); no click-ops. Each billable
Databricks action is gated behind explicit confirmation (§16).

Built and tested:
- `data/` — WM-811K streaming parse (`LSWMD.pkl`), label normalization, dedup,
  content-hash leakage-safe splits (MixedWM38 loader retained for reference)
- `tokenize/` — native per-die tokenizer, token cap for giant maps, die-preserving
  (orientation-invariant) augmentations
- `model/` — per-die ViT (patch=1), full-attention + ISAB, DINO head (weight-normed
  prototypes) / loss / EMA teacher, gradient checkpointing
- `train/` — device-agnostic DINO training loop with LR / teacher-temp / momentum
  schedules; embedding + Gate-G1 metrics (runs on CPU for tests, CUDA on AI Runtime)
- `eval/` — exact-search retrieval (mAP/nDCG/recall@k), clustering (ARI/NMI/silhouette),
  kNN probe, collapse / alignment-uniformity diagnostics
- `jobs/` — idempotent `bootstrap` (UC schema/volume) + streaming WM-811K `ingest`
  (Delta + parquet export) + `train` (AI Runtime GPU entry point)

Next (each billable → confirm first): a full **Gate-G1** training run (depth-12, longer
schedule) to clear the provisional targets (mAP@10 ≥ 0.80, NMI ≥ 0.85, no collapse), then
Lakebase Search integration (P2+).

## Layout

```
pyproject.toml            # uv project + wheel (hatchling)
databricks.yml            # Databricks Asset Bundle (target: dev → fevm-mmf-mlops-demo)
resources/                # bundle job definitions (bootstrap, ingest)
ai_runtime/train.yaml     # AI Runtime serverless-GPU training workload (databricks air)
src/wafer_embeddings/
  obs.py                  # logging / progress / timing helpers
  jobs/                   # entry points: bootstrap, ingest (WM-811K), train (GPU)
  data/                   # WM-811K parsing, dedup, leakage-safe splits
  tokenize/               # native per-die tokenizer + token cap + augmentations
  model/                  # per-die ViT (patch=1, ISAB), DINO head/loss/EMA
  train/                  # DINO training loop + schedules + Gate-G1 eval
  eval/                   # exact-search retrieval + clustering + health metrics
tests/                    # local unit tests (no Databricks, no GPU) — 73 tests
```

## Local dev

```bash
uv sync                     # create .venv and install deps (+ dev: black, ty, pytest)
uv run black --check .      # formatting (line length 100)
uv run ty check             # type checking (Astral ty)
uv run pytest               # unit tests (CPU-only, no Databricks) — 73 tests
databricks bundle validate -t dev --profile mmf   # validate the bundle config
```

CI (`.github/workflows/ci.yml`) runs black + ty + pytest on every push/PR.

## Databricks pipeline

```bash
# 1. Deploy the bundle (jobs + artifacts) to the dev target
databricks bundle deploy -t dev --profile mmf

# 2. Create UC schema + volume (idempotent)
databricks bundle run bootstrap -t dev --profile mmf

# 3. Ingest WM-811K → Delta + parquet export (streaming; var.ingest_limit=0 ingests all)
databricks bundle run ingest -t dev --profile mmf

# 4. DINO pretrain + Gate-G1 eval on serverless GPU (AI Runtime, no Spark session)
databricks air submit --profile mmf ai_runtime/train.yaml
```

The training job reads the parquet export from the UC volume (AI Runtime has no Spark
session), pretrains the per-die ViT with DINO, embeds the labeled subset, logs loss /
collapse / G1 metrics to MLflow, and saves the encoder weights. `ai_runtime/train.yaml`
documents smoke-test overrides (`--override env_variables.STEPS=50 ...`).

Target workspace: `fevm-mmf-mlops-demo.cloud.databricks.com` · catalog
`mmf_mlops_demo_catalog` · schema `wafer_embeddings`.
