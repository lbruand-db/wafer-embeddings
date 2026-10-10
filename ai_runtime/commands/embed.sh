#!/bin/bash
# Batch-embed every map with wafer_encoder@champion for the `publish` job (GPU).
set -euo pipefail
export PYTHONPATH="${CODE_SOURCE_PATH}/src:${CODE_SOURCE_PATH}"
python -m wafer_embeddings.jobs.embed \
  --catalog mmf_mlops_demo_catalog --schema wafer_embeddings --alias champion
