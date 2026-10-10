#!/bin/bash
# Cheap end-to-end check of the `pipeline` job (deploy with --var train_command=...smoke).
set -euo pipefail
export PYTHONPATH="${CODE_SOURCE_PATH}/src:${CODE_SOURCE_PATH}"
python -m wafer_embeddings.jobs.train \
  --catalog mmf_mlops_demo_catalog --schema wafer_embeddings \
  --run-name pipeline-train --steps 50 --max-train 2000 --track-every 0
