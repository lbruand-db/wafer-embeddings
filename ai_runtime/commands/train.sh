#!/bin/bash
# DINO pretraining for the `pipeline` job (ai_runtime_task: no parameters, so the R1
# recipe comes from jobs/train.py defaults). The run name is what `register` looks for.
set -euo pipefail
export PYTHONPATH="${CODE_SOURCE_PATH}/src:${CODE_SOURCE_PATH}"
python -m wafer_embeddings.jobs.train \
  --catalog mmf_mlops_demo_catalog --schema wafer_embeddings \
  --run-name pipeline-train
