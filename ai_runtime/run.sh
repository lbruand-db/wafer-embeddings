#!/usr/bin/env bash
# Submit an AI Runtime workload with the MLflow experiment in YOUR workspace home.
#
#   ai_runtime/run.sh <workload.yaml> [databricks air run args...]
#   ai_runtime/run.sh ai_runtime/train.yaml --profile mmf \
#     --override env_variables.RUN_NAME=nnpix-k5 \
#     --override "env_variables.EXTRA_ARGS=--nn-positives 5"
#
# The workload YAML has no variable substitution (`${...}` is passed literally), so the
# YAMLs carry a neutral mlflow_experiment_directory and this wrapper overrides it with
# /Workspace/Users/<current user>, the directory the bundle jobs and tools/compare_runs.py
# use. It goes first, so a later --override mlflow_experiment_directory=... still wins.
# Everything else (--profile, --dry-run, --watch, ...) is passed to `databricks air run`.
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: $0 <workload.yaml> [databricks air run args...]" >&2
  exit 2
fi
yaml="$1"
shift

# the profile the run will use, for the current-user lookup
profile=()
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --profile | -p) profile=(--profile "${args[i + 1]:-}") ;;
    --profile=*) profile=(--profile "${args[i]#--profile=}") ;;
  esac
done

user="$(databricks current-user me ${profile[@]+"${profile[@]}"} -o json |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["userName"])')"
if [[ -z "$user" ]]; then
  echo "$0: could not resolve the current user (are you logged in?)" >&2
  exit 1
fi

exec databricks air run --file "$yaml" \
  --override "mlflow_experiment_directory=/Workspace/Users/${user}" "$@"
