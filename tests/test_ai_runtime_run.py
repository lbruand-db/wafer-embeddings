"""ai_runtime/run.sh: submits `databricks air run` with the experiment in the user's home."""

import os
import stat
import subprocess
from pathlib import Path

RUN = Path(__file__).parents[1] / "ai_runtime" / "run.sh"

# A fake `databricks`: `current-user me` prints a user (and records its args); `air run`
# records its argv, one argument per line.
FAKE = """#!/usr/bin/env bash
if [[ "$1" == current-user ]]; then
  printf '%s\\n' "$@" > "$LOG.me"
  echo '{"userName": "a@b.com"}'
else
  printf '%s\\n' "$@" > "$LOG.air"
fi
"""


def _run(tmp_path, *args):
    fake = tmp_path / "databricks"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}", "LOG": str(tmp_path / "log")}
    res = subprocess.run(["bash", str(RUN), *args], env=env, capture_output=True, text=True)
    me = tmp_path / "log.me"
    air = tmp_path / "log.air"
    return (
        res,
        me.read_text().splitlines() if me.exists() else [],
        air.read_text().splitlines() if air.exists() else [],
    )


def test_overrides_the_experiment_directory_first(tmp_path):
    res, me, air = _run(tmp_path, "w.yaml", "--profile", "mmf", "--override", "x=a b", "--dry-run")
    assert res.returncode == 0, res.stderr
    assert me[:4] == ["current-user", "me", "--profile", "mmf"]  # the run's profile
    assert air == [
        "air",
        "run",
        "--file",
        "w.yaml",
        "--override",
        "mlflow_experiment_directory=/Workspace/Users/a@b.com",
        "--profile",
        "mmf",
        "--override",
        "x=a b",  # quoting survives
        "--dry-run",
    ]


def test_profile_equals_form_and_no_profile(tmp_path):
    _, me, _ = _run(tmp_path, "w.yaml", "--profile=dev")
    assert me[:4] == ["current-user", "me", "--profile", "dev"]
    _, me, air = _run(tmp_path, "w.yaml")
    assert me == ["current-user", "me", "-o", "json"]  # default profile / env
    assert air[-1] == "mlflow_experiment_directory=/Workspace/Users/a@b.com"


def test_usage_without_a_yaml(tmp_path):
    res, _, air = _run(tmp_path, "--profile", "mmf")
    assert res.returncode == 2 and "usage" in res.stderr and air == []
