"""End-to-end CLI tests for src.eval.cli."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str], env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "src.eval.cli", *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def tmp_eval_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "eval_runs"
    runs.mkdir()
    return runs


@pytest.fixture
def synthetic_squad(tmp_path: Path, monkeypatch) -> Path:
    """Write a tiny 3-question synthetic squad set and override the loader's path."""
    from src.eval.schemas import EvalQuestion
    questions = [
        EvalQuestion(
            id=f"q{i}", question=f"What is fact {i}?",
            gold_answer=f"Fact {i}.", gold_chunk_ids=[f"q{i}"],
            metadata={"context": f"Fact {i} is important.", "title": "t"},
        )
        for i in range(3)
    ]
    path = tmp_path / "squad.jsonl"
    with path.open("w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")
    return path


@pytest.fixture
def cli_config(tmp_path: Path, synthetic_squad: Path) -> Path:
    """Write a baseline-shaped YAML config; runner picks it up."""
    import yaml
    config_data = {
        "name": "cli-test", "description": "",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": 3},
            "generator": {"model": "gpt-4.1-nano", "reasoning_model": None},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200"],
            "judge_model": "gpt-4.1-nano",
            "bootstrap_n": 100, "permutation_n": 100, "seed": 42,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))
    return config_path


def _patch_squad_path(env: dict, squad_path: Path) -> dict:
    """Inject a python -c snippet to set DEFAULT_OUTPUT_PATH before runner runs.

    Easier: set EVAL_SQUAD_PATH and have the CLI honor it. We extend the CLI
    so EVAL_SQUAD_PATH overrides the default frozen path.
    """
    env["EVAL_SQUAD_PATH"] = str(squad_path)
    return env


class TestCliRun:
    def test_run_completes_and_prints_run_id(self, tmp_eval_runs, cli_config, synthetic_squad):
        env = {
            "EVAL_RUNS_DIR": str(tmp_eval_runs),
            "EVAL_LLM_OVERRIDE_DUMMY": "1",
            "EVAL_SQUAD_PATH": str(synthetic_squad),
        }
        result = _run_cli(["run", "--config", str(cli_config)], env_overrides=env)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "cli-test" in result.stdout
        # A run dir was created
        run_dirs = list(tmp_eval_runs.iterdir())
        assert len(run_dirs) == 1


class TestCliList:
    def test_list_shows_runs(self, tmp_eval_runs, cli_config, synthetic_squad):
        # First create a run
        env = {
            "EVAL_RUNS_DIR": str(tmp_eval_runs),
            "EVAL_LLM_OVERRIDE_DUMMY": "1",
            "EVAL_SQUAD_PATH": str(synthetic_squad),
        }
        _run_cli(["run", "--config", str(cli_config)], env_overrides=env)

        result = _run_cli(["list"], env_overrides=env)
        assert result.returncode == 0
        assert "cli-test" in result.stdout

    def test_list_empty(self, tmp_eval_runs):
        env = {"EVAL_RUNS_DIR": str(tmp_eval_runs)}
        result = _run_cli(["list"], env_overrides=env)
        assert result.returncode == 0
        # Should print something (header or "No runs"); just confirm no crash
        assert result.stdout != "" or result.stderr == ""


class TestCliShow:
    def test_show_prints_metrics(self, tmp_eval_runs, cli_config, synthetic_squad):
        env = {
            "EVAL_RUNS_DIR": str(tmp_eval_runs),
            "EVAL_LLM_OVERRIDE_DUMMY": "1",
            "EVAL_SQUAD_PATH": str(synthetic_squad),
        }
        run_result = _run_cli(["run", "--config", str(cli_config)], env_overrides=env)
        # Extract run_id from stdout (it's printed somewhere)
        run_id = next(line for line in run_result.stdout.split("\n")
                      if "cli-test" in line and "_" in line).strip().split()[-1]

        result = _run_cli(["show", run_id], env_overrides=env)
        assert result.returncode == 0
        # Show should print at least the run_id and at least one metric name
        assert run_id in result.stdout or "metric" in result.stdout.lower()

    def test_show_with_html_writes_file(self, tmp_eval_runs, cli_config, synthetic_squad):
        env = {
            "EVAL_RUNS_DIR": str(tmp_eval_runs),
            "EVAL_LLM_OVERRIDE_DUMMY": "1",
            "EVAL_SQUAD_PATH": str(synthetic_squad),
        }
        run_result = _run_cli(["run", "--config", str(cli_config)], env_overrides=env)
        run_id = next(line for line in run_result.stdout.split("\n")
                      if "cli-test" in line and "_" in line).strip().split()[-1]

        result = _run_cli(["show", run_id, "--html"], env_overrides=env)
        assert result.returncode == 0
        html_path = tmp_eval_runs / run_id / "report.html"
        assert html_path.exists()
        assert "<table" in html_path.read_text()


class TestCliCompare:
    def test_compare_prints_table(self, tmp_eval_runs, cli_config, synthetic_squad):
        env = {
            "EVAL_RUNS_DIR": str(tmp_eval_runs),
            "EVAL_LLM_OVERRIDE_DUMMY": "1",
            "EVAL_SQUAD_PATH": str(synthetic_squad),
        }
        # Create two runs
        r1 = _run_cli(["run", "--config", str(cli_config)], env_overrides=env)
        r2 = _run_cli(["run", "--config", str(cli_config)], env_overrides=env)
        run_ids = sorted(p.name for p in tmp_eval_runs.iterdir() if p.is_dir())
        assert len(run_ids) == 2

        result = _run_cli(["compare", run_ids[0], run_ids[1]], env_overrides=env)
        assert result.returncode == 0
        # Compare output should mention at least one metric
        assert "recall" in result.stdout.lower() or "delta" in result.stdout.lower()
