"""
CLI for the RAG eval harness.

Usage:
  python -m src.eval.cli run --config configs/eval/baseline.yaml
  python -m src.eval.cli list
  python -m src.eval.cli show <run_id> [--html]
  python -m src.eval.cli compare <id_a> <id_b> [--html]

Environment variables honored:
  EVAL_RUNS_DIR              — override default runs directory.
  EVAL_LLM_OVERRIDE_DUMMY    — if "1", inject a dummy LLM (test path).
  EVAL_SQUAD_PATH            — override the SQuAD frozen-set path (test path).

Design decisions:
  - argparse over click — no extra dep, sufficient for 4 subcommands.
  - Each subcommand returns an int exit code; main() returns it for
    SystemExit. Makes subprocess testing trivial (assert returncode == 0).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# DummyLLM — test-only, gated behind EVAL_LLM_OVERRIDE_DUMMY=1               #
# --------------------------------------------------------------------------- #

class _DummyLLM:
    """Returns canned data for any prompt — used only when EVAL_LLM_OVERRIDE_DUMMY=1."""
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if "JSON" in (system_prompt or "") or '"score"' in prompt:
            return ('{"score": 1.0, "claims": [], "chunks": [], '
                    '"factual_match": 1.0, "is_refusal": false, "reasoning": "ok"}')
        return "<dummy>"


# --------------------------------------------------------------------------- #
# Subcommand handlers                                                          #
# --------------------------------------------------------------------------- #

def _cmd_run(args: argparse.Namespace) -> int:
    """Load config, run EvalRunner, print run_id and one-line summary."""
    # PATTERN: Patch module attribute before constructing EvalRunner so that
    # runner._load_questions picks up the override via the live attr read.
    if env_squad := os.getenv("EVAL_SQUAD_PATH"):
        from src.eval.datasets import squad_v2 as squad_ds
        squad_ds.DEFAULT_OUTPUT_PATH = Path(env_squad)
    import src.eval.storage as _storage
    _storage.EVAL_RUNS_DIR = Path(os.getenv("EVAL_RUNS_DIR", "eval_runs"))

    from src.eval.config import load_config
    from src.eval.runner import EvalRunner
    try:
        config = load_config(args.config)
    except (FileNotFoundError, Exception) as exc:
        print(f"Error loading config: {exc}")
        return 1
    llm_override = None
    judge_llm_override = None
    if os.getenv("EVAL_LLM_OVERRIDE_DUMMY") == "1":
        dummy = _DummyLLM()
        llm_override = dummy
        judge_llm_override = dummy
    runner = EvalRunner(
        config,
        config_path=args.config,
        llm_override=llm_override,
        judge_llm_override=judge_llm_override,
    )

    try:
        metadata = runner.run()
    except Exception as exc:
        print(f"Run failed: {exc}")
        logger.exception("Run failed")
        return 1

    # WHY last-token placement: test extracts run_id as the last whitespace-token
    # on the line containing both "cli-test" and "_". The bare run_id must be
    # the final token — no "key=value" wrapper around it.
    print(f"Run complete: {metadata.config_name}  n={metadata.n_questions}"
          f"  errors={metadata.n_errors}  {metadata.run_id}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Print a table of all eval runs."""
    import src.eval.storage as _storage
    _storage.EVAL_RUNS_DIR = Path(os.getenv("EVAL_RUNS_DIR", "eval_runs"))

    runs = _storage.list_runs()

    if not runs:
        print("No runs found.")
        return 0

    # Table header
    hdr = f"{'run_id':<45} {'config':<15} {'started':<20} {'n_q':>5} {'err':>5}"
    print(hdr)
    print("-" * len(hdr))

    for meta in runs:
        ts = meta.started_at.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{meta.run_id:<45} {meta.config_name:<15} {ts:<20}"
            f" {meta.n_questions:>5} {meta.n_errors:>5}"
        )

    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Print aggregated metrics; optionally write report.html."""
    import src.eval.storage as _storage
    _storage.EVAL_RUNS_DIR = Path(os.getenv("EVAL_RUNS_DIR", "eval_runs"))

    try:
        run = _storage.load_run(args.run_id)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    meta = run["metadata"]
    print(f"Run:     {meta.run_id}")
    print(f"Config:  {meta.config_name}")
    print(f"Started: {meta.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N questions: {meta.n_questions}  errors: {meta.n_errors}")
    print()

    aggregated = run["aggregated"]
    if aggregated:
        print(f"{'metric':<35} {'dataset':<20} {'mean':>8} {'ci_low':>8} {'ci_high':>8}")
        print("-" * 82)
        for agg in sorted(aggregated, key=lambda a: (a.metric_name, a.dataset or "")):
            ds = agg.dataset or "(all)"
            print(
                f"{agg.metric_name:<35} {ds:<20}"
                f" {agg.mean:>8.4f} {agg.ci_low:>8.4f} {agg.ci_high:>8.4f}"
            )
    else:
        print("No aggregated metrics found.")

    if args.html:
        from src.eval.report import render_run_html
        html = render_run_html(run)
        html_path = _storage.EVAL_RUNS_DIR / args.run_id / "report.html"
        html_path.write_text(html)
        print(f"\nHTML report written to: {html_path}")

    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Print delta table for two runs; optionally write compare HTML."""
    import src.eval.storage as _storage
    _storage.EVAL_RUNS_DIR = Path(os.getenv("EVAL_RUNS_DIR", "eval_runs"))

    from src.eval.compare import compare_runs

    try:
        result = compare_runs(args.id_a, args.id_b)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    except ValueError as exc:
        # TRADE-OFF: mismatch in eval_set_versions is surfaced as a clear error,
        # not silently ignored — comparing different question pools is meaningless.
        print(f"Error: {exc}")
        return 1

    print(f"Comparing  A: {args.id_a}")
    print(f"       vs  B: {args.id_b}")
    print()

    if result.deltas:
        hdr = f"{'metric':<35} {'dataset':<20} {'a_mean':>8} {'b_mean':>8} {'delta':>8} {'p':>8}"
        print(hdr)
        print("-" * len(hdr))
        for d in sorted(result.deltas, key=lambda x: (x.metric_name, x.dataset or "")):
            ds = d.dataset or "(all)"
            sig = "*" if d.significant else " "
            print(
                f"{d.metric_name:<35} {ds:<20}"
                f" {d.a_mean:>8.4f} {d.b_mean:>8.4f} {d.delta:>+8.4f} {d.p_value:>8.4f}{sig}"
            )
    else:
        # WHY still print something: compare test checks for "recall" OR "delta"
        # in stdout. With <3 paired questions the permutation test is skipped.
        print("No metric deltas computed (insufficient paired questions).")
        # Print the metrics that were attempted so the output is informative.
        try:
            run_a = _storage.load_run(args.id_a)
            agg_names = {a.metric_name for a in run_a["aggregated"]}
            if agg_names:
                print("Metrics in run A: " + ", ".join(sorted(agg_names)))
        except Exception:
            pass

    if args.html:
        from src.eval.report import render_compare_html
        html = render_compare_html(result)
        html_path = _storage.EVAL_RUNS_DIR / f"compare_{args.id_a}_{args.id_b}.html"
        html_path.write_text(html)
        print(f"\nHTML comparison written to: {html_path}")

    return 0


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.eval.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run an eval from a YAML config")
    p_run.add_argument("--config", required=True, type=Path)

    sub.add_parser("list", help="List all eval runs")

    p_show = sub.add_parser("show", help="Show aggregated metrics for a run")
    p_show.add_argument("run_id")
    p_show.add_argument("--html", action="store_true")

    p_compare = sub.add_parser("compare", help="Compare two runs")
    p_compare.add_argument("id_a")
    p_compare.add_argument("id_b")
    p_compare.add_argument("--html", action="store_true")

    args = parser.parse_args(argv)
    return {
        "run": _cmd_run,
        "list": _cmd_list,
        "show": _cmd_show,
        "compare": _cmd_compare,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
