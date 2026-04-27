"""
EvalRunner — orchestrates a full evaluation run end-to-end.

Eval Harness Position:
  EvalConfig → [EvalRunner] → eval_runs/<run_id>/{metadata,questions,metrics,cost,config}

Lifecycle:
  1. Resolve git SHA, env hash, run_id.
  2. For each dataset: load questions → build pipeline → query+score each →
     teardown.
  3. Aggregate metrics with bootstrap CIs.
  4. Persist via storage.save_run.

Design decisions:
  - Per-question try/except so one broken question doesn't kill the run.
    Failures land in EvalResult.error and count toward RunMetadata.n_errors.
  - llm_override / judge_llm_override let tests inject DummyLLM without
    touching real API calls.
  - on_progress callback for the API's status polling endpoint
    (added in Sub-plan 1C).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from src.eval import storage as _storage
from src.eval.aggregator import aggregate
from src.eval.config import EvalConfig
from src.eval.datasets import ml_papers as ml_papers_ds
from src.eval.datasets import squad_v2 as squad_ds
from src.eval.metrics.generation import (
    answer_correctness,
    context_recall,
    judge_answer_relevancy,
    judge_context_precision,
    judge_faithfulness,
)
from src.eval.metrics.operational import aggregate_costs, aggregate_tokens
from src.eval.metrics.refusal import refusal_correctness
from src.eval.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k
from src.eval.pipeline_factory import build_pipeline
from src.eval.schemas import EvalQuestion, EvalResult, RunMetadata
from src.eval.storage import compute_run_id, save_run

logger = logging.getLogger(__name__)

_K_VALUES = [1, 3, 5, 10]


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file's bytes. Returns 'unknown' if missing."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unknown"


def _score_question(
    question: EvalQuestion,
    chunks: list,
    answer: str,
    judge_llm: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute all applicable metrics for one question.

    WHY extracted: keeps the per-question try/except in run() slim and
    makes metric logic independently testable.

    Args:
        question: The gold-labeled question.
        chunks: SearchResult list returned by pipeline.query().
        answer: Generated answer string.
        judge_llm: LLM judge (real or test double).

    Returns:
        Tuple of (metrics dict, metric_details dict).
    """
    retrieved_ids = [r.chunk_id for r in chunks]
    retrieved_texts = [r.content for r in chunks]

    metrics: dict[str, float] = {}
    details: dict[str, Any] = {}

    # --- Always: refusal correctness ---
    metrics["refusal_correctness"] = refusal_correctness(
        answer, question.is_unanswerable, judge_llm
    )

    # --- Retrieval metrics (only when gold chunk IDs are known) ---
    if question.gold_chunk_ids:
        for k in _K_VALUES:
            metrics[f"recall_at_{k}"] = recall_at_k(question.gold_chunk_ids, retrieved_ids, k)
            metrics[f"mrr_at_{k}"] = mrr_at_k(question.gold_chunk_ids, retrieved_ids, k)
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(question.gold_chunk_ids, retrieved_ids, k)

        metrics["context_recall"] = context_recall(question.gold_chunk_ids, retrieved_ids)

        # LLM-judge generation metrics
        faith_score, faith_details = judge_faithfulness(answer, retrieved_texts, judge_llm)
        metrics["judge_faithfulness"] = faith_score
        details["judge_faithfulness"] = faith_details

        cp_score, cp_details = judge_context_precision(question.question, retrieved_texts, judge_llm)
        metrics["judge_context_precision"] = cp_score
        details["judge_context_precision"] = cp_details

        ar_score, ar_details = judge_answer_relevancy(question.question, answer, judge_llm)
        metrics["judge_answer_relevancy"] = ar_score
        details["judge_answer_relevancy"] = ar_details

    # --- Answer correctness (only when gold answer is known) ---
    if question.gold_answer:
        ac_score, ac_details = answer_correctness(answer, question.gold_answer, judge_llm)
        metrics["answer_correctness"] = ac_score
        details["answer_correctness"] = ac_details

    return metrics, details


class EvalRunner:
    """Orchestrates a full end-to-end evaluation run from config to disk.

    Teaches: the orchestrator pattern — EvalRunner owns the lifecycle
    (load → ingest → query → score → aggregate → persist) but delegates
    each step to specialized components. This keeps each component
    independently testable and swappable.

    Pipeline position: TOP-LEVEL — receives EvalConfig, produces a
    run directory under EVAL_RUNS_DIR with five well-known files.
    """

    def __init__(
        self,
        config: EvalConfig,
        *,
        config_path: Path | None = None,
        llm_override: object | None = None,
        judge_llm_override: object | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        run_id_override: str | None = None,
    ) -> None:
        self._config = config
        self._config_path = str(config_path) if config_path else f"<inline:{config.name}>"
        self._llm_override = llm_override
        self._judge_llm_override = judge_llm_override
        self._on_progress = on_progress
        # WHY run_id_override: the API pre-computes the run_id so it can register
        # the run in RunRegistry BEFORE the runner starts (enabling status polling).
        # When set, we use this id instead of computing one from timestamp+sha.
        self._run_id_override = run_id_override

    def run(self) -> RunMetadata:
        """Execute the full eval lifecycle and return run provenance.

        Returns:
            RunMetadata with run_id, timing, error counts, and warnings.
        """
        config = self._config
        started_at = datetime.now(timezone.utc)

        # --- Git SHA ---
        # WHY try/except: the harness may run outside a git repo (CI containers,
        # zip-extracted deployments). Fall back to 'unknown' rather than crashing.
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            git_sha = "unknown"

        # --- Env hash (requirements.txt fingerprint) ---
        env_hash = _sha256_of_file(Path("requirements.txt"))[:16]

        # --- Run ID and directory ---
        # WHY: If run_id_override is set (from the API route), use it directly.
        # This ensures the registered registry run_id matches the saved directory.
        run_id = self._run_id_override or compute_run_id(config.name, started_at, git_sha)
        # WHY _storage.EVAL_RUNS_DIR at call time: the fixture reloads storage
        # after setting EVAL_RUNS_DIR env var, but runner's top-level import
        # already bound the old value. Reading from the live module attribute
        # ensures we pick up the reloaded (test-patched) path.
        run_dir = _storage.EVAL_RUNS_DIR / run_id

        # --- Eval-set version fingerprints ---
        # WHY live attribute read: squad_5 fixture patches DEFAULT_OUTPUT_PATH
        # after import; reading the module attribute picks up the patched value.
        eval_set_versions: dict[str, str] = {}
        for dataset_name in config.eval.datasets:
            if dataset_name == "squad_v2_dev_200":
                fp = squad_ds.DEFAULT_OUTPUT_PATH
            elif dataset_name == "ml_papers_v1":
                fp = ml_papers_ds.DEFAULT_QUESTIONS_PATH
            else:
                fp = None
            eval_set_versions[dataset_name] = (
                _sha256_of_file(fp)[:16] if fp is not None else "unknown"
            )

        # --- Load all datasets up front so we know total_questions ---
        # WHY pre-load: the progress callback needs total before the first
        # on_progress(1, total) call. Eager load also surfaces missing files
        # before any pipeline work starts.
        dataset_questions: dict[str, list[EvalQuestion]] = {}
        for dataset_name in config.eval.datasets:
            qs = self._load_questions(dataset_name)
            dataset_questions[dataset_name] = qs

        total_questions = sum(len(qs) for qs in dataset_questions.values())

        # --- Per-dataset pipeline loop ---
        all_results: list[EvalResult] = []

        for dataset_name, questions in dataset_questions.items():
            pipeline = build_pipeline(
                config,
                dataset_name,
                llm_override=self._llm_override,
                judge_llm_override=self._judge_llm_override,
            )
            try:
                pipeline.ingest(questions)
                for question in questions:
                    result = self._run_question(
                        question, dataset_name, pipeline, pipeline.judge_llm
                    )
                    all_results.append(result)
                    if self._on_progress is not None:
                        self._on_progress(len(all_results), total_questions)
            finally:
                # WHY finally: ensures teardown even if a question raises
                # an unhandled exception outside the per-question try block.
                pipeline.teardown()

        # --- Aggregate and persist ---
        aggregated, warnings = aggregate(all_results, config)
        cost_summary = {**aggregate_costs(all_results), **aggregate_tokens(all_results)}
        finished_at = datetime.now(timezone.utc)

        metadata = RunMetadata(
            run_id=run_id,
            config_name=config.name,
            config_path=self._config_path,
            git_sha=git_sha,
            started_at=started_at,
            finished_at=finished_at,
            env_hash=env_hash,
            eval_set_versions=eval_set_versions,
            n_questions=len(all_results),
            n_errors=sum(1 for r in all_results if r.error),
            warnings=warnings,
        )

        config_yaml_text = yaml.safe_dump(config.model_dump())
        save_run(run_dir, metadata, all_results, aggregated, cost_summary, config_yaml_text)

        return metadata

    def _load_questions(self, dataset_name: str) -> list[EvalQuestion]:
        """Load questions for one dataset, with graceful fallback for empty sets.

        WHY live attribute reads (e.g. squad_ds.DEFAULT_OUTPUT_PATH): test
        fixtures patch the module attribute after import; calling
        load_frozen(squad_ds.DEFAULT_OUTPUT_PATH) reads the patched value,
        whereas load_frozen() would use the default arg bound at def-time.
        """
        if dataset_name == "squad_v2_dev_200":
            try:
                return squad_ds.load_frozen(squad_ds.DEFAULT_OUTPUT_PATH)
            except FileNotFoundError as exc:
                logger.warning("SQuAD dataset not found: %s — skipping.", exc)
                return []

        if dataset_name == "ml_papers_v1":
            try:
                questions = ml_papers_ds.load_questions(ml_papers_ds.DEFAULT_QUESTIONS_PATH)
            except FileNotFoundError as exc:
                logger.warning("ML Papers questions not found: %s — skipping.", exc)
                return []
            if not questions:
                logger.warning("ML Papers dataset is empty (skeleton state) — skipping.")
            return questions

        logger.warning("Unknown dataset %r — skipping.", dataset_name)
        return []

    def _run_question(
        self,
        question: EvalQuestion,
        dataset_name: str,
        pipeline: Any,
        judge_llm: Any,
    ) -> EvalResult:
        """Query the pipeline and score one question; capture any exception as error.

        WHY outer try/except: a single malformed question or judge response
        must not abort the entire run. The error lands in EvalResult.error
        and increments RunMetadata.n_errors; the run continues.
        """
        try:
            chunks, answer, telemetry = pipeline.query(question.question)
            metrics, metric_details = _score_question(question, chunks, answer, judge_llm)
            return EvalResult(
                question_id=question.id,
                dataset=dataset_name,
                retrieved_chunk_ids=[r.chunk_id for r in chunks],
                retrieved_chunks=[r.content for r in chunks],
                generated_answer=answer,
                metrics=metrics,
                metric_details=metric_details,
                timings_ms=telemetry["timings_ms"],
                tokens=telemetry["tokens"],
                cost_usd=telemetry["cost_usd"],
                error=None,
            )
        except Exception as exc:
            logger.exception("Error on question %s: %s", question.id, exc)
            return EvalResult(
                question_id=question.id,
                dataset=dataset_name,
                retrieved_chunk_ids=[],
                retrieved_chunks=[],
                generated_answer="",
                metrics={},
                metric_details={},
                timings_ms={},
                tokens={},
                cost_usd=0.0,
                error=str(exc),
            )
