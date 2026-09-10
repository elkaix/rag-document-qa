"""Eval run submission — start a run, name it, and report its progress.

Eval Harness Position:
    config name -> [SUBMISSION] -> EvalRunner.run() -> run directory on disk
                        ^^^
    Everything between "a caller asked for a run" and "the runner is executing"
    lives here: locating the config, deriving the run id, selecting test
    doubles, and reporting lifecycle transitions to a progress sink.

What concept it teaches:
    Putting orchestration behind an interface so it is reachable from more than
    one caller. This logic previously lived inside a FastAPI route handler, so
    the *only* way to submit a run was an HTTP request — which is why its tests
    had to boot a TestClient and inject a fake LLM through an environment
    variable, even though EvalRunner accepts one directly.

Why this approach over alternatives:
    The module named ``src/api/services/eval_runs.py`` sounds like it owns this,
    but it is a thread-safe status store — a dict behind a mutex — and never
    imports the eval package. The split was inverted: the "service" held
    bookkeeping while the route held orchestration.

Design Decision:
    The progress sink is a structural Protocol, not the concrete registry, so
    the eval package does not import the API layer. ``RunRegistry`` satisfies it
    as written.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.eval.config import EvalConfig, load_config
from src.eval.doubles import resolve_llm_overrides
from src.eval.runner import EvalRunner
from src.eval.storage import compute_run_id, current_git_sha


# WHY runtime_checkable: matches the project's other seams (Retriever,
#      ProviderAdapter) and lets a test assert conformance directly.
@runtime_checkable
class RunProgressSink(Protocol):
    """Where a submission reports a run's lifecycle.

    ``RunRegistry`` satisfies this structurally. Declaring what is needed rather
    than importing the concrete class keeps the eval package free of any
    dependency on the web layer.
    """

    def register(self, run_id: str, n_total: int) -> None: ...

    def update_progress(
        self, run_id: str, n_completed: int, n_total: int | None = None
    ) -> None: ...

    def mark_completed(self, run_id: str) -> None: ...

    def mark_failed(self, run_id: str, error_message: str) -> None: ...


class ConfigNotFoundError(FileNotFoundError):
    """Raised when a named eval config has no file on disk."""


@dataclass(frozen=True)
class SubmittedRun:
    """The identity a caller needs to follow a run it just started."""

    run_id: str
    config_path: Path


def resolve_config(config_name: str, configs_dir: Path) -> tuple[EvalConfig, Path]:
    """Load the named config from a configs directory.

    Args:
        config_name: Config stem, without the ``.yaml`` suffix.
        configs_dir: Directory holding eval configs.

    Returns:
        The parsed config and the path it came from.

    Raises:
        ConfigNotFoundError: If no such file exists. A distinct type so callers
            can translate it — an HTTP caller into a 404 — without inspecting
            the message.
    """
    path = configs_dir / f"{config_name}.yaml"
    if not path.exists():
        raise ConfigNotFoundError(f"Config '{config_name}' not found in {configs_dir}.")
    return load_config(path), path


def reserve_run_id(config_name: str, started_at: datetime | None = None) -> str:
    """Derive the id a run will be saved under, before it starts.

    Args:
        config_name: Name of the config being run.
        started_at: Submission time. Defaults to now, in UTC.

    Returns:
        The run id, matching what ``EvalRunner`` would derive for itself.

    WHY reserve it up front: a caller that wants to report status must know the
        id before the run begins. This used to be computed here *and* inside the
        runner, from two independent ``datetime.now()`` calls that had to agree
        to the second — an agreement papered over by a ``run_id_override``
        parameter that existed only to reconcile the duplicate.
    """
    return compute_run_id(
        config_name, started_at or datetime.now(timezone.utc), current_git_sha()
    )


def submit_run(
    config_name: str,
    *,
    configs_dir: Path,
    progress: RunProgressSink,
    run_id: str | None = None,
) -> SubmittedRun:
    """Run an evaluation to completion, reporting lifecycle to ``progress``.

    This call is synchronous — it returns when the run has finished or failed.
    Callers that need to return sooner (an HTTP handler) dispatch it to a
    worker; the run id is reserved before dispatch so status can be polled
    immediately.

    Args:
        config_name: Config stem to run.
        configs_dir: Directory holding eval configs.
        progress: Sink receiving register / progress / completed / failed.
        run_id: A previously reserved id. Derived here when omitted.

    Returns:
        The run's identity and the config path it used.

    Raises:
        ConfigNotFoundError: If the named config does not exist. Raised before
            anything is registered, so a bad name leaves no orphan entry.
    """
    config, config_path = resolve_config(config_name, configs_dir)
    resolved_id = run_id or reserve_run_id(config_name)

    # WHY n_total=0: the question count is not known until the runner loads its
    #      datasets. The first progress report carries the real total.
    progress.register(resolved_id, n_total=0)

    overrides = resolve_llm_overrides()
    runner = EvalRunner(
        config,
        config_path=config_path,
        llm_override=overrides.llm,
        judge_llm_override=overrides.judge_llm,
        on_progress=lambda done, total: progress.update_progress(
            resolved_id, done, n_total=total
        ),
        run_id=resolved_id,
    )

    try:
        runner.run()
    except Exception as exc:
        progress.mark_failed(resolved_id, str(exc))
    else:
        progress.mark_completed(resolved_id)

    return SubmittedRun(run_id=resolved_id, config_path=config_path)
