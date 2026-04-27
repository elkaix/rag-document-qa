"""
HTML report renderer — single-run and two-run reports via jinja2.

Eval Harness Position:
  storage.load_run() / compare.compare_runs() → [REPORT] → standalone HTML

Design decisions:
  - Self-contained HTML (inline CSS, no external assets) so a report is
    portable — copyable to a gist, attachable to an email, openable from
    a checked-out repo.
  - jinja2 templates live in templates/eval/ so designers can edit them
    without touching Python.
  - Templates KEEP IT SIMPLE: tables, basic CSS, no JS. The React UI
    (Sub-plan 1C) is the rich interactive surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.eval.schemas import CompareResult

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates" / "eval"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render_run_html(run: dict[str, Any]) -> str:
    """Render a single eval run dict to HTML."""
    template = _env.get_template("run_report.html.j2")
    return template.render(
        metadata=run["metadata"],
        aggregated=run["aggregated"],
        results=run["results"],
        cost=run["cost"],
    )


def render_compare_html(compare: CompareResult) -> str:
    """Render a CompareResult to HTML."""
    template = _env.get_template("compare_report.html.j2")
    return template.render(
        run_a=compare.run_a,
        run_b=compare.run_b,
        deltas=compare.deltas,
        per_question_diff=compare.per_question_diff,
    )
