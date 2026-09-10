"""Test doubles for eval runs — a canned LLM and the switch that selects it.

RAG Pipeline Position:
    config -> [DOUBLES] -> EvalRunner -> retrieve -> generate -> judge
                  ^^^
    Substituted for the real generator and judge so an eval run exercises the
    full harness without provider calls, cost, or network.

Design Decision:
    This lives in ``src/eval/`` rather than in the CLI because two callers need
    it — the CLI and the HTTP submission path. The HTTP layer used to reach into
    ``src.eval.cli`` for a *private* ``_DummyLLM``, and both callers repeated the
    same environment-variable dispatch. One public home, one dispatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# WHY a module constant: the variable name was previously spelled out at two
#      call sites, so a rename would have silently disabled the double at one
#      of them.
DUMMY_OVERRIDE_ENV = "EVAL_LLM_OVERRIDE_DUMMY"

_JUDGE_RESPONSE = (
    '{"score": 1.0, "claims": [], "chunks": [], '
    '"factual_match": 1.0, "is_refusal": false, "reasoning": "ok"}'
)


class DummyEvalLLM:
    """An LLM stand-in that answers every prompt with canned text.

    Serves as both the generator and the judge: a prompt that asks for JSON (or
    carries a ``"score"`` field) gets a well-formed judge verdict, anything else
    gets a short placeholder answer.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return canned text shaped to whichever role the prompt implies.

        Args:
            prompt: The user prompt the harness would have sent.
            system_prompt: The system prompt, used to detect a judge call.

        Returns:
            A judge verdict as JSON, or a placeholder answer.
        """
        if "JSON" in (system_prompt or "") or '"score"' in prompt:
            return _JUDGE_RESPONSE
        return "<dummy>"


@dataclass(frozen=True)
class LLMOverrides:
    """The generator and judge substitutes an eval run should use, if any."""

    llm: DummyEvalLLM | None = None
    judge_llm: DummyEvalLLM | None = None


def resolve_llm_overrides() -> LLMOverrides:
    """Return the LLM doubles selected by the environment.

    Returns:
        Both slots filled with one shared :class:`DummyEvalLLM` when
        ``EVAL_LLM_OVERRIDE_DUMMY=1``, otherwise both empty so the runner builds
        real handlers.

    WHY one function: the CLI and the HTTP submission path both need this
        decision, and they had drifted into two copies of the same ``if``.
    """
    if os.getenv(DUMMY_OVERRIDE_ENV) != "1":
        return LLMOverrides()
    dummy = DummyEvalLLM()
    return LLMOverrides(llm=dummy, judge_llm=dummy)
