"""
Model pricing table and cost arithmetic.

Eval Harness Position:
  Pipeline → tokens → [PRICING] → cost_usd → EvalResult.cost_usd
                       ^^^^^^^^
  Pure data + a 4-line function. Hard-coded prices are fine for Phase 1;
  if prices change frequently we move to a JSON file in a later phase.

Design decisions:
  - Hard-coded table, not env-driven, so price changes are visible in
    git diffs and reviewable in PRs.
  - Unknown model returns 0.0 (with a logged warning) rather than
    raising — eval should not crash because of an outdated price table;
    it should surface the gap in logs.
  - Prices in USD per 1M tokens (industry-standard quoting unit).
  - Prices intentionally MAY be slightly stale; this is a portfolio
    project, not a billing system. Update them when convenient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens, separate prompt and completion rates."""

    prompt_per_1m: float
    completion_per_1m: float


# WHY hard-coded: see module docstring. Update as needed; rare event.
# Source for rates: each provider's public pricing page as of 2026-04.
MODEL_PRICES: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-5-mini": ModelPrice(prompt_per_1m=0.25, completion_per_1m=2.00),
    "gpt-5-nano": ModelPrice(prompt_per_1m=0.05, completion_per_1m=0.40),
    "gpt-4.1-mini": ModelPrice(prompt_per_1m=0.40, completion_per_1m=1.60),
    "gpt-4.1-nano": ModelPrice(prompt_per_1m=0.10, completion_per_1m=0.40),
    "gpt-4o-mini": ModelPrice(prompt_per_1m=0.15, completion_per_1m=0.60),
    # Anthropic
    "claude-haiku-4-5": ModelPrice(prompt_per_1m=1.00, completion_per_1m=5.00),
    "claude-sonnet-4-6": ModelPrice(prompt_per_1m=3.00, completion_per_1m=15.00),
    # GLM (Zhipu)
    "glm-5.1": ModelPrice(prompt_per_1m=0.50, completion_per_1m=1.50),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute total cost in USD for one inference call.

    Args:
        model: Model id matching a key in :data:`MODEL_PRICES`.
        prompt_tokens: Input/prompt token count, ``>= 0``.
        completion_tokens: Output/completion token count, ``>= 0``.

    Returns:
        Total cost in USD. ``0.0`` if the model is unknown (a warning
        is logged) — the eval should not crash on missing prices.

    Raises:
        ValueError: If either token count is negative.
    """
    # PATTERN: validate at the boundary. Internal callers shouldn't
    #          send negatives, but this function is reachable from the
    #          API too (via the runner) so we fail loud on bad input.
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError(
            f"Token counts must be non-negative: "
            f"prompt={prompt_tokens}, completion={completion_tokens}"
        )

    price = MODEL_PRICES.get(model)
    if price is None:
        logger.warning(
            "Unknown model %r in cost_usd — returning 0.0. "
            "Add it to MODEL_PRICES to record real cost.",
            model,
        )
        return 0.0

    # Prices are quoted per 1M tokens.
    return (prompt_tokens / 1_000_000) * price.prompt_per_1m + (
        completion_tokens / 1_000_000
    ) * price.completion_per_1m
