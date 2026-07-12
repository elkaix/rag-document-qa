"""Token counting — core telemetry utility.

Owned by the core (``src/telemetry/``) and used by both production
(``LLMHandler`` adapters, as the usage fallback) and the eval harness. It used
to live under ``src/eval/``, which forced production code to import from the eval
package — the dependency this module's move inverts (core never imports eval).

Token Counting:
  This module tries tiktoken first (exact model-aware tokenization) and falls
  back to a word-count heuristic if tiktoken doesn't know the model. The fallback
  ensures nothing hard-fails on a new model release before tiktoken is updated.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# WHY: one-time warning flag for tiktoken fallback — we don't want the
#      warning to spam on every token-count call throughout an eval run.
_tiktoken_warned = False

try:
    import tiktoken as _tiktoken  # type: ignore
except ImportError:
    _tiktoken = None  # type: ignore


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text, falling back to word-count * 1.3 if tiktoken fails.

    WHY the fallback: tiktoken doesn't know every model (new OpenAI releases
    ship before tiktoken is updated). Eval should not hard-fail on a missing
    tokenizer — a ±30% estimate is fine for cost/latency tracking.

    Args:
        text: The text to count tokens for.
        model: Model name used to select the tiktoken encoding.

    Returns:
        Estimated token count (int).
    """
    global _tiktoken_warned
    if _tiktoken is not None:
        try:
            enc = _tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # Unknown model for tiktoken — fall through to word estimate
            pass

    if not _tiktoken_warned:
        logger.warning(
            "tiktoken not installed or model %r unknown — "
            "using word-count × 1.3 for token estimates.",
            model,
        )
        _tiktoken_warned = True
    return int(len(text.split()) * 1.3)
