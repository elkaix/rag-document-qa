"""
SQuAD v2 dev-set sampling and freezing for the eval harness.

Eval Harness Position:
  HuggingFace `squad_v2` → seeded sample of 200 → frozen JSONL artifact
                                                  ↓
                                           runner reads via load_frozen

Design decisions:
  - Sample is FROZEN to JSONL on disk (checked into git) so the dev
    set is byte-identical across machines and runs. The seed is recorded
    separately so anyone can regenerate.
  - Each sampled context becomes ONE chunk_id == question_id. This
    keeps the corpus unit and the gold-chunk unit aligned for SQuAD,
    where every question has exactly one supporting context. The runner
    is responsible for ingesting these contexts into a separate Chroma
    collection (covered in sub-plan 1B).
  - Unanswerable rows preserve the empty-answers / no-gold-chunk shape
    so the refusal metric works downstream.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from src.eval.schemas import EvalQuestion

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_PATH = Path("eval_data/squad_v2_dev_200/questions.jsonl")


def _stable_id(question_text: str, context: str) -> str:
    """Stable hash of (question, context) — used as both question_id and chunk_id."""
    h = hashlib.sha256()
    h.update(question_text.encode("utf-8"))
    h.update(b"\x00")
    h.update(context.encode("utf-8"))
    return h.hexdigest()[:16]


def sample_and_freeze(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[EvalQuestion]:
    """Sample ``sample_size`` rows from squad_v2 dev split and freeze to JSONL.

    Args:
        output_path: Destination JSONL file.
        sample_size: Number of (question, context, answers) tuples.
        seed: PRNG seed for the sample.

    Returns:
        The sampled :class:`EvalQuestion` list (also written to disk).
    """
    # Local import: `datasets` pulls heavy transitive deps; keep
    # `import src.eval.datasets.squad_v2` cheap.
    from datasets import load_dataset

    logger.info("Loading squad_v2 validation split (caches in ~/.cache/huggingface)...")
    ds = load_dataset("squad_v2", split="validation")
    shuffled = ds.shuffle(seed=seed)
    sample = shuffled.select(range(sample_size))

    questions: list[EvalQuestion] = []
    for row in sample:
        question_text: str = row["question"]
        context: str = row["context"]
        answer_texts: list[str] = row["answers"]["text"]
        is_unanswerable = len(answer_texts) == 0

        chunk_id = _stable_id(question_text, context)

        questions.append(
            EvalQuestion(
                id=chunk_id,
                question=question_text,
                gold_answer=None if is_unanswerable else answer_texts[0],
                gold_chunk_ids=[] if is_unanswerable else [chunk_id],
                is_unanswerable=is_unanswerable,
                metadata={
                    "title": row.get("title", ""),
                    # context kept in metadata so the ingestion step can
                    # find the text without re-querying HF.
                    "context": context,
                },
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")

    logger.info(
        "Froze %d SQuAD v2 questions to %s (seed=%d)",
        len(questions),
        output_path,
        seed,
    )
    return questions


def load_frozen(path: Path = DEFAULT_OUTPUT_PATH) -> list[EvalQuestion]:
    """Load a previously-frozen JSONL of EvalQuestion rows.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen SQuAD set not found at {path}. "
            f"Run `python -m src.eval.datasets.squad_v2` to generate it."
        )
    with path.open() as f:
        return [EvalQuestion.model_validate_json(line) for line in f if line.strip()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_and_freeze()
