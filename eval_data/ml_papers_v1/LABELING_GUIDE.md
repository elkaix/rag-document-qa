# ML Papers v1 — Labeling Guide

This document is the authoritative rubric for hand-labeling the 50-question
ML Papers v1 dev set. Every question added to `questions.jsonl` MUST
satisfy these rules. The file itself is part of the eval contract — if
the rubric changes, the dev set version (`v1`) must change.

## Corpus

The corpus is 5–10 ML/AI papers listed in `corpus_manifest.json`. Each
entry pins the exact PDF by SHA-256 so the corpus is byte-stable.

## Question schema

Each row in `questions.jsonl` is a single `EvalQuestion` (Pydantic):

```json
{
  "id": "<sha256(question)[:16]>",
  "question": "Natural-language question.",
  "gold_answer": "Concise reference answer (or null if unanswerable).",
  "gold_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "is_unanswerable": false,
  "metadata": {
    "source_paper": "attention-is-all-you-need",
    "section": "3.2 Multi-Head Attention",
    "difficulty": "definition" | "reasoning" | "multi_hop",
    "labeler": "<initials>"
  }
}
```

`chunk_id` is the SHA-256-prefix ID assigned by the chunker when the
PDF is ingested. To find the right `chunk_id`, ingest the corpus first
(`python -m src.eval.datasets.ml_papers --ingest`) and inspect the
collection.

## What makes a good question

A good question:

- **Has exactly one defensible answer** in natural language. If two
  knowledgeable readers would write different answers, rephrase.
- **Cannot be answered from the question text alone.** "What is
  attention?" is bad — too vague. "In Vaswani et al. 2017, what is the
  scaling factor applied to QK^T before softmax?" is good.
- **Has gold_chunk_ids that *causally* support the answer.** A chunk
  that merely mentions the topic doesn't qualify; the chunk must
  contain the information needed to derive the answer.
- **Has at most 3 gold_chunk_ids.** If you need more, the question is
  too broad — split it.

## Difficulty buckets (target distribution: ~20 / ~20 / ~10)

- **definition** (20 questions) — single-fact lookup. *"What activation
  function is used in the standard Transformer FFN?"*
- **reasoning** (20 questions) — requires understanding within one
  passage. *"Why does scaled dot-product attention divide by √d_k?"*
- **multi_hop** (10 questions) — requires synthesizing across two or
  more chunks (possibly across papers). *"How does ColBERT's late
  interaction differ from DPR's bi-encoder design?"*

## Unanswerable questions (target: 10 of 50)

These must be:
- Plausibly on-topic for the corpus (e.g. an ML question about a model
  the corpus does not cover).
- **Not** trivially off-topic (e.g. "what's the capital of France?"
  doesn't test refusal usefully).
- Marked with `is_unanswerable: true`, `gold_answer: null`,
  `gold_chunk_ids: []`.

Examples:
- *"What batch size does BGE-M3 use during pre-training?"* — BGE-M3
  is not in the v1 corpus → unanswerable.
- *"What learning rate does the LoRA paper recommend for ResNet-50?"*
  — LoRA paper does not cover ResNet → unanswerable.

## Anti-patterns (DO NOT label these)

- Questions whose answer is in the question. ("What is Section 3
  about?" if the chunk header is "Section 3: Attention".)
- Questions requiring outside knowledge not in the corpus.
- Questions with ambiguous wording ("Is X better?" — better at what?).
- Questions where the gold chunk is the question's own paraphrase.

## Workflow

1. Read one paper at a time. Take notes on questions that arise naturally.
2. For each question, find the supporting chunk in the ingested
   collection (the runner exposes a helper).
3. Write the question, gold answer, and gold_chunk_ids to a draft file.
4. Self-review against this guide.
5. Append to `questions.jsonl` (one JSON object per line).
6. After each batch, run `python -m src.eval.datasets.ml_papers --validate`
   to check schema and SHA-stability.

## Versioning

Any change to this guide that affects what counts as a valid question
requires bumping the version (`v1` → `v2`) and creating a new directory.
Do not edit `v1` after first publication.
