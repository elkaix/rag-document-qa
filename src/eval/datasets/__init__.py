"""Eval dataset loaders.

Each loader returns an iterable of EvalQuestion objects and freezes
its sample to eval_data/<dataset_name>/questions.jsonl for reproducibility.
"""

from __future__ import annotations
