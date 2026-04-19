# RAG Evaluation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a custom LLM-as-judge evaluation system that scores RAG answers for faithfulness (real-time), answer relevancy, and context precision (on-demand).

**Architecture:** Three evaluator functions in `src/evaluation.py` each construct a structured prompt, call the judge LLM, and parse JSON scores. Real-time faithfulness fires after streaming completes via WebSocket. On-demand full evaluation is triggered by a REST endpoint. Scores persist in a new `MessageEvaluation` SQLModel table. Frontend shows an inline badge on answers and a detailed evaluation section in the Sources panel.

**Tech Stack:** Python (LLMHandler, SQLModel, FastAPI), React/TypeScript (existing component library)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/config.py` | Add `EVAL_MODEL` constant |
| Create | `src/models/evaluation.py` | `MessageEvaluation` SQLModel table |
| Modify | `src/models/__init__.py` | Register new model for table creation |
| Replace | `src/evaluation.py` | Three evaluator functions (faithfulness, relevancy, precision) |
| Modify | `src/backend.py` | `evaluate_faithfulness_realtime()` and `evaluate_message()` methods |
| Create | `src/api/routes/evaluation.py` | REST endpoints for on-demand evaluation |
| Modify | `src/api/routes/__init__.py` | Register evaluation router |
| Modify | `src/api/routes/query.py` | Fire real-time faithfulness after streaming done event |
| Modify | `src/api/main.py` | Mount evaluation router |
| Create | `tests/test_evaluation.py` | Backend evaluation tests |
| Modify | `frontend/src/api/types.ts` | Add `EvaluationScore`, `WsEvaluationMessage` types |
| Modify | `frontend/src/api/client.ts` | Add `evaluateMessage()` and `getEvaluation()` API functions |
| Modify | `frontend/src/hooks/use-chat.ts` | Handle `evaluation` WebSocket event |
| Create | `frontend/src/components/chat/evaluation-badge.tsx` | Inline faithfulness dot + score |
| Create | `frontend/src/components/chat/evaluation-section.tsx` | Sources panel evaluation detail |
| Modify | `frontend/src/components/chat/chat-message.tsx` | Add EvaluationBadge + Evaluate button |
| Modify | `frontend/src/components/chat/sources-panel.tsx` | Add EvaluationSection above sources |

---

### Task 1: Add EVAL_MODEL config constant

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add the constant**

In `src/config.py`, add after the `REASONING_MODEL` block (after line 116):

```python
# WHY a dedicated evaluation model: Using the same model that generated the
#      answer to judge itself creates self-evaluation bias -- models are less
#      likely to flag their own hallucinations. A mid-tier model like
#      gpt-4.1-mini is cheap enough for real-time faithfulness checks while
#      strong enough to catch factual errors.
EVAL_MODEL: str = "gpt-4.1-mini"
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.config import EVAL_MODEL; print(EVAL_MODEL)"`
Expected: `gpt-4.1-mini`

- [ ] **Step 3: Commit**

```
git add src/config.py
git commit -m "feat(config): add EVAL_MODEL constant for evaluation judge"
```

---

### Task 2: Create MessageEvaluation SQLModel

**Files:**
- Create: `src/models/evaluation.py`
- Modify: `src/models/__init__.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluation.py`:

```python
"""Tests for the RAG evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from sqlmodel import Session, SQLModel, select

from src.database import get_engine
from src.models.evaluation import MessageEvaluation
from src.models.message import Message
from src.models.conversation import Conversation


def _setup_db():
    """Create an in-memory DB with all tables."""
    engine = get_engine("sqlite://")
    import src.models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    return engine


def test_message_evaluation_crud():
    """MessageEvaluation can be created, read, and cascades on message delete."""
    engine = _setup_db()
    with Session(engine) as session:
        conv = Conversation(title="test")
        session.add(conv)
        session.commit()
        session.refresh(conv)

        msg = Message(conversation_id=conv.id, role="assistant", content="test answer")
        session.add(msg)
        session.commit()
        session.refresh(msg)

        evaluation = MessageEvaluation(
            message_id=msg.id,
            metric="faithfulness",
            score=0.85,
            reasoning="All claims supported.",
            details='{"claims": []}',
            judge_model="gpt-4.1-mini",
        )
        session.add(evaluation)
        session.commit()

        result = session.exec(
            select(MessageEvaluation).where(MessageEvaluation.message_id == msg.id)
        ).first()
        assert result is not None
        assert result.metric == "faithfulness"
        assert result.score == 0.85
        assert result.judge_model == "gpt-4.1-mini"

    with Session(engine) as session:
        msg = session.exec(select(Message)).first()
        session.delete(msg)
        session.commit()

        evals = session.exec(select(MessageEvaluation)).all()
        assert len(evals) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluation.py::test_message_evaluation_crud -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.evaluation'`

- [ ] **Step 3: Create the model**

Create `src/models/evaluation.py`:

```python
"""
MessageEvaluation SQLModel table -- LLM-as-judge scores for RAG answers.

RAG Pipeline Position:
  Document -> Chunks -> Embeddings -> Vector Store -> Retrieval -> Generator
                                                                      |
                                                              [MESSAGE] -> [EVALUATION]

What concept it teaches:
  LLM-as-judge evaluation -- using a separate LLM to score the quality of
  generated answers across multiple dimensions (faithfulness, relevancy,
  context precision).

Why this approach over alternatives:
  A dedicated table (vs. a JSON column on Message) keeps evaluation data
  normalized and queryable -- e.g. "show all low-faithfulness answers" or
  "average scores over time" are simple SQL queries.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class MessageEvaluation(SQLModel, table=True):
    """One evaluation metric score for an assistant message.

    Each row stores a single metric (faithfulness, answer_relevancy, or
    context_precision) with the judge's score, reasoning, and optional
    structured details (claim-level or chunk-level breakdown as JSON).
    """

    __tablename__ = "message_evaluations"

    id: Optional[int] = Field(default=None, primary_key=True)

    message_id: str = Field(
        foreign_key="messages.id",
        ondelete="CASCADE",
        index=True,
    )

    metric: str

    score: float

    reasoning: str = Field(default="")

    details: Optional[str] = Field(default=None)

    judge_model: str

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
```

- [ ] **Step 4: Register in models/__init__.py**

Add to `src/models/__init__.py`:

```python
from src.models.evaluation import MessageEvaluation
```

And update `__all__` to include `"MessageEvaluation"`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluation.py::test_message_evaluation_crud -v`
Expected: PASS

- [ ] **Step 6: Commit**

```
git add src/models/evaluation.py src/models/__init__.py tests/test_evaluation.py
git commit -m "feat(models): add MessageEvaluation table for LLM-as-judge scores"
```

---

### Task 3: Implement evaluation functions

**Files:**
- Replace: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests for all three evaluators**

Append to `tests/test_evaluation.py`:

```python
import json
from unittest.mock import MagicMock

from src.evaluation import (
    evaluate_faithfulness,
    evaluate_answer_relevancy,
    evaluate_context_precision,
)


def _mock_llm(response_json: dict) -> MagicMock:
    """Create a mock LLMHandler that returns a fixed JSON string."""
    llm = MagicMock()
    llm.generate.return_value = json.dumps(response_json)
    return llm


def test_evaluate_faithfulness_all_supported():
    llm = _mock_llm({
        "claims": [
            {"claim": "LoRA freezes weights", "supported": True, "evidence": "context says so"},
            {"claim": "LoRA uses low-rank matrices", "supported": True, "evidence": "mentioned"},
        ],
        "score": 1.0,
        "reasoning": "All claims supported.",
    })
    score, reasoning, details = evaluate_faithfulness(
        answer="LoRA freezes weights and uses low-rank matrices.",
        contexts=["LoRA freezes the original weights and adds low-rank matrices."],
        llm=llm,
    )
    assert score == 1.0
    assert details is not None
    parsed = json.loads(details)
    assert len(parsed["claims"]) == 2


def test_evaluate_faithfulness_partial():
    llm = _mock_llm({
        "claims": [
            {"claim": "LoRA freezes weights", "supported": True, "evidence": "yes"},
            {"claim": "LoRA was invented in 2025", "supported": False, "evidence": None},
        ],
        "score": 0.5,
        "reasoning": "One claim unsupported.",
    })
    score, reasoning, details = evaluate_faithfulness(
        answer="LoRA freezes weights. LoRA was invented in 2025.",
        contexts=["LoRA freezes the original weights."],
        llm=llm,
    )
    assert score == 0.5


def test_evaluate_faithfulness_malformed_json():
    llm = MagicMock()
    llm.generate.return_value = "This is not JSON at all"
    score, reasoning, details = evaluate_faithfulness(
        answer="test", contexts=["test"], llm=llm,
    )
    assert score == 0.0
    assert "failed" in reasoning.lower() or "error" in reasoning.lower()
    assert details is None


def test_evaluate_answer_relevancy():
    llm = _mock_llm({
        "score": 0.9,
        "reasoning": "Answer directly addresses the question.",
    })
    score, reasoning = evaluate_answer_relevancy(
        question="What is LoRA?",
        answer="LoRA is a fine-tuning technique.",
        llm=llm,
    )
    assert score == 0.9
    assert len(reasoning) > 0


def test_evaluate_context_precision():
    llm = _mock_llm({
        "chunks": [
            {"chunk_index": 0, "relevant": True},
            {"chunk_index": 1, "relevant": False},
        ],
        "score": 0.5,
        "reasoning": "Only one chunk was relevant.",
    })
    score, reasoning, details = evaluate_context_precision(
        question="What is LoRA?",
        contexts=["LoRA is about low-rank adaptation.", "The weather is nice today."],
        llm=llm,
    )
    assert score == 0.5
    parsed = json.loads(details)
    assert len(parsed["chunks"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluation.py -k "evaluate" -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_faithfulness'`

- [ ] **Step 3: Implement the evaluation module**

Replace `src/evaluation.py` entirely. The module contains:
- `_parse_json_response(text)` -- strips markdown fences, parses JSON
- `evaluate_faithfulness(answer, contexts, llm)` -- claim decomposition + verification
- `evaluate_answer_relevancy(question, answer, llm)` -- directness scoring
- `evaluate_context_precision(question, contexts, llm)` -- per-chunk relevance

Each function constructs a system prompt + user prompt requesting strict JSON, calls `llm.generate()`, and parses the response. On malformed JSON, returns `(0.0, error_message, None)`.

Prompt templates use double-brace escaping for the JSON schema examples (e.g., `{{"claim": "..."}}`) since the prompts are Python f-strings with `.format()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat(evaluation): implement faithfulness, relevancy, and context precision evaluators"
```

---

### Task 4: Add evaluation methods to RAGBackend

**Files:**
- Modify: `src/backend.py`

- [ ] **Step 1: Add imports**

Add to the imports section of `src/backend.py`:

```python
from src.config import EVAL_MODEL
from src.evaluation import (
    evaluate_answer_relevancy,
    evaluate_context_precision,
    evaluate_faithfulness,
)
from src.models.evaluation import MessageEvaluation
```

- [ ] **Step 2: Add eval_llm to __init__**

In `RAGBackend.__init__()`, after `self.reasoning_llm`, add:

```python
self.eval_llm = LLMHandler(model=EVAL_MODEL)
```

- [ ] **Step 3: Add three new methods**

Add to RAGBackend:
- `evaluate_faithfulness_realtime(message_id, answer, contexts)` -- runs faithfulness, persists to MessageEvaluation, returns score dict
- `evaluate_message(message_id)` -- loads message + sources + preceding question, runs all 3 metrics, persists, returns list of score dicts. Skips faithfulness if already exists.
- `get_evaluation(message_id)` -- fetches existing MessageEvaluation rows, returns list of score dicts

- [ ] **Step 4: Verify imports**

Run: `python -c "from src.backend import RAGBackend; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```
git add src/backend.py
git commit -m "feat(backend): add evaluation methods to RAGBackend"
```

---

### Task 5: Create evaluation API routes

**Files:**
- Create: `src/api/routes/evaluation.py`
- Modify: `src/api/routes/__init__.py`
- Modify: `src/api/main.py`

- [ ] **Step 1: Create evaluation router**

Create `src/api/routes/evaluation.py` with two endpoints:
- `POST /api/messages/{message_id}/evaluate` -- calls `backend.evaluate_message()`, returns 404 if no results
- `GET /api/messages/{message_id}/evaluation` -- calls `backend.get_evaluation()`

- [ ] **Step 2: Register in routes __init__.py**

Add `from .evaluation import router as evaluation_router` and update `__all__`.

- [ ] **Step 3: Mount in main.py**

Import `evaluation_router` and add `app.include_router(evaluation_router)`.

- [ ] **Step 4: Verify routes registered**

Run: `python -c "from src.api.main import app; routes = [r.path for r in app.routes]; print([r for r in routes if 'eval' in r])"`
Expected: Two evaluation paths in the list

- [ ] **Step 5: Commit**

```
git add src/api/routes/evaluation.py src/api/routes/__init__.py src/api/main.py
git commit -m "feat(api): add evaluation endpoints for on-demand RAG scoring"
```

---

### Task 6: Wire real-time faithfulness into WebSocket streaming

**Files:**
- Modify: `src/api/routes/query.py`

- [ ] **Step 1: Add accumulation variables**

Before the streaming loop, add:
```python
full_answer_parts = []
retrieved_contexts = []
done_data = {}
```

- [ ] **Step 2: Accumulate answer tokens and contexts during streaming**

In the `token` case, append `data` to `full_answer_parts`.
In the `done` case, capture `data` as `done_data` and extract excerpt strings into `retrieved_contexts`.

- [ ] **Step 3: Fire faithfulness after streaming completes**

After the streaming try/except/finally block, check if `message_id`, `full_answer`, and `retrieved_contexts` are available. If so, run `backend.evaluate_faithfulness_realtime()` via `run_in_executor` and send an `evaluation` WebSocket event with the result. Wrap in try/except so failures don't break the connection.

- [ ] **Step 4: Commit**

```
git add src/api/routes/query.py
git commit -m "feat(streaming): fire real-time faithfulness evaluation after answer completes"
```

---

### Task 7: Frontend types and API client

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Add EvaluationScore interface and WsEvaluationMessage**

In `types.ts`, add the `EvaluationScore` interface (metric, score, reasoning, details?, judge_model?, evaluated_at?). Add `WsEvaluationMessage` type. Update `WsMessage` union to include it. Add `evaluation?: EvaluationScore[]` field to `ChatMessage`.

- [ ] **Step 2: Add API functions**

In `client.ts`, add `evaluateMessage(messageId)` (POST) and `getEvaluation(messageId)` (GET) to the `api` object.

- [ ] **Step 3: Commit**

```
git add frontend/src/api/types.ts frontend/src/api/client.ts
git commit -m "feat(frontend): add evaluation types and API client functions"
```

---

### Task 8: Handle evaluation WebSocket event in use-chat

**Files:**
- Modify: `frontend/src/hooks/use-chat.ts`

- [ ] **Step 1: Add evaluation event handler**

In the `ws.onmessage` handler, add a case for `data.type === "evaluation"` that appends `data.content` to the message's `evaluation` array.

- [ ] **Step 2: Add updateEvaluation function**

Add a `updateEvaluation(messageId, scores)` callback to the hook that sets `evaluation` on the matching message. Return it from the hook.

- [ ] **Step 3: Commit**

```
git add frontend/src/hooks/use-chat.ts
git commit -m "feat(frontend): handle evaluation WebSocket event in useChat hook"
```

---

### Task 9: Create EvaluationBadge component

**Files:**
- Create: `frontend/src/components/chat/evaluation-badge.tsx`

- [ ] **Step 1: Create the component**

Build `EvaluationBadge` with:
- Shows faithfulness score as colored icon (ShieldCheck green >= 0.8, Shield yellow >= 0.5, ShieldAlert red < 0.5) + numeric score
- "Evaluate" button when full evaluation hasn't run (evaluation.length < 3)
- Loading spinner during API call
- Calls `api.evaluateMessage()` and fires `onEvaluationComplete` callback

- [ ] **Step 2: Commit**

```
git add frontend/src/components/chat/evaluation-badge.tsx
git commit -m "feat(frontend): add EvaluationBadge component with inline score and evaluate button"
```

---

### Task 10: Create EvaluationSection component for Sources panel

**Files:**
- Create: `frontend/src/components/chat/evaluation-section.tsx`

- [ ] **Step 1: Create the component**

Build `EvaluationSection` with:
- Header "Evaluation" in uppercase
- Score bars for each metric (colored fill: green >= 0.8, yellow >= 0.5, red < 0.5)
- "View Details" toggle that expands reasoning text per metric
- ClaimBreakdown sub-component for faithfulness (green/red dots per claim)

- [ ] **Step 2: Commit**

```
git add frontend/src/components/chat/evaluation-section.tsx
git commit -m "feat(frontend): add EvaluationSection with score bars and claim breakdown"
```

---

### Task 11: Integrate badge into ChatMessage and evaluation into SourcesPanel

**Files:**
- Modify: `frontend/src/components/chat/chat-message.tsx`
- Modify: `frontend/src/components/chat/sources-panel.tsx`
- Modify: `frontend/src/hooks/use-chat.ts`
- Modify: `frontend/src/pages/chat.tsx`

- [ ] **Step 1: Add EvaluationBadge to ChatMessage**

Import `EvaluationBadge`. Add `onEvaluate` prop to `ChatMessageProps`. Render `EvaluationBadge` below the answer bubble for completed assistant messages, passing `onEvaluationComplete` through to `onEvaluate`.

- [ ] **Step 2: Thread evaluation state through ChatPage**

Destructure `updateEvaluation` from `useChat()`. Pass it down through ChatThread to ChatMessage as `onEvaluate`. Track the latest evaluation scores and pass them to SourcesPanel.

- [ ] **Step 3: Add EvaluationSection to SourcesPanel**

Import `EvaluationSection`. Add `evaluation?: EvaluationScore[]` to `SourcesPanelProps`. Render `EvaluationSection` above the source cards when evaluation data exists.

- [ ] **Step 4: Verify UI renders**

Start dev server, send a chat query, verify:
1. Faithfulness badge appears after answer completes
2. "Evaluate" button triggers full scoring
3. Scores appear in Sources panel with bars and details

- [ ] **Step 5: Commit**

```
git add frontend/src/components/chat/chat-message.tsx frontend/src/components/chat/sources-panel.tsx frontend/src/hooks/use-chat.ts frontend/src/pages/chat.tsx
git commit -m "feat(frontend): integrate evaluation badge and sources panel section"
```

---

### Task 12: Restart backend and end-to-end test

**Files:** None (infrastructure)

- [ ] **Step 1: Restart the backend container**

```
docker restart rag-qa-api-1
```

Wait for healthy: `sleep 5 && docker ps --filter name=rag-qa-api-1 --format "{{.Status}}"`

- [ ] **Step 2: End-to-end test**

1. Open `http://localhost:3000/chat`
2. Send a question about the indexed documents
3. Verify: Thinking streams -> answer streams -> faithfulness badge appears
4. Click "Evaluate" -> loading -> full scores in Sources panel
5. Click "View Details" -> reasoning + claim breakdown expand

- [ ] **Step 3: Run all backend tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Final commit and push**

```
git add -A
git commit -m "feat: complete RAG evaluation pipeline with real-time faithfulness and on-demand scoring"
git push
```
