"""Tests for src.eval.pipeline_factory."""

from __future__ import annotations

import json

from src.eval.config import EvalConfig
from src.eval.pipeline_factory import EvalPipeline, build_pipeline
from src.eval.schemas import EvalQuestion


class DummyLLM:
    """Returns a fixed answer; tracks calls."""
    def __init__(self, answer: str = "<dummy>"):
        self.answer = answer
        self.model = "gpt-4.1-nano"  # engine reads .model for spans + cost pricing
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return self.answer

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, int, int]:
        self.calls.append((prompt, system_prompt))
        return self.answer, max(1, len(prompt.split())), len(self.answer.split())


def _baseline_config() -> EvalConfig:
    return EvalConfig.model_validate({
        "name": "test",
        "description": "",
        "pipeline": {
            "chunker": {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
            "retriever": {"top_k": 3},
            "generator": {"model": "gpt-4.1-nano", "reasoning_model": None},
        },
        "eval": {
            "datasets": ["squad_v2_dev_200"],
            "judge_model": "gpt-4.1-nano",
            "bootstrap_n": 100, "permutation_n": 100, "seed": 7,
        },
    })


def _squad_question(qid: str, ctx: str, q: str = "Q?") -> EvalQuestion:
    return EvalQuestion(
        id=qid, question=q, gold_answer="A", gold_chunk_ids=[qid],
        metadata={"context": ctx, "title": "t"},
    )


class TestBuildPipeline:
    def test_returns_pipeline_with_components(self):
        cfg = _baseline_config()
        p = build_pipeline(cfg, "squad_v2_dev_200",
                           llm_override=DummyLLM("answer"),
                           judge_llm_override=DummyLLM("{}"))
        assert isinstance(p, EvalPipeline)
        assert p.config is cfg
        assert p.dataset_name == "squad_v2_dev_200"
        p.teardown()


class TestIngestAndQuery:
    def test_squad_ingest_then_query(self):
        cfg = _baseline_config()
        p = build_pipeline(cfg, "squad_v2_dev_200",
                           llm_override=DummyLLM("Paris"),
                           judge_llm_override=DummyLLM("{}"))
        try:
            qs = [
                _squad_question("q1", "Paris is the capital of France.", "What is the capital of France?"),
                _squad_question("q2", "The Eiffel Tower is in Paris.", "Where is the Eiffel Tower?"),
            ]
            p.ingest(qs)

            chunks, answer, telemetry = p.query("What is the capital of France?")
            assert isinstance(chunks, list)
            assert len(chunks) >= 1
            assert answer == "Paris"
            assert "timings_ms" in telemetry
            assert "tokens" in telemetry
            assert "cost_usd" in telemetry
            assert telemetry["timings_ms"]["retrieve"] >= 0.0
            assert telemetry["timings_ms"]["generate"] >= 0.0
            assert telemetry["tokens"]["prompt"] > 0
            assert telemetry["tokens"]["completion"] >= 0
            assert telemetry["cost_usd"] >= 0.0
        finally:
            p.teardown()


class TestTeardown:
    def test_teardown_does_not_raise(self):
        cfg = _baseline_config()
        p = build_pipeline(cfg, "squad_v2_dev_200",
                           llm_override=DummyLLM(),
                           judge_llm_override=DummyLLM())
        p.teardown()  # should not raise


class TestMLPapersIngest:
    """The 58-line ML-papers branch, reachable now that the path is injectable.

    The manifest path used to be a literal inside _ingest_ml_papers, so the only
    branch a test could reach was the missing-manifest no-op.
    """

    def _pipeline(self, tmp_path, manifest, **kw):
        import uuid

        import chromadb

        from src.eval.pipeline_factory import EvalPipeline
        from src.ingestion import TextChunker
        from src.vector_store import ChromaVectorStore

        # WHY a unique name: EphemeralClient shares one in-process store, so a
        # fixed name would leak chunks between tests in this class.
        store = ChromaVectorStore.open(
            chromadb.EphemeralClient(), f"ml_papers_{uuid.uuid4().hex}"
        )
        return EvalPipeline(
            chunker=TextChunker(chunk_size=128, chunk_overlap=16),
            vector_store=store,
            llm=None,
            judge_llm=None,
            config=_baseline_config(),
            dataset_name="ml_papers_v1",
            ml_papers_manifest=manifest,
            **kw,
        )

    def test_missing_manifest_is_a_no_op(self, tmp_path):
        pipeline = self._pipeline(tmp_path, tmp_path / "absent.json")
        pipeline._ingest_ml_papers()
        assert pipeline.vector_store.get_stats()["total_chunks"] == 0

    def test_empty_manifest_is_a_no_op(self, tmp_path):
        manifest = tmp_path / "corpus_manifest.json"
        manifest.write_text(json.dumps({"papers": []}))
        pipeline = self._pipeline(tmp_path, manifest)
        pipeline._ingest_ml_papers()
        assert pipeline.vector_store.get_stats()["total_chunks"] == 0

    def test_a_listed_paper_is_chunked_and_upserted(self, tmp_path):
        paper = tmp_path / "paper.txt"
        paper.write_text(
            "Retrieval augmented generation combines a retriever with a generator. "
            * 20
        )
        manifest = tmp_path / "corpus_manifest.json"
        manifest.write_text(
            json.dumps({"papers": [{"id": "p1", "local_path": str(paper)}]})
        )

        pipeline = self._pipeline(tmp_path, manifest)
        pipeline._ingest_ml_papers()

        assert pipeline.vector_store.get_stats()["total_chunks"] > 0

    def test_a_missing_paper_file_does_not_abort_the_run(self, tmp_path):
        manifest = tmp_path / "corpus_manifest.json"
        manifest.write_text(
            json.dumps({"papers": [{"id": "gone", "local_path": str(tmp_path / "nope.pdf")}]})
        )
        pipeline = self._pipeline(tmp_path, manifest)
        pipeline._ingest_ml_papers()
        assert pipeline.vector_store.get_stats()["total_chunks"] == 0
