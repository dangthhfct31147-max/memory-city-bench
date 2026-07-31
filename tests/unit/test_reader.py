"""Unit tests for the optional reader layer and end-to-end metrics."""

from __future__ import annotations

from memcity.readers.reader import (
    ReaderAnswer,
    ReaderResult,
    build_prompt,
    parse_reader_response,
    OracleReader,
)
from memcity.evaluation.e2e_metrics import exact_match, token_f1, compute_e2e_metrics


class TestJSONParsing:
    def test_clean_json_parses(self):
        raw = '{"answer": "Paris", "evidence_ids": ["ep-1"], "abstained": false, "confidence": 0.9}'
        ans, ok, repaired = parse_reader_response(raw)
        assert ok is True
        assert repaired is False
        assert ans is not None
        assert ans.answer == "Paris"
        assert ans.evidence_ids == ["ep-1"]

    def test_json_with_surrounding_text_is_repaired(self):
        raw = 'Sure! Here is my answer:\n{"answer": "Rome", "evidence_ids": ["ep-2"]}\nHope that helps.'
        ans, ok, repaired = parse_reader_response(raw)
        assert ok is True
        assert repaired is True
        assert ans is not None
        assert ans.answer == "Rome"

    def test_broken_json_fails_cleanly(self):
        raw = "I don't know how to answer this question at all."
        ans, ok, repaired = parse_reader_response(raw)
        assert ok is False
        assert ans is None


class TestPromptBudget:
    def test_prompt_respects_token_budget(self):
        evidence = [{"id": f"ep-{i}", "text": "word " * 100} for i in range(50)]
        prompt = build_prompt("What happened?", evidence, token_budget=300)
        # Should not include all 50 passages given the small budget
        assert prompt.count("[ep-") < 50
        assert "QUESTION: What happened?" in prompt

    def test_prompt_handles_empty_evidence(self):
        prompt = build_prompt("Anything?", [], token_budget=1000)
        assert "(no evidence provided)" in prompt


class TestE2EMetricHelpers:
    def test_exact_match_normalises(self):
        assert exact_match("The Answer.", "the answer") == 1.0
        assert exact_match("no", "yes") == 0.0

    def test_token_f1_partial(self):
        f1 = token_f1("the quick brown fox", "the slow brown fox")
        assert 0.0 < f1 < 1.0

    def test_token_f1_perfect(self):
        assert token_f1("hello world", "hello world") == 1.0


class TestOracleReader:
    def test_oracle_always_grounded(self):
        reader = OracleReader()
        result = reader.answer(
            sample_id="s1",
            query="q",
            evidence=[{"id": "ep-1", "text": "x"}],
            ground_truth_answer="42",
            evidence_ids=["ep-1"],
        )
        assert result.schema_ok is True
        assert result.answer is not None
        assert result.answer.answer == "42"
        assert result.answer.evidence_ids == ["ep-1"]


class _FakeSample:
    def __init__(self, sample_id, answer, category, evidence_ids):
        self.sample_id = sample_id
        self.answer = answer
        self.evidence_episode_ids = evidence_ids

        class _Cat:
            def __init__(self, v):
                self.value = v

        self.question_category = _Cat(category)


class TestComputeE2EMetrics:
    def test_perfect_oracle_scores_high(self):
        samples = [
            _FakeSample("s1", "Paris", "direct_fact", ["ep-1"]),
            _FakeSample("s2", "Rome", "direct_fact", ["ep-2"]),
        ]
        results = [
            ReaderResult(
                sample_id="s1", retriever="oracle", schema_ok=True,
                answer=ReaderAnswer(answer="Paris", evidence_ids=["ep-1"]),
            ).model_dump(),
            ReaderResult(
                sample_id="s2", retriever="oracle", schema_ok=True,
                answer=ReaderAnswer(answer="Rome", evidence_ids=["ep-2"]),
            ).model_dump(),
        ]
        metrics = compute_e2e_metrics(results, samples)
        assert metrics["exact_match"] == 1.0
        assert metrics["json_schema_success_rate"] == 1.0
        assert metrics["grounded_answer_rate"] == 1.0

    def test_abstention_handling(self):
        samples = [_FakeSample("s1", "INSUFFICIENT_EVIDENCE", "abstention", [])]
        results = [
            ReaderResult(
                sample_id="s1", retriever="bm25", schema_ok=True,
                answer=ReaderAnswer(answer="INSUFFICIENT_EVIDENCE", abstained=True),
            ).model_dump(),
        ]
        metrics = compute_e2e_metrics(results, samples)
        assert metrics["abstention_recall"] == 1.0
