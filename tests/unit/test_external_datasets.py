"""Regression tests for official external dataset shapes."""

from __future__ import annotations

import json

from memcity.datasets.locomo import LoCoMoDataset
from memcity.datasets.longmemeval import LongMemEvalDataset
from memcity.datasets.protocol import QuestionCategory


def test_longmemeval_cleaned_session_evidence(tmp_path):
    root = tmp_path / "longmemeval"
    root.mkdir()
    payload = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "What happened?",
            "question_date": "2023/05/30 (Tue) 23:40",
            "answer": "A result",
            "answer_session_ids": ["answer-session"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_session_ids": ["answer-session"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Remember this"},
                    {"role": "assistant", "content": "A result"},
                ]
            ],
        }
    ]
    (root / "longmemeval_s_cleaned.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = LongMemEvalDataset(limit=1)
    samples = dataset.load(tmp_path)

    assert dataset.validate(samples) == []
    assert samples[0].evidence_episode_ids == ["lme:answer-session"]
    assert samples[0].history[0].turn_index == 0
    assert samples[0].category == QuestionCategory.MULTI_SESSION


def test_locomo_official_file_and_compound_evidence(tmp_path):
    root = tmp_path / "locomo"
    root.mkdir()
    payload = [
        {
            "sample_id": "conv-1",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "1:00 PM on 7 May, 2023",
                "session_1": [
                    {"speaker": "A", "dia_id": "D1:1", "text": "First fact"},
                    {"speaker": "B", "dia_id": "D1:2", "text": "Second fact"},
                ],
            },
            "qa": [
                {
                    "question": "Combine them",
                    "answer": "Both facts",
                    "evidence": ["D1:1; D1:2"],
                    "category": 3,
                }
            ],
        }
    ]
    (root / "locomo10.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = LoCoMoDataset()
    samples = dataset.load(tmp_path)

    assert dataset.validate(samples) == []
    assert samples[0].evidence_episode_ids == ["locomo:conv-1:D1:1", "locomo:conv-1:D1:2"]
    assert samples[0].category == QuestionCategory.MULTI_HOP
