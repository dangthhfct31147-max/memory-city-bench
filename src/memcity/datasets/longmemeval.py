"""Adapter for the official cleaned LongMemEval release."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memcity.datasets.protocol import (
    BaseDataset,
    DatasetManifest,
    EpisodeTurn,
    QASample,
    QuestionCategory,
)
from memcity.utils.helpers import stable_hash

_CATEGORY_MAP: dict[str, QuestionCategory] = {
    "single-session-user": QuestionCategory.DIRECT_FACT,
    "single-session-assistant": QuestionCategory.DIRECT_FACT,
    "single-session-preference": QuestionCategory.DIRECT_FACT,
    "multi-session": QuestionCategory.MULTI_SESSION,
    "knowledge-update": QuestionCategory.FACT_UPDATE,
    "temporal-reasoning": QuestionCategory.TEMPORAL,
}


def _timestamp(value: str, fallback: float) -> float:
    try:
        dt = datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=UTC)
        return dt.timestamp()
    except (TypeError, ValueError):
        return fallback


class LongMemEvalDataset(BaseDataset):
    """Load LongMemEval while preserving its session-level evidence labels."""

    name = "longmemeval"

    def __init__(self, variant: str = "s", limit: int | None = None) -> None:
        self.variant = variant.removesuffix("_cleaned")
        self.limit = limit
        self._samples: list[QASample] = []
        self._source_path: Path | None = None

    def _candidate_paths(self, data_dir: Path) -> list[Path]:
        stem = f"longmemeval_{self.variant}"
        return [
            data_dir / "longmemeval" / f"{stem}_cleaned.json",
            data_dir / "longmemeval" / f"{stem}.json",
            data_dir / "raw" / "longmemeval" / f"{stem}_cleaned.json",
        ]

    def load(self, data_dir: Path, **_: Any) -> list[QASample]:
        path = next((p for p in self._candidate_paths(data_dir) if p.exists()), None)
        if path is None:
            expected = self._candidate_paths(data_dir)[0]
            raise FileNotFoundError(
                f"LongMemEval '{self.variant}' not found at {expected}. "
                f"Run: memcity datasets fetch longmemeval --variant {self.variant}"
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"LongMemEval root must be a list: {path}")
        if self.limit is not None:
            raw = raw[: self.limit]

        self._source_path = path
        self._samples = [self._convert(entry) for entry in raw]
        return self._samples

    def _convert(self, entry: dict[str, Any]) -> QASample:
        question_id = str(entry.get("question_id", stable_hash(str(entry))))
        sessions = entry.get("haystack_sessions") or []
        session_ids = entry.get("haystack_session_ids") or []
        dates = entry.get("haystack_dates") or []
        if not (len(sessions) == len(session_ids) == len(dates)):
            raise ValueError(f"{question_id}: session, id, and date counts differ")

        history: list[EpisodeTurn] = []
        session_to_episode: dict[str, str] = {}
        for index, (session, session_id, date_text) in enumerate(
            zip(sessions, session_ids, dates, strict=True)
        ):
            episode_id = f"lme:{session_id}"
            session_to_episode[str(session_id)] = episode_id
            user_parts: list[str] = []
            assistant_parts: list[str] = []
            for message in session:
                role = str(message.get("role", "")).lower()
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                if role == "assistant":
                    assistant_parts.append(content)
                else:
                    user_parts.append(content)
            history.append(
                EpisodeTurn(
                    episode_id=episode_id,
                    session_id=str(session_id),
                    turn_index=index,
                    user_text="\n".join(user_parts),
                    assistant_text="\n".join(assistant_parts),
                    timestamp=_timestamp(str(date_text), 1_700_000_000.0 + index),
                    metadata={"source_session_id": str(session_id), "source_date": date_text},
                )
            )

        evidence_ids = [
            session_to_episode[str(session_id)]
            for session_id in entry.get("answer_session_ids", [])
            if str(session_id) in session_to_episode
        ]
        category = (
            QuestionCategory.ABSTENTION
            if question_id.endswith("_abs")
            else _CATEGORY_MAP.get(
                str(entry.get("question_type", "")), QuestionCategory.DIRECT_FACT
            )
        )
        return QASample(
            sample_id=f"lme:{question_id}",
            query=str(entry.get("question", "")),
            answer=str(entry.get("answer", "")),
            history=history,
            corpus_scope_id=f"lme:{question_id}",
            evidence_episode_ids=evidence_ids,
            category=category,
            timestamp=_timestamp(str(entry.get("question_date", "")), 0.0),
            required_evidence_count=len(evidence_ids),
            metadata={
                "question_type": entry.get("question_type", ""),
                "answer_session_ids": entry.get("answer_session_ids", []),
            },
        )

    def manifest(self) -> DatasetManifest:
        categories = Counter(sample.category.value for sample in self._samples)
        source_hash = (
            stable_hash(self._source_path.read_text(encoding="utf-8")) if self._source_path else ""
        )
        return DatasetManifest(
            name=f"longmemeval-{self.variant}",
            version="cleaned",
            description="Official cleaned LongMemEval release",
            scale=self.variant,
            sample_count=len(self._samples),
            category_counts=dict(categories),
            source_hash=source_hash,
            provenance="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        )

    def validate(self, samples: list[QASample]) -> list[str]:
        errors: list[str] = []
        for sample in samples:
            episode_ids = {episode.episode_id for episode in sample.history}
            if not sample.query.strip():
                errors.append(f"{sample.sample_id}: empty query")
            if not sample.history:
                errors.append(f"{sample.sample_id}: no history")
            missing = set(sample.evidence_episode_ids) - episode_ids
            if missing:
                errors.append(
                    f"{sample.sample_id}: evidence missing from history: {sorted(missing)}"
                )
            if sample.category != QuestionCategory.ABSTENTION and not sample.evidence_episode_ids:
                errors.append(f"{sample.sample_id}: no evidence session")
        return errors
