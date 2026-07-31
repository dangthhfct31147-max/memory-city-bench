"""Adapter for the official LoCoMo ``data/locomo10.json`` release."""

from __future__ import annotations

import json
import re
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

_CATEGORY_MAP = {
    1: QuestionCategory.DIRECT_FACT,
    2: QuestionCategory.TEMPORAL,
    3: QuestionCategory.MULTI_HOP,
    4: QuestionCategory.PROCEDURAL,
    5: QuestionCategory.ABSTENTION,
}


def _timestamp(value: str, fallback: float) -> float:
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC).timestamp()
        except (TypeError, ValueError):
            continue
    return fallback


def _evidence_refs(values: list[Any]) -> list[str]:
    refs: list[str] = []
    for value in values:
        refs.extend(part.strip() for part in re.split(r"[;,]", str(value)) if part.strip())
    return refs


class LoCoMoDataset(BaseDataset):
    """Flatten LoCoMo QA annotations while sharing each conversation history."""

    name = "locomo"

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self._samples: list[QASample] = []
        self._source_path: Path | None = None

    def _candidate_paths(self, data_dir: Path) -> list[Path]:
        return [
            data_dir / "locomo" / "locomo10.json",
            data_dir / "raw" / "locomo" / "locomo10.json",
        ]

    def load(self, data_dir: Path, **_: Any) -> list[QASample]:
        path = next((p for p in self._candidate_paths(data_dir) if p.exists()), None)
        if path is None:
            expected = self._candidate_paths(data_dir)[0]
            raise FileNotFoundError(
                f"LoCoMo not found at {expected}. Run: memcity datasets fetch locomo"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"LoCoMo root must be a list: {path}")

        samples: list[QASample] = []
        for conversation in raw:
            conversation_id = str(conversation.get("sample_id", len(samples)))
            history, evidence_map = self._history(
                conversation_id, conversation.get("conversation", {})
            )
            for question_index, qa in enumerate(conversation.get("qa", [])):
                if self.limit is not None and len(samples) >= self.limit:
                    self._source_path = path
                    self._samples = samples
                    return samples
                refs = _evidence_refs(qa.get("evidence", []))
                evidence_ids = [evidence_map[ref] for ref in refs if ref in evidence_map]
                category_number = int(qa.get("category", 1))
                samples.append(
                    QASample(
                        sample_id=f"locomo:{conversation_id}:q{question_index}",
                        query=str(qa.get("question", "")),
                        answer=str(qa.get("answer", "")),
                        history=history,
                        corpus_scope_id=f"locomo:{conversation_id}",
                        evidence_episode_ids=evidence_ids,
                        category=_CATEGORY_MAP.get(category_number, QuestionCategory.DIRECT_FACT),
                        required_evidence_count=len(evidence_ids),
                        metadata={"category_number": category_number, "evidence_refs": refs},
                    )
                )

        self._source_path = path
        self._samples = samples
        return samples

    def _history(
        self, conversation_id: str, conversation: dict[str, Any]
    ) -> tuple[list[EpisodeTurn], dict[str, str]]:
        history: list[EpisodeTurn] = []
        evidence_map: dict[str, str] = {}
        session_numbers = sorted(
            int(match.group(1))
            for key in conversation
            if (match := re.fullmatch(r"session_(\d+)", key))
        )
        turn_index = 0
        for session_number in session_numbers:
            session_key = f"session_{session_number}"
            date_text = str(conversation.get(f"{session_key}_date_time", ""))
            for turn in conversation.get(session_key, []):
                dialog_id = str(turn.get("dia_id", f"D{session_number}:{turn_index}"))
                episode_id = f"locomo:{conversation_id}:{dialog_id}"
                text = str(turn.get("text") or turn.get("blip_caption") or "")
                history.append(
                    EpisodeTurn(
                        episode_id=episode_id,
                        session_id=f"{conversation_id}:{session_key}",
                        turn_index=turn_index,
                        user_text=text,
                        assistant_text="",
                        timestamp=_timestamp(date_text, 1_700_000_000.0 + turn_index),
                        metadata={
                            "speaker": turn.get("speaker", ""),
                            "dialog_id": dialog_id,
                            "source_date": date_text,
                        },
                    )
                )
                evidence_map[dialog_id] = episode_id
                turn_index += 1
        return history, evidence_map

    def manifest(self) -> DatasetManifest:
        categories = Counter(sample.category.value for sample in self._samples)
        source_hash = (
            stable_hash(self._source_path.read_text(encoding="utf-8")) if self._source_path else ""
        )
        return DatasetManifest(
            name="locomo",
            version="locomo10",
            description="Official ten-conversation LoCoMo release",
            scale="official",
            sample_count=len(self._samples),
            category_counts=dict(categories),
            source_hash=source_hash,
            provenance="https://github.com/snap-research/locomo/blob/main/data/locomo10.json",
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
        return errors
