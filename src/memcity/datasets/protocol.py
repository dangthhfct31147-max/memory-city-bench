"""Dataset protocol and sample schema."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    DIRECT_FACT = "direct_fact"
    PARAPHRASE = "paraphrase"
    MULTI_SESSION = "multi_session"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    FACT_UPDATE = "fact_update"
    CONTRADICTION = "contradiction"
    ABSTENTION = "abstention"
    GLOBAL_COMMUNITY = "global_community"
    PROCEDURAL = "procedural"
    DISTRACTOR = "distractor"


class EpisodeTurn(BaseModel):
    episode_id: str
    session_id: str
    turn_index: int
    user_text: str
    assistant_text: str
    timestamp: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QASample(BaseModel):
    sample_id: str
    query: str
    answer: str
    history: list[EpisodeTurn]
    evidence_episode_ids: list[str] = Field(default_factory=list)
    evidence_node_ids: list[str] = Field(default_factory=list)
    category: QuestionCategory = QuestionCategory.DIRECT_FACT
    timestamp: float = 0.0
    expected_current_fact: str = ""
    stale_fact_ids: list[str] = Field(default_factory=list)
    required_evidence_count: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    scale: str = "tiny"
    seed: int = 42
    sample_count: int = 0
    category_counts: dict[str, int] = Field(default_factory=dict)
    source_hash: str = ""
    provenance: str = ""
    splits: dict[str, list[str]] = Field(default_factory=dict)


class BaseDataset(ABC):
    name: str = "base"

    @abstractmethod
    def load(self, data_dir: Path, **kwargs) -> list[QASample]: ...

    @abstractmethod
    def manifest(self) -> DatasetManifest: ...

    def validate(self, samples: list[QASample]) -> list[str]:
        """Return a list of validation errors."""
        errors = []
        for s in samples:
            if not s.evidence_episode_ids:
                errors.append(f"{s.sample_id}: no evidence_episode_ids")
            if not s.query.strip():
                errors.append(f"{s.sample_id}: empty query")
        return errors
