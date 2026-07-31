"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from memcity.datasets.synthetic import SyntheticDataset


@pytest.fixture
def tiny_samples(tmp_path: Path):
    ds = SyntheticDataset(scale="tiny", seed=42)
    return ds.load(tmp_path)


@pytest.fixture
def tiny_corpus(tiny_samples):
    seen: set[str] = set()
    corpus: list[dict] = []
    for sample in tiny_samples:
        for ep in sample.history:
            if ep.episode_id not in seen:
                seen.add(ep.episode_id)
                corpus.append({
                    "id": ep.episode_id,
                    "node_type": "episode",
                    "text": f"{ep.user_text} {ep.assistant_text}".strip(),
                    "user_text": ep.user_text,
                    "assistant_text": ep.assistant_text,
                    "session_id": ep.session_id,
                    "timestamp": ep.timestamp,
                    "source_episode_ids": [ep.episode_id],
                })
    return corpus
